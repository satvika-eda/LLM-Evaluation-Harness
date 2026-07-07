"""
RAGAS scorer — faithfulness and answer_relevance.

Uses GPT-4o as the judge LLM via LangChain's OpenAI wrapper.

Metric names written to the DB
-------------------------------
    ragas/faithfulness       (skipped when context is empty, e.g. TruthfulQA —
                              faithfulness grades the answer's claims against
                              the retrieved context, which is undefined
                              without one)
    ragas/answer_relevance

context_recall was removed deliberately: it compares the ground truth against
the retrieved context and never looks at the model's answer, so with a fixed
dataset every model gets an identical score — it measures the dataset, not
the model. It only becomes meaningful with a real retrieval pipeline where
contexts vary per run.

Failed or NaN judge scores are omitted from the result dict (never coerced
to 0.0), so they can't poison AVG() aggregates in the DB.

Notes on RAGAS dataset schema (>=0.2)
--------------------------------------
    SingleTurnSample fields:
        user_input          str          — the question
        response            str          — model's answer
        retrieved_contexts  list[str]    — supporting passages
        reference           str          — ground-truth answer

    evaluate() returns an EvaluationResult whose .scores attribute is
    a list of per-sample dicts keyed by metric name.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from typing import Any

from sqlalchemy.orm import Session

from src.scorers import ScoringInput

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "ragas"


def _install_vertexai_shim() -> None:
    """Compatibility shim for ragas importing a removed langchain module.

    ``ragas.llms.base`` hard-imports ``ChatVertexAI`` from
    ``langchain_community.chat_models.vertexai`` — a submodule that
    langchain-community >=0.3 removed (Vertex AI moved to the standalone
    ``langchain-google-vertexai`` package). This harness only uses ChatOpenAI,
    and ragas references ``ChatVertexAI`` merely as a type in its
    "supports multiple completions" detection tuple — never instantiating it.

    Registering a stub module lets ``import ragas`` succeed without pinning an
    end-of-life langchain-community. If the real standalone package is present,
    its class is used instead of the stub.
    """
    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name in sys.modules:
        return
    try:
        import langchain_community.chat_models as _chat_models
    except Exception:
        return
    if hasattr(_chat_models, "vertexai"):
        return
    try:
        from langchain_google_vertexai import ChatVertexAI  # type: ignore
    except Exception:
        class ChatVertexAI:  # minimal stub; never instantiated in this harness
            pass
    shim = types.ModuleType(mod_name)
    shim.ChatVertexAI = ChatVertexAI
    sys.modules[mod_name] = shim
    _chat_models.vertexai = shim  # type: ignore[attr-defined]


_install_vertexai_shim()


class RAGASScorer:
    """Wraps RAGAS evaluate() and persists scores to PostgreSQL."""

    def __init__(self, openai_api_key: str | None = None) -> None:
        self._api_key = openai_api_key or os.environ["OPENAI_API_KEY"]
        # Heavy imports deferred to first use so module loads are cheap.
        self._llm: Any = None
        self._embeddings: Any = None

    # ── Lazy setup ────────────────────────────────────────────────────────────

    def _get_llm(self) -> Any:
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper

            from src.scorers import judge_config

            cfg = judge_config(self._api_key)
            self._llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model=cfg["model"],
                    api_key=cfg["api_key"],
                    base_url=cfg["base_url"],
                    temperature=0,
                )
            )
        return self._llm

    def _get_embeddings(self) -> Any:
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper

            self._embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model="text-embedding-3-small", api_key=self._api_key)
            )
        return self._embeddings

    # ── Core scoring (sync, runs in executor) ─────────────────────────────────

    def _score_sync(self, inputs: list[ScoringInput]) -> list[dict[str, float]]:
        """
        Run RAGAS evaluate() synchronously over a batch of inputs.

        Returns a list parallel to `inputs`, each element being a dict of
        {metric_name: score}. Metrics that are inapplicable (faithfulness
        without context) or whose judge call failed (NaN) are absent from
        the dict rather than coerced to 0.
        """
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from ragas.run_config import RunConfig

        from src.scorers import judge_max_concurrency

        llm = self._get_llm()
        embeddings = self._get_embeddings()

        # Cap RAGAS's internal parallelism so bursts of judge calls stay under
        # the OpenAI tokens-per-minute ceiling (default max_workers is 16).
        run_config = RunConfig(max_workers=judge_max_concurrency())

        faithfulness_metric = Faithfulness(llm=llm)
        relevancy_metric = ResponseRelevancy(llm=llm, embeddings=embeddings)

        # Split inputs into those with / without context. Faithfulness is only
        # defined against a non-empty context: with an empty one, no claim can
        # ever be supported, so every answer would score ~0 regardless of
        # quality. No-context rows get answer_relevance only.
        with_context: list[tuple[int, ScoringInput]] = []
        no_context: list[tuple[int, ScoringInput]] = []
        for idx, inp in enumerate(inputs):
            (with_context if inp.context.strip() else no_context).append((idx, inp))

        results: list[dict[str, float]] = [{} for _ in inputs]

        # ── Batch with context (faithfulness + relevancy) ─────────────────────
        if with_context:
            samples = [
                SingleTurnSample(
                    user_input=inp.question,
                    response=inp.response_text,
                    retrieved_contexts=[inp.context],
                    reference=inp.ground_truth,
                )
                for _, inp in with_context
            ]
            dataset = EvaluationDataset(samples=samples)
            result = evaluate(
                dataset,
                metrics=[faithfulness_metric, relevancy_metric],
                run_config=run_config,
            )
            for batch_pos, (orig_idx, _) in enumerate(with_context):
                row: dict[str, Any] = result.scores[batch_pos]
                results[orig_idx] = _clean_metric_map(
                    {
                        f"{_METRIC_PREFIX}/faithfulness":     row.get("faithfulness"),
                        f"{_METRIC_PREFIX}/answer_relevance": row.get("answer_relevancy"),
                    }
                )

        # ── Batch without context (relevancy only) ────────────────────────────
        if no_context:
            samples = [
                SingleTurnSample(
                    user_input=inp.question,
                    response=inp.response_text,
                    retrieved_contexts=[""],  # required field; unused by relevancy
                    reference=inp.ground_truth,
                )
                for _, inp in no_context
            ]
            dataset = EvaluationDataset(samples=samples)
            result = evaluate(
                dataset,
                metrics=[relevancy_metric],
                run_config=run_config,
            )
            for batch_pos, (orig_idx, _) in enumerate(no_context):
                row = result.scores[batch_pos]
                results[orig_idx] = _clean_metric_map(
                    {f"{_METRIC_PREFIX}/answer_relevance": row.get("answer_relevancy")}
                )

        return results

    # ── Public async interface ────────────────────────────────────────────────

    async def score(
        self,
        inputs: list[ScoringInput],
        session: Session,
    ) -> list[dict[str, Any]]:
        """
        Score a batch of responses with RAGAS and persist results to DB.

        Parameters
        ----------
        inputs  : list of ScoringInput (one per model response)
        session : active SQLAlchemy Session; caller commits

        Returns
        -------
        list of dicts, each with keys response_id, metric_name, score.
        """
        if not inputs:
            return []

        logger.info("RAGASScorer: scoring %d responses…", len(inputs))

        try:
            metric_maps: list[dict[str, float]] = await asyncio.to_thread(
                self._score_sync, inputs
            )
        except Exception as exc:
            logger.error("RAGASScorer batch failed: %s", exc, exc_info=True)
            return []

        return _persist_scores(metric_maps, inputs, session, "RAGASScorer")


# ── Shared helpers (used by all three scorer modules) ─────────────────────────

def _clean_metric_map(raw: dict[str, Any]) -> dict[str, float]:
    """
    Drop missing/NaN judge scores instead of coercing them to numbers.

    RAGAS returns NaN for samples where the judge call or output parsing
    failed. A NaN stored in Postgres poisons any AVG() over the column, and
    coercing a missing value to 0.0 is indistinguishable from a genuine
    zero — both silently corrupt aggregates. Omitting the metric leaves an
    honest gap that per-metric COUNT() makes visible.
    """
    import math

    cleaned: dict[str, float] = {}
    for name, val in raw.items():
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            logger.warning("Non-numeric judge score for %s: %r — dropped.", name, val)
            continue
        if math.isnan(f):
            logger.warning("NaN judge score for %s — dropped.", name)
            continue
        cleaned[name] = f
    return cleaned


def _persist_scores(
    metric_maps: list[dict[str, float]],
    inputs: list[ScoringInput],
    session: Session,
    scorer_label: str,
) -> list[dict[str, Any]]:
    """
    Bulk-insert Score rows and return the inserted dicts.

    Imported and reused by deepeval_scorer and bert_scorer to avoid
    duplicating the DB write logic.

    Skips (response_id, metric_name) pairs that already exist so a resumed
    run never double-counts previously committed chunks — the scores table
    also enforces this with a unique constraint, but skipping here keeps
    resume from aborting the whole flush.
    """
    from src.db import Score

    response_ids = {inp.response_id for inp in inputs}
    existing: set[tuple[int, str]] = set(
        session.query(Score.response_id, Score.metric_name)
        .filter(Score.response_id.in_(response_ids))
        .all()
    )

    records: list[dict[str, Any]] = []
    skipped = 0
    for inp, metric_map in zip(inputs, metric_maps):
        for metric_name, score_val in metric_map.items():
            if (inp.response_id, metric_name) in existing:
                skipped += 1
                continue
            row = Score(
                response_id=inp.response_id,
                metric_name=metric_name,
                score=score_val,
            )
            session.add(row)
            records.append(
                {
                    "response_id": inp.response_id,
                    "metric_name": metric_name,
                    "score": score_val,
                }
            )

    session.flush()
    if skipped:
        logger.info(
            "%s: skipped %d already-persisted scores (resume).", scorer_label, skipped
        )
    logger.debug("%s: persisted %d score rows.", scorer_label, len(records))
    return records
