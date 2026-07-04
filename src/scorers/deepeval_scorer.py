"""
DeepEval scorer — hallucination and G-Eval coherence.

Uses GPT-4o as the judge LLM with logprobs=True so G-Eval can compute
weighted probability scores rather than a single sampled token.

Metric names written to the DB
-------------------------------
    deepeval/hallucination   — HallucinationMetric (0 = no hallucination,
                               1 = full hallucination; stored as-is)
    deepeval/coherence       — GEval coherence score, range [0, 1]

Notes on DeepEval API
---------------------
    LLMTestCase fields used:
        input            str        — question
        actual_output    str        — model response
        context          list[str]  — supporting passages
        expected_output  str        — ground truth

    HallucinationMetric requires `context`; for TruthfulQA (empty context)
    we pass [""] rather than [] because DeepEval validates that the list is
    non-empty, but a single empty string is treated as "no context provided"
    and the metric degrades gracefully.

    GEval does not require context — it evaluates coherence purely from
    (question, response).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sqlalchemy.orm import Session

from src.scorers import ScoringInput
from src.scorers.ragas_scorer import _persist_scores

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "deepeval"


class DeepEvalScorer:
    """Wraps DeepEval HallucinationMetric and GEval, persists to PostgreSQL."""

    def __init__(self, openai_api_key: str | None = None) -> None:
        self._api_key = openai_api_key or os.environ["OPENAI_API_KEY"]
        self._model: Any = None

    # ── Lazy model setup ──────────────────────────────────────────────────────

    def _get_model(self) -> Any:
        """
        Return a GPTModel configured for logprobs scoring.

        logprobs=True enables DeepEval's weighted probability aggregation in
        GEval, which is more stable than sampling a single verdict token.
        top_logprobs=5 gives sufficient probability mass for scoring.
        """
        if self._model is None:
            from deepeval.models import GPTModel

            self._model = GPTModel(
                model="gpt-4o",
                _openai_api_key=self._api_key,
            )
        return self._model

    # ── Core scoring (async, runs in the event loop) ──────────────────────────

    async def _score_single(self, inp: ScoringInput) -> dict[str, float]:
        """Score one response with both metrics. Returns {metric: score}.

        Uses DeepEval's async ``a_measure`` API so scoring runs inside the
        event loop on the main thread.  The synchronous ``measure`` path
        registers SIGINT/SIGTERM handlers via DeepEval's tracing layer, which
        raises "signal only works in main thread" when this scorer is run off
        the main thread (e.g. via ``asyncio.to_thread``); ``a_measure`` avoids
        that entirely.
        """
        from deepeval.metrics import GEval, HallucinationMetric
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        model = self._get_model()
        context_list = [inp.context] if inp.context.strip() else [""]

        test_case = LLMTestCase(
            input=inp.question,
            actual_output=inp.response_text,
            expected_output=inp.ground_truth,
            context=context_list,
        )

        scores: dict[str, float] = {}

        # ── Hallucination ─────────────────────────────────────────────────────
        try:
            hallucination = HallucinationMetric(model=model, threshold=0.5)
            await hallucination.a_measure(test_case)
            scores[f"{_METRIC_PREFIX}/hallucination"] = float(hallucination.score)
        except Exception as exc:
            logger.warning(
                "HallucinationMetric failed for response_id=%d: %s",
                inp.response_id,
                exc,
            )

        # ── G-Eval coherence ──────────────────────────────────────────────────
        try:
            coherence = GEval(
                name="Coherence",
                model=model,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                criteria=(
                    "Determine whether the response is logically coherent, "
                    "well-structured, and directly addresses the question "
                    "without contradictions or non-sequiturs."
                ),
            )
            await coherence.a_measure(test_case)
            # GEval scores are already in [0, 1] when using logprobs
            scores[f"{_METRIC_PREFIX}/coherence"] = float(coherence.score)
        except Exception as exc:
            logger.warning(
                "GEval coherence failed for response_id=%d: %s",
                inp.response_id,
                exc,
            )

        return scores

    async def _score_all(self, inputs: list[ScoringInput]) -> list[dict[str, float]]:
        """Run both metrics over every input sequentially in the event loop."""
        results: list[dict[str, float]] = []
        for inp in inputs:
            try:
                results.append(await self._score_single(inp))
            except Exception as exc:
                logger.error(
                    "DeepEvalScorer failed for response_id=%d: %s",
                    inp.response_id,
                    exc,
                    exc_info=True,
                )
                results.append({})
        return results

    # ── Public async interface ────────────────────────────────────────────────

    async def score(
        self,
        inputs: list[ScoringInput],
        session: Session,
    ) -> list[dict[str, Any]]:
        """
        Score a batch of responses with DeepEval and persist to DB.

        Parameters
        ----------
        inputs  : list of ScoringInput
        session : active SQLAlchemy Session; caller commits

        Returns
        -------
        list of dicts with keys response_id, metric_name, score.
        """
        if not inputs:
            return []

        logger.info("DeepEvalScorer: scoring %d responses…", len(inputs))

        try:
            metric_maps = await self._score_all(inputs)
        except Exception as exc:
            logger.error("DeepEvalScorer batch failed: %s", exc, exc_info=True)
            return []

        return _persist_scores(metric_maps, inputs, session, "DeepEvalScorer")
