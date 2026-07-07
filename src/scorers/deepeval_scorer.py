"""
DeepEval scorer — hallucination and G-Eval coherence.

Uses a configurable judge LLM (see ``src.scorers.judge_config``).

Metric names written to the DB
-------------------------------
    deepeval/hallucination   — HallucinationMetric (0 = no hallucination,
                               1 = full hallucination; stored as-is).
                               NOTE: inverted polarity vs every other metric
                               (lower is better). Skipped when the question
                               has no retrieval context (e.g. TruthfulQA) —
                               the metric asks whether the output contradicts
                               the provided passages, which is undefined
                               against an empty one.
    deepeval/coherence       — GEval coherence score, range [0, 1].
                               Does not require context; always computed.

Notes on DeepEval API
---------------------
    LLMTestCase fields used:
        input            str        — question
        actual_output    str        — model response
        context          list[str]  — supporting passages (only set when
                                      the dataset provides them)
        expected_output  str        — ground truth
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from sqlalchemy.orm import Session

from src.scorers import ScoringInput, judge_max_concurrency
from src.scorers.ragas_scorer import _clean_metric_map, _persist_scores

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
        Return the judge model used by HallucinationMetric and GEval.

        The judge (model id, base URL, API key) is configured via environment
        variables — see ``src.scorers.judge_config`` — so it can be swapped for
        a cheaper / higher-rate-limit model (e.g. ``gpt-4o-mini``) or an
        OpenAI-compatible endpoint without touching this code. Defaults to
        GPT-4o via OpenAI.
        """
        if self._model is None:
            from deepeval.models import GPTModel

            from src.scorers import judge_config

            cfg = judge_config(self._api_key)
            self._model = GPTModel(
                model=cfg["model"],
                _openai_api_key=cfg["api_key"],
                base_url=cfg["base_url"],
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
        has_context = bool(inp.context.strip())

        test_case = LLMTestCase(
            input=inp.question,
            actual_output=inp.response_text,
            expected_output=inp.ground_truth,
            context=[inp.context] if has_context else None,
        )

        scores: dict[str, float] = {}

        # ── Hallucination (context-grounded; skipped without context) ────────
        if has_context:
            try:
                # include_reason=False skips the separate reason-generation
                # judge call — one fewer LLM request per response, and the
                # reason text was never persisted anyway.
                hallucination = HallucinationMetric(
                    model=model, threshold=0.5, include_reason=False
                )
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
            # Explicit evaluation_steps instead of `criteria`: with criteria
            # alone, every fresh GEval instance spends an extra judge call
            # generating steps — one wasted request per response. Fixed steps
            # also make the rubric deterministic across responses and runs.
            coherence = GEval(
                name="Coherence",
                model=model,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                evaluation_steps=[
                    "Read the question (input) and the response (actual output).",
                    "Check whether the response directly addresses the question.",
                    "Check the response for logical flow and internal "
                    "consistency — no contradictions or non-sequiturs.",
                    "Penalize disorganized or rambling answers; reward clear, "
                    "well-structured ones.",
                ],
            )
            await coherence.a_measure(test_case)
            scores[f"{_METRIC_PREFIX}/coherence"] = float(coherence.score)
        except Exception as exc:
            logger.warning(
                "GEval coherence failed for response_id=%d: %s",
                inp.response_id,
                exc,
            )

        return _clean_metric_map(scores)

    async def _score_all(self, inputs: list[ScoringInput]) -> list[dict[str, float]]:
        """Score every input, capped at judge_max_concurrency() in flight.

        A semaphore bounds how many responses are judged at once so bursts of
        judge LLM calls stay under the OpenAI tokens-per-minute ceiling. Results
        preserve input order; a per-input failure yields an empty score dict.
        """
        sem = asyncio.Semaphore(judge_max_concurrency())

        async def _one(inp: ScoringInput) -> dict[str, float]:
            async with sem:
                try:
                    return await self._score_single(inp)
                except Exception as exc:
                    logger.error(
                        "DeepEvalScorer failed for response_id=%d: %s",
                        inp.response_id,
                        exc,
                        exc_info=True,
                    )
                    return {}

        return await asyncio.gather(*(_one(inp) for inp in inputs))

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
