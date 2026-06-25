"""
BERTScore scorer — precision, recall, F1.

Uses microsoft/deberta-xlarge-mnli as the embedding model, which
consistently outranks BERT/RoBERTa on WMT and summarisation benchmarks
because DeBERTa's disentangled attention better captures token importance.

Metric names written to the DB
-------------------------------
    bertscore/precision
    bertscore/recall
    bertscore/f1

Notes on bert-score library
-----------------------------
    bert_score.score(cands, refs, model_type=..., device=...) returns
    (P, R, F1) as torch Tensors of shape [N].

    The model is loaded once and cached in self._scorer to avoid the
    ~3 s warmup on every call. On CPU this is still the slowest scorer
    in the pipeline (~0.5 s per response); use device="cuda" when a GPU
    is available.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy.orm import Session

from src.scorers import ScoringInput
from src.scorers.ragas_scorer import _persist_scores

logger = logging.getLogger(__name__)

_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"
_METRIC_PREFIX = "bertscore"


class BERTScorer:
    """Wraps bert-score and persists precision / recall / F1 to PostgreSQL."""

    def __init__(self, device: str = "cpu") -> None:
        """
        Parameters
        ----------
        device : torch device string — "cpu", "cuda", or "cuda:0" etc.
                 Defaults to "cpu" for portability; override with "cuda"
                 when a GPU is available to get ~10× speedup.
        """
        self._device = device
        self._scorer: Any = None   # bert_score.BERTScorer instance, lazy

    # ── Lazy model setup ──────────────────────────────────────────────────────

    def _get_scorer(self) -> Any:
        """
        Load DeBERTa-xlarge-mnli once and cache it on self._scorer.

        rescale_with_baseline=True maps raw cosine similarities onto a
        [0, 1] range calibrated against human judgements, making scores
        directly comparable across models and datasets.
        """
        if self._scorer is None:
            from bert_score import BERTScorer as _BERTScorer

            logger.info(
                "Loading BERTScorer model %s on %s (first call only)…",
                _MODEL_TYPE,
                self._device,
            )
            self._scorer = _BERTScorer(
                model_type=_MODEL_TYPE,
                device=self._device,
                rescale_with_baseline=True,
            )
            logger.info("BERTScorer model loaded.")
        return self._scorer

    # ── Core scoring (sync, runs in executor) ─────────────────────────────────

    def _score_sync(self, inputs: list[ScoringInput]) -> list[dict[str, float]]:
        """
        Compute BERTScore for all inputs in one batched call.

        bert_score processes the whole list in a single forward pass
        (batched internally), so this is more efficient than a per-sample
        loop. The scorer is not re-loaded between calls.
        """
        scorer = self._get_scorer()

        candidates: list[str] = [inp.response_text for inp in inputs]
        references: list[str] = [inp.ground_truth for inp in inputs]

    # Returns three Tensors of shape [N]
        P, R, F1 = scorer.score(candidates, references)

        results: list[dict[str, float]] = []
        for p, r, f1 in zip(P.tolist(), R.tolist(), F1.tolist()):
            results.append(
                {
                    f"{_METRIC_PREFIX}/precision": round(float(p), 6),
                    f"{_METRIC_PREFIX}/recall":    round(float(r), 6),
                    f"{_METRIC_PREFIX}/f1":        round(float(f1), 6),
                }
            )
        return results

    # ── Public async interface ────────────────────────────────────────────────

    async def score(
        self,
        inputs: list[ScoringInput],
        session: Session,
    ) -> list[dict[str, Any]]:
        """
        Score a batch of responses with BERTScore and persist to DB.

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

        logger.info("BERTScorer: scoring %d responses…", len(inputs))

        try:
            metric_maps = await asyncio.to_thread(self._score_sync, inputs)
        except Exception as exc:
            logger.error("BERTScorer batch failed: %s", exc, exc_info=True)
            return []

        return _persist_scores(metric_maps, inputs, session, "BERTScorer")
