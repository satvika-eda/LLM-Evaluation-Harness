"""
BERTScore scorer — precision, recall, F1.

The embedding model is configurable via the BERTSCORE_MODEL env var.
Default is microsoft/deberta-xlarge-mnli (~900M params, ~3.5 GB RAM in
fp32 on CPU), which consistently outranks BERT/RoBERTa on WMT and
summarisation benchmarks. On memory-constrained hosts (≤8 GB) set
BERTSCORE_MODEL=roberta-large (~1.4 GB, the bert-score default for
English) — scores are not comparable across embedding models, so pick
one and keep it for a whole benchmark.

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
import os
from typing import Any

from sqlalchemy.orm import Session

from src.scorers import ScoringInput
from src.scorers.ragas_scorer import _persist_scores

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"
_METRIC_PREFIX = "bertscore"


def _model_type() -> str:
    """Embedding model for BERTScore; override with BERTSCORE_MODEL."""
    return os.environ.get("BERTSCORE_MODEL", _DEFAULT_MODEL_TYPE)


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

        rescale_with_baseline=True re-centres raw cosine similarities
        against a per-language baseline so scores spread over a wider,
        more interpretable range. Rescaled scores are NOT bounded to
        [0, 1] — texts less similar than the baseline (e.g. short answers
        vs long references) come out negative. Downstream consumers must
        not assume a unit interval for bertscore/* metrics.
        """
        if self._scorer is None:
            from bert_score import BERTScorer as _BERTScorer

            model_type = _model_type()
            logger.info(
                "Loading BERTScorer model %s on %s (first call only)…",
                model_type,
                self._device,
            )
            self._scorer = _BERTScorer(
                model_type=model_type,
                device=self._device,
                lang="en",
                rescale_with_baseline=True,
            )
            self._clamp_tokenizer_max_length()
            logger.info("BERTScorer model loaded.")
        return self._scorer

    @staticmethod
    def _sentinel_max_length(tokenizer: Any) -> bool:
        return getattr(tokenizer, "model_max_length", 0) > 100_000

    def _clamp_tokenizer_max_length(self) -> None:
        """Clamp an unset tokenizer max length to the model's real context size.

        When a tokenizer has no configured max length, transformers reports it
        as a huge sentinel (VERY_LARGE_INTEGER ≈ 1e30). bert-score passes that
        value straight to the Rust tokenizer's truncation, which raises
        ``OverflowError: int too big to convert``. Fall back to the model's
        ``max_position_embeddings`` (512 for DeBERTa-xlarge-mnli).
        """
        tokenizer = getattr(self._scorer, "_tokenizer", None)
        if tokenizer is None or not self._sentinel_max_length(tokenizer):
            return
        model = getattr(self._scorer, "_model", None)
        max_len = getattr(getattr(model, "config", None), "max_position_embeddings", 512)
        tokenizer.model_max_length = int(max_len)
        logger.info("Clamped BERTScore tokenizer model_max_length to %d.", max_len)

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
