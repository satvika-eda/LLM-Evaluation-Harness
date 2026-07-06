"""
Scoring orchestrator for the LLM evaluation harness.

Runs RAGASScorer, DeepEvalScorer, and BERTScorer concurrently over a list
of model response records using asyncio.gather, then persists all scores
to the PostgreSQL scores table.

Typical call site (inside an eval loop)
-----------------------------------------
    from src.scorers.orchestrator import ScoringOrchestrator

    orchestrator = ScoringOrchestrator()

    # Build ScoringInputs from joined Response + Question rows
    inputs = orchestrator.build_inputs(responses, questions_by_id)

    async with async_session() as session:
        summary = await orchestrator.score_all(inputs, session)
        session.commit()

Public API
----------
    ScoringOrchestrator.build_inputs(responses, questions_by_id) -> list[ScoringInput]
    ScoringOrchestrator.score_all(inputs, session)               -> ScoringSummary
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from src.db import Question, Response
from src.scorers import ScoringInput
from src.scorers.bert_scorer import BERTScorer
from src.scorers.deepeval_scorer import DeepEvalScorer
from src.scorers.ragas_scorer import RAGASScorer

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ScoringSummary:
    """
    Aggregate outcome of one orchestrator.score_all() call.

    Fields
    ------
    total_inputs        : number of ScoringInputs processed
    total_scores_saved  : total Score rows inserted (all scorers combined)
    scorer_counts       : per-scorer count of saved rows
    errors              : any scorer-level errors encountered
    """

    total_inputs: int = 0
    total_scores_saved: int = 0
    scorer_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ScoringOrchestrator:
    """
    Runs all three scorers concurrently and aggregates the results.

    Scorer instances are created once at construction time so their
    lazy-loaded models (BERTScorer) are cached across multiple
    score_all() calls within the same process.
    """

    def __init__(self, bert_device: str = "cpu", openai_api_key: str | None = None) -> None:
        """
        Parameters
        ----------
        bert_device     : torch device for BERTScorer, e.g. "cpu" or "cuda"
        openai_api_key  : override for OPENAI_API_KEY env var (useful in tests)
        """
        self.ragas = RAGASScorer(openai_api_key=openai_api_key)
        self.deepeval = DeepEvalScorer(openai_api_key=openai_api_key)
        self.bert = BERTScorer(device=bert_device)

    # ── Input construction ────────────────────────────────────────────────────

    @staticmethod
    def build_inputs(
        responses: list[Response],
        questions_by_id: dict[int, Question],
    ) -> list[ScoringInput]:
        """
        Join Response ORM rows with their parent Question rows.

        Parameters
        ----------
        responses       : list of Response ORM objects (must have .id,
                          .question_id, .response_text, .model_name set)
        questions_by_id : dict mapping question_id -> Question ORM object;
                          build with {q.id: q for q in questions}

        Returns
        -------
        list[ScoringInput] — one entry per response, skipping any whose
        question_id is not found in questions_by_id (logged as a warning).
        """
        inputs: list[ScoringInput] = []
        for resp in responses:
            q = questions_by_id.get(resp.question_id)
            if q is None:
                logger.warning(
                    "Question id=%d not found for response id=%d — skipping.",
                    resp.question_id,
                    resp.id,
                )
                continue
            inputs.append(
                ScoringInput(
                    response_id=resp.id,
                    response_text=resp.response_text,
                    question=q.question,
                    ground_truth=q.ground_truth,
                    context=q.context or "",
                    model_name=resp.model_name,
                    dataset_name=q.dataset_name,
                )
            )
        return inputs

    # ── Concurrent scoring ────────────────────────────────────────────────────

    async def score_all(
        self,
        inputs: list[ScoringInput],
        session: Session,
    ) -> ScoringSummary:
        """
        Run all three scorers concurrently and save results to the DB.

        Each scorer receives the full input list and writes independently
        to the scores table. asyncio.gather is used with
        return_exceptions=True so a failure in one scorer does not cancel
        the others.

        Parameters
        ----------
        inputs  : list[ScoringInput] — build with build_inputs()
        session : active SQLAlchemy Session; caller is responsible for
                  committing after this method returns

        Returns
        -------
        ScoringSummary with per-scorer counts and any error messages.
        """
        if not inputs:
            logger.warning("score_all called with empty inputs list.")
            return ScoringSummary()

        chunk_size = int(os.environ.get("SCORING_CHUNK_SIZE", "40"))
        logger.info(
            "ScoringOrchestrator: scoring %d inputs sequentially in chunks of %d "
            "(memory-bounded, committed per chunk)…",
            len(inputs),
            chunk_size,
        )

        # Pre-resolve bert_score / transformers imports before scoring.
        # transformers uses lazy-module imports that are not thread-safe; doing
        # this once here (single-threaded) avoids an "AutoTokenizer" import race
        # when BERTScore runs in its worker thread.
        try:
            import bert_score  # noqa: F401
            from transformers import AutoModel, AutoTokenizer  # noqa: F401
        except Exception as exc:  # pragma: no cover - defensive; both are deps
            logger.warning("Could not pre-import bert_score/transformers: %s", exc)

        # Score sequentially, one small chunk at a time, committing each chunk
        # before the next. This bounds peak memory (only one scorer's working set
        # on ~chunk_size inputs is live at once) so scoring fits on memory-limited
        # hosts, and makes progress durable — a crash/OOM loses at most the current
        # chunk, and already-scored responses are skipped on resume.
        scorers = (
            ("BERTScorer", self.bert),
            ("RAGASScorer", self.ragas),
            ("DeepEvalScorer", self.deepeval),
        )
        summary = ScoringSummary(total_inputs=len(inputs))
        summary.scorer_counts = {name: 0 for name, _ in scorers}

        for start in range(0, len(inputs), chunk_size):
            chunk = inputs[start:start + chunk_size]
            for name, scorer in scorers:
                try:
                    saved = await scorer.score(chunk, session)
                    summary.scorer_counts[name] += len(saved)
                    summary.total_scores_saved += len(saved)
                except Exception as exc:
                    msg = f"{name} failed on chunk starting at {start}: {exc}"
                    logger.error(msg, exc_info=True)
                    summary.errors.append(msg)
            session.commit()  # durable checkpoint per chunk
            logger.info(
                "Scored chunk %d–%d of %d (%d scores so far).",
                start,
                min(start + chunk_size, len(inputs)),
                len(inputs),
                summary.total_scores_saved,
            )

        logger.info(
            "ScoringOrchestrator complete: %d total scores saved, %d errors.",
            summary.total_scores_saved,
            len(summary.errors),
        )
        return summary
