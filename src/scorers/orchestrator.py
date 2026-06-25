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

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

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

        logger.info(
            "ScoringOrchestrator: running 3 scorers on %d inputs concurrently…",
            len(inputs),
        )

        ragas_task = self.ragas.score(inputs, session)
        deepeval_task = self.deepeval.score(inputs, session)
        bert_task = self.bert.score(inputs, session)

        raw_results: tuple[Any, Any, Any] = await asyncio.gather(
            ragas_task,
            deepeval_task,
            bert_task,
            return_exceptions=True,
        )

        scorer_names = ("RAGASScorer", "DeepEvalScorer", "BERTScorer")
        summary = ScoringSummary(total_inputs=len(inputs))

        for name, result in zip(scorer_names, raw_results):
            if isinstance(result, BaseException):
                msg = f"{name} raised an unhandled exception: {result}"
                logger.error(msg, exc_info=result)
                summary.errors.append(msg)
                summary.scorer_counts[name] = 0
            else:
                count = len(result)
                summary.scorer_counts[name] = count
                summary.total_scores_saved += count
                logger.info("%s: saved %d scores.", name, count)

        logger.info(
            "ScoringOrchestrator complete: %d total scores saved, %d errors.",
            summary.total_scores_saved,
            len(summary.errors),
        )
        return summary
