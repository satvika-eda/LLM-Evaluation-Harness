"""
RQ task definitions for the LLM evaluation harness.

The single public task — run_eval_pipeline — is enqueued by callers
(e.g. the FastAPI API layer or a CLI script) and executed by an RQ worker
process.  It owns the full end-to-end pipeline for one evaluation run:

    load questions  →  run models (async)  →  score responses (async)
    →  mark run completed / failed

Pipeline design
---------------
* Questions are loaded from HuggingFace datasets and upserted into the DB.
* Model inference runs with asyncio inside a single asyncio.run() call so
  all three providers run concurrently per question; questions are processed
  sequentially to keep DB session usage simple and to allow partial results
  to be visible in the dashboard while the job is still running.
* Scoring runs with asyncio.run() after all responses are collected; all
  three scorers run concurrently via asyncio.gather inside the orchestrator.
* The EvalRun.status column (pending → running → completed | failed) lets
  the dashboard reflect real-time job state even before the RQ job metadata
  is polled.

Enqueuing example
-----------------
    from rq import Queue
    from src.worker.tasks import run_eval_pipeline
    from src.cache.cache import get_redis_client

    q = Queue("eval_jobs", connection=get_redis_client())
    job = q.enqueue(
        run_eval_pipeline,
        run_id=42,
        dataset_name="hotpotqa",
        n_questions=50,
        models=["gpt-4o", "claude-3-5-sonnet", "mistral-7b"],
        job_timeout=7200,
    )
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timezone

from src.datasets.loader import (
    HOTPOTQA_NAME,
    TRUTHFULQA_NAME,
    load_hotpotqa,
    load_truthfulqa,
    sample_questions,
    save_questions_to_db,
)
from src.db import EvalRun, Question, Response, RunStatus, SessionLocal
from src.runners.runner import run_all_models
from src.scorers.orchestrator import ScoringOrchestrator

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

_SAMPLE_SEED = 42


def _load_questions(dataset_name: str, n: int) -> list[dict]:
    """
    Load the full split and take a deterministic seed-42 random sample.

    Sampling from the full split (rather than the head) keeps the subset
    representative, and the fixed seed means every run of size n evaluates
    exactly the same questions — the property the cross-run mean ± std
    methodology depends on.
    """
    if dataset_name == TRUTHFULQA_NAME:
        questions = load_truthfulqa()
    elif dataset_name == HOTPOTQA_NAME:
        questions = load_hotpotqa()
    else:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Supported: {TRUTHFULQA_NAME!r}, {HOTPOTQA_NAME!r}"
        )
    return sample_questions(questions, n, seed=_SAMPLE_SEED)


def _set_status(
    run_id: int,
    status: RunStatus,
    *,
    error_message: str | None = None,
    completed_at: datetime | None = None,
) -> None:
    """
    Open a short-lived session to update EvalRun.status.

    Kept separate from the main pipeline session so a status update is
    committed immediately and visible to the dashboard without waiting for
    the full transaction to close.
    """
    with SessionLocal() as session:
        run = session.get(EvalRun, run_id)
        if run is None:
            logger.error("EvalRun id=%d not found when trying to set status %s", run_id, status)
            return
        run.status = status
        if error_message is not None:
            run.error_message = error_message[:2000]   # guard against huge tracebacks
        if completed_at is not None:
            run.completed_at = completed_at
        session.commit()
    logger.info("EvalRun id=%d → status=%s", run_id, status)


async def _run_questions_async(
    questions: list[Question],
    run_id: int,
    models: list[str] | None,
) -> list:
    """
    Run the selected models for every question, one question at a time.

    Questions are processed sequentially so each question gets its own
    short session, keeping flush/commit granularity tight and making
    partial results immediately queryable.  Within each question the
    selected model calls run concurrently via asyncio.gather (inside
    run_all_models). The question's retrieval context is passed through
    so context-bearing datasets (HotpotQA) are answered from their
    passages, matching what the context-grounded metrics grade against.
    """
    all_results = []
    for q in questions:
        with SessionLocal() as session:
            try:
                results = await run_all_models(
                    question=q.question,
                    question_id=q.id,
                    run_id=run_id,
                    session=session,
                    context=q.context or "",
                    models=models,
                )
                session.commit()
                all_results.extend(results)
                logger.info(
                    "question_id=%d: %d model responses saved.", q.id, len(results)
                )
            except Exception:
                session.rollback()
                logger.exception("run_all_models failed for question_id=%d", q.id)
                # Continue with remaining questions rather than aborting the run.
    return all_results


async def _score_responses_async(
    run_id: int,
    questions: list[Question],
    orchestrator: ScoringOrchestrator,
) -> None:
    """
    Query all responses for this run, build ScoringInputs, run orchestrator.

    A fresh session is opened for the scoring phase so the scorer flushes
    land in a clean transaction that is committed here.
    """
    with SessionLocal() as session:
        responses: list[Response] = (
            session.query(Response)
            .filter(Response.run_id == run_id, Response.response_text != "")
            .all()
        )
        if not responses:
            logger.warning("EvalRun id=%d: no responses found for scoring.", run_id)
            return

        questions_by_id: dict[int, Question] = {q.id: q for q in questions}
        scoring_inputs = orchestrator.build_inputs(responses, questions_by_id)

        summary = await orchestrator.score_all(scoring_inputs, session)
        session.commit()

        logger.info(
            "EvalRun id=%d scoring complete: %d scores saved across %d scorers. Errors: %s",
            run_id,
            summary.total_scores_saved,
            len(summary.scorer_counts),
            summary.errors or "none",
        )


# ── Public RQ task ────────────────────────────────────────────────────────────

def run_eval_pipeline(
    run_id: int,
    dataset_name: str,
    n_questions: int,
    models: list[str],
) -> dict:
    """
    Full evaluation pipeline as a single RQ task.

    Parameters
    ----------
    run_id       : PK of an existing EvalRun row (status must be "pending")
    dataset_name : "truthfulqa" or "hotpotqa"
    n_questions  : number of questions to sample from the dataset
    models       : model identifiers to evaluate (validated against the
                   runner registry; unknown names fail the run). An empty
                   list or None runs every registered model.

    Returns
    -------
    dict with summary keys: run_id, status, questions_loaded,
    responses_attempted, scores_saved (best-effort; 0 on failure).

    Side effects
    ------------
    * Updates EvalRun.status in PostgreSQL throughout execution.
    * Writes Response and Score rows.
    * Populates the Redis response cache.
    """
    logger.info(
        "run_eval_pipeline START  run_id=%d  dataset=%s  n=%d  models=%s",
        run_id, dataset_name, n_questions, models,
    )

    # ── 1. Mark run as running ────────────────────────────────────────────────
    _set_status(run_id, RunStatus.RUNNING)

    questions: list[Question] = []

    try:
        # ── 2. Load questions from HuggingFace ────────────────────────────────
        logger.info("Loading %d questions from %s…", n_questions, dataset_name)
        raw_questions = _load_questions(dataset_name, n_questions)

        with SessionLocal() as session:
            session.expire_on_commit = False  # keep attribute values after commit
            # Returns one ORM row per sampled question (existing rows are
            # reused), so inference runs on exactly the seed-42 sample —
            # never an arbitrary LIMIT-n subset of the table.
            questions = save_questions_to_db(raw_questions, session)
            session.commit()
            session.expunge_all()

        logger.info("%d questions ready (new + existing).", len(questions))

        # ── 3. Run model inference (async) ────────────────────────────────────
        logger.info("Running model inference for %d questions…", len(questions))
        all_results = asyncio.run(
            _run_questions_async(questions, run_id, models or None)
        )

        successful = [r for r in all_results if not r.error]
        failed     = [r for r in all_results if r.error]
        logger.info(
            "Inference complete: %d succeeded, %d failed.", len(successful), len(failed)
        )
        if all_results and not successful:
            raise RuntimeError(
                f"All {len(all_results)} model calls failed — marking run as "
                "failed instead of scoring an empty result set."
            )

        # ── 4. Score responses (async) ────────────────────────────────────────
        orchestrator = ScoringOrchestrator()
        logger.info("Running scoring orchestrator…")
        asyncio.run(_score_responses_async(run_id, questions, orchestrator))

        # ── 5. Mark completed ─────────────────────────────────────────────────
        _set_status(
            run_id,
            RunStatus.COMPLETED,
            completed_at=datetime.now(timezone.utc),
        )

        summary = {
            "run_id":               run_id,
            "status":               RunStatus.COMPLETED,
            "questions_loaded":     len(questions),
            "responses_attempted":  len(all_results),
            "responses_succeeded":  len(successful),
            "responses_failed":     len(failed),
        }
        logger.info("run_eval_pipeline DONE  %s", summary)
        return summary

    except Exception as exc:
        # ── 6. Mark failed ────────────────────────────────────────────────────
        tb = traceback.format_exc()
        logger.exception("run_eval_pipeline FAILED run_id=%d", run_id)
        _set_status(
            run_id,
            RunStatus.FAILED,
            error_message=tb,
            completed_at=datetime.now(timezone.utc),
        )
        # Re-raise so RQ records the job as failed and stores the traceback.
        raise
