"""
FastAPI application for the LLM Evaluation Harness.

Endpoints
---------
POST /run-eval       — create an eval run and enqueue the pipeline job
GET  /runs           — list all eval runs with status and metadata
GET  /results/{id}   — scores for a run grouped by model → metric
GET  /leaderboard    — models ranked by avg faithfulness with all metric avgs
GET  /cache/stats    — Redis cache hit rate and memory usage
GET  /health         — PostgreSQL and Redis connectivity
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.cache.cache import cache_stats, get_redis_client
from src.db import EvalRun, Question, Response, RunStatus, Score, create_tables, get_db
from src.worker.tasks import run_eval_pipeline
from src.worker.worker import get_queue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(title="LLM Evaluation Harness", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class RunEvalRequest(BaseModel):
    run_name: str
    dataset_name: str
    n_questions: int
    models: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/run-eval", status_code=201)
def run_eval(req: RunEvalRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Create an eval_run record with status PENDING, enqueue the pipeline job
    to RQ on the "eval_jobs" queue, and return run_id + job_id immediately.
    """
    run = EvalRun(
        run_name=req.run_name,
        dataset_name=req.dataset_name,
        models_evaluated=req.models,
        status=RunStatus.PENDING,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    q = get_queue()
    job = q.enqueue(
        run_eval_pipeline,
        run_id=run.id,
        dataset_name=req.dataset_name,
        n_questions=req.n_questions,
        models=req.models,
    )

    return {"run_id": run.id, "job_id": job.id, "status": "queued"}


@app.get("/runs")
def list_runs(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """Return all eval runs ordered by creation time (newest first)."""
    runs = db.query(EvalRun).order_by(EvalRun.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "run_name": r.run_name,
            "dataset_name": r.dataset_name,
            "models_evaluated": r.models_evaluated,
            "status": r.status,
            "error_message": r.error_message,
            "created_at": r.created_at,
            "completed_at": r.completed_at,
        }
        for r in runs
    ]


@app.get("/results/{run_id}")
def get_results(run_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Return all scores for a run grouped by model and metric as a nested dict.

    Response shape
    --------------
    {
        "run_id": 1,
        "results": {
            "gpt-4o": {
                "ragas/faithfulness": 0.91,
                "bertscore/f1": 0.84,
                ...
            },
            ...
        }
    }
    """
    run = db.get(EvalRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    rows = (
        db.query(Response.model_name, Score.metric_name, Score.score)
        .join(Score, Score.response_id == Response.id)
        .filter(Response.run_id == run_id)
        .all()
    )

    # Accumulate per (model, metric)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for model_name, metric_name, score in rows:
        grouped[model_name][metric_name].append(score)

    results = {
        model: {metric: sum(scores) / len(scores) for metric, scores in metrics.items()}
        for model, metrics in grouped.items()
    }

    return {"run_id": run_id, "results": results}


@app.get("/leaderboard")
def leaderboard(dataset: str | None = None, db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    Return all models ranked by average faithfulness score (descending).

    Each entry includes all metric averages and average cost per question.

    Query params
    ------------
    dataset : optional dataset name (e.g. "hotpotqa"). When given, only scores
              and costs from responses to that dataset's questions are counted.
              Omit to aggregate across all datasets. Filtering by dataset avoids
              blending metrics that are dataset-dependent — e.g. faithfulness is
              ~0 on context-free TruthfulQA but meaningful on HotpotQA.

    Response shape
    --------------
    {
        "leaderboard": [
            {
                "rank": 1,
                "model": "llama-3.1-8b",
                "avg_faithfulness": 0.92,
                "avg_cost_per_question": 0.0031,
                "metrics": {
                    "ragas/faithfulness": 0.92,
                    "ragas/answer_relevance": 0.88,
                    ...
                }
            },
            ...
        ]
    }
    """
    _FAITHFULNESS = "ragas/faithfulness"

    score_q = (
        db.query(Response.model_name, Score.metric_name, Score.score)
        .join(Score, Score.response_id == Response.id)
    )
    cost_q = (
        db.query(Response.model_name, Response.cost_usd)
        .filter(Response.cost_usd.isnot(None))
    )
    if dataset:
        score_q = score_q.join(Question, Question.id == Response.question_id).filter(
            Question.dataset_name == dataset
        )
        cost_q = cost_q.join(Question, Question.id == Response.question_id).filter(
            Question.dataset_name == dataset
        )

    score_rows = score_q.all()

    metric_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for model_name, metric_name, score in score_rows:
        metric_scores[model_name][metric_name].append(score)

    cost_rows = cost_q.all()
    cost_lists: dict[str, list[float]] = defaultdict(list)
    for model_name, cost in cost_rows:
        cost_lists[model_name].append(cost)

    entries: list[dict[str, Any]] = []
    for model, metrics in metric_scores.items():
        avg_metrics = {m: sum(s) / len(s) for m, s in metrics.items()}
        costs = cost_lists.get(model, [])
        entries.append(
            {
                "model": model,
                "avg_faithfulness": avg_metrics.get(_FAITHFULNESS, 0.0),
                "avg_cost_per_question": sum(costs) / len(costs) if costs else None,
                "metrics": avg_metrics,
            }
        )

    entries.sort(key=lambda e: e["avg_faithfulness"], reverse=True)
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank

    return {"leaderboard": entries}


@app.get("/cache/stats")
def get_cache_stats() -> dict[str, Any]:
    """
    Return Redis cache statistics including hit rate and memory usage.

    hit_rate is computed from Redis keyspace_hits / (keyspace_hits + keyspace_misses).
    Returns null when no lookups have been recorded yet.
    """
    try:
        stats = cache_stats()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {exc}") from exc

    # Augment with hit rate from Redis INFO stats section
    try:
        info = get_redis_client().info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total_lookups = hits + misses
        stats["keyspace_hits"] = hits
        stats["keyspace_misses"] = misses
        stats["hit_rate"] = round(hits / total_lookups, 4) if total_lookups else None
    except Exception:
        stats.setdefault("hit_rate", None)

    return stats


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return connectivity status for PostgreSQL and Redis."""
    pg_ok = False
    pg_error: str | None = None
    try:
        db.execute(text("SELECT 1"))
        pg_ok = True
    except Exception as exc:
        pg_error = str(exc)

    redis_ok = False
    redis_error: str | None = None
    try:
        get_redis_client().ping()
        redis_ok = True
    except Exception as exc:
        redis_error = str(exc)

    return {
        "status": "ok" if (pg_ok and redis_ok) else "degraded",
        "postgres": {"ok": pg_ok, "error": pg_error},
        "redis": {"ok": redis_ok, "error": redis_error},
    }
