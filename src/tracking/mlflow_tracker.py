"""
MLflow experiment tracking for the LLM evaluation harness.

Every completed EvalRun logs a parent MLflow run (dataset, question count,
judge model, models benchmarked) with one nested child run per benchmarked
model (its per-metric averages + cost). This makes runs comparable and
browsable in MLflow's own UI without touching the Postgres schema that powers
the FastAPI/Streamlit side — MLflow is a second, independent view onto the
same eval results, not a replacement for the DB.

Configuration
-------------
    MLFLOW_TRACKING_URI     where runs are logged (default: local SQLite file
                             ``sqlite:///mlflow.db`` — no server needed).
                             MLflow's plain filesystem backend (``file:./mlruns``)
                             is in maintenance mode as of MLflow 3.x and refuses
                             new runs, so a database backend is used by default.
    MLFLOW_EXPERIMENT_NAME  groups runs in the MLflow UI
                             (default ``llm-eval-harness``).

Browsing runs
-------------
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

Comparing across the judge switch
----------------------------------
Each run is tagged with judge_model (see EvalRun.judge_model / the
/leaderboard judge_model filter) — filter or group by it in the MLflow UI
before comparing metrics across runs, since different judges score
ragas/*/deepeval/* differently (see docs/judge notes in src/scorers).

Failure handling
-----------------
Logging is best-effort: any MLflow error is caught and logged, never raised,
so a tracking outage can't fail an eval run that already scored successfully.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
_DEFAULT_EXPERIMENT_NAME = "llm-eval-harness"

_configured = False


def _configure() -> None:
    """Set MLflow's tracking URI + experiment once per process."""
    global _configured
    if _configured:
        return
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", _DEFAULT_EXPERIMENT_NAME))
    _configured = True


def log_eval_run(
    *,
    run_id: int,
    run_name: str,
    dataset_name: str,
    n_questions: int,
    models: list[str],
    judge_model: str | None,
    results: dict[str, dict[str, float]],
    avg_cost_by_model: dict[str, float],
) -> None:
    """
    Log one completed EvalRun to MLflow: a parent run for the EvalRun itself,
    with one nested child run per benchmarked model.

    Parameters
    ----------
    run_id            : EvalRun.id (Postgres PK) — recorded as a tag so an
                        MLflow run can always be traced back to its source row.
    run_name          : EvalRun.run_name
    dataset_name      : "hotpotqa" | "truthfulqa"
    n_questions       : sample size for this run
    models            : model identifiers actually scored in this run
    judge_model       : LLM_JUDGE_MODEL active when this run was scored (see
                        EvalRun.judge_model) — logged so runs are only
                        compared against others scored under the same judge.
    results           : {model_name: {metric_name: avg_score}} — same shape
                        as GET /results/{run_id}.
    avg_cost_by_model : {model_name: avg_cost_per_question}

    Never raises — a tracking failure is logged and swallowed so it can't
    fail an eval run whose scoring already succeeded.
    """
    try:
        import mlflow

        _configure()

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "eval_run_id": str(run_id),
                    "dataset": dataset_name,
                    "judge_model": judge_model or "unknown",
                }
            )
            mlflow.log_params(
                {
                    "dataset": dataset_name,
                    "n_questions": n_questions,
                    "models": ",".join(models),
                    "judge_model": judge_model or "unknown",
                }
            )

            for model_name, metrics in results.items():
                with mlflow.start_run(run_name=model_name, nested=True):
                    mlflow.set_tags({"eval_run_id": str(run_id), "model": model_name})
                    mlflow.log_param("model", model_name)
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value)
                    cost = avg_cost_by_model.get(model_name)
                    if cost is not None:
                        mlflow.log_metric("avg_cost_per_question", cost)

        logger.info("EvalRun id=%d logged to MLflow.", run_id)
    except Exception:
        logger.exception("MLflow logging failed for EvalRun id=%d (non-fatal).", run_id)
