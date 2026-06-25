"""
RQ Worker entry-point for the LLM evaluation harness.

Starts a worker process that listens on the "eval_jobs" queue and executes
tasks defined in src/worker/tasks.py.

Usage
-----
    # From the project root (with the virtualenv active):
    python -m src.worker.worker

    # Or via the rq CLI (equivalent, picks up REDIS_URL from the environment):
    rq worker eval_jobs --url $REDIS_URL

Configuration
-------------
REDIS_URL  : Redis connection string (default: redis://localhost:6379/0)
LOG_LEVEL  : Python log level name (default: INFO)

Worker behaviour
----------------
* Listens on a single queue "eval_jobs" with default priority ordering.
* Uses a burst=False loop so the process stays alive between jobs.
* Job timeout defaults to 2 hours (7 200 s) — override per-job at enqueue
  time with job_timeout=<seconds>.
* Failed jobs land in the "failed" queue and are kept for 30 days so their
  tracebacks are inspectable via rq-dashboard or the RQ CLI.
"""

from __future__ import annotations

import logging
import os
import signal
import sys

from dotenv import load_dotenv
from redis import Redis
from rq import Queue, Worker
from rq.job import Retry

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

QUEUE_NAME        = "eval_jobs"
DEFAULT_JOB_TIMEOUT = 7_200        # 2 hours in seconds
FAILED_TTL        = 60 * 60 * 24 * 30   # keep failed jobs for 30 days


def get_connection() -> Redis:
    """Return a Redis connection from REDIS_URL."""
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    conn: Redis = Redis.from_url(url)
    logger.info("Worker connecting to Redis at %s", url)
    return conn


def get_queue(connection: Redis | None = None) -> Queue:
    """
    Return the eval_jobs Queue.

    Useful for enqueuing tasks from application code without importing
    Redis connection management separately:

        from src.worker.worker import get_queue
        q = get_queue()
        job = q.enqueue(run_eval_pipeline, run_id=1, ...)
    """
    conn = connection or get_connection()
    return Queue(QUEUE_NAME, connection=conn, default_timeout=DEFAULT_JOB_TIMEOUT)


def start_worker() -> None:
    """
    Start an RQ Worker and block until the process is killed.

    Handles SIGTERM / SIGINT gracefully — the worker finishes the current
    job before shutting down (RQ's default warm-shutdown behaviour).
    """
    connection = get_connection()
    queues = [Queue(QUEUE_NAME, connection=connection, default_timeout=DEFAULT_JOB_TIMEOUT)]

    worker = Worker(
        queues=queues,
        connection=connection,
        # Store failed jobs so they are visible in rq-dashboard.
        serializer=None,   # use default pickle serializer
    )

    logger.info(
        "Worker %s starting — listening on queue %r",
        worker.name,
        QUEUE_NAME,
    )

    # warm_shutdown_delay: finish the current job before exiting on SIGTERM
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    start_worker()
