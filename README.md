# LLM Evaluation Harness

Benchmarks open-weight LLMs (Llama, Qwen, DeepSeek) served through the HuggingFace Inference Providers router, scoring every response with RAGAS, DeepEval, and BERTScore against a separately-configurable judge LLM. Runs, responses, and scores persist to PostgreSQL; a FastAPI backend and Streamlit dashboard expose a live leaderboard, and every completed run is also logged to MLflow for experiment tracking.

## Overview

- **Model runner** (`src/runners`) — benchmarks open-weight models via the HF router (OpenAI-compatible chat-completions API), with Redis response caching, retry/backoff, and per-call cost/latency telemetry.
- **Scorers** (`src/scorers`) — three independent scoring frameworks, orchestrated together and run in memory-bounded chunks:
  - **RAGAS** — faithfulness, answer relevance
  - **DeepEval** — hallucination detection, G-Eval coherence
  - **BERTScore** — semantic similarity (precision/recall/F1)
- **Judge LLM** — RAGAS and DeepEval both grade against a configurable judge model, independent of the models being benchmarked (defaults to an open-weight model via the HF router, not OpenAI — see [Judge model](#judge-model)).
- **API + dashboard** — a FastAPI backend (`src/api`) backed by PostgreSQL, and a Streamlit dashboard (`dashboard/app.py`) with a leaderboard, cost-vs-quality scatter plot, per-metric explorer, and a form to launch new eval runs.
- **Async job queue** — eval runs execute as RQ (Redis Queue) background jobs (`src/worker`) so the API stays responsive while a run is in progress.
- **Experiment tracking** — every completed run is also logged to MLflow (`src/tracking`) as a parent run with one nested child run per model.

## Project Structure

```
llm-evaluation-harness/
├── src/
│   ├── runners/         # Async model runner (HF Inference Providers router)
│   ├── scorers/         # RAGAS / DeepEval / BERTScore wrappers + orchestrator
│   ├── datasets/        # HotpotQA / TruthfulQA loaders
│   ├── tracking/        # MLflow experiment logging
│   ├── cache/           # Redis response cache
│   ├── worker/          # RQ worker + the run_eval_pipeline task
│   ├── api/             # FastAPI app + SQLAlchemy models (src/db.py)
│   └── db.py            # EvalRun / Question / Response / Score ORM models
├── dashboard/            # Streamlit dashboard (app.py)
├── tests/                # pytest suite (mocked provider calls, no live creds needed)
├── docker-compose.yml    # postgres + redis + api + worker + dashboard
├── Dockerfile.api
├── Dockerfile.dashboard
├── pyproject.toml
├── .env.example
└── README.md
```

## Setup

### 1. Install

```bash
git clone <repo-url>
cd llm-evaluation-harness
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,dashboard]"
```

### 2. Configure environment

```bash
cp .env.example .env
```

`.env.example` is the source of truth for every setting (each one is documented inline); the essentials:

| Variable              | Description                                                          |
|-----------------------|------------------------------------------------------------------------|
| `HUGGINGFACE_API_KEY` | HF token — used both to run the benchmarked models and (by default) as the judge's API key |
| `OPENAI_API_KEY`      | Only needed for RAGAS's embedding step (`text-embedding-3-small`) and if you switch the judge back to an OpenAI model |
| `DATABASE_URL`        | PostgreSQL connection string |
| `REDIS_URL`           | Redis connection string (cache + job queue) |
| `LLM_JUDGE_MODEL` / `JUDGE_BASE_URL` / `JUDGE_API_KEY` | Judge LLM config — see [Judge model](#judge-model) |
| `MLFLOW_TRACKING_URI` | Defaults to a local `sqlite:///mlflow.db` file, no server required |

### 3. Start Postgres and Redis

**Docker Compose** (runs everything — postgres, redis, api, worker, dashboard):

```bash
docker compose up --build
```

**Or natively**, if you'd rather run the Python processes yourself (e.g. no Docker available): start Postgres and Redis however you normally would (Homebrew services, a local install, etc.), matching the `DATABASE_URL`/`REDIS_URL` in your `.env`, then create the database/role once:

```sql
CREATE ROLE llm_eval LOGIN PASSWORD 'llm_eval';
CREATE DATABASE llm_eval OWNER llm_eval;
```

Tables are created automatically on first API startup (`create_tables()` in the FastAPI lifespan) — no separate migration step or CLI command.

### 4. Run the three processes

Only needed if you're not using `docker compose up` — each in its own terminal:

```bash
# API — also creates DB tables on startup
uvicorn src.api.main:app --reload --port 8000

# Worker — executes eval runs asynchronously via Redis Queue
python -m src.worker.worker

# Dashboard
streamlit run dashboard/app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) for the dashboard (leaderboard, cost/quality, metric explorer, run launcher) or [http://localhost:8000/docs](http://localhost:8000/docs) for the raw API.

## Running an evaluation

Either use the **Run New Eval** tab in the dashboard, or hit the API directly:

```bash
curl -X POST http://localhost:8000/run-eval \
  -H "Content-Type: application/json" \
  -d '{
    "run_name": "hotpotqa-test",
    "dataset_name": "hotpotqa",
    "n_questions": 50,
    "models": ["llama-3.1-8b", "qwen2.5-72b", "deepseek-v3.2"]
  }'
```

This creates an `EvalRun` row and enqueues `run_eval_pipeline` on the RQ worker: it samples `n_questions` from the dataset, runs every listed model concurrently per question, scores all responses with RAGAS + DeepEval + BERTScore, logs the run to MLflow, and marks the run `completed`. Poll `GET /runs` or watch the dashboard for status.

## API Endpoints

| Method | Path                | Description                                              |
|--------|---------------------|------------------------------------------------------------|
| POST   | `/run-eval`         | Create an eval run and enqueue the pipeline job            |
| GET    | `/runs`             | List all eval runs with status, dataset, and judge model   |
| GET    | `/results/{run_id}` | Scores for one run, grouped by model → metric              |
| GET    | `/leaderboard`      | Models ranked by faithfulness; filterable by `dataset` and `judge_model` |
| GET    | `/cache/stats`      | Redis cache hit rate and memory usage                       |
| GET    | `/health`           | PostgreSQL and Redis connectivity                            |

## Supported Models

All benchmarked models are open-weight, served through the HuggingFace Inference Providers router (`src/runners/runner.py`) — no local downloads or GPU required.

| Model ID        | HF model                          | Approx. cost / 1K output tokens |
|------------------|------------------------------------|----------------------------------|
| `llama-3.1-8b`   | `meta-llama/Llama-3.1-8B-Instruct` | ~$0.0002 |
| `qwen2.5-72b`    | `Qwen/Qwen2.5-72B-Instruct`        | ~$0.0008 |
| `deepseek-v3.2`  | `deepseek-ai/DeepSeek-V3.2`        | ~$0.0003 |

## Judge model

RAGAS and DeepEval both need an LLM to grade responses (faithfulness, hallucination, coherence) — this judge is configured independently of the models being benchmarked via `LLM_JUDGE_MODEL` / `JUDGE_BASE_URL` / `JUDGE_API_KEY` (see `src/scorers/__init__.py::judge_config`). It defaults to `google/gemma-3-27b-it`, also via the HF router — an open-weight judge, not OpenAI — chosen deliberately over the three benchmarked models to avoid a model grading its own output.

Every run records which judge scored it (`EvalRun.judge_model`), and `/leaderboard` only aggregates runs scored by the same judge by default (BERTScore is judge-independent and always included); pass `?judge_model=__all__` to explicitly blend every judge ever used, or name a specific one to view its runs alone.

## Metrics

| Metric                    | Framework  | Description                                                    |
|---------------------------|------------|------------------------------------------------------------------|
| `ragas/faithfulness`      | RAGAS      | Does the answer's claims hold up against the retrieved context? (skipped on context-free datasets) |
| `ragas/answer_relevance`  | RAGAS      | Is the answer relevant to the question?                        |
| `deepeval/hallucination`  | DeepEval   | Rate of contradiction with the provided context — **lower is better** (skipped on context-free datasets) |
| `deepeval/coherence`      | DeepEval (G-Eval) | Logical flow and internal consistency of the response      |
| `bertscore/precision`, `/recall`, `/f1` | BERTScore | Semantic similarity to the reference answer (embedding model configurable via `BERTSCORE_MODEL`) |

`ragas/context_recall` was deliberately removed — with a fixed dataset it compares ground truth against retrieved context without ever looking at the model's answer, so every model gets an identical score.

## Experiment tracking (MLflow)

Every completed run also logs to MLflow: one parent run per `EvalRun`, with a nested child run per benchmarked model carrying its full metric set and cost. Browse it with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

then open [http://localhost:5001](http://localhost:5001) — switch the left sidebar toggle from "GenAI" to "Model training" to see the classic runs table.

## Running Tests

```bash
pytest                        # all tests
pytest tests/ -v --cov=src    # with coverage
```

External provider calls (HuggingFace router, judge LLM) are mocked throughout — no live credentials are required to run the test suite.

## License

MIT
