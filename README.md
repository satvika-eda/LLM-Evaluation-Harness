# LLM Evaluation Harness

Multi-model LLM evaluation framework benchmarking GPT-4o, Claude, and Mistral on faithfulness, hallucination, and answer relevance using RAGAS, DeepEval, and G-Eval — with a live leaderboard dashboard.

## Overview

This harness runs structured evaluations across multiple LLM providers and scores responses using several complementary frameworks:

- **RAGAS** — faithfulness, answer relevance, context precision/recall
- **DeepEval** — hallucination detection, G-Eval, answer correctness
- **BERTScore** — semantic similarity scoring
- **FastAPI dashboard** — live leaderboard with per-model score breakdowns

## Project Structure

```
llm-evaluation-harness/
├── src/
│   ├── runners/        # Model runners for OpenAI, Anthropic, HuggingFace
│   ├── scorers/        # RAGAS, DeepEval, BERTScore wrappers
│   ├── datasets/       # Dataset loaders and preprocessing
│   └── api/            # FastAPI routes and database models
├── dashboard/          # Frontend assets for the leaderboard UI
├── tests/              # Unit and integration tests
├── pyproject.toml
├── .env.example
└── README.md
```

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd llm-evaluation-harness
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

| Variable              | Description                        |
|-----------------------|------------------------------------|
| `OPENAI_API_KEY`      | OpenAI API key (for GPT-4o)        |
| `ANTHROPIC_API_KEY`   | Anthropic API key (for Claude)     |
| `HUGGINGFACE_API_KEY` | HuggingFace token (for datasets / Mistral) |
| `DATABASE_URL`        | PostgreSQL connection string       |

### 3. Set up the database

```bash
# Start PostgreSQL (example with Docker)
docker run -d \
  --name llm-eval-db \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=llm_eval \
  -p 5432:5432 \
  postgres:16

# Run migrations
python -m src.api.db migrate
```

### 4. Run evaluations

```bash
# Run the full benchmark suite
python -m src.runners.run_eval --models gpt-4o claude-3-5-sonnet mistral-7b

# Run with a specific dataset
python -m src.runners.run_eval --dataset squad --models gpt-4o
```

### 5. Start the dashboard

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) to view the live leaderboard.

## Running Tests

```bash
pytest                        # all tests
pytest tests/ -v --cov=src    # with coverage
pytest -k "test_scorers"      # filter by name
```

## Supported Models

| Provider    | Model IDs                              |
|-------------|----------------------------------------|
| OpenAI      | `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo` |
| Anthropic   | `claude-3-5-sonnet`, `claude-3-haiku`  |
| HuggingFace | `mistral-7b`, `llama-3-8b`             |

## Metrics

| Metric              | Framework  | Description                                      |
|---------------------|------------|--------------------------------------------------|
| Faithfulness        | RAGAS      | Does the answer stay faithful to the context?    |
| Answer Relevance    | RAGAS      | Is the answer relevant to the question?          |
| Context Recall      | RAGAS      | Is the relevant context retrieved?               |
| Hallucination Score | DeepEval   | Rate of factually incorrect statements           |
| Answer Correctness  | DeepEval   | Factual correctness vs. ground truth             |
| BERTScore F1        | BERTScore  | Semantic similarity to reference answer          |

## License

MIT
