"""
Async model runner for the LLM evaluation harness.

Benchmarks a set of open-weight models served through the HuggingFace
Inference Providers router (OpenAI-compatible chat-completions endpoint),
concurrently for a single question, with:

  - Redis cache check before every API call (cache_hit=True skips inference)
  - Exponential backoff retry on rate-limit / transient errors
  - Per-call telemetry: latency_ms, input_tokens, output_tokens, cost_usd
  - Persistence of all results to the PostgreSQL responses table

All models are open-weight and reached via a single provider (the HF router),
so adding or swapping a model is just an entry in the _MODELS registry below —
no new caller code. The LLM judge used for scoring (RAGAS / DeepEval) is
configured separately in the scorers and is independent of this runner.

Model registry (canonical name -> HF model id, cost per 1 000 output tokens)
----------------------------------------------------------------------------
  llama-3.1-8b    meta-llama/Llama-3.1-8B-Instruct   ~$0.0002 / 1K out
  qwen2.5-72b     Qwen/Qwen2.5-72B-Instruct          ~$0.0008 / 1K out
  deepseek-v3.2   deepseek-ai/DeepSeek-V3.2           ~$0.0003 / 1K out

Public API
----------
    run_all_models(question, question_id, run_id, session) -> list[ModelResult]
    run_single_model(model_name, question, question_id,
                     run_id, session)                      -> ModelResult
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from src.cache.cache import get_cached_response, set_cached_response
from src.db import Response

load_dotenv()

logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────────
# Canonical name -> HF Inference Providers model id + cost per 1 000 output
# tokens (USD). Costs are approximate router rates and vary by the provider the
# router selects; they only affect the reported cost_usd telemetry. To add a
# model, confirm it is a chat model on the router (GET /v1/models) and add a row.

_MODELS: dict[str, dict[str, Any]] = {
    "llama-3.1-8b":  {"hf_id": "meta-llama/Llama-3.1-8B-Instruct", "cost_per_1k": 0.0002},
    "qwen2.5-72b":   {"hf_id": "Qwen/Qwen2.5-72B-Instruct",        "cost_per_1k": 0.0008},
    "deepseek-v3.2": {"hf_id": "deepseek-ai/DeepSeek-V3.2",        "cost_per_1k": 0.0003},
}

_HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# ── Retry config ──────────────────────────────────────────────────────────────

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 1.0   # seconds; doubles each attempt
_RETRY_MAX_DELAY = 30.0

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """
    Holds the outcome of one model inference call (or cache hit).

    Fields
    ------
    model_name     : canonical model identifier
    question_id    : FK to the questions table
    run_id         : FK to the eval_runs table
    response_text  : generated text
    latency_ms     : wall-clock milliseconds for the API round-trip
                     (0 for cache hits)
    input_tokens   : prompt tokens reported by the provider
    output_tokens  : completion tokens reported by the provider
    cost_usd       : estimated USD cost based on output tokens
    cache_hit      : True when the result was served from Redis
    error          : non-None if the call failed after all retries
    db_response_id : primary key assigned after DB insert (None until saved)
    """

    model_name: str
    question_id: int
    run_id: int
    response_text: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    error: str | None = None
    db_response_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_cost(model_name: str, output_tokens: int) -> float:
    spec = _MODELS.get(model_name)
    rate = spec["cost_per_1k"] if spec else 0.0
    return round(rate * output_tokens / 1_000, 8)


def _result_to_cache_dict(result: ModelResult) -> dict[str, Any]:
    return {
        "response_text": result.response_text,
        "latency_ms":    result.latency_ms,
        "input_tokens":  result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd":      result.cost_usd,
    }


def _cache_dict_to_result(
    data: dict[str, Any],
    model_name: str,
    question_id: int,
    run_id: int,
) -> ModelResult:
    return ModelResult(
        model_name=model_name,
        question_id=question_id,
        run_id=run_id,
        response_text=data.get("response_text", ""),
        latency_ms=data.get("latency_ms", 0),
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        cost_usd=data.get("cost_usd", 0.0),
        cache_hit=True,
    )


async def _backoff_sleep(attempt: int) -> None:
    """Async exponential backoff with jitter, capped at _RETRY_MAX_DELAY."""
    delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
    # add ±20 % jitter to avoid thundering-herd on concurrent retries
    import random
    jitter = delay * 0.2 * (random.random() * 2 - 1)
    await asyncio.sleep(max(0.0, delay + jitter))


# ── HF Inference Providers router caller ──────────────────────────────────────

async def _call_hf_router(
    model_name: str,
    question: str,
    session: aiohttp.ClientSession,
) -> ModelResult:
    """
    Call an open-weight model via the HuggingFace Inference Providers router.

    Uses the OpenAI-compatible chat-completions endpoint at
    router.huggingface.co. The chat template is applied server-side and the
    response carries real token usage. Retries on HTTP 429 (rate limit) and
    503 (model loading / cold start); a slow first call is normal while a
    provider cold-starts a large model.
    """
    hf_token = os.environ["HUGGINGFACE_API_KEY"]
    hf_id = _MODELS[model_name]["hf_id"]
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "model": hf_id,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": 512,
        "temperature": 0.01,   # near-zero; the router doesn't accept exactly 0
        "stream": False,
    }
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.monotonic()
            async with session.post(_HF_ROUTER_URL, headers=headers, json=payload) as resp:
                latency_ms = int((time.monotonic() - t0) * 1_000)

                if resp.status in (429, 503):
                    body = await resp.text()
                    last_exc = RuntimeError(f"HTTP {resp.status}: {body[:200]}")
                    logger.warning(
                        "HF router rate limit/loading %s (attempt %d/%d): %s",
                        model_name,
                        attempt + 1,
                        _MAX_RETRIES,
                        last_exc,
                    )
                    await _backoff_sleep(attempt)
                    continue

                if not resp.ok:
                    body = await resp.text()
                    err = f"HTTP {resp.status}: {body[:400]}"
                    logger.error("HF router non-retryable error for %s: %s", model_name, err)
                    return ModelResult(model_name=model_name, question_id=0, run_id=0, error=err)

                data = await resp.json()
                choices = data.get("choices") or []
                text: str = (
                    (choices[0].get("message") or {}).get("content", "") if choices else ""
                )

                # The router is OpenAI-compatible and returns real token usage;
                # fall back to a char-length estimate if a provider omits it.
                usage = data.get("usage") or {}
                in_tok = usage.get("prompt_tokens") or max(1, len(question) // 4)
                out_tok = usage.get("completion_tokens") or max(1, len(text) // 4)

                return ModelResult(
                    model_name=model_name,
                    question_id=0,   # caller patches this
                    run_id=0,        # caller patches this
                    response_text=text,
                    latency_ms=latency_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=_compute_cost(model_name, out_tok),
                )

        except aiohttp.ClientError as exc:
            last_exc = exc
            logger.warning(
                "HF router connection error %s (attempt %d/%d): %s",
                model_name,
                attempt + 1,
                _MAX_RETRIES,
                exc,
            )
            await _backoff_sleep(attempt)

    return ModelResult(
        model_name=model_name,
        question_id=0,
        run_id=0,
        error=f"Exceeded {_MAX_RETRIES} retries: {last_exc}",
    )


# ── DB persistence ────────────────────────────────────────────────────────────

def _save_result_to_db(result: ModelResult, session: Session) -> None:
    """
    Insert a ModelResult into the responses table and set result.db_response_id.

    Skips errored results (response_text is empty and error is set).
    The caller owns commit/rollback.
    """
    if result.error:
        logger.warning(
            "Skipping DB save for %s (error: %s)", result.model_name, result.error
        )
        return

    row = Response(
        run_id=result.run_id,
        question_id=result.question_id,
        model_name=result.model_name,
        response_text=result.response_text,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )
    session.add(row)
    session.flush()  # populate row.id without committing
    result.db_response_id = row.id
    logger.debug(
        "Saved response id=%d model=%s question_id=%d",
        row.id,
        result.model_name,
        result.question_id,
    )


# ── Public API ────────────────────────────────────────────────────────────────

async def run_all_models(
    question: str,
    question_id: int,
    run_id: int,
    session: Session,
) -> list[ModelResult]:
    """
    Call every registered model concurrently for a single question.

    For each model:
      1. Check Redis cache — return cached result immediately if hit.
      2. Call the HF router with exponential-backoff retry.
      3. On success, write result to Redis cache.
      4. Save result to the PostgreSQL responses table.

    Parameters
    ----------
    question    : question text sent to each model
    question_id : PK of the Question row (used as cache key and FK)
    run_id      : PK of the EvalRun row (FK for Response rows)
    session     : active SQLAlchemy Session; caller commits after this returns

    Returns
    -------
    list[ModelResult] — one entry per model in _MODELS registry order. Errors
    are recorded in ModelResult.error; the list always has one element per
    registered model.
    """
    async with aiohttp.ClientSession() as http:
        results = await asyncio.gather(
            *(
                run_single_model(name, question, question_id, run_id, session, http)
                for name in _MODELS
            ),
            return_exceptions=False,
        )
    return list(results)


async def run_single_model(
    model_name: str,
    question: str,
    question_id: int,
    run_id: int,
    session: Session,
    http_session: aiohttp.ClientSession | None = None,
) -> ModelResult:
    """
    Run one model for one question, with cache check, retry, and persistence.

    Parameters
    ----------
    model_name   : a key in the _MODELS registry
    question     : question text
    question_id  : FK / cache key
    run_id       : FK for the Response row
    session      : active SQLAlchemy Session
    http_session : shared aiohttp session for router calls; created
                   internally if not provided (use run_all_models instead
                   to share the session across concurrent calls)

    Returns
    -------
    ModelResult with all fields populated.

    Raises
    ------
    ValueError if model_name is not in the registry.
    """
    if model_name not in _MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}. Supported: {list(_MODELS)}"
        )

    # ── 1. Cache check ────────────────────────────────────────────────────────
    cached = get_cached_response(question_id, model_name)
    if cached is not None:
        result = _cache_dict_to_result(cached, model_name, question_id, run_id)
        logger.info("Cache hit  %-22s question_id=%d", model_name, question_id)
        _save_result_to_db(result, session)
        return result

    logger.info("Cache miss %-22s question_id=%d — calling API", model_name, question_id)

    # ── 2. API call ───────────────────────────────────────────────────────────
    _own_http = http_session is None
    if _own_http:
        http_session = aiohttp.ClientSession()

    try:
        result = await _call_hf_router(model_name, question, http_session)
    finally:
        if _own_http:
            await http_session.close()

    # Patch question/run IDs that the caller leaves as 0
    result.question_id = question_id
    result.run_id = run_id

    # ── 3. Populate cache on success ──────────────────────────────────────────
    if not result.error:
        set_cached_response(question_id, model_name, _result_to_cache_dict(result))

    # ── 4. Persist to DB ──────────────────────────────────────────────────────
    _save_result_to_db(result, session)

    return result
