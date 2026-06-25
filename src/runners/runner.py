"""
Async model runner for the LLM evaluation harness.

Calls OpenAI GPT-4o, Anthropic Claude Sonnet, and Mistral-7B (via HuggingFace
Inference API) concurrently for a single question, with:

  - Redis cache check before every API call (cache_hit=True skips inference)
  - Exponential backoff retry on rate-limit / transient errors
  - Per-call telemetry: latency_ms, input_tokens, output_tokens, cost_usd
  - Persistence of all results to the PostgreSQL responses table

Pricing constants (per 1 000 output tokens, USD)
-------------------------------------------------
  GPT-4o          $0.005
  Claude Sonnet   $0.003
  Mistral-7B      $0.0002

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
import anthropic
import openai
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from src.cache.cache import get_cached_response, set_cached_response
from src.db import Response

load_dotenv()

logger = logging.getLogger(__name__)

# ── Pricing (USD per 1 000 output tokens) ────────────────────────────────────

_COST_PER_1K: dict[str, float] = {
    "gpt-4o":              0.005,
    "claude-3-5-sonnet":   0.003,
    "mistral-7b":          0.0002,
}

# ── Retry config ──────────────────────────────────────────────────────────────

_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 1.0   # seconds; doubles each attempt
_RETRY_MAX_DELAY = 30.0

# ── Rate-limit exception types per provider ───────────────────────────────────

_OPENAI_RATE_LIMIT_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)
_ANTHROPIC_RATE_LIMIT_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
)

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
    rate = _COST_PER_1K.get(model_name, 0.0)
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


# ── Per-provider callers ──────────────────────────────────────────────────────

async def _call_openai(question: str) -> ModelResult:
    """Call GPT-4o via the async OpenAI client."""
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.monotonic()
            resp = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": question}],
                temperature=0.0,
            )
            latency_ms = int((time.monotonic() - t0) * 1_000)

            text = resp.choices[0].message.content or ""
            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0

            return ModelResult(
                model_name="gpt-4o",
                question_id=0,   # caller patches this
                run_id=0,        # caller patches this
                response_text=text,
                latency_ms=latency_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=_compute_cost("gpt-4o", out_tok),
            )

        except _OPENAI_RATE_LIMIT_ERRORS as exc:
            last_exc = exc
            logger.warning("OpenAI rate limit (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, exc)
            await _backoff_sleep(attempt)

        except openai.OpenAIError as exc:
            # Non-retryable provider error
            logger.error("OpenAI non-retryable error: %s", exc)
            return ModelResult(model_name="gpt-4o", question_id=0, run_id=0, error=str(exc))

    return ModelResult(
        model_name="gpt-4o",
        question_id=0,
        run_id=0,
        error=f"Exceeded {_MAX_RETRIES} retries: {last_exc}",
    )


async def _call_anthropic(question: str) -> ModelResult:
    """Call Claude 3.5 Sonnet via the async Anthropic client."""
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.monotonic()
            resp = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": question}],
            )
            latency_ms = int((time.monotonic() - t0) * 1_000)

            text = resp.content[0].text if resp.content else ""
            in_tok = resp.usage.input_tokens if resp.usage else 0
            out_tok = resp.usage.output_tokens if resp.usage else 0

            return ModelResult(
                model_name="claude-3-5-sonnet",
                question_id=0,
                run_id=0,
                response_text=text,
                latency_ms=latency_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=_compute_cost("claude-3-5-sonnet", out_tok),
            )

        except _ANTHROPIC_RATE_LIMIT_ERRORS as exc:
            last_exc = exc
            logger.warning(
                "Anthropic rate limit (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, exc
            )
            await _backoff_sleep(attempt)

        except anthropic.APIError as exc:
            logger.error("Anthropic non-retryable error: %s", exc)
            return ModelResult(
                model_name="claude-3-5-sonnet", question_id=0, run_id=0, error=str(exc)
            )

    return ModelResult(
        model_name="claude-3-5-sonnet",
        question_id=0,
        run_id=0,
        error=f"Exceeded {_MAX_RETRIES} retries: {last_exc}",
    )


async def _call_mistral_hf(question: str, session: aiohttp.ClientSession) -> ModelResult:
    """
    Call Mistral-7B-Instruct via the HuggingFace Inference API.

    Uses the text-generation endpoint with the standard [INST] prompt format.
    Retries on HTTP 429 (rate limit) and 503 (model loading).
    """
    hf_token = os.environ["HUGGINGFACE_API_KEY"]
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": f"[INST] {question} [/INST]",
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.01,   # near-zero; HF API doesn't accept exactly 0
            "return_full_text": False,
        },
    }
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.monotonic()
            async with session.post(url, headers=headers, json=payload) as resp:
                latency_ms = int((time.monotonic() - t0) * 1_000)

                if resp.status in (429, 503):
                    body = await resp.text()
                    last_exc = RuntimeError(f"HTTP {resp.status}: {body[:200]}")
                    logger.warning(
                        "HuggingFace rate limit/loading (attempt %d/%d): %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        last_exc,
                    )
                    await _backoff_sleep(attempt)
                    continue

                if not resp.ok:
                    body = await resp.text()
                    err = f"HTTP {resp.status}: {body[:400]}"
                    logger.error("HuggingFace non-retryable error: %s", err)
                    return ModelResult(model_name="mistral-7b", question_id=0, run_id=0, error=err)

                data = await resp.json()
                text: str = data[0].get("generated_text", "") if isinstance(data, list) else ""

                # HF text-generation does not return token counts; estimate from
                # character length (≈ 4 chars per token) as a billing proxy.
                out_tok = max(1, len(text) // 4)
                in_tok = max(1, len(question) // 4)

                return ModelResult(
                    model_name="mistral-7b",
                    question_id=0,
                    run_id=0,
                    response_text=text,
                    latency_ms=latency_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=_compute_cost("mistral-7b", out_tok),
                )

        except aiohttp.ClientError as exc:
            last_exc = exc
            logger.warning(
                "HuggingFace connection error (attempt %d/%d): %s",
                attempt + 1,
                _MAX_RETRIES,
                exc,
            )
            await _backoff_sleep(attempt)

    return ModelResult(
        model_name="mistral-7b",
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
    Call all three models concurrently for a single question.

    For each model:
      1. Check Redis cache — return cached result immediately if hit.
      2. Call provider API with exponential-backoff retry.
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
    list[ModelResult] — one entry per model, in the order
    [gpt-4o, claude-3-5-sonnet, mistral-7b]. Errors are recorded in
    ModelResult.error; the list always has exactly 3 elements.
    """
    async with aiohttp.ClientSession() as http:
        results = await asyncio.gather(
            run_single_model("gpt-4o",            question, question_id, run_id, session, http),
            run_single_model("claude-3-5-sonnet", question, question_id, run_id, session, http),
            run_single_model("mistral-7b",        question, question_id, run_id, session, http),
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
    model_name   : one of "gpt-4o", "claude-3-5-sonnet", "mistral-7b"
    question     : question text
    question_id  : FK / cache key
    run_id       : FK for the Response row
    session      : active SQLAlchemy Session
    http_session : shared aiohttp session for Mistral calls; created
                   internally if not provided (use run_all_models instead
                   to share the session across concurrent calls)

    Returns
    -------
    ModelResult with all fields populated.

    Raises
    ------
    ValueError if model_name is not recognised.
    """
    if model_name not in _COST_PER_1K:
        raise ValueError(
            f"Unknown model {model_name!r}. Supported: {list(_COST_PER_1K)}"
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
        if model_name == "gpt-4o":
            result = await _call_openai(question)
        elif model_name == "claude-3-5-sonnet":
            result = await _call_anthropic(question)
        else:  # mistral-7b
            result = await _call_mistral_hf(question, http_session)
    finally:
        if _own_http:
            await http_session.close()

    # Patch question/run IDs that the internal callers leave as 0
    result.question_id = question_id
    result.run_id = run_id

    # ── 3. Populate cache on success ──────────────────────────────────────────
    if not result.error:
        set_cached_response(question_id, model_name, _result_to_cache_dict(result))

    # ── 4. Persist to DB ──────────────────────────────────────────────────────
    _save_result_to_db(result, session)

    return result
