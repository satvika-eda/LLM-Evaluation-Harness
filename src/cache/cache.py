"""
Redis cache layer for the LLM evaluation harness.

Caches model responses by (question_id, model_name) to avoid redundant and
costly inference calls when re-running or extending an evaluation.

Key format
----------
    response:{question_id}:{model_name}

    e.g. response:42:gpt-4o
         response:42:claude-3-5-sonnet

Public API
----------
    get_redis_client()                                    -> Redis
    get_cached_response(question_id, model_name)          -> dict | None
    set_cached_response(question_id, model_name,
                        response_dict, ttl)               -> bool
    cache_stats()                                         -> dict
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

import redis
from redis import Redis

logger = logging.getLogger(__name__)

_KEY_PREFIX = "response"
_DEFAULT_TTL = 86_400  # 24 hours in seconds


# ── Client factory ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_redis_client() -> Redis:
    """
    Return a Redis client connected to REDIS_URL.

    The client is created once and reused across calls (module-level singleton
    via lru_cache). Connection health is verified with a PING on first call so
    misconfiguration is caught at startup rather than on the first cache miss.

    Environment
    -----------
    REDIS_URL : Redis connection string, e.g. redis://localhost:6379/0
                Defaults to redis://localhost:6379/0 if not set.

    Raises
    ------
    redis.exceptions.ConnectionError
        If Redis is unreachable.
    """
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    client: Redis = redis.from_url(url, decode_responses=True)
    client.ping()  # fail fast if unreachable
    logger.info("Redis client connected to %s", url)
    return client


# ── Key helper ────────────────────────────────────────────────────────────────

def _make_key(question_id: int | str, model_name: str) -> str:
    return f"{_KEY_PREFIX}:{question_id}:{model_name}"


# ── Cache operations ──────────────────────────────────────────────────────────

def get_cached_response(
    question_id: int | str,
    model_name: str,
) -> dict[str, Any] | None:
    """
    Look up a cached model response.

    Parameters
    ----------
    question_id:
        Primary key of the question row (int) or any stable string identifier.
    model_name:
        Model identifier string, e.g. "gpt-4o" or "claude-3-5-sonnet".

    Returns
    -------
    The deserialized response dict if the key exists and has not expired,
    otherwise None.
    """
    key = _make_key(question_id, model_name)
    try:
        raw = get_redis_client().get(key)
    except redis.RedisError as exc:
        logger.warning("Redis GET failed for key %r: %s", key, exc)
        return None

    if raw is None:
        logger.debug("Cache miss: %s", key)
        return None

    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Cache hit but JSON decode failed for key %r: %s", key, exc)
        return None

    logger.debug("Cache hit: %s", key)
    return data


def set_cached_response(
    question_id: int | str,
    model_name: str,
    response_dict: dict[str, Any],
    ttl: int = _DEFAULT_TTL,
) -> bool:
    """
    Serialize and store a response dict in Redis.

    Parameters
    ----------
    question_id:
        Primary key or stable identifier for the question.
    model_name:
        Model identifier string.
    response_dict:
        Arbitrary JSON-serializable dict. Typically contains keys such as
        response_text, latency_ms, input_tokens, output_tokens, cost_usd.
    ttl:
        Time-to-live in seconds. Defaults to 86 400 (24 hours).
        Pass ttl=0 to store without expiry (not recommended for prod).

    Returns
    -------
    True on success, False if the write failed (Redis error is logged but
    not re-raised so callers don't need try/except for every inference call).
    """
    key = _make_key(question_id, model_name)
    try:
        serialized = json.dumps(response_dict, default=str)
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize response for key %r: %s", key, exc)
        return False

    try:
        client = get_redis_client()
        if ttl > 0:
            client.setex(key, ttl, serialized)
        else:
            client.set(key, serialized)
    except redis.RedisError as exc:
        logger.warning("Redis SET failed for key %r: %s", key, exc)
        return False

    logger.debug("Cached response at %s (ttl=%ds)", key, ttl)
    return True


# ── Stats ─────────────────────────────────────────────────────────────────────

def cache_stats() -> dict[str, Any]:
    """
    Return a summary of cache utilisation from Redis INFO.

    Queries the keyspace for keys matching the response prefix rather than
    using DBSIZE so the count reflects only eval-harness entries, not any
    other keys that may exist in the same Redis instance.

    Returns
    -------
    dict with keys:
        total_cached_responses (int)  — number of response:* keys
        used_memory_bytes      (int)  — bytes currently used by Redis
        used_memory_human      (str)  — human-readable form, e.g. "3.12M"
        connected_clients      (int)  — active client connections
        redis_version          (str)  — server version string
        uptime_seconds         (int)  — server uptime in seconds

    Raises
    ------
    redis.RedisError
        Re-raised if the INFO command itself fails (unlike the read/write
        helpers, a stats failure is worth surfacing to the caller).
    """
    client = get_redis_client()
    info: dict[str, Any] = client.info("all")

    # Count only keys belonging to this application.
    # SCAN is non-blocking and safe in production (unlike KEYS).
    total = sum(1 for _ in client.scan_iter(match=f"{_KEY_PREFIX}:*", count=100))

    return {
        "total_cached_responses": total,
        "used_memory_bytes": info.get("used_memory", 0),
        "used_memory_human": info.get("used_memory_human", "N/A"),
        "connected_clients": info.get("connected_clients", 0),
        "redis_version": info.get("redis_version", "N/A"),
        "uptime_seconds": info.get("uptime_in_seconds", 0),
    }
