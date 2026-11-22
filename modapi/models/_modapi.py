"""Shared helpers for the ModAPI model provider: fixed endpoints, key check and catalog lookups."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import requests

API_BASE = "https://modapi.ai/v1"
CATALOG_URL = "https://modapi.ai/api/pricing"
PRICING_PAGE = "https://modapi.ai/pricing"
USER_AGENT = "modapi-dify-plugin/0.1.0"
TIMEOUT = (10, 30)
_CATALOG_TTL = 600
_catalog_cache: dict[str, Any] = {"at": 0.0, "models": {}}


class ModAPIError(Exception):
    """A problem talking to ModAPI, with a message that is safe to show to the user."""


def api_key(credentials: Mapping[str, Any]) -> str:
    key = (credentials.get("api_key") or "").strip()
    if not key:
        raise ModAPIError("Enter your ModAPI API key.")
    return key


def check_key(credentials: Mapping[str, Any]) -> None:
    """Check the key with GET /v1/models, which is free, instead of a paid chat request."""
    try:
        response = requests.get(
            f"{API_BASE}/models",
            headers={"Authorization": f"Bearer {api_key(credentials)}", "User-Agent": USER_AGENT},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ModAPIError(f"Could not reach ModAPI: {type(exc).__name__}.") from exc
    if response.status_code in (401, 403):
        raise ModAPIError("ModAPI rejected this API key. Create a key at https://modapi.ai/keys and paste it again.")
    if response.status_code >= 400:
        raise ModAPIError(f"ModAPI returned HTTP {response.status_code} while checking the key. Try again later.")


def catalog() -> dict[str, dict]:
    """Public model catalog keyed by model ID. Cached in memory for ten minutes; carries no user data."""
    now = time.monotonic()
    if _catalog_cache["models"] and now - _catalog_cache["at"] < _CATALOG_TTL:
        return _catalog_cache["models"]
    try:
        response = requests.get(CATALOG_URL, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json().get("data") or []
    except (requests.RequestException, ValueError) as exc:
        raise ModAPIError("Could not read the ModAPI model catalog. Try again later.") from exc
    models = {item["model_name"]: item for item in data if isinstance(item, dict) and item.get("model_name")}
    _catalog_cache.update(at=now, models=models)
    return models


def find_model(model: str) -> dict | None:
    return catalog().get(model)


def supports_openai_chat(entry: Mapping[str, Any]) -> bool:
    return "openai" in (entry.get("supported_endpoint_types") or [])


def supports_embeddings(entry: Mapping[str, Any]) -> bool:
    return "embedding" in (entry.get("supported_endpoint_types") or [])


def usd_per_million(entry: Mapping[str, Any]) -> tuple[str, str] | None:
    """Standard-group list price per million tokens, or None for models not billed per token."""
    if entry.get("quota_type") != 0 or entry.get("model_ratio") is None:
        return None
    input_price = float(entry["model_ratio"]) * 2
    output_price = input_price * float(entry.get("completion_ratio") or 1)
    return f"{input_price:.6f}".rstrip("0").rstrip("."), f"{output_price:.6f}".rstrip("0").rstrip(".")


def model_tags(entry: Mapping[str, Any]) -> set[str]:
    return {t.strip() for t in (entry.get("tags") or "").split(",") if t.strip()}
