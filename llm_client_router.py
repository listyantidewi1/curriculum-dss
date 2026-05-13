"""
llm_client_router.py

Provider-routing rule (durable; see memory/project_llm_provider_routing.md):

    GPT models → Jatevo (api_keys/jatevo.txt, sk-clb-... format).
    Everything else → OpenRouter (api_keys/OpenRouter.txt, sk-or-... format).

Past incident (commit 2aced75) — Jatevo keys were being sent to OpenRouter's
base URL → 401 "Missing Authentication header". This module dispatches by
model family so that bug class can't recur.

Public API:

    client, provider = get_client_for_model("openai/gpt-5")
    # → (OpenAI client pointing at Jatevo, provider="jatevo")

    client, provider = get_client_for_model("deepseek/deepseek-v3.2")
    # → (OpenAI client pointing at OpenRouter, provider="openrouter")

Both Jatevo and OpenRouter speak the OpenAI-compatible chat-completions API,
so callers can use a single `client.chat.completions.create(...)` regardless
of the provider.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Jatevo's OpenAI-compatible endpoint. Confirmed by user 2026-05-12.
JATEVO_BASE_URL = "https://lb.jatevo.ai/v1"


# A model name is treated as a GPT family member if it contains one of these
# tokens (case-insensitive). Add to this list when new GPT-family models
# become routable through Jatevo.
_GPT_FAMILY_TOKENS = re.compile(
    r"(^|/|-)(gpt-|o1|o3|chatgpt)",
    re.IGNORECASE,
)


def is_gpt_family(model_name: str) -> bool:
    """Decide whether `model_name` belongs to the GPT family.

    Conservative — matches common GPT/o-series naming patterns. Examples:
        "gpt-4o-mini"            -> True
        "openai/gpt-5"           -> True
        "gpt-4-turbo-preview"    -> True
        "o1-preview"             -> True
        "o3-mini"                -> True
        "deepseek/deepseek-v3.2" -> False
        "claude-3-7-sonnet"      -> False
        "meta-llama/Llama-3.1-70b-instruct" -> False
    """
    if not model_name:
        return False
    return bool(_GPT_FAMILY_TOKENS.search(model_name))


def _read_key(file_path: Path) -> str:
    try:
        key = file_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"API key file not found: {file_path}. "
            f"Create it or set the appropriate environment variable."
        ) from exc
    if not key:
        raise RuntimeError(f"API key file is empty: {file_path}")
    return key


def _load_jatevo_key() -> str:
    env_key = os.getenv("JATEVO_API_KEY")
    if env_key:
        return env_key
    return _read_key(Path("api_keys") / "jatevo.txt")


def _load_openrouter_key() -> str:
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    return _read_key(Path("api_keys") / "OpenRouter.txt")


def get_client_for_model(
    model_name: str,
    request_timeout: float = 120.0,
) -> Tuple[OpenAI, str]:
    """Return (OpenAI client, provider_tag) for the given model.

    Args:
        model_name: e.g. "deepseek/deepseek-v3.2", "openai/gpt-5".
        request_timeout: per-request timeout in seconds.

    Returns:
        (client, provider) where provider is "jatevo" | "openrouter".
    """
    if is_gpt_family(model_name):
        api_key = _load_jatevo_key()
        base_url = JATEVO_BASE_URL
        provider = "jatevo"
    else:
        api_key = _load_openrouter_key()
        base_url = OPENROUTER_BASE_URL
        provider = "openrouter"
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
    return client, provider


def strip_provider_prefix(model_name: str) -> str:
    """Some endpoints want the bare model name without an "openai/" prefix.

    Returns:
        "openai/gpt-5"          -> "gpt-5"
        "deepseek/deepseek-v3.2"-> "deepseek/deepseek-v3.2"  (kept — OpenRouter wants the prefix)

    Only strips the prefix when routing to Jatevo (which doesn't use the
    provider/ namespace).
    """
    if is_gpt_family(model_name) and "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name
