import contextlib
from typing import Any

import requests

from utils.exceptions import ValidationError
from utils.logging import log_message

from ._http import COMMON_STATUS_HINTS, post_with_retries
from .openai_compatible import (
    build_chat_messages,
    read_chat_reply,
    routed_sampling_params,
)

# OpenRouter model metadata cache & reasoning detection
_OPENROUTER_MODELS_META: dict[str, dict[str, Any]] = {}


def _ensure_openrouter_models_meta_loaded(debug: bool = False) -> None:
    """Loads and caches OpenRouter models metadata once per process.

    Populates a mapping of model id -> model dict (including description/architecture).
    """
    if _OPENROUTER_MODELS_META:
        return
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        response.raise_for_status()
        data = response.json()
        all_models = data.get("data", [])
        for model in all_models:
            mid = str(model.get("id", "")).lower()
            if mid:
                _OPENROUTER_MODELS_META[mid] = model
    except Exception as e:
        # Do not raise; lack of metadata should not block translations
        log_message(
            f"Could not load OpenRouter models metadata: {e}", always_print=True
        )


def openrouter_is_reasoning_model(model_name: str, debug: bool = False) -> bool:
    """Detects whether an OpenRouter model is a reasoning model.

    Returns True if the model's 'supported_parameters' contains 'include_reasoning'.
    Returns False if model_name is empty, metadata is not found, or the check fails.
    """
    if not model_name:
        return False
    with contextlib.suppress(Exception):
        _ensure_openrouter_models_meta_loaded(debug=debug)

    lm = model_name.lower()
    meta = _OPENROUTER_MODELS_META.get(lm)
    if not meta:
        return False

    # Check if supported_parameters contains "include_reasoning"
    supported_parameters = meta.get("supported_parameters", [])
    return bool(
        isinstance(supported_parameters, list)
        and "include_reasoning" in supported_parameters
    )


def _reasoning_config(
    reasoning_effort: str | None, metadata: dict[str, Any], is_reasoning_model: bool
) -> dict[str, Any]:
    """Build OpenRouter's `reasoning` object; empty leaves reasoning at the model default."""
    if not reasoning_effort:
        return {}
    is_claude_max = metadata.get("is_claude_effort_max", False)
    # Fable 5+ has always-on thinking, so it gets no reasoning config
    omit_thinking = metadata.get("is_claude_omit_thinking", False)

    if metadata.get("is_claude_adaptive_default", False) and not omit_thinking:
        # Thinking is on by default, so "none" has to switch it off explicitly
        return {"enabled": reasoning_effort != "none"}
    if is_claude_max:
        # Claude 4.6+: `enabled` turns on adaptive thinking
        on = reasoning_effort != "none" and not omit_thinking
        return {"enabled": True} if on else {}
    if is_reasoning_model:
        return {"effort": reasoning_effort}
    return {}


def call_openrouter_endpoint(
    api_key: str,
    model_name: str,
    parts: list[dict[str, Any]],
    generation_config: dict[str, Any],
    system_prompt: str | None = None,
    debug: bool = False,
    timeout: int = 300,
    max_retries: int = 3,
    base_delay: float = 1.0,
    enable_web_search: bool = False,
) -> str | None:
    """
    Calls the OpenRouter Chat Completions API endpoint (OpenAI compatible) and handles retries.

    Args:
        api_key (str): OpenRouter API key.
        model_name (str): OpenRouter model ID (e.g., "openai/gpt-4o", "anthropic/claude-3-haiku").
        parts (List[Dict[str, Any]]): List of content parts (text, images).
                                      # Assumes the first part is the text prompt, subsequent are images.
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, top_k, max_tokens).
                                             # Parameter restrictions (temp clamp, no top_k) are applied
                                             # based on the model_name if it indicates OpenAI or Anthropic.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable web search for up-to-date information using native provider engine.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if blocked by content filter or if no content is found after retries.

    Raises:
        ValueError: If API key is missing or parts format is invalid.
        RuntimeError: If API call fails after retries for non-rate-limited HTTP errors,
                      connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for OpenRouter endpoint")

    metadata = generation_config.get("_metadata", {})
    is_openai_model = metadata.get("is_openai_model", False)
    messages = build_chat_messages(
        parts,
        system_prompt,
        "OpenRouter",
        image_detail=generation_config.get("image_detail") if is_openai_model else None,
    )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/meangrinch/MangaTranslator",
        "X-OpenRouter-Title": "MangaTranslator",
        "X-OpenRouter-Categories": "writing-assistant,image-gen",
    }

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": generation_config.get("max_tokens", 4096),
    }

    if enable_web_search:
        payload["model"] = payload["model"].removesuffix(":online")
        payload["plugins"] = [{"id": "web", "engine": "native"}]

    payload.update(
        routed_sampling_params(
            model_name, generation_config, cap_openai_temperature=True
        )
    )

    # OpenRouter verbosity parameter: used by both Claude (effort) and GPT-5 (verbosity)
    effort = generation_config.get("effort")
    supports_verbosity = metadata.get("supports_verbosity", False) or metadata.get(
        "is_gpt5_model", False
    )
    if effort and metadata.get("is_claude_effort", False):
        payload["verbosity"] = effort
    elif supports_verbosity and generation_config.get("verbosity"):
        payload["verbosity"] = generation_config["verbosity"]

    reasoning_config = _reasoning_config(
        generation_config.get("reasoning_effort"),
        metadata,
        openrouter_is_reasoning_model(model_name, debug=debug),
    )

    # Claude rejects a custom temperature while thinking is on
    if metadata.get("is_anthropic_model", False) and (
        reasoning_config.get("enabled")
        or reasoning_config.get("effort", "none") != "none"
    ):
        payload.pop("temperature", None)

    if reasoning_config:
        reasoning_config["exclude"] = True
        payload["reasoning"] = reasoning_config

    payload = {k: v for k, v in payload.items() if v is not None}

    return post_with_retries(
        url,
        headers,
        payload,
        lambda result: read_chat_reply(result, "OpenRouter", debug),
        label="OpenRouter",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
        status_hints=COMMON_STATUS_HINTS,
    )
