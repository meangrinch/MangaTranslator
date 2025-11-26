import json
import time
from typing import Any, Dict, List, Optional

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message

# OpenRouter model metadata cache & reasoning detection
_OPENROUTER_MODELS_META: Dict[str, Dict[str, Any]] = {}


def _ensure_openrouter_models_meta_loaded(debug: bool = False) -> None:
    """Loads and caches OpenRouter models metadata once per process.

    Populates a mapping of model id -> model dict (including description/architecture).
    """
    global _OPENROUTER_MODELS_META
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
    try:
        _ensure_openrouter_models_meta_loaded(debug=debug)
    except Exception:
        pass

    lm = model_name.lower()
    meta = _OPENROUTER_MODELS_META.get(lm)
    if not meta:
        return False

    # Check if supported_parameters contains "include_reasoning"
    supported_parameters = meta.get("supported_parameters", [])
    if (
        isinstance(supported_parameters, list)
        and "include_reasoning" in supported_parameters
    ):
        return True

    return False


def call_openrouter_endpoint(
    api_key: str,
    model_name: str,
    parts: List[Dict[str, Any]],
    generation_config: Dict[str, Any],
    system_prompt: Optional[str] = None,
    debug: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
    base_delay: float = 1.0,
    enable_web_search: bool = False,
) -> Optional[str]:
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
        enable_web_search (bool): Enable web search for up-to-date information (Gemini models via :online suffix).

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
    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]
    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for OpenRouter: No text prompt found."
        )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/meangrinch/MangaTranslator",
        "X-Title": "MangaTranslator",
    }

    messages = []
    user_content = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for part in image_parts:
        if (
            "inline_data" in part
            and "data" in part["inline_data"]
            and "mime_type" in part["inline_data"]
        ):
            mime_type = part["inline_data"]["mime_type"]
            base64_image = part["inline_data"]["data"]
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )
        else:
            log_message(f"Invalid image part format: {part}", always_print=True)
    user_content.append({"type": "text", "text": text_part["text"]})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": generation_config.get("max_tokens", 4096),
    }

    if enable_web_search:
        if not model_name.endswith(":online"):
            payload["model"] = f"{model_name}:online"

    metadata = generation_config.get("_metadata", {})
    is_openai_model = metadata.get("is_openai_model", False)
    is_anthropic_model = metadata.get("is_anthropic_model", False)
    is_gemini_3 = metadata.get("is_gemini_3", False)
    is_google_model = metadata.get("is_google_model", False)

    temp = generation_config.get("temperature")
    if temp is not None:
        if is_anthropic_model or is_openai_model:
            payload["temperature"] = min(temp, 1.0)
        else:
            payload["temperature"] = temp

    top_p = generation_config.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p

    top_k = generation_config.get("top_k")
    if top_k is not None and not is_openai_model and not is_anthropic_model:
        payload["top_k"] = top_k

    reasoning_config = {}
    # For reasoning-capable Google models (excluding Gemini 3), "auto" should route to reasoning={"enabled": true}
    reasoning_effort = generation_config.get("reasoning_effort")
    if reasoning_effort == "auto" and is_google_model and not is_gemini_3:
        reasoning_config["enabled"] = True

    # Skip if we already set enabled=True for Google auto case
    if reasoning_effort and not (
        reasoning_effort == "auto" and is_google_model and not is_gemini_3
    ):
        is_openai_reasoning = metadata.get("is_openai_reasoning", False)
        is_anthropic_reasoning = metadata.get("is_anthropic_reasoning", False)
        is_grok_reasoning = metadata.get("is_grok_reasoning", False)

        if reasoning_effort and reasoning_effort != "none":
            if is_openai_reasoning or is_anthropic_reasoning or is_grok_reasoning:
                # OpenAI models support "minimal" in addition to high/medium/low
                if is_openai_reasoning and reasoning_effort in [
                    "low",
                    "medium",
                    "high",
                    "minimal",
                ]:
                    reasoning_config["effort"] = reasoning_effort
                elif is_anthropic_reasoning and reasoning_effort in [
                    "low",
                    "medium",
                    "high",
                ]:
                    reasoning_config["effort"] = reasoning_effort
                elif is_grok_reasoning and reasoning_effort in ["high", "low"]:
                    reasoning_config["effort"] = reasoning_effort

    if is_gemini_3 and generation_config.get("reasoning_effort"):
        reasoning_effort_gemini3 = generation_config.get("reasoning_effort")
        if reasoning_effort_gemini3 in ["low", "high"]:
            reasoning_config["effort"] = reasoning_effort_gemini3

    # Exception: Google models with auto (enabled=True) don't need exclude set
    if reasoning_config:
        # Only set exclude if we're using effort-based reasoning, not enabled-based
        if reasoning_config.get("enabled") is not True:
            reasoning_config["exclude"] = True
        payload["reasoning"] = reasoning_config

    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"OpenRouter API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message("Processing OpenRouter response", verbose=debug)
            try:
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    finish_reason = choice.get("finish_reason")

                    if finish_reason == "content_filter":
                        log_message(
                            "Content blocked by OpenRouter filter", always_print=True
                        )
                        return None

                    message = choice.get("message")
                    if message and "content" in message:
                        content = message["content"]
                        return content.strip() if content else ""
                    else:
                        log_message("No message content in response", verbose=debug)
                        return ""
                else:
                    if "error" in result:
                        error_msg = result.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        raise TranslationError(
                            f"OpenRouter API returned error: {error_msg}"
                        )
                    return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing OpenRouter API response: {str(e)}"
                ) from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]

            if status_code == 429 and attempt < max_retries:
                log_message(
                    f"Rate limited, retrying in {current_delay:.1f}s", verbose=debug
                )
                time.sleep(current_delay)
                continue
            else:
                error_reason = f"Status {status_code}: {error_text}"
                if status_code == 429 and attempt == max_retries:
                    error_reason = (
                        f"Rate limited after {max_retries + 1} attempts: {error_text}"
                    )
                elif status_code == 400:
                    error_reason += " (Check model name and payload)"
                elif status_code == 401:
                    error_reason += " (Check API key)"
                elif status_code == 403:
                    error_reason += " (Permission denied, check API key/plan)"

                log_message(
                    f"OpenRouter API HTTP Error: {error_reason}", always_print=True
                )
                raise TranslationError(
                    f"OpenRouter API HTTP Error: {error_reason}"
                ) from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"Connection error, retrying in {current_delay:.1f}s: {str(e)}",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            else:
                log_message(
                    f"OpenRouter connection failed after {max_retries + 1} attempts: {str(e)}",
                    always_print=True,
                )
                raise TranslationError(
                    f"OpenRouter API Connection Error after retries: {str(e)}"
                ) from e

    raise TranslationError(
        f"Failed to get response from OpenRouter API after {max_retries + 1} attempts."
    )
