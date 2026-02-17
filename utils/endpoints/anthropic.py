import json
import time
from typing import Any, Dict, List, Optional

import requests

from core.config import calculate_reasoning_budget
from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message


def call_anthropic_endpoint(
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
    Calls the Anthropic Messages API endpoint with the provided data and handles retries.

    Args:
        api_key (str): Anthropic API key.
        model_name (str): Anthropic model to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
                                      # Assumes the first part is the system/main prompt, subsequent are images.
        generation_config (Dict[str, Any]): Configuration for generation (temp <= 1.0, top_p, top_k, max_tokens).
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if an error occurs or no content is found after retries.

    Raises:
        ValueError: If API key is missing or parts format is invalid.
        RuntimeError: If API call fails after retries for non-rate-limited HTTP errors,
                      connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for Anthropic endpoint")
    user_prompt_part = None
    image_parts = []
    for part in reversed(parts):
        if "text" in part and user_prompt_part is None:
            user_prompt_part = part
        elif "inline_data" in part:
            image_parts.insert(0, part)

    if not user_prompt_part:
        raise ValidationError(
            "Invalid 'parts' format for Anthropic: No text prompt found for user message."
        )

    content_parts = image_parts

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    messages = []
    user_prompt_text = user_prompt_part["text"]
    user_content = []
    for part in content_parts:
        if (
            "inline_data" in part
            and "data" in part["inline_data"]
            and "mime_type" in part["inline_data"]
        ):
            mime_type = part["inline_data"]["mime_type"]
            base64_image = part["inline_data"]["data"]
            user_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image,
                    },
                }
            )
        else:
            log_message(f"Invalid image part format: {part}", always_print=True)

    user_content.append({"type": "text", "text": user_prompt_text})
    if not user_content:
        raise ValidationError(
            "No valid content (images/text) could be prepared for Anthropic user message."
        )

    messages.append({"role": "user", "content": user_content})

    temp = generation_config.get("temperature")
    clamped_temp = min(temp, 1.0) if temp is not None else None

    payload = {
        "model": model_name,
        "system": system_prompt,
        "messages": messages,
        "temperature": clamped_temp,
        "top_k": generation_config.get("top_k"),
        "max_tokens": generation_config.get("max_tokens", 4096),
    }

    try:
        thinking_type = generation_config.get("thinking_type")
        reasoning_effort = generation_config.get("reasoning_effort")
        if thinking_type == "adaptive":
            # Opus 4.6: Adaptive thinking - Claude decides reasoning depth
            payload["thinking"] = {"type": "adaptive"}
        elif thinking_type == "enabled":
            # Older models: Budget-based thinking
            if reasoning_effort and reasoning_effort != "none":
                max_tokens_value = generation_config.get("max_tokens", 4096)
                budget_tokens = calculate_reasoning_budget(
                    max_tokens_value, reasoning_effort
                )
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
            elif reasoning_effort == "none":
                payload["thinking"] = {"type": "enabled", "budget_tokens": 0}
    except Exception:
        pass

    try:
        effort = generation_config.get("effort")
        is_46 = generation_config.get("is_46_model", False)
        valid_efforts = (
            ("max", "high", "medium", "low")
            if is_46
            else ("high", "medium", "low")
        )
        if effort and effort in valid_efforts:
            payload["output_config"] = {"effort": effort}
    except Exception:
        pass

    if enable_web_search:
        payload["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"Anthropic API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message("Processing Anthropic response", verbose=debug)
            try:
                result = response.json()

                if result.get("type") == "error":
                    error_data = result.get("error", {})
                    error_type = error_data.get("type", "unknown_error")
                    error_message = error_data.get(
                        "message", "No error message provided."
                    )
                    raise TranslationError(
                        f"Anthropic API returned error: {error_type} - {error_message}"
                    )

                if (
                    "content" in result
                    and isinstance(result["content"], list)
                    and len(result["content"]) > 0
                ):
                    text_content = ""
                    for block in result["content"]:
                        if block.get("type") == "text":
                            text_content = block.get("text", "")
                            break

                    return text_content.strip()

                else:
                    stop_reason = result.get("stop_reason")
                    log_message(
                        f"No text content in Anthropic response. Stop reason: {stop_reason}",
                        always_print=True,
                    )
                    log_message(
                        f"Full response: {json.dumps(result, indent=2)}",
                        verbose=debug,
                    )
                    return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful Anthropic API response: {str(e)}"
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
                    error_reason += (
                        " (Check model name, API key, payload, or max_tokens)"
                    )
                elif status_code == 401:
                    error_reason += " (Check API key)"
                elif status_code == 403:
                    error_reason += " (Permission denied, check API key/plan)"
                log_message(
                    f"Anthropic API HTTP Error: {error_reason}", always_print=True
                )
                raise TranslationError(
                    f"Anthropic API HTTP Error: {error_reason}"
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
                raise TranslationError(
                    f"Anthropic API Connection Error after retries: {str(e)}"
                ) from e

    raise TranslationError(
        f"Failed to get response from Anthropic API after {max_retries + 1} attempts."
    )
