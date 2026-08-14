import json
import time
from typing import Any, Dict, List, Optional

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message


def get_opencode_endpoint_url(tier: str = "zen") -> str:
    """Return the Chat Completions URL for the specified OpenCode tier."""
    normalized_tier = (tier or "zen").strip().lower()
    if "go" in normalized_tier:
        return "https://opencode.ai/zen/go/v1/chat/completions"
    return "https://opencode.ai/zen/v1/chat/completions"


def call_opencode_endpoint(
    api_key: str,
    model_name: str,
    parts: List[Dict[str, Any]],
    generation_config: Dict[str, Any],
    system_prompt: Optional[str] = None,
    tier: str = "zen",
    debug: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
    base_delay: float = 1.0,
    enable_web_search: bool = False,
) -> Optional[str]:
    """Call the OpenCode Chat Completions API with the given prompt and image parts.

    Args:
        api_key (str): OpenCode API key.
        model_name (str): OpenCode model identifier.
        parts (List[Dict[str, Any]]): List containing text prompt and optional image parts.
        generation_config (Dict[str, Any]): Generation parameters (temperature, max_tokens, etc.).
        system_prompt (Optional[str]): Optional system instruction.
        tier (str): "zen" or "go" endpoint tier.
        debug (bool): Enable verbose logging.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum retry attempts for rate limits (429).
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable model's built-in web search tool if supported.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if an error occurs or no content is found after retries.

    Raises:
        ValidationError: If API key is missing or parts format is invalid.
        TranslationError: If API call fails after retries for non-rate-limited HTTP errors,
                          connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for OpenCode endpoint")

    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]

    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for OpenCode: No text prompt found."
        )

    url = get_opencode_endpoint_url(tier)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content: List[Dict[str, Any]] = []
    for img in image_parts:
        inline = img.get("inline_data", {})
        mime = inline.get("mime_type", "image/png")
        data = inline.get("data", "")
        if data:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                }
            )

    user_content.append({"type": "text", "text": text_part["text"]})
    messages.append({"role": "user", "content": user_content})

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": generation_config.get("max_tokens", 4096),
    }

    temp = generation_config.get("temperature")
    if temp is not None:
        payload["temperature"] = temp

    top_p = generation_config.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p

    reasoning_effort = generation_config.get("reasoning_effort")
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort

    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]

    payload = {k: v for k, v in payload.items() if v is not None}

    tier_label = "Go" if "go" in (tier or "").lower() else "Zen"

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"OpenCode ({tier_label}) API request to {url} (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message(f"Processing OpenCode ({tier_label}) response", verbose=debug)
            try:
                result = response.json()

                if "error" in result:
                    error_obj = result.get("error", {})
                    error_msg = (
                        error_obj.get("message", "Unknown error")
                        if isinstance(error_obj, dict)
                        else str(error_obj)
                    )
                    raise TranslationError(
                        f"OpenCode ({tier_label}) API returned error: {error_msg}"
                    )

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    message = choice.get("message")
                    if message and "content" in message:
                        content = message["content"]
                        return content.strip() if content else ""

                    output_text = choice.get("text")
                    if isinstance(output_text, str) and output_text.strip():
                        return output_text.strip()

                    finish_reason = choice.get("finish_reason") or "unknown"
                    log_message(
                        f"No message content in OpenCode response. Finish reason: {finish_reason}",
                        always_print=True,
                    )
                    log_message(
                        f"Full response: {json.dumps(result, indent=2)}",
                        verbose=debug,
                    )
                    return ""

                # Fallback check for output_text at top level
                if "output_text" in result and isinstance(result["output_text"], str):
                    return result["output_text"].strip()

                log_message(
                    f"No choices in OpenCode ({tier_label}) response",
                    always_print=True,
                )
                return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful OpenCode ({tier_label}) API response: {str(e)}"
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

            error_reason = f"Status {status_code}: {error_text}"
            if status_code == 429 and attempt == max_retries:
                error_reason = (
                    f"Rate limited after {max_retries + 1} attempts: {error_text}"
                )
            elif status_code == 400:
                error_reason += " (Check payload / model support)"
            elif status_code == 401:
                error_reason += " (Check OpenCode API key)"
            elif status_code == 403:
                error_reason += f" (Permission denied for {tier_label} tier, check subscription or credits)"
            elif status_code == 404:
                error_reason += (
                    f" (Model '{model_name}' not found on {tier_label} tier)"
                )

            raise TranslationError(
                f"OpenCode ({tier_label}) API HTTP Error: {error_reason}"
            ) from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"Connection error, retrying in {current_delay:.1f}s: {str(e)}",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            raise TranslationError(
                f"OpenCode ({tier_label}) API Connection Error after retries: {str(e)}"
            ) from e

    raise TranslationError(
        f"Failed to get response from OpenCode ({tier_label}) API after {max_retries + 1} attempts."
    )
