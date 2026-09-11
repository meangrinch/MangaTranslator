import json
import time
from typing import Any

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message


def call_deepseek_endpoint(
    api_key: str,
    model_name: str,
    parts: list[dict[str, Any]],
    generation_config: dict[str, Any],
    system_prompt: str | None = None,
    debug: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
    base_delay: float = 1.0,
    enable_web_search: bool = False,
) -> str | None:
    """
    Calls the DeepSeek Responses API endpoint with the provided data and handles retries.
    Supports text and multimodal images (e.g., deepseek-flash).

    Args:
        api_key (str): DeepSeek API key.
        model_name (str): DeepSeek model to use (e.g., deepseek-flash, deepseek-v4-pro).
        parts (List[Dict[str, Any]]): List of content parts (text and optional images).
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, max_tokens/max_output_tokens,
            thinking, reasoning_effort).
        system_prompt (Optional[str]): System prompt for the conversation.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable DeepSeek's web search tool.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if an error occurs or no content is found after retries.

    Raises:
        ValidationError: If API key is missing or parts format is invalid.
        TranslationError: If API call fails after retries for non-rate-limited HTTP errors,
                          connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for DeepSeek endpoint")

    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]

    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for DeepSeek: No text prompt found."
        )

    url = "https://api.deepseek.com/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if image_parts:
        content_list = []
        for part in image_parts:
            if (
                "inline_data" in part
                and "data" in part["inline_data"]
                and "mime_type" in part["inline_data"]
            ):
                mime_type = part["inline_data"]["mime_type"]
                base64_image = part["inline_data"]["data"]
                content_list.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{base64_image}",
                    }
                )
            else:
                log_message(
                    f"Invalid image part format for DeepSeek: {part}",
                    always_print=True,
                )
        content_list.append({"type": "input_text", "text": text_part["text"]})
        input_messages = [{"role": "user", "content": content_list}]
    else:
        input_messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text_part["text"]}],
            }
        ]

    max_output_tokens = generation_config.get(
        "max_output_tokens"
    ) or generation_config.get("max_tokens", 4096)

    payload: dict[str, Any] = {
        "model": model_name,
        "input": input_messages,
        "max_output_tokens": max_output_tokens,
    }

    if system_prompt:
        payload["instructions"] = system_prompt

    # Add thinking parameter if present
    thinking_config = generation_config.get("thinking")
    thinking_enabled = thinking_config and thinking_config.get("type") == "enabled"
    if thinking_config:
        payload["thinking"] = thinking_config

    reasoning_effort = generation_config.get("reasoning_effort")
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    # Thinking mode does not support temperature/top_p (no error, but no effect)
    if not thinking_enabled:
        temp = generation_config.get("temperature")
        if temp is not None:
            payload["temperature"] = min(temp, 2.0)

        top_p = generation_config.get("top_p")
        if top_p is not None:
            payload["top_p"] = top_p

    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]

    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"DeepSeek API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message("Processing DeepSeek response", verbose=debug)
            try:
                result = response.json()

                if "error" in result:
                    error_msg = result.get("error", {}).get("message", "Unknown error")
                    raise TranslationError(f"DeepSeek API returned error: {error_msg}")

                output_text = result.get("output_text")
                if isinstance(output_text, str) and output_text.strip():
                    return output_text.strip()

                finish_reason = result.get("finish_reason") or "unknown"
                log_message(
                    f"No text content in DeepSeek response. Finish reason: {finish_reason}",
                    always_print=True,
                )
                log_message(
                    f"Full response: {json.dumps(result, indent=2)}",
                    verbose=debug,
                )
                return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful DeepSeek API response: {e!s}"
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
                    error_reason += " (Invalid request format)"
                elif status_code == 401:
                    error_reason += " (Check API key)"
                elif status_code == 402:
                    error_reason += " (Insufficient balance, top up your account)"
                elif status_code == 403:
                    error_reason += " (Permission denied, check API key/plan)"
                elif status_code == 404:
                    error_reason += " (Endpoint or model not found)"
                elif status_code == 422:
                    error_reason += " (Invalid parameters)"

                raise TranslationError(
                    f"DeepSeek API HTTP Error: {error_reason}"
                ) from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"Connection error, retrying in {current_delay:.1f}s: {e!s}",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            else:
                raise TranslationError(
                    f"DeepSeek API Connection Error after retries: {e!s}"
                ) from e

    raise TranslationError(
        f"Failed to get response from DeepSeek API after {max_retries + 1} attempts."
    )
