import json
import time
from typing import Any, Dict, List, Optional

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message
from utils.model_metadata import supports_qwencloud_reasoning_effort


def call_qwencloud_endpoint(
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
    Calls the QwenCloud (Alibaba Cloud Model Studio Responses API) endpoint with the provided data and handles retries.

    Args:
        api_key (str): QwenCloud API key.
        model_name (str): QwenCloud model to use.
        parts (List[Dict[str, Any]]): List of content parts (text and optional images).
        generation_config (Dict[str, Any]): Configuration for generation (temperature, top_p, max_output_tokens/max_tokens, thinking, reasoning_effort).
        system_prompt (Optional[str]): System prompt for the conversation.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable QwenCloud's built-in web search tool.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if an error occurs or no content is found after retries.

    Raises:
        ValidationError: If API key is missing or parts format is invalid.
        TranslationError: If API call fails after retries for non-rate-limited HTTP errors,
                          connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for QwenCloud endpoint")

    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]

    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for QwenCloud: No text prompt found."
        )

    url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses"
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
                log_message(f"Invalid image part format: {part}", always_print=True)
        content_list.append({"type": "input_text", "text": text_part["text"]})
        input_messages = [{"role": "user", "content": content_list}]
    else:
        input_messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text_part["text"]}],
            }
        ]

    payload: Dict[str, Any] = {
        "model": model_name,
        "input": input_messages,
        "max_output_tokens": generation_config.get("max_tokens", 4096),
    }

    if system_prompt:
        payload["instructions"] = system_prompt

    temp = generation_config.get("temperature")
    if temp is not None:
        payload["temperature"] = min(temp, 1.0)

    top_p = generation_config.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p

    thinking = generation_config.get("thinking")
    reasoning_effort = generation_config.get("reasoning_effort")
    if thinking:
        thinking_enabled = (
            thinking.get("type") == "enabled"
            if isinstance(thinking, dict)
            else bool(thinking)
        )
        if not thinking_enabled:
            payload["reasoning"] = {"effort": "none"}
        else:
            if reasoning_effort and reasoning_effort != "auto":
                if reasoning_effort in ("xhigh", "max", "high"):
                    payload["reasoning"] = {"effort": "high"}
                elif reasoning_effort in ("medium", "low", "minimal", "none"):
                    payload["reasoning"] = {"effort": reasoning_effort}
            else:
                payload["reasoning"] = {
                    "effort": (
                        "high"
                        if supports_qwencloud_reasoning_effort(model_name)
                        else "medium"
                    )
                }
    elif reasoning_effort:
        if reasoning_effort == "none":
            payload["reasoning"] = {"effort": "none"}
        elif reasoning_effort in ("xhigh", "max", "high"):
            payload["reasoning"] = {"effort": "high"}
        elif reasoning_effort in ("medium", "low", "minimal"):
            payload["reasoning"] = {"effort": reasoning_effort}

    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]

    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"QwenCloud API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message("Processing QwenCloud response", verbose=debug)
            try:
                result = response.json()

                if "error" in result:
                    error_obj = result.get("error", {})
                    error_msg = (
                        error_obj.get("message", "Unknown error")
                        if isinstance(error_obj, dict)
                        else str(error_obj)
                    )
                    raise TranslationError(f"QwenCloud API returned error: {error_msg}")

                output_text = result.get("output_text")
                if isinstance(output_text, str) and output_text.strip():
                    return output_text.strip()

                finish_reason = result.get("finish_reason") or "unknown"
                log_message(
                    f"No text content in QwenCloud response. Finish reason: {finish_reason}",
                    always_print=True,
                )
                log_message(
                    f"Full response: {json.dumps(result, indent=2)}",
                    verbose=debug,
                )
                return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful QwenCloud API response: {str(e)}"
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
                error_reason += " (Check payload)"
            elif status_code == 401:
                error_reason += " (Check API key)"
            elif status_code == 403:
                error_reason += " (Permission denied, check API key/plan)"
            elif status_code == 404:
                error_reason += " (Model not found or permission denied)"

            raise TranslationError(f"QwenCloud API HTTP Error: {error_reason}") from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"Connection error, retrying in {current_delay:.1f}s: {str(e)}",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            raise TranslationError(
                f"QwenCloud API Connection Error after retries: {str(e)}"
            ) from e

    raise TranslationError(
        f"Failed to get response from QwenCloud API after {max_retries + 1} attempts."
    )
