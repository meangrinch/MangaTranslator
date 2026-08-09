import json
import time
from typing import Any, Dict, List, Optional

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message
from utils.model_metadata import supports_meta_reasoning_effort


def call_meta_model_endpoint(
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
    Calls the Meta Model API chat completions endpoint with the provided data and handles retries.

    Args:
        api_key (str): Meta Model API key.
        model_name (str): Meta Model API model to use (e.g., muse-spark-1.2, muse-spark-1.1).
        parts (List[Dict[str, Any]]): List of content parts (text and optional images).
        generation_config (Dict[str, Any]): Configuration for generation.
        system_prompt (Optional[str]): System prompt for the conversation.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable web search tool for Meta Model API.

    Returns:
        Optional[str]: The raw text content from the API response if successful.
    """
    if not api_key:
        raise ValidationError("API key is required for Meta Model API endpoint")

    url = "https://api.meta.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]

    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for Meta Model API: No text prompt found."
        )

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
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    }
                )
            else:
                log_message(
                    f"Invalid image part format for Meta Model API: {part}",
                    always_print=True,
                )
        content_list.append({"type": "text", "text": text_part["text"]})
        user_content = content_list
    else:
        user_content = text_part["text"]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": generation_config.get("max_tokens", 4096),
    }

    temp = generation_config.get("temperature")
    if temp is not None:
        payload["temperature"] = min(temp, 1.0)

    top_p = generation_config.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p

    reasoning_effort = generation_config.get("reasoning_effort")
    if (
        reasoning_effort
        and reasoning_effort not in ("auto", "none")
        and supports_meta_reasoning_effort(model_name)
    ):
        payload["reasoning_effort"] = reasoning_effort

    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]

    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"Meta Model API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message("Processing Meta Model API response", verbose=debug)
            try:
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    finish_reason = choice.get("finish_reason")

                    message = choice.get("message")
                    if message and "content" in message:
                        content = message["content"]
                        return content.strip() if content else ""
                    log_message(
                        f"No message content in Meta Model API response. Finish reason: {finish_reason}",
                        always_print=True,
                    )
                    log_message(
                        f"Full response: {json.dumps(result, indent=2)}",
                        verbose=debug,
                    )
                    return ""
                log_message("No choices in Meta Model API response", always_print=True)
                if "error" in result:
                    error_obj = result.get("error", {})
                    error_msg = (
                        error_obj.get("message", "Unknown error")
                        if isinstance(error_obj, dict)
                        else str(error_obj)
                    )
                    raise TranslationError(
                        f"Meta Model API returned error: {error_msg}"
                    )
                return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful Meta Model API response: {str(e)}"
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

            raise TranslationError(f"Meta Model API HTTP Error: {error_reason}") from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"Connection error, retrying in {current_delay:.1f}s: {str(e)}",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            raise TranslationError(
                f"Meta Model API Connection Error after retries: {str(e)}"
            ) from e

    raise TranslationError(
        f"Failed to get response from Meta Model API after {max_retries + 1} attempts."
    )
