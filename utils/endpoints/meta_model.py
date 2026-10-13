import json
import time
from typing import Any

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message
from utils.model_metadata import supports_meta_reasoning_effort

from .openai import extract_responses_text


def call_meta_model_endpoint(
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
    Calls the Meta Model API Responses endpoint with the provided data and handles retries.

    Args:
        api_key (str): Meta Model API key.
        model_name (str): Meta Model API model to use (e.g., muse-wonder-1.0, muse-spark-1.3, muse-spark-1.2).
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

    url = "https://api.meta.ai/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # The last text part is the prompt; earlier ones label the image after them
    text_part = next((p for p in reversed(parts) if "text" in p), None)
    image_parts = [p for p in parts if p is not text_part]

    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for Meta Model API: No text prompt found."
        )

    content_list = []
    for part in image_parts:
        if "text" in part:
            content_list.append({"type": "input_text", "text": part["text"]})
            continue
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
                f"Invalid image part format for Meta Model API: {part}",
                always_print=True,
            )
    content_list.append({"type": "input_text", "text": text_part["text"]})

    payload: dict[str, Any] = {
        "model": model_name,
        "input": [{"role": "user", "content": content_list}],
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

    reasoning_effort = generation_config.get("reasoning_effort")
    if (
        reasoning_effort
        and reasoning_effort not in ("auto", "none")
        and supports_meta_reasoning_effort(model_name)
    ):
        payload["reasoning"] = {"effort": reasoning_effort}

    # Built-in web search only exists on the Responses API
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

                if result.get("error"):
                    error_obj = result["error"]
                    error_msg = (
                        error_obj.get("message", "Unknown error")
                        if isinstance(error_obj, dict)
                        else str(error_obj)
                    )
                    raise TranslationError(
                        f"Meta Model API returned error: {error_msg}"
                    )

                output_text = extract_responses_text(result)
                if output_text:
                    return output_text

                log_message(
                    f"No text content in Meta Model API response. Status: {result.get('status')}, "
                    f"details: {result.get('incomplete_details')}",
                    always_print=True,
                )
                log_message(
                    f"Full response: {json.dumps(result, indent=2)}",
                    verbose=debug,
                )
                return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful Meta Model API response: {e!s}"
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
                    f"Connection error, retrying in {current_delay:.1f}s: {e!s}",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            raise TranslationError(
                f"Meta Model API Connection Error after retries: {e!s}"
            ) from e

    raise TranslationError(
        f"Failed to get response from Meta Model API after {max_retries + 1} attempts."
    )
