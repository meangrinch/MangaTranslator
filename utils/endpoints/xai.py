import json
import time
from typing import Any, Dict, List, Optional

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message


def call_xai_endpoint(
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
    Calls the xAI Responses API endpoint with the provided data and handles retries.

    Args:
        api_key (str): xAI API key.
        model_name (str): xAI model to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, max_tokens, reasoning_tokens).
        system_prompt (Optional[str]): System prompt for the model.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if blocked by content filter or if no content is found after retries.

    Raises:
        ValueError: If API key is missing or parts format is invalid.
        RuntimeError: If API call fails after retries for non-rate-limited HTTP errors,
                      connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for xAI endpoint")

    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]
    if not text_part:
        raise ValidationError("Invalid 'parts' format for xAI: No text prompt found.")

    url = "https://api.x.ai/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    input_messages = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})

    if image_parts:
        user_content = []
        for part in image_parts:
            if (
                "inline_data" in part
                and "data" in part["inline_data"]
                and "mime_type" in part["inline_data"]
            ):
                mime_type = part["inline_data"]["mime_type"]
                base64_image = part["inline_data"]["data"]

                # Check for per-image resolution first (from media_resolution metadata)
                part_res = part.get("media_resolution", {}).get("level")
                if part_res:
                    res_map = {
                        "MEDIA_RESOLUTION_UNSPECIFIED": "auto",
                        "MEDIA_RESOLUTION_LOW": "low",
                        "MEDIA_RESOLUTION_MEDIUM": "high",
                        "MEDIA_RESOLUTION_HIGH": "high",
                    }
                    detail = res_map.get(part_res, "high")
                else:
                    media_res = (
                        generation_config.get("media_resolution") or "auto"
                    ).lower()
                    detail = (
                        media_res if media_res in ["auto", "high", "low"] else "high"
                    )

                user_content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{base64_image}",
                        "detail": detail,
                    }
                )
            else:
                log_message(f"Invalid image part format: {part}", always_print=True)

        user_content.append({"type": "input_text", "text": text_part["text"]})
        input_messages.append({"role": "user", "content": user_content})
    else:
        input_messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text_part["text"]}],
            }
        )

    payload = {
        "model": model_name,
        "input": input_messages,
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
    }

    model_lower = (model_name or "").lower()
    is_reasoning_model = "reasoning" in model_lower or model_lower.startswith(
        "grok-4-fast-reasoning"
    )

    if is_reasoning_model:
        max_tokens_value = generation_config.get("max_tokens", 4096)
        reasoning_effort = generation_config.get("reasoning_effort", "high")
        # Mimics OpenAI's functionality
        if reasoning_effort and reasoning_effort in ["high", "low"]:
            payload["reasoning_effort"] = reasoning_effort
        payload["max_output_tokens"] = max_tokens_value
    else:
        payload["max_output_tokens"] = generation_config.get("max_tokens", 4096)

    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]
    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"xAI API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message("Processing xAI response", verbose=debug)
            try:
                result = response.json()

                if "output" in result and isinstance(result["output"], list):
                    for output_item in result["output"]:
                        if isinstance(output_item, dict) and "content" in output_item:
                            content = output_item["content"]
                            if isinstance(content, str) and content.strip():
                                return content.strip()
                            elif isinstance(content, list):
                                for content_block in content:
                                    if (
                                        isinstance(content_block, dict)
                                        and "text" in content_block
                                    ):
                                        text = content_block["text"]
                                        if text and text.strip():
                                            return text.strip()

                if "error" in result:
                    error_msg = result.get("error", {}).get("message", "Unknown error")
                    raise TranslationError(f"xAI API returned error: {error_msg}")

                log_message("No text content in xAI response", verbose=debug)
                return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing xAI API response: {str(e)}"
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

                log_message(f"xAI API HTTP Error: {error_reason}", always_print=True)
                raise TranslationError(f"xAI API HTTP Error: {error_reason}") from e

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
                    f"xAI connection failed after {max_retries + 1} attempts: {str(e)}",
                    always_print=True,
                )
                raise TranslationError(
                    f"xAI API Connection Error after retries: {str(e)}"
                ) from e

    raise TranslationError(
        f"Failed to get response from xAI API after {max_retries + 1} attempts."
    )
