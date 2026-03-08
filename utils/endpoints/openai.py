import json
import time
from typing import Any, Dict, List, Optional

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message
from utils.model_metadata import get_gpt5_generation, is_gpt5_series


def call_openai_endpoint(
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
    Calls the OpenAI Responses API endpoint with the provided data and handles retries.

    Args:
        api_key (str): OpenAI API key.
        model_name (str): OpenAI model to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
                                      # Assumes the first part is the text prompt, subsequent are images.
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, max_tokens).
                                            # 'top_k' is ignored by OpenAI Chat API.
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
        raise ValidationError("API key is required for OpenAI endpoint")
    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]
    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for OpenAI: No text prompt found."
        )

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    input_content = []
    for part in image_parts:
        if (
            "inline_data" in part
            and "data" in part["inline_data"]
            and "mime_type" in part["inline_data"]
        ):
            mime_type = part["inline_data"]["mime_type"]
            base64_image = part["inline_data"]["data"]
            input_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{base64_image}",
                }
            )
        else:
            log_message(f"Invalid image part format: {part}", always_print=True)
    input_content.append({"type": "input_text", "text": text_part["text"]})

    payload = {
        "model": model_name,
        "input": [{"role": "user", "content": input_content}],
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
        "max_output_tokens": generation_config.get("max_output_tokens", 4096),
    }
    if system_prompt:
        payload["instructions"] = system_prompt
    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        lower_model = (model_name or "").lower()
        is_chat_variant = "chat" in lower_model
        is_gpt5 = is_gpt5_series(model_name)
        gen = get_gpt5_generation(model_name)
        is_reasoning_capable = (
            is_gpt5
            or lower_model.startswith("o1")
            or lower_model.startswith("o3")
            or lower_model.startswith("o4-mini")
        )

        if is_reasoning_capable and not is_chat_variant:
            effort = generation_config.get("reasoning_effort")
            if effort:
                xhigh_capable = gen in ("5.2", "5.3", "5.4")
                none_capable = gen is not None and gen != "5"

                if none_capable and effort == "none":
                    payload["reasoning"] = {"effort": "none"}
                elif effort != "none":
                    effort_to_send = effort
                    if effort_to_send == "xhigh" and not xhigh_capable:
                        effort_to_send = "high"
                    if none_capable and effort_to_send == "minimal":
                        effort_to_send = "none"
                    elif effort_to_send == "minimal" and not is_gpt5:
                        effort_to_send = "low"
                    payload["reasoning"] = {"effort": effort_to_send}

        if is_gpt5 and not is_chat_variant:
            verbosity = generation_config.get("verbosity", "low")
            payload["text"] = {"verbosity": verbosity}

            # temp/top_p only allowed when effort is "none" (gpt-5.1+) or "minimal" (base gpt-5)
            current_effort = payload.get("reasoning", {}).get("effort")
            allow_sampling = (
                gen is not None and gen != "5" and current_effort == "none"
            ) or (gen == "5" and current_effort == "minimal")
            if not allow_sampling:
                payload.pop("temperature", None)
                payload.pop("top_p", None)

        elif is_reasoning_capable and not is_chat_variant:
            # Non-GPT-5 reasoning models (o1, o3, o4-mini) don't support temp/top_p
            payload.pop("temperature", None)
            payload.pop("top_p", None)
    except Exception:
        pass

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"OpenAI API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message("Processing OpenAI response", verbose=debug)
            try:
                result = response.json()

                # Prefer convenience field if available
                output_text = result.get("output_text")
                if isinstance(output_text, str) and output_text.strip():
                    return output_text.strip()

                # Fallback: parse output list content
                output_items = result.get("output")
                if isinstance(output_items, list):
                    for item in output_items:
                        content_blocks = (
                            item.get("content") if isinstance(item, dict) else None
                        )
                        if isinstance(content_blocks, list):
                            for block in content_blocks:
                                if isinstance(block, dict):
                                    text_val = block.get("text") or block.get(
                                        "output_text"
                                    )
                                    if isinstance(text_val, str) and text_val.strip():
                                        return text_val.strip()

                finish_reason = result.get("finish_reason") or "unknown"
                log_message(
                    f"No text content in OpenAI response. Finish reason: {finish_reason}",
                    always_print=True,
                )
                log_message(
                    f"Full response: {json.dumps(result, indent=2)}", verbose=debug
                )
                return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful OpenAI API response: {str(e)}"
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
                    error_reason += " (Check payload)"

                raise TranslationError(f"OpenAI API HTTP Error: {error_reason}") from e

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
                    f"OpenAI API Connection Error after retries: {str(e)}"
                ) from e

    raise TranslationError(
        f"Failed to get response from OpenAI Responses API after {max_retries + 1} attempts."
    )
