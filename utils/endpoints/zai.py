import json
import time
from typing import Any, Dict, List, Optional

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message


def call_zai_endpoint(
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
    Calls the Z.ai API endpoint with the provided data and handles retries.

    Args:
        api_key (str): Z.ai API key.
        model_name (str): Z.ai model to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, max_tokens, thinking).
        system_prompt (Optional[str]): System prompt for the model.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable Z.ai's web search for up-to-date information.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if blocked by content filter or if no content is found after retries.

    Raises:
        ValueError: If API key is missing or parts format is invalid.
        RuntimeError: If API call fails after retries for non-rate-limited HTTP errors,
                      connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for Z.ai endpoint")

    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]
    if not text_part:
        raise ValidationError("Invalid 'parts' format for Z.ai: No text prompt found.")

    url = "https://api.z.ai/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Check if this is a vision model
    model_lower = (model_name or "").lower()
    is_vision_model = "glm-4.5v" in model_lower or "glm-4.6v" in model_lower

    if image_parts and is_vision_model:
        # Build multimodal content for vision models (glm-4.5v / glm-4.6v)
        user_content = []
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
    else:
        # Text-only content for non vision models
        messages.append({"role": "user", "content": text_part["text"]})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
        "top_k": generation_config.get("top_k"),
        "max_tokens": generation_config.get("max_tokens", 4096),
        "stream": False,
    }

    # Handle thinking/reasoning parameter
    thinking_config = generation_config.get("thinking")
    if thinking_config:
        payload["thinking"] = thinking_config

    # Handle web search
    if enable_web_search:
        payload["tools"] = [
            {
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_engine": "search_pro_jina",
                },
            }
        ]

    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"Z.ai API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message("Processing Z.ai response", verbose=debug)
            try:
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")

                    # Check for finish reason
                    finish_reason = choice.get("finish_reason", "")
                    if finish_reason == "sensitive":
                        log_message(
                            "Z.ai response blocked due to sensitive content",
                            always_print=True,
                        )
                        return None

                    if content and content.strip():
                        return content.strip()

                if "error" in result:
                    error_msg = result.get("error", {})
                    if isinstance(error_msg, dict):
                        error_msg = error_msg.get("message", "Unknown error")
                    raise TranslationError(f"Z.ai API returned error: {error_msg}")

                log_message("No text content in Z.ai response", verbose=debug)
                return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing Z.ai API response: {str(e)}"
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

                log_message(f"Z.ai API HTTP Error: {error_reason}", always_print=True)
                raise TranslationError(f"Z.ai API HTTP Error: {error_reason}") from e

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
                    f"Z.ai connection failed after {max_retries + 1} attempts: {str(e)}",
                    always_print=True,
                )
                raise TranslationError(
                    f"Z.ai API Connection Error after retries: {str(e)}"
                ) from e

    raise TranslationError(
        f"Failed to get response from Z.ai API after {max_retries + 1} attempts."
    )
