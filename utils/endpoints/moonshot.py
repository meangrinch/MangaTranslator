import json
import time
from typing import Any, Dict, List, Optional

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message


def call_moonshot_endpoint(
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
    Calls the Moonshot AI (Kimi) API endpoint with the provided data and handles retries.
    Moonshot uses OpenAI-compatible API format and is text-only (no image support for kimi-k2 models).

    Args:
        api_key (str): Moonshot API key.
        model_name (str): Moonshot model to use (e.g., kimi-k2-thinking, kimi-k2-turbo-preview).
        parts (List[Dict[str, Any]]): List of content parts (text only, images are ignored).
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, max_tokens).
        system_prompt (Optional[str]): System prompt for the conversation.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable Moonshot's built-in web search for up-to-date information.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if an error occurs or no content is found after retries.

    Raises:
        ValueError: If API key is missing or parts format is invalid.
        RuntimeError: If API call fails after retries for non-rate-limited HTTP errors,
                      connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for Moonshot endpoint")

    # Moonshot kimi-k2 models are text-only, so we only extract text parts
    text_part = next((p for p in parts if "text" in p), None)
    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for Moonshot: No text prompt found."
        )

    url = "https://api.moonshot.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text_part["text"]})

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": generation_config.get("max_tokens", 4096),
    }

    # Temperature: Moonshot supports 0-1, default 0.6 for kimi-k2, 1.0 for kimi-k2-thinking
    temp = generation_config.get("temperature")
    if temp is not None:
        payload["temperature"] = min(temp, 1.0)  # Moonshot caps at 1.0

    top_p = generation_config.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p

    # Handle web search via Moonshot's builtin_search tool
    if enable_web_search:
        payload["tools"] = [{"type": "builtin_search"}]

    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"Moonshot API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message("Processing Moonshot response", verbose=debug)
            try:
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    finish_reason = choice.get("finish_reason")

                    # Handle content filter rejection
                    if finish_reason == "content_filter":
                        log_message(
                            "Moonshot response blocked due to content filter",
                            always_print=True,
                        )
                        return None

                    message = choice.get("message")
                    if message and "content" in message:
                        content = message["content"]
                        return content.strip() if content else ""
                    else:
                        log_message(
                            "No message content in Moonshot response", verbose=debug
                        )
                        return ""
                else:
                    log_message("No choices in Moonshot response", always_print=True)
                    if "error" in result:
                        error_obj = result.get("error", {})
                        error_msg = (
                            error_obj.get("message", "Unknown error")
                            if isinstance(error_obj, dict)
                            else str(error_obj)
                        )
                        raise TranslationError(
                            f"Moonshot API returned error: {error_msg}"
                        )
                    return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful Moonshot API response: {str(e)}"
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
                elif status_code == 404:
                    error_reason += " (Model not found or permission denied)"

                raise TranslationError(
                    f"Moonshot API HTTP Error: {error_reason}"
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
                    f"Moonshot API Connection Error after retries: {str(e)}"
                ) from e

    raise TranslationError(
        f"Failed to get response from Moonshot API after {max_retries + 1} attempts."
    )
