import json
import time
from typing import Any, Dict, List, Optional

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message


def call_deepseek_endpoint(
    api_key: str,
    model_name: str,
    parts: List[Dict[str, Any]],
    generation_config: Dict[str, Any],
    system_prompt: Optional[str] = None,
    debug: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Optional[str]:
    """
    Calls the DeepSeek Chat Completions API endpoint with the provided data and handles retries.
    DeepSeek uses OpenAI-compatible API format and is text-only (no image support).

    Args:
        api_key (str): DeepSeek API key.
        model_name (str): DeepSeek model to use.
        parts (List[Dict[str, Any]]): List of content parts (text only, images are ignored).
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, max_tokens).
        system_prompt (Optional[str]): System prompt for the conversation.
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
        raise ValidationError("API key is required for DeepSeek endpoint")

    # DeepSeek is text-only, so we only extract text parts
    text_part = next((p for p in parts if "text" in p), None)
    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for DeepSeek: No text prompt found."
        )

    url = "https://api.deepseek.com/v1/chat/completions"
    api_model_name = model_name

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text_part["text"]})

    payload = {
        "model": api_model_name,
        "messages": messages,
        "max_tokens": generation_config.get("max_tokens", 4096),
    }

    temp = generation_config.get("temperature")
    if temp is not None:
        payload["temperature"] = min(temp, 2.0)  # DeepSeek supports up to 2.0

    top_p = generation_config.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p

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

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    finish_reason = choice.get("finish_reason")

                    if finish_reason == "content_filter":
                        return None
                    if finish_reason == "safety":
                        return None

                    message = choice.get("message")
                    if message and "content" in message:
                        content = message["content"]
                        return content.strip() if content else ""
                    else:
                        log_message(
                            "No message content in DeepSeek response", verbose=debug
                        )
                        return ""
                else:
                    log_message("No choices in DeepSeek response", always_print=True)
                    if "error" in result:
                        error_msg = result.get("error", {}).get(
                            "message", "Unknown error"
                        )
                        raise TranslationError(
                            f"DeepSeek API returned error: {error_msg}"
                        )
                    return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful DeepSeek API response: {str(e)}"
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

                raise TranslationError(
                    f"DeepSeek API HTTP Error: {error_reason}"
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
                    f"DeepSeek API Connection Error after retries: {str(e)}"
                ) from e

    raise TranslationError(
        f"Failed to get response from DeepSeek API after {max_retries + 1} attempts."
    )
