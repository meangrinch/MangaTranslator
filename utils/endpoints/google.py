import json
import time
from typing import Any, Dict, List, Optional

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message


def call_gemini_endpoint(
    api_key: str,
    model_name: str,
    parts: List[Dict[str, Any]],
    generation_config: Dict[str, Any],
    system_prompt: Optional[str] = None,
    debug: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
    base_delay: float = 1.0,
    enable_grounding: bool = False,
) -> Optional[str]:
    """
    Calls the Google API endpoint with the provided data and handles retries.

    Args:
        api_key (str): Google API key.
        model_name (str): Google model to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, etc.).
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if blocked by safety settings or if no content is found after retries.

    Raises:
        ValueError: If API key is missing.
        RuntimeError: If API call fails after retries for non-rate-limited HTTP errors,
                      connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for Google endpoint")

    # Detect Gemini 3 models - they require v1alpha API for per-part media_resolution
    is_gemini_3 = "gemini-3" in (model_name or "").lower()
    api_version = "v1alpha" if is_gemini_3 else "v1beta"
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={api_key}"

    safety_settings_config = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
        "safetySettings": safety_settings_config,
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    if enable_grounding:
        payload["tools"] = [{"googleSearch": {}}]

    for attempt in range(max_retries + 1):
        current_delay = min(
            base_delay * (2**attempt), 16.0
        )  # Exponential backoff, max 16s
        try:
            log_message(
                f"Google API request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()

            log_message("Processing Google response", verbose=debug)
            try:
                result = response.json()
                prompt_feedback = result.get("promptFeedback")
                if prompt_feedback and prompt_feedback.get("blockReason"):
                    block_reason = prompt_feedback.get("blockReason")
                    return None

                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if candidate.get("finishReason") == "SAFETY":
                        safety_ratings = candidate.get("safetyRatings", [])
                        block_reason = "Unknown Safety Reason"
                        if safety_ratings:
                            block_reason = safety_ratings[0].get(
                                "category", block_reason
                            )
                        return None

                    content_parts = candidate.get("content", {}).get("parts", [{}])
                    if content_parts and "text" in content_parts[0]:
                        return content_parts[0].get("text", "").strip()
                    else:
                        log_message("No text content in response", verbose=debug)
                        return ""

                else:
                    log_message("No candidates in Google response", always_print=True)
                    if prompt_feedback and prompt_feedback.get("blockReason"):
                        block_reason = prompt_feedback.get("blockReason")
                        return None
                    return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful Google API response: {str(e)}"
                ) from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]  # Limit error text length

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

                raise TranslationError(f"Google API HTTP Error: {error_reason}") from e

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
                    f"Google API Connection Error after retries: {str(e)}"
                ) from e

    raise TranslationError(
        f"Failed to get response from Google API after {max_retries + 1} attempts."
    )
