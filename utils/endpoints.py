import json
import time
from typing import Any, Dict, List, Optional

import requests

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
) -> Optional[str]:
    """
    Calls the Gemini API endpoint with the provided data and handles retries.

    Args:
        api_key (str): Gemini API key.
        model_name (str): Gemini model to use.
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
        raise ValueError("API key is required for Gemini endpoint")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

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

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)  # Exponential backoff, max 16s
        try:
            log_message(f"Making Gemini API request (Attempt {attempt + 1}/{max_retries + 1})...", verbose=debug)
            # print(f"Payload: {json.dumps(payload, indent=2)}")

            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()

            log_message("Gemini API response OK (200), processing result...", verbose=debug)
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
                            block_reason = safety_ratings[0].get("category", block_reason)
                        return None

                    content_parts = candidate.get("content", {}).get("parts", [{}])
                    if content_parts and "text" in content_parts[0]:
                        return content_parts[0].get("text", "").strip()
                    else:
                        log_message("API Warning: No text content found in the first candidate part.", verbose=debug)
                        return ""

                else:
                    log_message("API Warning: No candidates found in successful response.", always_print=True)
                    if prompt_feedback and prompt_feedback.get("blockReason"):
                        block_reason = prompt_feedback.get("blockReason")
                        return None
                    return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"Error processing successful Gemini API response: {str(e)}") from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]  # Limit error text length

            if status_code == 429 and attempt < max_retries:
                log_message(
                    f"API Error: 429 Rate Limit Exceeded. Retrying in {current_delay:.1f} seconds...", verbose=debug
                )
                time.sleep(current_delay)
                continue
            else:
                error_reason = f"Status Code: {status_code}, Response: {error_text}"
                if status_code == 429 and attempt == max_retries:
                    error_reason = (
                        f"Persistent rate limiting after {max_retries + 1} attempts. Last error: {error_text}"
                    )

                raise RuntimeError(f"Gemini API HTTP Error: {error_reason}") from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"Gemini API Connection Error: {str(e)}. Retrying in {current_delay:.1f} seconds...", verbose=debug
                )
                time.sleep(current_delay)
                continue
            else:
                raise RuntimeError(f"Gemini API Connection Error after retries: {str(e)}") from e

    raise RuntimeError(f"Failed to get response from Gemini API after {max_retries + 1} attempts.")


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
) -> Optional[str]:
    """
    Calls the OpenAI Responses API endpoint with the provided data and handles retries.

    Args:
        api_key (str): OpenAI API key.
        model_name (str): OpenAI model to use (e.g., "gpt-4o").
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
        raise ValueError("API key is required for OpenAI endpoint")
    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]
    if not text_part:
        raise ValueError("Invalid 'parts' format for OpenAI: No text prompt found.")

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Build Responses API input content
    input_content = []
    for part in image_parts:
        if "inline_data" in part and "data" in part["inline_data"] and "mime_type" in part["inline_data"]:
            mime_type = part["inline_data"]["mime_type"]
            base64_image = part["inline_data"]["data"]
            # Responses API image input
            input_content.append({
                "type": "input_image",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            })
        else:
            log_message(f"Warning: Skipping invalid image part format in OpenAI request: {part}", always_print=True)
    input_content.append({"type": "input_text", "text": text_part["text"]})

    payload = {
        "model": model_name,
        "input": [{"role": "user", "content": input_content}],
        "temperature": generation_config.get("temperature") if model_name != "o4-mini" else 1.0,
        "top_p": generation_config.get("top_p") if model_name != "o4-mini" else None,
        "max_output_tokens": generation_config.get("max_output_tokens", 2048),
    }
    if system_prompt:
        payload["instructions"] = system_prompt
    payload = {k: v for k, v in payload.items() if v is not None}

    # Conditionally include OpenAI reasoning and verbosity controls
    try:
        lower_model = (model_name or "").lower()
        is_gpt5_series = lower_model.startswith("gpt-5")
        is_reasoning_capable = (
            is_gpt5_series
            or lower_model.startswith("o1")
            or lower_model.startswith("o3")
            or lower_model.startswith("o4-mini")
        )

        if is_reasoning_capable:
            effort = generation_config.get("reasoning_effort")
            if effort:
                if effort == "minimal" and not is_gpt5_series:
                    effort_to_send = "low"
                else:
                    effort_to_send = effort
                payload["reasoning"] = {"effort": effort_to_send}

        if is_gpt5_series:
            payload["text"] = {"verbosity": "low"}
    except Exception:
        # Do not fail the request if model detection or mapping has issues
        pass

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            attempt_info = f"(Attempt {attempt + 1}/{max_retries + 1})"
            log_message(
                f"Making OpenAI Responses API request {attempt_info}...",
                verbose=debug,
            )
            # print(f"Payload: {json.dumps(payload, indent=2)}")

            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()

            log_message("OpenAI Responses API response OK (200), processing result...", verbose=debug)
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
                        content_blocks = item.get("content") if isinstance(item, dict) else None
                        if isinstance(content_blocks, list):
                            for block in content_blocks:
                                if isinstance(block, dict):
                                    text_val = block.get("text") or block.get("output_text")
                                    if isinstance(text_val, str) and text_val.strip():
                                        return text_val.strip()

                log_message("API Warning: No textual content found in Responses output.", always_print=True)
                return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"Error processing successful OpenAI API response: {str(e)}") from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]

            if status_code == 429 and attempt < max_retries:
                log_message(
                    f"API Error: 429 Rate Limit Exceeded. Retrying in {current_delay:.1f} seconds...", verbose=debug
                )
                time.sleep(current_delay)
                continue
            else:
                error_reason = f"Status Code: {status_code}, Response: {error_text}"
                if status_code == 429 and attempt == max_retries:
                    error_reason = (
                        f"Persistent rate limiting after {max_retries + 1} attempts. Last error: {error_text}"
                    )
                elif status_code == 400:
                    error_reason += " (Check model name and request payload)"  # Often indicates a bad request

                raise RuntimeError(f"OpenAI Responses API HTTP Error: {error_reason}") from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"OpenAI API Connection Error: {str(e)}. Retrying in {current_delay:.1f} seconds...", verbose=debug
                )
                time.sleep(current_delay)
                continue
            else:
                raise RuntimeError(f"OpenAI Responses API Connection Error after retries: {str(e)}") from e

    raise RuntimeError(f"Failed to get response from OpenAI Responses API after {max_retries + 1} attempts.")


def call_anthropic_endpoint(
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
    Calls the Anthropic Messages API endpoint with the provided data and handles retries.

    Args:
        api_key (str): Anthropic API key.
        model_name (str): Anthropic model to use (e.g., "claude-3-opus-20240229").
        parts (List[Dict[str, Any]]): List of content parts (text, images).
                                      # Assumes the first part is the system/main prompt, subsequent are images.
        generation_config (Dict[str, Any]): Configuration for generation (temp <= 1.0, top_p, top_k, max_tokens).
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
        raise ValueError("API key is required for Anthropic endpoint")
    user_prompt_part = None
    image_parts = []
    for part in reversed(parts):
        if "text" in part and user_prompt_part is None:
            user_prompt_part = part
        elif "inline_data" in part:
            image_parts.insert(0, part)

    if not user_prompt_part:
        raise ValueError("Invalid 'parts' format for Anthropic: No text prompt found for user message.")

    content_parts = image_parts

    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}

    messages = []
    user_prompt_text = user_prompt_part["text"]
    user_content = []
    for part in content_parts:
        if "inline_data" in part and "data" in part["inline_data"] and "mime_type" in part["inline_data"]:
            mime_type = part["inline_data"]["mime_type"]
            base64_image = part["inline_data"]["data"]
            user_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image,
                    },
                }
            )
        else:
            log_message(f"Warning: Skipping invalid image part format in Anthropic request: {part}", always_print=True)

    user_content.append({"type": "text", "text": user_prompt_text})
    if not user_content:
        raise ValueError("No valid content (images/text) could be prepared for Anthropic user message.")

    messages.append({"role": "user", "content": user_content})

    temp = generation_config.get("temperature")
    clamped_temp = min(temp, 1.0) if temp is not None else None

    payload = {
        "model": model_name,
        "system": system_prompt,
        "messages": messages,
        "temperature": clamped_temp,
        "top_p": generation_config.get("top_p"),
        "top_k": generation_config.get("top_k"),
        "max_tokens": generation_config.get("max_tokens", 2048),
    }
    # Include Anthropic thinking parameter for supported models when requested
    try:
        if generation_config.get("anthropic_thinking"):
            payload["thinking"] = {"type": "enabled", "budget_tokens": 8192}
            beta_header_value = generation_config.get("anthropic_beta") or "thinking-v1"
            headers["anthropic-beta"] = beta_header_value
    except Exception:
        pass
    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(f"Making Anthropic API request (Attempt {attempt + 1}/{max_retries + 1})...", verbose=debug)
            # print(f"Payload: {json.dumps(payload, indent=2)}")

            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()

            log_message("Anthropic API response OK (200), processing result...", verbose=debug)
            try:
                result = response.json()

                if result.get("type") == "error":
                    error_data = result.get("error", {})
                    error_type = error_data.get("type", "unknown_error")
                    error_message = error_data.get("message", "No error message provided.")
                    raise RuntimeError(f"Anthropic API returned error: {error_type} - {error_message}")

                if "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0:
                    text_content = ""
                    for block in result["content"]:
                        if block.get("type") == "text":
                            text_content = block.get("text", "")
                            break

                    stop_reason = result.get("stop_reason")
                    if stop_reason == "max_tokens":
                        log_message(
                            "API Warning: Anthropic response truncated due to max_tokens limit.", always_print=True
                        )
                    elif stop_reason == "stop_sequence":
                        pass
                    elif stop_reason:
                        log_message(f"Anthropic response finished with reason: {stop_reason}", verbose=debug)

                    return text_content.strip()

                else:
                    log_message(
                        "API Warning: No text content block found in successful Anthropic response.", always_print=True
                    )
                    return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"Error processing successful Anthropic API response: {str(e)}") from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]

            if status_code == 429 and attempt < max_retries:
                log_message(
                    f"API Error: 429 Rate Limit Exceeded. Retrying in {current_delay:.1f} seconds...", verbose=debug
                )
                time.sleep(current_delay)
                continue
            else:
                error_reason = f"Status Code: {status_code}, Response: {error_text}"
                if status_code == 429 and attempt == max_retries:
                    error_reason = (
                        f"Persistent rate limiting after {max_retries + 1} attempts. Last error: {error_text}"
                    )
                elif status_code == 400:
                    error_reason += " (Check model name, API key, payload structure, or max_tokens)"
                elif status_code == 401:
                    error_reason += " (Check API key)"
                elif status_code == 403:
                    error_reason += " (Permission denied, check API key/plan)"
                log_message(f"Anthropic API HTTP Error: {error_reason}", always_print=True)
                raise RuntimeError(f"Anthropic API HTTP Error: {error_reason}") from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"Anthropic API Connection Error: {str(e)}. Retrying in {current_delay:.1f} seconds...",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            else:
                raise RuntimeError(f"Anthropic API Connection Error after retries: {str(e)}") from e

    raise RuntimeError(f"Failed to get response from Anthropic API after {max_retries + 1} attempts.")


def call_openrouter_endpoint(
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
    Calls the OpenRouter Chat Completions API endpoint (OpenAI compatible) and handles retries.

    Args:
        api_key (str): OpenRouter API key.
        model_name (str): OpenRouter model ID (e.g., "openai/gpt-4o", "anthropic/claude-3-haiku").
        parts (List[Dict[str, Any]]): List of content parts (text, images).
                                      # Assumes the first part is the text prompt, subsequent are images.
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, top_k, max_tokens).
                                             # Parameter restrictions (temp clamp, no top_k) are applied
                                             # based on the model_name if it indicates OpenAI or Anthropic.
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
        raise ValueError("API key is required for OpenRouter endpoint")
    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]
    if not text_part:
        raise ValueError("Invalid 'parts' format for OpenRouter: No text prompt found.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/meangrinch/MangaTranslator",
        "X-Title": "MangaTranslator",
    }

    # Transform parts to OpenAI messages format
    messages = []
    user_content = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for part in image_parts:
        if "inline_data" in part and "data" in part["inline_data"] and "mime_type" in part["inline_data"]:
            mime_type = part["inline_data"]["mime_type"]
            base64_image = part["inline_data"]["data"]
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
        else:
            log_message(f"Warning: Skipping invalid image part format in OpenRouter request: {part}", always_print=True)
    user_content.append({"type": "text", "text": text_part["text"]})
    messages.append({"role": "user", "content": user_content})

    # Map generation config with provider-specific restrictions
    # Reasoning-aware max tokens: allow higher limit if indicated by caller
    payload = {"model": model_name, "messages": messages, "max_tokens": generation_config.get("max_tokens", 2048)}

    is_openai_model = "openai/" in model_name or model_name.startswith("gpt-")
    is_anthropic_model = "anthropic/" in model_name or model_name.startswith("claude-")

    temp = generation_config.get("temperature")
    if temp is not None:
        if is_anthropic_model or is_openai_model:
            payload["temperature"] = min(temp, 1.0)
        else:
            payload["temperature"] = temp

    top_p = generation_config.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p

    top_k = generation_config.get("top_k")
    if top_k is not None and not is_openai_model and not is_anthropic_model:
        payload["top_k"] = top_k

    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(f"Making OpenRouter API request (Attempt {attempt + 1}/{max_retries + 1})...", verbose=debug)
            # print(f"Payload: {json.dumps(payload, indent=2)}")

            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()

            log_message("OpenRouter API response OK (200), processing result...", verbose=debug)
            try:
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    finish_reason = choice.get("finish_reason")

                    if finish_reason == "content_filter":
                        log_message(
                            "API Error: Content generation blocked by OpenRouter content filter.", always_print=True
                        )
                        return None

                    message = choice.get("message")
                    if message and "content" in message:
                        content = message["content"]
                        return content.strip() if content else ""
                    else:
                        log_message("API Warning: No message content found in the first choice.", verbose=debug)
                        return ""
                else:
                    if "error" in result:
                        error_msg = result.get("error", {}).get("message", "Unknown error")
                        raise RuntimeError(f"OpenRouter API returned error: {error_msg}")
                    return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                log_message(f"Error processing successful OpenRouter API response: {str(e)}", always_print=True)
                raise RuntimeError(f"Error processing successful OpenRouter API response: {str(e)}") from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]

            if status_code == 429 and attempt < max_retries:
                log_message(
                    f"API Error: 429 Rate Limit Exceeded. Retrying in {current_delay:.1f} seconds...", verbose=debug
                )
                time.sleep(current_delay)
                continue
            else:
                error_reason = f"Status Code: {status_code}, Response: {error_text}"
                if status_code == 429 and attempt == max_retries:
                    error_reason = (
                        f"Persistent rate limiting after {max_retries + 1} attempts. Last error: {error_text}"
                    )
                elif status_code == 400:
                    error_reason += " (Check model name and request payload)"  # Potential 400 reasons
                elif status_code == 401:
                    error_reason += " (Check API key)"  # Potential 401 reason
                elif status_code == 403:
                    error_reason += " (Permission denied, check API key/plan)"  # Potential 403 reason

                log_message(f"OpenRouter API HTTP Error: {error_reason}", always_print=True)
                raise RuntimeError(f"OpenRouter API HTTP Error: {error_reason}") from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"OpenRouter API Connection Error: {str(e)}. Retrying in {current_delay:.1f} seconds...",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            else:
                log_message(
                    f"OpenRouter API Connection Error after {max_retries + 1} attempts: {str(e)}", always_print=True
                )
                raise RuntimeError(f"OpenRouter API Connection Error after retries: {str(e)}") from e

    log_message(
        f"Failed to get response from OpenRouter API after {max_retries + 1} attempts due to repeated errors.",
        always_print=True,
    )
    raise RuntimeError(f"Failed to get response from OpenRouter API after {max_retries + 1} attempts.")


# --- OpenRouter model metadata cache & reasoning detection ---
_OPENROUTER_MODELS_META: Dict[str, Dict[str, Any]] = {}


def _ensure_openrouter_models_meta_loaded(debug: bool = False) -> None:
    """Loads and caches OpenRouter models metadata once per process.

    Populates a mapping of model id -> model dict (including description/architecture).
    """
    global _OPENROUTER_MODELS_META
    if _OPENROUTER_MODELS_META:
        return
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        response.raise_for_status()
        data = response.json()
        all_models = data.get("data", [])
        for model in all_models:
            mid = str(model.get("id", "")).lower()
            if mid:
                _OPENROUTER_MODELS_META[mid] = model
    except Exception as e:
        # Do not raise; lack of metadata should not block translations
        log_message(f"Warning: Could not load OpenRouter models metadata: {e}", always_print=True)


def openrouter_is_reasoning_model(model_name: str, debug: bool = False) -> bool:
    """Heuristically detects whether an OpenRouter model is a reasoning model.

    Uses keywords on model id and description, similar to how vision-capable models are detected.
    Keywords include: reasoning, reason, thinking, think, cot, chain-of-thought.
    """
    if not model_name:
        return False
    try:
        _ensure_openrouter_models_meta_loaded(debug=debug)
    except Exception:
        pass

    lm = model_name.lower()
    # Exclude models that are instruction-tuned (contain 'instruct' in the model id)
    if "instruct" in lm:
        return False
    description = ""
    meta = _OPENROUTER_MODELS_META.get(lm)
    if meta:
        description = str(meta.get("description", "")).lower()
    keywords = [
        "reasoning",
        "reason",
        "thinking",
        "think",
        "cot",
        "chain-of-thought",
        "chain_of_thought",
    ]
    text = lm + " " + description
    return any(kw in text for kw in keywords)


def call_openai_compatible_endpoint(
    base_url: str,
    api_key: Optional[str],
    model_name: str,
    parts: List[Dict[str, Any]],
    generation_config: Dict[str, Any],
    system_prompt: Optional[str] = None,
    debug: bool = False,
    timeout: int = 300,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> Optional[str]:
    """
    Calls a generic OpenAI-Compatible Chat Completions API endpoint and handles retries.

    Args:
        base_url (str): The base URL of the compatible endpoint (e.g., "http://localhost:11434/v1").
        api_key (Optional[str]): The API key, if required by the endpoint.
        model_name (str): The model ID to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
                                      # Assumes the first part is the text prompt, subsequent are images.
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, top_k, max_tokens).
                                            # Parameter restrictions (temp clamp, no top_k) might apply
                                            # depending on the underlying model.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if blocked by content filter or if no content is found after retries.

    Raises:
        ValueError: If base_url is missing or parts format is invalid.
        RuntimeError: If API call fails after retries for non-rate-limited HTTP errors,
                      connection errors, or response processing fails.
    """
    if not base_url:
        raise ValueError("Base URL is required for OpenAI-Compatible endpoint")
    text_part = next((p for p in parts if "text" in p), None)
    image_parts = [p for p in parts if "inline_data" in p]
    if not text_part:
        raise ValueError("Invalid 'parts' format for OpenAI-Compatible: No text prompt found.")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Transform parts to OpenAI messages format
    messages = []
    user_content = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for part in image_parts:
        if "inline_data" in part and "data" in part["inline_data"] and "mime_type" in part["inline_data"]:
            mime_type = part["inline_data"]["mime_type"]
            base64_image = part["inline_data"]["data"]
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
        else:
            log_message(
                f"Warning: Skipping invalid image part format in OpenAI-Compatible request: {part}", always_print=True
            )
    user_content.append({"type": "text", "text": text_part["text"]})
    messages.append({"role": "user", "content": user_content})

    payload = {"model": model_name, "messages": messages, "max_tokens": generation_config.get("max_tokens", 2048)}

    temp = generation_config.get("temperature")
    if temp is not None:
        payload["temperature"] = min(temp, 1.0)

    top_p = generation_config.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p

    top_k = generation_config.get("top_k")
    if top_k is not None:
        payload["top_k"] = top_k

    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"Making OpenAI-Compatible API request to {url} (Attempt {attempt + 1}/{max_retries + 1})...",
                verbose=debug,
            )
            # print(f"Payload: {json.dumps(payload, indent=2)}")

            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()

            log_message(f"OpenAI-Compatible API response OK (200) from {url}, processing result...", verbose=debug)
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
                        log_message("API Warning: No message content found in the first choice.", verbose=debug)
                        return ""
                else:
                    log_message(
                        "API Warning: No choices found in successful OpenAI-Compatible response.", always_print=True
                    )
                    if "error" in result:
                        error_msg = result.get("error", {}).get("message", "Unknown error")
                        raise RuntimeError(f"OpenAI-Compatible API returned error: {error_msg}")
                    return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"Error processing successful OpenAI-Compatible API response: {str(e)}") from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]

            if status_code == 429 and attempt < max_retries:
                log_message(
                    f"API Error: 429 Rate Limit Exceeded. Retrying in {current_delay:.1f} seconds...", verbose=debug
                )
                time.sleep(current_delay)
                continue
            else:
                error_reason = f"Status Code: {status_code}, Response: {error_text}"
                if status_code == 429 and attempt == max_retries:
                    error_reason = (
                        f"Persistent rate limiting after {max_retries + 1} attempts. Last error: {error_text}"
                    )
                elif status_code == 400:
                    error_reason += " (Check model name and request payload)"
                elif status_code == 401:
                    error_reason += " (Check API key if provided)"
                elif status_code == 403:
                    error_reason += " (Permission denied)"

                raise RuntimeError(f"OpenAI-Compatible API HTTP Error: {error_reason}") from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"OpenAI-Compatible API Connection Error: {str(e)}. Retrying in {current_delay:.1f} seconds...",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            else:
                raise RuntimeError(f"OpenAI-Compatible API Connection Error after retries: {str(e)}") from e

    raise RuntimeError(f"Failed to get response from OpenAI-Compatible API ({url}) after {max_retries + 1} attempts.")
