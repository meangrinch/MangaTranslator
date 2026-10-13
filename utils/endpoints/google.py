import json
from typing import Any

from utils.exceptions import ValidationError
from utils.logging import log_message

from ._http import post_with_retries

_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


def _extract_text(result: dict[str, Any], debug: bool) -> str | None:
    """Return the answer text, "" if the candidate has none, or None if the prompt was blocked."""
    block_reason = (result.get("promptFeedback") or {}).get("blockReason")
    candidates = result.get("candidates")
    if block_reason or not candidates:
        log_message(
            f"No candidates in Google response. Block reason: {block_reason or 'unknown'}",
            always_print=True,
        )
        log_message(f"Full response: {json.dumps(result, indent=2)}", verbose=debug)
        return None

    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts") or []
    # Skip thought parts (gemma-4); code execution parts separate
    # a preamble from the answer that follows them
    text = "".join(
        part["text"] if "text" in part and not part.get("thought") else "\n"
        for part in parts
    ).strip()
    # Fall back to the first part when every text part is a thought
    text = text or (parts[0].get("text", "").strip() if parts else "")
    if not text:
        log_message(
            f"No text content in Google response. Finish reason: {candidate.get('finishReason') or 'unknown'}",
            always_print=True,
        )
        log_message(f"Full response: {json.dumps(result, indent=2)}", verbose=debug)
    return text


def call_gemini_endpoint(
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
    enable_code_execution: bool = False,
) -> str | None:
    """
    Calls the Google API endpoint with the provided data and handles retries.

    Args:
        api_key (str): Google API key.
        model_name (str): Google model to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, etc.).
        system_prompt (Optional[str]): System prompt for the conversation.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting and connection errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable web search (Google Search) for up-to-date information.
        enable_code_execution (bool): Enable Gemini's code execution tool for image zoom/inspection.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       "" if the response has no text, or None if it was blocked or has no candidates.

    Raises:
        ValidationError: If API key is missing.
        TranslationError: If API call fails after retries for non-rate-limited HTTP errors,
                          connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for Google endpoint")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    # Key goes in a header so it never appears in URLs echoed by request errors
    headers = {"x-goog-api-key": api_key}

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
        "safetySettings": _SAFETY_SETTINGS,
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    tools = []
    if enable_web_search:
        tools.append({"googleSearch": {}})
    if enable_code_execution:
        tools.append({"code_execution": {}})
    if tools:
        payload["tools"] = tools

    return post_with_retries(
        url,
        headers,
        payload,
        lambda result: _extract_text(result, debug),
        label="Google",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
    )
