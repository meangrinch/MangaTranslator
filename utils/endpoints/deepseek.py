from typing import Any

from utils.exceptions import ValidationError

from ._http import post_with_retries
from .openai import build_responses_content, read_responses_reply


def call_deepseek_endpoint(
    api_key: str,
    model_name: str,
    parts: list[dict[str, Any]],
    generation_config: dict[str, Any],
    system_prompt: str | None = None,
    debug: bool = False,
    timeout: int = 300,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str | None:
    """
    Calls the DeepSeek Responses API endpoint with the provided data and handles retries.
    Supports text and multimodal images (e.g., deepseek-flash).

    Args:
        api_key (str): DeepSeek API key.
        model_name (str): DeepSeek model to use (e.g., deepseek-flash, deepseek-v4-pro).
        parts (List[Dict[str, Any]]): List of content parts (text and optional images).
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, max_tokens/max_output_tokens,
            thinking, reasoning_effort).
        system_prompt (Optional[str]): System prompt for the conversation.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if an error occurs or no content is found after retries.

    Raises:
        ValidationError: If API key is missing or parts format is invalid.
        TranslationError: If API call fails after retries for non-rate-limited HTTP errors,
                          connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for DeepSeek endpoint")

    content_list = build_responses_content(parts, "DeepSeek")

    url = "https://api.deepseek.com/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    max_output_tokens = generation_config.get(
        "max_output_tokens"
    ) or generation_config.get("max_tokens", 4096)

    payload: dict[str, Any] = {
        "model": model_name,
        "input": [{"role": "user", "content": content_list}],
        "max_output_tokens": max_output_tokens,
    }

    if system_prompt:
        payload["instructions"] = system_prompt

    # The Responses API takes reasoning.effort; "none" turns thinking off
    thinking_config = generation_config.get("thinking")
    if thinking_config and thinking_config.get("type") == "enabled":
        # Thinking mode ignores temperature/top_p, so they are left out
        reasoning_effort = generation_config.get("reasoning_effort")
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
    else:
        if thinking_config:
            payload["reasoning"] = {"effort": "none"}

        temp = generation_config.get("temperature")
        if temp is not None:
            payload["temperature"] = min(temp, 2.0)

        top_p = generation_config.get("top_p")
        if top_p is not None:
            payload["top_p"] = top_p

    payload = {k: v for k, v in payload.items() if v is not None}

    return post_with_retries(
        url,
        headers,
        payload,
        lambda result: read_responses_reply(result, "DeepSeek", debug),
        label="DeepSeek",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
        status_hints={
            400: "Invalid request format",
            401: "Check API key",
            402: "Insufficient balance, top up your account",
            403: "Permission denied, check API key/plan",
            404: "Endpoint or model not found",
            422: "Invalid parameters",
        },
    )
