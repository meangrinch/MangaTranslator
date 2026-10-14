from typing import Any

from utils.exceptions import ValidationError
from utils.model_metadata import supports_qwencloud_reasoning_effort

from ._http import COMMON_STATUS_HINTS, post_with_retries
from .openai import build_responses_content, read_responses_reply


def call_qwencloud_endpoint(
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
    Calls the QwenCloud (Alibaba Cloud Model Studio Responses API) endpoint with the provided data and handles retries.

    Args:
        api_key (str): QwenCloud API key.
        model_name (str): QwenCloud model to use.
        parts (List[Dict[str, Any]]): List of content parts (text and optional images).
        generation_config (Dict[str, Any]): Configuration for generation (temperature, top_p, max_output_tokens/max_tokens, thinking, reasoning_effort).
        system_prompt (Optional[str]): System prompt for the conversation.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable QwenCloud's built-in web search tool.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if an error occurs or no content is found after retries.

    Raises:
        ValidationError: If API key is missing or parts format is invalid.
        TranslationError: If API call fails after retries for non-rate-limited HTTP errors,
                          connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for QwenCloud endpoint")

    content_list = build_responses_content(parts, "QwenCloud")

    url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

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

    # The Responses API has no thinking switch; reasoning.effort "none" turns it off
    thinking = generation_config.get("thinking")
    reasoning_effort = generation_config.get("reasoning_effort")
    if thinking:
        if isinstance(thinking, dict) and thinking.get("type") != "enabled":
            reasoning_effort = "none"
        elif not reasoning_effort or reasoning_effort == "auto":
            reasoning_effort = (
                "high" if supports_qwencloud_reasoning_effort(model_name) else "medium"
            )
    if reasoning_effort in ("xhigh", "max"):
        reasoning_effort = "high"
    if reasoning_effort in ("high", "medium", "low", "minimal", "none"):
        payload["reasoning"] = {"effort": reasoning_effort}

    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]

    payload = {k: v for k, v in payload.items() if v is not None}

    return post_with_retries(
        url,
        headers,
        payload,
        lambda result: read_responses_reply(result, "QwenCloud", debug),
        label="QwenCloud",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
        status_hints={
            **COMMON_STATUS_HINTS,
            404: "Model not found or permission denied",
        },
    )
