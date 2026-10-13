import json
from typing import Any

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message
from utils.model_metadata import supports_qwencloud_reasoning_effort

from ._http import COMMON_STATUS_HINTS, post_with_retries
from .openai import extract_responses_text


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

    # The last text part is the prompt; earlier ones label the image after them
    text_part = next((p for p in reversed(parts) if "text" in p), None)
    image_parts = [p for p in parts if p is not text_part]

    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for QwenCloud: No text prompt found."
        )

    url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if image_parts:
        content_list = []
        for part in image_parts:
            if "text" in part:
                content_list.append({"type": "input_text", "text": part["text"]})
                continue
            if (
                "inline_data" in part
                and "data" in part["inline_data"]
                and "mime_type" in part["inline_data"]
            ):
                mime_type = part["inline_data"]["mime_type"]
                base64_image = part["inline_data"]["data"]
                content_list.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{base64_image}",
                    }
                )
            else:
                log_message(f"Invalid image part format: {part}", always_print=True)
        content_list.append({"type": "input_text", "text": text_part["text"]})
        input_messages = [{"role": "user", "content": content_list}]
    else:
        input_messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text_part["text"]}],
            }
        ]

    payload: dict[str, Any] = {
        "model": model_name,
        "input": input_messages,
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

    thinking = generation_config.get("thinking")
    reasoning_effort = generation_config.get("reasoning_effort")
    if thinking:
        thinking_enabled = (
            thinking.get("type") == "enabled"
            if isinstance(thinking, dict)
            else bool(thinking)
        )
        if not thinking_enabled:
            payload["reasoning"] = {"effort": "none"}
        else:
            if reasoning_effort and reasoning_effort != "auto":
                if reasoning_effort in ("xhigh", "max", "high"):
                    payload["reasoning"] = {"effort": "high"}
                elif reasoning_effort in ("medium", "low", "minimal", "none"):
                    payload["reasoning"] = {"effort": reasoning_effort}
            else:
                payload["reasoning"] = {
                    "effort": (
                        "high"
                        if supports_qwencloud_reasoning_effort(model_name)
                        else "medium"
                    )
                }
    elif reasoning_effort:
        if reasoning_effort == "none":
            payload["reasoning"] = {"effort": "none"}
        elif reasoning_effort in ("xhigh", "max", "high"):
            payload["reasoning"] = {"effort": "high"}
        elif reasoning_effort in ("medium", "low", "minimal"):
            payload["reasoning"] = {"effort": reasoning_effort}

    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]

    payload = {k: v for k, v in payload.items() if v is not None}

    def parse(result: dict[str, Any]) -> str | None:
        if result.get("error"):
            error_obj = result.get("error", {})
            error_msg = (
                error_obj.get("message", "Unknown error")
                if isinstance(error_obj, dict)
                else str(error_obj)
            )
            raise TranslationError(f"QwenCloud API returned error: {error_msg}")

        output_text = extract_responses_text(result)
        if output_text:
            return output_text

        log_message(
            f"No text content in QwenCloud response. Status: {result.get('status')}, "
            f"details: {result.get('incomplete_details')}",
            always_print=True,
        )
        log_message(
            f"Full response: {json.dumps(result, indent=2)}",
            verbose=debug,
        )
        return None

    return post_with_retries(
        url,
        headers,
        payload,
        parse,
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
