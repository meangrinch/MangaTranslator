import json
from typing import Any

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message

from ._http import post_with_retries
from .openai import extract_responses_text


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

    # The last text part is the prompt; earlier ones label the image after them
    text_part = next((p for p in reversed(parts) if "text" in p), None)
    image_parts = [p for p in parts if p is not text_part]

    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for DeepSeek: No text prompt found."
        )

    url = "https://api.deepseek.com/responses"
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
                log_message(
                    f"Invalid image part format for DeepSeek: {part}",
                    always_print=True,
                )
        content_list.append({"type": "input_text", "text": text_part["text"]})
        input_messages = [{"role": "user", "content": content_list}]
    else:
        input_messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text_part["text"]}],
            }
        ]

    max_output_tokens = generation_config.get(
        "max_output_tokens"
    ) or generation_config.get("max_tokens", 4096)

    payload: dict[str, Any] = {
        "model": model_name,
        "input": input_messages,
        "max_output_tokens": max_output_tokens,
    }

    if system_prompt:
        payload["instructions"] = system_prompt

    # The Responses API takes reasoning.effort; "none" turns thinking off
    thinking_config = generation_config.get("thinking")
    thinking_enabled = thinking_config and thinking_config.get("type") == "enabled"
    reasoning_effort = (
        generation_config.get("reasoning_effort") if thinking_enabled else "none"
    )
    if thinking_config and reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    # Thinking mode does not support temperature/top_p (no error, but no effect)
    if not thinking_enabled:
        temp = generation_config.get("temperature")
        if temp is not None:
            payload["temperature"] = min(temp, 2.0)

        top_p = generation_config.get("top_p")
        if top_p is not None:
            payload["top_p"] = top_p

    payload = {k: v for k, v in payload.items() if v is not None}

    def parse(result: dict[str, Any]) -> str | None:
        if result.get("error"):
            error_obj = result["error"]
            error_msg = (
                error_obj.get("message", "Unknown error")
                if isinstance(error_obj, dict)
                else str(error_obj)
            )
            raise TranslationError(f"DeepSeek API returned error: {error_msg}")

        output_text = extract_responses_text(result)
        if output_text:
            return output_text

        log_message(
            f"No text content in DeepSeek response. Status: {result.get('status')}, "
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
