import json
from typing import Any

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message
from utils.model_metadata import supports_meta_reasoning_effort

from ._http import COMMON_STATUS_HINTS, post_with_retries
from .openai import extract_responses_text


def call_meta_model_endpoint(
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
    Calls the Meta Model API Responses endpoint with the provided data and handles retries.

    Args:
        api_key (str): Meta Model API key.
        model_name (str): Meta Model API model to use (e.g., muse-wonder-1.0, muse-spark-1.3, muse-spark-1.2).
        parts (List[Dict[str, Any]]): List of content parts (text and optional images).
        generation_config (Dict[str, Any]): Configuration for generation.
        system_prompt (Optional[str]): System prompt for the conversation.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable web search tool for Meta Model API.

    Returns:
        Optional[str]: The raw text content from the API response if successful.
    """
    if not api_key:
        raise ValidationError("API key is required for Meta Model API endpoint")

    url = "https://api.meta.ai/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # The last text part is the prompt; earlier ones label the image after them
    text_part = next((p for p in reversed(parts) if "text" in p), None)
    image_parts = [p for p in parts if p is not text_part]

    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for Meta Model API: No text prompt found."
        )

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
                f"Invalid image part format for Meta Model API: {part}",
                always_print=True,
            )
    content_list.append({"type": "input_text", "text": text_part["text"]})

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

    reasoning_effort = generation_config.get("reasoning_effort")
    if (
        reasoning_effort
        and reasoning_effort not in ("auto", "none")
        and supports_meta_reasoning_effort(model_name)
    ):
        payload["reasoning"] = {"effort": reasoning_effort}

    # Built-in web search only exists on the Responses API
    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]

    payload = {k: v for k, v in payload.items() if v is not None}

    def parse(result: dict[str, Any]) -> str | None:
        if result.get("error"):
            error_obj = result["error"]
            error_msg = (
                error_obj.get("message", "Unknown error")
                if isinstance(error_obj, dict)
                else str(error_obj)
            )
            raise TranslationError(f"Meta Model API returned error: {error_msg}")

        output_text = extract_responses_text(result)
        if output_text:
            return output_text

        log_message(
            f"No text content in Meta Model API response. Status: {result.get('status')}, "
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
        label="Meta Model",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
        status_hints={
            **COMMON_STATUS_HINTS,
            404: "Model not found or permission denied",
        },
    )
