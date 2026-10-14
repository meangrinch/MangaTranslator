from typing import Any

from utils.exceptions import ValidationError
from utils.model_metadata import is_mimo_multimodal_model

from ._http import COMMON_STATUS_HINTS, post_with_retries
from .openai_compatible import read_chat_reply


def call_mimo_endpoint(
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
    Calls the Xiaomi MiMo API endpoint with the provided data and handles retries.
    MiMo uses OpenAI-compatible API format. Both mimo-v2.6-flash and
    mimo-v2.6-pro are multimodal and support hybrid reasoning via the thinking parameter.

    Args:
        api_key (str): MiMo API key.
        model_name (str): MiMo model to use.
        parts (List[Dict[str, Any]]): List of content parts (text and optional images).
        generation_config (Dict[str, Any]): Configuration for generation.
        system_prompt (Optional[str]): System prompt for the conversation.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.
        enable_web_search (bool): Enable MiMo's web search tool.

    Returns:
        Optional[str]: The raw text content from the API response if successful.
    """
    if not api_key:
        raise ValidationError("API key is required for MiMo endpoint")

    url = "https://api.xiaomimimo.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # The last text part is the prompt; earlier ones label the image after them
    text_part = next((p for p in reversed(parts) if "text" in p), None)
    image_parts = [p for p in parts if p is not text_part]

    if not text_part:
        raise ValidationError("Invalid 'parts' format for MiMo: No text prompt found.")

    is_multimodal = is_mimo_multimodal_model(model_name)

    if is_multimodal and image_parts:
        content_list = []
        for part in image_parts:
            if "text" in part:
                content_list.append({"type": "text", "text": part["text"]})
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
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    }
                )
        content_list.append({"type": "text", "text": text_part["text"]})
        user_content = content_list
    else:
        user_content = text_part["text"]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model_name,
        "messages": messages,
        "max_completion_tokens": generation_config.get("max_tokens", 4096),
    }

    temp = generation_config.get("temperature")
    if temp is not None:
        payload["temperature"] = min(temp, 1.0)

    top_p = generation_config.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p

    thinking = generation_config.get("thinking")
    if thinking:
        payload["thinking"] = thinking

    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]

    payload = {k: v for k, v in payload.items() if v is not None}

    return post_with_retries(
        url,
        headers,
        payload,
        lambda result: read_chat_reply(result, "MiMo", debug),
        label="MiMo",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
        status_hints={
            **COMMON_STATUS_HINTS,
            404: "Model not found or permission denied",
        },
    )
