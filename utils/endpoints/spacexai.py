from typing import Any

from utils.exceptions import ValidationError
from utils.model_metadata import xai_reasoning_efforts

from ._http import COMMON_STATUS_HINTS, post_with_retries
from .openai import build_responses_content, read_responses_reply

# `detail` is auto, low, or high, so a medium resolution is sent as high
_DETAIL_BY_MEDIA_RESOLUTION_LEVEL = {
    "MEDIA_RESOLUTION_UNSPECIFIED": "auto",
    "MEDIA_RESOLUTION_LOW": "low",
    "MEDIA_RESOLUTION_MEDIUM": "high",
    "MEDIA_RESOLUTION_HIGH": "high",
}


def _image_detail(part: dict[str, Any], generation_config: dict[str, Any]) -> str:
    """Pick an image's `detail` from its own media resolution, else the request-wide one."""
    level = part.get("media_resolution", {}).get("level")
    if level:
        return _DETAIL_BY_MEDIA_RESOLUTION_LEVEL.get(level, "high")
    media_resolution = (generation_config.get("media_resolution") or "auto").lower()
    return media_resolution if media_resolution in ("auto", "high", "low") else "high"


def call_xai_endpoint(
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
    Calls the SpaceXAI Responses API endpoint with the provided data and handles retries.

    Args:
        api_key (str): SpaceXAI API key.
        model_name (str): SpaceXAI model to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, max_tokens).
        system_prompt (Optional[str]): System prompt for the model.
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
        raise ValidationError("API key is required for SpaceXAI endpoint")

    user_content = build_responses_content(
        parts,
        "SpaceXAI",
        image_detail=lambda part: _image_detail(part, generation_config),
    )

    url = "https://api.x.ai/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    input_messages = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})
    input_messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model_name,
        "input": input_messages,
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
    }

    payload["max_output_tokens"] = generation_config.get("max_tokens", 4096)

    prompt_cache_key = generation_config.get("prompt_cache_key")
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key

    reasoning_effort = generation_config.get("reasoning_effort")
    if reasoning_effort in xai_reasoning_efforts(model_name):
        payload["reasoning"] = {"effort": reasoning_effort}

    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]
    payload = {k: v for k, v in payload.items() if v is not None}

    return post_with_retries(
        url,
        headers,
        payload,
        lambda result: read_responses_reply(result, "SpaceXAI", debug),
        label="SpaceXAI",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
        status_hints=COMMON_STATUS_HINTS,
    )
