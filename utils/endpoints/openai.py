import json
from collections.abc import Callable
from typing import Any

from utils.exceptions import ValidationError
from utils.logging import log_message
from utils.model_metadata import (
    clamp_openai_reasoning_effort,
    is_gpt5_series,
    is_gpt6_series,
    is_openai_reasoning_model,
    resolve_openai_api_model_name,
)

from ._http import api_error, post_with_retries


def extract_responses_text(result: dict[str, Any]) -> str | None:
    """Read the answer text from a Responses API body.

    Documented raw JSON has no top-level `output_text` (that field is SDK-only), and
    reasoning items carry their own text parts, so only `output_text` parts are read.
    Parts of one item are joined directly; items go on separate lines so a preamble
    before a tool call does not run into the answer.
    """
    items = [
        "".join(
            block["text"]
            for block in item.get("content") or []
            if isinstance(block, dict)
            and block.get("type") == "output_text"
            and isinstance(block.get("text"), str)
        )
        for item in result.get("output") or []
        if isinstance(item, dict)
    ]
    text = "\n".join(item for item in items if item).strip()
    if not text and isinstance(result.get("output_text"), str):
        text = result["output_text"].strip()
    return text or None


def read_responses_text(
    result: dict[str, Any], label: str, debug: bool = False
) -> str | None:
    """Return a Responses API body's answer text, or log that it has none and return None."""
    output_text = extract_responses_text(result)
    if output_text:
        return output_text

    log_message(
        f"No text content in {label} response. Status: {result.get('status')}, "
        f"details: {result.get('incomplete_details')}",
        always_print=True,
    )
    log_message(f"Full response: {json.dumps(result, indent=2)}", verbose=debug)
    return None


def read_responses_reply(
    result: dict[str, Any], label: str, debug: bool = False
) -> str | None:
    """Read a Responses API body like `read_responses_text`, raising if it carries an error."""
    if result.get("error"):
        raise api_error(result["error"], label)
    return read_responses_text(result, label, debug)


def build_responses_content(
    parts: list[dict[str, Any]],
    label: str,
    image_detail: str | Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Convert `parts` into the user message content of a Responses API request.

    The last text part is the prompt and goes last; earlier text parts label the image
    after them.

    Args:
        parts: Text parts (`{"text": ...}`) and image parts (`{"inline_data": ...}`).
        label: Provider name for the error raised when `parts` has no text.
        image_detail: Sent as each image's `detail` when set. A function is called with
            the image part, to pick the detail per image.

    Raises:
        ValidationError: If `parts` has no text part.
    """
    text_part = next((p for p in reversed(parts) if "text" in p), None)
    if not text_part:
        raise ValidationError(
            f"Invalid 'parts' format for {label}: No text prompt found."
        )

    content = []
    for part in parts:
        if part is text_part:
            continue
        if "text" in part:
            content.append({"type": "input_text", "text": part["text"]})
            continue
        if (
            "inline_data" in part
            and "data" in part["inline_data"]
            and "mime_type" in part["inline_data"]
        ):
            mime_type = part["inline_data"]["mime_type"]
            base64_image = part["inline_data"]["data"]
            image_item = {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{base64_image}",
            }
            detail = image_detail(part) if callable(image_detail) else image_detail
            if detail:
                image_item["detail"] = detail
            content.append(image_item)
        else:
            log_message(f"Invalid image part format: {part}", always_print=True)
    content.append({"type": "input_text", "text": text_part["text"]})
    return content


def call_openai_endpoint(
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
    Calls the OpenAI Responses API endpoint with the provided data and handles retries.

    Args:
        api_key (str): OpenAI API key.
        model_name (str): OpenAI model to use.
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
        raise ValidationError("API key is required for OpenAI endpoint")
    input_content = build_responses_content(
        parts, "OpenAI", image_detail=generation_config.get("image_detail")
    )

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    api_model_name = resolve_openai_api_model_name(model_name) or model_name
    payload = {
        "model": api_model_name,
        "input": [{"role": "user", "content": input_content}],
        "temperature": generation_config.get("temperature"),
        "top_p": generation_config.get("top_p"),
        "max_output_tokens": generation_config.get("max_output_tokens", 4096),
    }
    if system_prompt:
        payload["instructions"] = system_prompt
    if enable_web_search:
        payload["tools"] = [{"type": "web_search"}]
    payload = {k: v for k, v in payload.items() if v is not None}

    is_chat_variant = "chat" in (model_name or "").lower()
    is_gpt5_or_gpt6 = is_gpt5_series(model_name) or is_gpt6_series(model_name)

    if is_openai_reasoning_model(model_name) and not is_chat_variant:
        reasoning: dict[str, Any] = {}
        effort = generation_config.get("reasoning_effort")
        if effort:
            reasoning["effort"] = clamp_openai_reasoning_effort(model_name, effort)
        if generation_config.get("reasoning_mode") == "pro":
            reasoning["mode"] = "pro"
        if reasoning:
            payload["reasoning"] = reasoning

    if is_gpt5_or_gpt6 and not is_chat_variant:
        payload["text"] = {"verbosity": generation_config.get("verbosity", "low")}
        # temp/top_p only allowed when effort is "none"
        if payload.get("reasoning", {}).get("effort") != "none":
            payload.pop("temperature", None)
            payload.pop("top_p", None)

    return post_with_retries(
        url,
        headers,
        payload,
        # Unlike its siblings, a body with an `error` is still read for text
        lambda result: read_responses_text(result, "OpenAI", debug),
        label="OpenAI",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
        status_hints={400: "Check payload"},
    )
