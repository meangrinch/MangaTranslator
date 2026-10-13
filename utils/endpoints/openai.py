import json
from typing import Any

from utils.exceptions import ValidationError
from utils.logging import log_message
from utils.model_metadata import (
    is_gpt5_series,
    is_gpt6_astra,
    is_gpt6_series,
    is_gpt61_sol,
    is_openai_reasoning_model,
    resolve_openai_api_model_name,
    supports_gpt5_max_effort,
    supports_gpt5_xhigh_effort,
)

from ._http import post_with_retries


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
    # The last text part is the prompt; earlier ones label the image after them
    text_part = next((p for p in reversed(parts) if "text" in p), None)
    image_parts = [p for p in parts if p is not text_part]
    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for OpenAI: No text prompt found."
        )

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    input_content = []
    image_detail = generation_config.get("image_detail")
    for part in image_parts:
        if "text" in part:
            input_content.append({"type": "input_text", "text": part["text"]})
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
            if image_detail:
                image_item["detail"] = image_detail
            input_content.append(image_item)
        else:
            log_message(f"Invalid image part format: {part}", always_print=True)
    input_content.append({"type": "input_text", "text": text_part["text"]})

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

    try:
        lower_model = (model_name or "").lower()
        is_chat_variant = "chat" in lower_model
        is_gpt5 = is_gpt5_series(model_name)
        is_gpt6 = is_gpt6_series(model_name)
        is_reasoning_capable = is_openai_reasoning_model(model_name)
        reasoning_mode = generation_config.get("reasoning_mode")

        if is_reasoning_capable and not is_chat_variant:
            effort = generation_config.get("reasoning_effort")
            reasoning_payload: dict[str, Any] = {}
            if effort:
                if is_gpt6:
                    effort_to_send = (
                        "low"
                        if effort == "minimal"
                        or (
                            (is_gpt6_astra(model_name) or is_gpt61_sol(model_name))
                            and effort == "none"
                        )
                        else effort
                    )
                    reasoning_payload["effort"] = effort_to_send
                else:
                    if effort == "none":
                        reasoning_payload["effort"] = "none"
                    else:
                        effort_to_send = effort
                        if effort_to_send == "max" and not supports_gpt5_max_effort(
                            model_name
                        ):
                            effort_to_send = (
                                "xhigh"
                                if supports_gpt5_xhigh_effort(model_name)
                                else "high"
                            )
                        if effort_to_send == "xhigh" and not supports_gpt5_xhigh_effort(
                            model_name
                        ):
                            effort_to_send = "high"
                        reasoning_payload["effort"] = effort_to_send

            if reasoning_mode == "pro":
                reasoning_payload["mode"] = "pro"

            if reasoning_payload:
                payload["reasoning"] = reasoning_payload

        if (is_gpt5 or is_gpt6) and not is_chat_variant:
            verbosity = generation_config.get("verbosity", "low")
            payload["text"] = {"verbosity": verbosity}

            # temp/top_p only allowed when effort is "none"
            current_effort = payload.get("reasoning", {}).get("effort")
            allow_sampling = current_effort == "none"
            if not allow_sampling:
                payload.pop("temperature", None)
                payload.pop("top_p", None)
    except Exception:
        pass

    def parse(result: dict[str, Any]) -> str | None:
        output_text = extract_responses_text(result)
        if output_text:
            return output_text

        log_message(
            f"No text content in OpenAI response. Status: {result.get('status')}, "
            f"details: {result.get('incomplete_details')}",
            always_print=True,
        )
        log_message(f"Full response: {json.dumps(result, indent=2)}", verbose=debug)
        return None

    return post_with_retries(
        url,
        headers,
        payload,
        parse,
        label="OpenAI",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
        status_hints={400: "Check payload"},
    )
