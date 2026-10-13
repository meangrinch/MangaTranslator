import json
from typing import Any

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message

from ._http import COMMON_STATUS_HINTS, post_with_retries


def _claude_capability_flag(generation_config: dict[str, Any], key: str) -> bool:
    metadata = generation_config.get("_metadata")
    if isinstance(metadata, dict) and key in metadata:
        return bool(metadata[key])
    return bool(generation_config.get(key))


def call_anthropic_endpoint(
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
    Calls the Anthropic Messages API endpoint with the provided data and handles retries.

    Args:
        api_key (str): Anthropic API key.
        model_name (str): Anthropic model to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
                                      # Assumes the first part is the system/main prompt, subsequent are images.
        generation_config (Dict[str, Any]): Configuration for generation (temp <= 1.0, top_p, top_k, max_tokens).
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if an error occurs or no content is found after retries.

    Raises:
        ValueError: If API key is missing or parts format is invalid.
        RuntimeError: If API call fails after retries for non-rate-limited HTTP errors,
                      connection errors, or response processing fails.
    """
    if not api_key:
        raise ValidationError("API key is required for Anthropic endpoint")
    user_prompt_part = None
    image_parts = []
    # The last text part is the prompt; earlier ones label the image after them
    for part in reversed(parts):
        if "text" in part and user_prompt_part is None:
            user_prompt_part = part
        else:
            image_parts.insert(0, part)

    if not user_prompt_part:
        raise ValidationError(
            "Invalid 'parts' format for Anthropic: No text prompt found for user message."
        )

    content_parts = image_parts

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    messages = []
    user_prompt_text = user_prompt_part["text"]
    user_content = []
    for part in content_parts:
        if "text" in part:
            user_content.append({"type": "text", "text": part["text"]})
            continue
        if (
            "inline_data" in part
            and "data" in part["inline_data"]
            and "mime_type" in part["inline_data"]
        ):
            mime_type = part["inline_data"]["mime_type"]
            base64_image = part["inline_data"]["data"]
            user_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image,
                    },
                }
            )
        else:
            log_message(f"Invalid image part format: {part}", always_print=True)

    user_content.append({"type": "text", "text": user_prompt_text})
    if not user_content:
        raise ValidationError(
            "No valid content (images/text) could be prepared for Anthropic user message."
        )

    messages.append({"role": "user", "content": user_content})

    temp = generation_config.get("temperature")
    clamped_temp = min(temp, 1.0) if temp is not None else None

    payload = {
        "model": model_name,
        "system": system_prompt,
        "messages": messages,
        "temperature": clamped_temp,
        "top_k": generation_config.get("top_k"),
        "max_tokens": generation_config.get("max_tokens", 4096),
    }

    if _claude_capability_flag(generation_config, "is_claude_effort_xhigh") or (
        _claude_capability_flag(generation_config, "is_claude_no_sampling")
    ):
        payload.pop("temperature", None)
        payload.pop("top_k", None)

    if not _claude_capability_flag(generation_config, "is_claude_omit_thinking"):
        try:
            thinking_type = generation_config.get("thinking_type")
            reasoning_effort = generation_config.get("reasoning_effort")
            if thinking_type == "adaptive":
                # Opus 4.6+: Adaptive thinking - Claude decides reasoning depth
                payload["thinking"] = {"type": "adaptive"}
            elif thinking_type == "between_tools" or (
                thinking_type == "disabled"
                and _claude_capability_flag(
                    generation_config, "is_claude_between_tools"
                )
            ):
                # Sonnet 5.5: Turn off up-front thinking
                payload["thinking"] = {"type": "between_tools"}
            elif thinking_type == "disabled":
                # Sonnet 5: Explicitly turn off adaptive thinking (on by default)
                payload["thinking"] = {"type": "disabled"}
            elif thinking_type == "enabled":
                # Older models: Budget-based thinking
                if reasoning_effort and reasoning_effort != "none":
                    from core.config import calculate_reasoning_budget

                    max_tokens_value = generation_config.get("max_tokens", 4096)
                    # The API rejects budgets under 1024, or not below max_tokens
                    budget_tokens = max(
                        1024,
                        calculate_reasoning_budget(max_tokens_value, reasoning_effort),
                    )
                    if budget_tokens < max_tokens_value:
                        payload["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": budget_tokens,
                        }
                    else:
                        log_message(
                            f"max_tokens {max_tokens_value} leaves no room for a "
                            "thinking budget; sending without thinking",
                            always_print=True,
                        )
                elif reasoning_effort == "none":
                    payload["thinking"] = {"type": "enabled", "budget_tokens": 0}
        except Exception:
            pass

    # Models that still accept sampling params reject them while thinking is on
    if payload.get("thinking", {}).get("type") in ("adaptive", "enabled"):
        payload.pop("temperature", None)
        payload.pop("top_k", None)

    try:
        effort = generation_config.get("effort")
        if _claude_capability_flag(generation_config, "is_claude_effort_xhigh"):
            valid_efforts = ("max", "xhigh", "high", "medium", "low")
        elif _claude_capability_flag(generation_config, "is_claude_effort_max"):
            valid_efforts = ("max", "high", "medium", "low")
        else:
            valid_efforts = ("high", "medium", "low")
        if effort and effort in valid_efforts:
            payload["output_config"] = {"effort": effort}
    except Exception:
        pass

    if payload.get("thinking") in (
        {"type": "disabled"},
        {"type": "between_tools"},
    ) and payload.get("output_config", {}).get("effort") in ("xhigh", "max"):
        payload["output_config"]["effort"] = "high"

    if enable_web_search:
        # 4.6+ models (the max-effort tier) filter results in code execution first
        web_search_type = (
            "web_search_20260209"
            if _claude_capability_flag(generation_config, "is_claude_effort_max")
            else "web_search_20250305"
        )
        payload["tools"] = [{"type": web_search_type, "name": "web_search"}]
    payload = {k: v for k, v in payload.items() if v is not None}

    content_blocks: list[dict[str, Any]] = []

    def parse(result: dict[str, Any]) -> tuple[str | None, bool]:
        """Return the text received so far and whether the turn paused."""
        if result.get("type") == "error":
            error_data = result.get("error", {})
            error_type = error_data.get("type", "unknown_error")
            error_message = error_data.get("message", "No error message provided.")
            raise TranslationError(
                f"Anthropic API returned error: {error_type} - {error_message}"
            )

        if isinstance(result.get("content"), list):
            content_blocks.extend(result["content"])
        if not content_blocks:
            stop_reason = result.get("stop_reason")
            stop_details_text = ""
            if stop_reason == "refusal" and result.get("stop_details"):
                stop_details_text = (
                    f". Stop details: "
                    f"{json.dumps(result['stop_details'], ensure_ascii=False)}"
                )
            log_message(
                f"No text content in Anthropic response. Stop reason: {stop_reason}{stop_details_text}",
                always_print=True,
            )
            log_message(
                f"Full response: {json.dumps(result, indent=2)}",
                verbose=debug,
            )
            return None, False

        # Citations split one answer across text blocks, and tool blocks
        # separate a preamble from the answer that follows them
        text = "".join(
            block.get("text", "") if block.get("type") == "text" else "\n"
            for block in content_blocks
        ).strip()
        return text, result.get("stop_reason") == "pause_turn"

    # A long server-tool turn pauses; sending it back as-is resumes it
    continuations_left = max_retries
    while True:
        text, paused = post_with_retries(
            url,
            headers,
            payload,
            parse,
            label="Anthropic",
            debug=debug,
            timeout=timeout,
            max_retries=max_retries,
            base_delay=base_delay,
            status_hints=COMMON_STATUS_HINTS,
        )
        if not paused or continuations_left <= 0:
            return text
        continuations_left -= 1
        messages[1:] = [{"role": "assistant", "content": content_blocks}]
