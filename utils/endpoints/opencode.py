import json
from typing import Any

from utils.exceptions import ValidationError
from utils.logging import log_message

from ._http import api_error, post_with_retries


def get_opencode_endpoint_url(tier: str = "zen") -> str:
    """Return the Chat Completions URL for the specified OpenCode tier."""
    normalized_tier = (tier or "zen").strip().lower()
    if "go" in normalized_tier:
        return "https://opencode.ai/zen/go/v1/chat/completions"
    return "https://opencode.ai/zen/v1/chat/completions"


def call_opencode_endpoint(
    api_key: str,
    model_name: str,
    parts: list[dict[str, Any]],
    generation_config: dict[str, Any],
    system_prompt: str | None = None,
    tier: str = "zen",
    debug: bool = False,
    timeout: int = 300,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str | None:
    """Call the OpenCode Chat Completions API with the given prompt and image parts.

    Args:
        api_key (str): OpenCode API key.
        model_name (str): OpenCode model identifier.
        parts (List[Dict[str, Any]]): List containing text prompt and optional image parts.
        generation_config (Dict[str, Any]): Generation parameters (temperature, max_tokens, etc.).
        system_prompt (Optional[str]): Optional system instruction.
        tier (str): "zen" or "go" endpoint tier.
        debug (bool): Enable verbose logging.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum retry attempts for rate limits (429).
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
        raise ValidationError("API key is required for OpenCode endpoint")

    # The last text part is the prompt; earlier ones label the image after them
    text_part = next((p for p in reversed(parts) if "text" in p), None)
    image_parts = [p for p in parts if p is not text_part]

    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for OpenCode: No text prompt found."
        )

    url = get_opencode_endpoint_url(tier)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenCode asks clients to identify themselves instead of sending a library default
        "User-Agent": "MangaTranslator",
    }

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content: list[dict[str, Any]] = []
    for img in image_parts:
        if "text" in img:
            user_content.append({"type": "text", "text": img["text"]})
            continue
        mime = img["inline_data"]["mime_type"]
        data = img["inline_data"]["data"]
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }
        )

    user_content.append({"type": "text", "text": text_part["text"]})
    messages.append({"role": "user", "content": user_content})

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": generation_config.get("max_tokens", 4096),
    }

    temp = generation_config.get("temperature")
    if temp is not None:
        payload["temperature"] = temp

    top_p = generation_config.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p

    reasoning_effort = generation_config.get("reasoning_effort")
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort

    payload = {k: v for k, v in payload.items() if v is not None}

    tier_label = "Go" if "go" in (tier or "").lower() else "Zen"

    def parse(result: dict[str, Any]) -> str | None:
        if "error" in result:
            raise api_error(result.get("error"), f"OpenCode ({tier_label})")

        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            message = choice.get("message")
            if message and "content" in message:
                content = message["content"]
                return content.strip() if content else ""

            output_text = choice.get("text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()

            finish_reason = choice.get("finish_reason") or "unknown"
            log_message(
                f"No message content in OpenCode response. Finish reason: {finish_reason}",
                always_print=True,
            )
            log_message(
                f"Full response: {json.dumps(result, indent=2)}",
                verbose=debug,
            )
            return ""

        # Fallback check for output_text at top level
        if "output_text" in result and isinstance(result["output_text"], str):
            return result["output_text"].strip()

        log_message(
            f"No choices in OpenCode ({tier_label}) response",
            always_print=True,
        )
        return None

    return post_with_retries(
        url,
        headers,
        payload,
        parse,
        label=f"OpenCode ({tier_label})",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
        status_hints={
            400: "Check payload / model support",
            401: "Check OpenCode API key",
            403: f"Permission denied for {tier_label} tier, check subscription or credits",
            404: f"Model '{model_name}' not found on {tier_label} tier",
        },
    )
