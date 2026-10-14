import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from utils.exceptions import ValidationError
from utils.logging import log_message
from utils.model_metadata import (
    is_azure_url,
    is_gemini_no_sampling_model,
)

from ._http import api_error, post_with_retries


def build_openai_compatible_url(base_url: str) -> str:
    """Builds the full chat completions URL for generic or Azure OpenAI endpoints."""
    if not base_url:
        raise ValidationError("Base URL is required for OpenAI-Compatible endpoint")

    url = base_url.strip()

    if is_azure_url(url):
        # v1 route: no api-version, and the deployment name goes in the body's `model`
        parsed = urlsplit(url)
        prefix = parsed.path.split("/openai")[0].rstrip("/")
        prefix = prefix.removesuffix("/chat/completions").removesuffix("/v1")
        query = "&".join(
            param
            for param in parsed.query.split("&")
            if param and not param.startswith("api-version=")
        )
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                f"{prefix}/openai/v1/chat/completions",
                query,
                "",
            )
        )

    if url.endswith("/chat/completions"):
        return url
    return f"{url.rstrip('/')}/chat/completions"


def build_chat_messages(
    parts: list[dict[str, Any]],
    system_prompt: str | None,
    label: str,
    image_detail: str | None = None,
) -> list[dict[str, Any]]:
    """Convert `parts` into Chat Completions `messages`.

    Returns an optional system message followed by one user message. The last text part
    is the prompt and goes last; earlier text parts label the image after them.

    Args:
        parts: Text parts (`{"text": ...}`) and image parts (`{"inline_data": ...}`).
        system_prompt: System instruction, if any.
        label: Provider name for the error raised when `parts` has no text.
        image_detail: Sent as each image's `detail` when set.

    Raises:
        ValidationError: If `parts` has no text part.
    """
    text_part = next((p for p in reversed(parts) if "text" in p), None)
    if not text_part:
        raise ValidationError(
            f"Invalid 'parts' format for {label}: No text prompt found."
        )

    user_content = []
    for part in parts:
        if part is text_part:
            continue
        if "text" in part:
            user_content.append({"type": "text", "text": part["text"]})
            continue
        mime_type = part["inline_data"]["mime_type"]
        base64_image = part["inline_data"]["data"]
        image_url = {"url": f"data:{mime_type};base64,{base64_image}"}
        if image_detail:
            image_url["detail"] = image_detail
        user_content.append({"type": "image_url", "image_url": image_url})
    user_content.append({"type": "text", "text": text_part["text"]})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def read_chat_reply(
    result: dict[str, Any], label: str, debug: bool = False
) -> str | None:
    """Read the answer text from a Chat Completions body.

    Returns "" when the first choice has no content, and None when there are no choices.

    Raises:
        TranslationError: If there are no choices and the body carries an error.
    """
    if "choices" in result and len(result["choices"]) > 0:
        choice = result["choices"][0]
        finish_reason = choice.get("finish_reason")

        message = choice.get("message")
        if message and "content" in message:
            content = message["content"]
            return content.strip() if content else ""
        log_message(
            f"No message content in {label} response. Finish reason: {finish_reason}",
            always_print=True,
        )
        log_message(f"Full response: {json.dumps(result, indent=2)}", verbose=debug)
        return ""
    log_message(f"No choices in {label} response", always_print=True)
    if "error" in result:
        raise api_error(result.get("error"), label)
    return None


def routed_sampling_params(
    model_name: str, generation_config: dict[str, Any], cap_openai_temperature: bool
) -> dict[str, Any]:
    """Pick the sampling parameters a model routed through an aggregator accepts.

    The model family comes from `generation_config["_metadata"]`. Claude takes no top_p
    or top_k and a temperature of at most 1.0, OpenAI takes no top_k, and the Claude and
    Gemini models that fix their own sampling take none of the three.

    Args:
        model_name: Model ID, used to spot Gemini models with fixed sampling.
        generation_config: Holds `temperature`, `top_p`, `top_k` and `_metadata`.
        cap_openai_temperature: Also cap an OpenAI model's temperature at 1.0.
    """
    metadata = generation_config.get("_metadata", {})
    is_openai_model = metadata.get("is_openai_model", False)
    is_anthropic_model = metadata.get("is_anthropic_model", False)
    # Read from the name too, because callers may pass no `_metadata`
    gemini_no_sampling = is_gemini_no_sampling_model(model_name)
    no_sampling = (
        gemini_no_sampling
        or metadata.get("is_gemini_no_sampling", False)
        or metadata.get("is_claude_effort_xhigh", False)
        or metadata.get("is_claude_no_sampling", False)
    )
    params = {}

    temperature = generation_config.get("temperature")
    if (
        temperature is not None
        and not gemini_no_sampling
        and not (is_anthropic_model and no_sampling)
    ):
        if is_anthropic_model or (cap_openai_temperature and is_openai_model):
            temperature = min(temperature, 1.0)
        params["temperature"] = temperature

    top_p = generation_config.get("top_p")
    if top_p is not None and not is_anthropic_model and not no_sampling:
        params["top_p"] = top_p

    top_k = generation_config.get("top_k")
    if (
        top_k is not None
        and not is_openai_model
        and not is_anthropic_model
        and not no_sampling
    ):
        params["top_k"] = top_k

    return params


def call_openai_compatible_endpoint(
    base_url: str,
    api_key: str | None,
    model_name: str,
    parts: list[dict[str, Any]],
    generation_config: dict[str, Any],
    system_prompt: str | None = None,
    debug: bool = False,
    timeout: int = 480,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> str | None:
    """
    Calls a generic or Azure OpenAI-Compatible Chat Completions API endpoint and handles retries.

    Args:
        base_url (str): The base URL of the compatible endpoint (e.g., "http://localhost:8080/v1" or Azure URL).
        api_key (Optional[str]): The API key, if required by the endpoint.
        model_name (str): The model ID or deployment name to use.
        parts (List[Dict[str, Any]]): List of content parts (text, images).
                                      # Assumes the first part is the text prompt, subsequent are images.
        generation_config (Dict[str, Any]): Configuration for generation (temp, top_p, top_k, max_tokens).
                                            # Parameter restrictions (temp clamp, no top_k) are applied
                                            # based on model metadata.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting errors.
        base_delay (float): Initial delay for retries in seconds.

    Returns:
        Optional[str]: The raw text content from the API response if successful,
                       None if blocked by content filter or if no content is found after retries.

    Raises:
        ValueError: If base_url is missing or parts format is invalid.
        RuntimeError: If API call fails after retries for non-rate-limited HTTP errors,
                      connection errors, or response processing fails.
    """
    if not base_url:
        raise ValidationError("Base URL is required for OpenAI-Compatible endpoint")

    messages = build_chat_messages(
        parts,
        system_prompt,
        "OpenAI-Compatible",
        image_detail=generation_config.get("image_detail"),
    )

    url = build_openai_compatible_url(base_url)
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        if is_azure_url(base_url):
            headers["api-key"] = api_key

    metadata = generation_config.get("_metadata", {})
    # OpenAI reasoning models reject max_tokens on chat completions
    token_limit_key = (
        "max_completion_tokens"
        if metadata.get("is_openai_reasoning", False)
        else "max_tokens"
    )
    payload = {
        "model": model_name,
        "messages": messages,
        token_limit_key: generation_config.get("max_tokens", 4096),
    }

    payload.update(
        routed_sampling_params(
            model_name, generation_config, cap_openai_temperature=False
        )
    )

    reasoning_effort = generation_config.get("reasoning_effort")
    if reasoning_effort and metadata.get("is_anthropic_model", False):
        reasoning_config = {"enabled": reasoning_effort != "none"}
        if reasoning_effort not in ("none", "auto"):
            reasoning_config["effort"] = reasoning_effort
        payload["reasoning"] = reasoning_config
    elif reasoning_effort and reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort

    supports_verbosity = metadata.get("supports_verbosity", False) or metadata.get(
        "is_gpt5_model", False
    )
    if supports_verbosity and generation_config.get("verbosity"):
        payload["verbosity"] = generation_config["verbosity"]

    effort = generation_config.get("effort")
    if effort and metadata.get("is_claude_effort", False):
        payload["effort"] = effort

    payload = {k: v for k, v in payload.items() if v is not None}

    return post_with_retries(
        url,
        headers,
        payload,
        lambda result: read_chat_reply(result, "OpenAI-Compatible", debug),
        label="OpenAI-Compatible",
        debug=debug,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=base_delay,
        status_hints={
            400: "Check payload",
            401: "Check API key if provided",
            403: "Permission denied",
        },
    )
