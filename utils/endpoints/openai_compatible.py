import json
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from utils.exceptions import TranslationError, ValidationError
from utils.logging import log_message
from utils.model_metadata import (
    is_azure_url,
    is_gemini_no_sampling_model,
)


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
    # The last text part is the prompt; earlier ones label the image after them
    text_part = next((p for p in reversed(parts) if "text" in p), None)
    image_parts = [p for p in parts if p is not text_part]
    if not text_part:
        raise ValidationError(
            "Invalid 'parts' format for OpenAI-Compatible: No text prompt found."
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
    messages = []
    user_content = []
    image_detail = (
        generation_config.get("image_detail")
        if metadata.get("is_openai_model", False)
        or generation_config.get("image_detail")
        else None
    )
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for part in image_parts:
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
            image_url = {"url": f"data:{mime_type};base64,{base64_image}"}
            if image_detail:
                image_url["detail"] = image_detail
            user_content.append({"type": "image_url", "image_url": image_url})
        else:
            log_message(f"Invalid image part format: {part}", always_print=True)
    user_content.append({"type": "text", "text": text_part["text"]})
    messages.append({"role": "user", "content": user_content})

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

    is_openai_model = metadata.get("is_openai_model", False)
    is_anthropic_model = metadata.get("is_anthropic_model", False)

    temp = generation_config.get("temperature")
    no_sampling = (
        metadata.get("is_claude_effort_xhigh", False)
        or metadata.get("is_claude_no_sampling", False)
        or metadata.get("is_gemini_no_sampling", False)
        or is_gemini_no_sampling_model(model_name)
    )
    if (
        temp is not None
        and not is_gemini_no_sampling_model(model_name)
        and not (is_anthropic_model and no_sampling)
    ):
        if is_anthropic_model:
            payload["temperature"] = min(temp, 1.0)
        else:
            payload["temperature"] = temp

    top_p = generation_config.get("top_p")
    if top_p is not None and not is_anthropic_model and not no_sampling:
        payload["top_p"] = top_p

    top_k = generation_config.get("top_k")
    if (
        top_k is not None
        and not is_openai_model
        and not is_anthropic_model
        and not no_sampling
    ):
        payload["top_k"] = top_k

    reasoning_effort = generation_config.get("reasoning_effort")
    if is_anthropic_model and reasoning_effort:
        reasoning_config = {}
        if reasoning_effort == "none":
            reasoning_config["enabled"] = False
        elif reasoning_effort == "auto":
            reasoning_config["enabled"] = True
        else:
            reasoning_config["enabled"] = True
            reasoning_config["effort"] = reasoning_effort
        payload["reasoning"] = reasoning_config
    elif reasoning_effort and reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort

    supports_verbosity = metadata.get("supports_verbosity", False) or metadata.get(
        "is_gpt5_model", False
    )
    if supports_verbosity and generation_config.get("verbosity"):
        payload["verbosity"] = generation_config["verbosity"]

    claude_effort = metadata.get("is_claude_effort", False)
    effort = generation_config.get("effort")
    if effort and claude_effort:
        payload["effort"] = effort

    payload = {k: v for k, v in payload.items() if v is not None}

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)
        try:
            log_message(
                f"OpenAI-Compatible API request to {url} (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            log_message(f"Processing response from {url}", verbose=debug)
            try:
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    finish_reason = choice.get("finish_reason")

                    message = choice.get("message")
                    if message and "content" in message:
                        content = message["content"]
                        return content.strip() if content else ""
                    else:
                        log_message(
                            f"No message content in response. Finish reason: {finish_reason}",
                            always_print=True,
                        )
                        log_message(
                            f"Full response: {json.dumps(result, indent=2)}",
                            verbose=debug,
                        )
                        return ""
                else:
                    log_message(
                        "No choices in OpenAI-Compatible response", always_print=True
                    )
                    if "error" in result:
                        error_obj = result["error"]
                        error_msg = (
                            error_obj.get("message", "Unknown error")
                            if isinstance(error_obj, dict)
                            else str(error_obj)
                        )
                        raise TranslationError(
                            f"OpenAI-Compatible API returned error: {error_msg}"
                        )
                    return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful OpenAI-Compatible API response: {e!s}"
                ) from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]

            if status_code == 429 and attempt < max_retries:
                log_message(
                    f"Rate limited, retrying in {current_delay:.1f}s", verbose=debug
                )
                time.sleep(current_delay)
                continue
            else:
                error_reason = f"Status {status_code}: {error_text}"
                if status_code == 429 and attempt == max_retries:
                    error_reason = (
                        f"Rate limited after {max_retries + 1} attempts: {error_text}"
                    )
                elif status_code == 400:
                    error_reason += " (Check payload)"
                elif status_code == 401:
                    error_reason += " (Check API key if provided)"
                elif status_code == 403:
                    error_reason += " (Permission denied)"

                raise TranslationError(
                    f"OpenAI-Compatible API HTTP Error: {error_reason}"
                ) from e

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                log_message(
                    f"Connection error, retrying in {current_delay:.1f}s: {e!s}",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue
            else:
                raise TranslationError(
                    f"OpenAI-Compatible API Connection Error after retries: {e!s}"
                ) from e

    raise TranslationError(
        f"Failed to get response from OpenAI-Compatible API ({url}) after {max_retries + 1} attempts."
    )
