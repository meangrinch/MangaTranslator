import json
import time
from collections.abc import Callable
from typing import Any, TypeVar

import requests

from utils.cancellation import raise_if_cancelled
from utils.exceptions import TranslationError
from utils.logging import log_message

T = TypeVar("T")

COMMON_STATUS_HINTS = {
    400: "Check payload",
    401: "Check API key",
    403: "Permission denied, check API key/plan",
}
# Rate limits and transient server overloads (529 is Anthropic's "overloaded")
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


def api_error(error: Any, label: str) -> TranslationError:
    """Build the error to raise for the `error` a response body carries."""
    message = (
        error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
    )
    return TranslationError(f"{label} API returned error: {message}")


def post_with_retries(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    parse: Callable[[dict[str, Any]], T],
    *,
    label: str,
    debug: bool = False,
    timeout: int = 300,
    max_retries: int = 3,
    base_delay: float = 1.0,
    status_hints: dict[int, str] | None = None,
) -> T:
    """
    POST `payload` and return `parse(response_json)`, retrying rate limits (429),
    server overloads (5xx) and connection errors with exponential backoff.

    Args:
        url (str): Endpoint URL.
        headers (Dict[str, str]): Request headers.
        payload (Dict[str, Any]): JSON request body.
        parse (Callable): Turns the JSON response body into the endpoint's return value.
        label (str): Provider name used in log and error messages.
        debug (bool): Whether to print debugging information.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum number of retries for rate limiting and connection errors.
        base_delay (float): Initial delay for retries in seconds.
        status_hints (Optional[Dict[int, str]]): Advice appended to the error for a given HTTP status.

    Raises:
        TranslationError: If the call fails after retries, returns a non-429 HTTP error,
                          or `parse` cannot read the response body.
    """
    # core/pipeline.py matches on the wording of these errors ("429", "rate limit", ...)
    attempts = max_retries + 1
    for attempt in range(attempts):
        raise_if_cancelled()
        delay = min(base_delay * (2**attempt), 16.0)  # Exponential backoff, max 16s
        can_retry = attempt < max_retries
        log_message(
            f"{label} API request to {url} (attempt {attempt + 1}/{attempts})",
            verbose=debug,
        )
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]  # Limit error text length
            if status_code not in RETRY_STATUSES:
                reason = f"Status {status_code}: {error_text}"
                hint = (status_hints or {}).get(status_code)
                if hint:
                    reason += f" ({hint})"
                raise TranslationError(f"{label} API HTTP Error: {reason}") from e
            if not can_retry:
                if status_code == 429:
                    raise TranslationError(
                        f"{label} API HTTP Error: Rate limited after {attempts} attempts: {error_text}"
                    ) from e
                raise TranslationError(
                    f"{label} API HTTP Error: Status {status_code} after {attempts} attempts: {error_text}"
                ) from e
            log_message(f"HTTP {status_code}, retrying in {delay:.1f}s", verbose=debug)
        except requests.exceptions.RequestException as e:
            if not can_retry:
                raise TranslationError(
                    f"{label} API Connection Error after retries: {e!s}"
                ) from e
            log_message(
                f"Connection error, retrying in {delay:.1f}s: {e!s}", verbose=debug
            )
        else:
            log_message(f"Processing {label} response", verbose=debug)
            try:
                return parse(response.json())
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                raise TranslationError(
                    f"Error processing successful {label} API response: {e!s}"
                ) from e
        time.sleep(delay)

    raise TranslationError(
        f"Failed to get response from {label} API after {attempts} attempts."
    )
