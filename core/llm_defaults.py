"""Provider-specific default sampling parameters."""

from __future__ import annotations

from typing import Dict, Optional

# Canonical provider names used across the app
DEFAULT_LLM_PROVIDER = "Google"

_PROVIDER_SAMPLING_DEFAULTS: Dict[str, Dict[str, float | int]] = {
    "Google": {"temperature": 0.1, "top_p": 0.95, "top_k": 64},
    "OpenAI": {"temperature": 0.1, "top_p": 1.0, "top_k": 0},
    "Anthropic": {"temperature": 0.1, "top_p": 1.0, "top_k": 0},
    "xAI": {"temperature": 0.1, "top_p": 1.0, "top_k": 0},
    "DeepSeek": {"temperature": 0.1, "top_p": 0.95, "top_k": 0},
    "Z.ai": {"temperature": 0.1, "top_p": 0.95, "top_k": 40},
    "Moonshot": {"temperature": 0.1, "top_p": 1.0, "top_k": 0},
    "OpenRouter": {"temperature": 0.1, "top_p": 0.95, "top_k": 64},
    "OpenAI-Compatible": {"temperature": 0.1, "top_p": 0.95, "top_k": 40},
}


def get_provider_sampling_defaults(provider: Optional[str]) -> Dict[str, float | int]:
    """Return a copy of the sampling defaults for the specified provider."""
    fallback = _PROVIDER_SAMPLING_DEFAULTS[DEFAULT_LLM_PROVIDER]
    if not provider:
        return fallback.copy()
    return _PROVIDER_SAMPLING_DEFAULTS.get(provider, fallback).copy()


__all__ = ["DEFAULT_LLM_PROVIDER", "get_provider_sampling_defaults"]
