import pytest

from core.config import TranslationConfig
from core.services.translation import _build_generation_config


@pytest.mark.parametrize(
    ("provider", "model", "effort", "key", "expected"),
    [
        # The CLI has no "auto", so any effort but "none" turns adaptive thinking on
        ("Anthropic", "claude-opus-4-6", "high", "thinking_type", "adaptive"),
        ("Anthropic", "claude-opus-4-8", "high", "thinking_type", "adaptive"),
        ("Anthropic", "claude-opus-4-6", "none", "thinking_type", None),
        ("SpaceXAI", "grok-4.7", "none", "reasoning_effort", "low"),
        ("SpaceXAI", "grok-4.6", "minimal", "reasoning_effort", "low"),
        ("SpaceXAI", "grok-4.6", "max", "reasoning_effort", "xhigh"),
        ("SpaceXAI", "grok-4.5", "xhigh", "reasoning_effort", "high"),
        ("SpaceXAI", "grok-4.5", "none", "reasoning_effort", "none"),
        ("Z.ai", "glm-5.3", "none", "reasoning_effort", "low"),
        ("Z.ai", "glm-5.3-flash", "medium", "reasoning_effort", "high"),
        ("Z.ai", "glm-5.3", "xhigh", "reasoning_effort", "max"),
        ("Z.ai", "glm-5.3", "low", "reasoning_effort", "low"),
        ("Moonshot AI", "kimi-k3", "none", "reasoning_effort", "low"),
        ("Moonshot AI", "kimi-k3", "xhigh", "reasoning_effort", "max"),
        ("Moonshot AI", "kimi-k3", "medium", "reasoning_effort", "high"),
        ("Z.ai", "glm-5.3", "auto", "reasoning_effort", "high"),
        ("Google", "gemini-4-argon", "minimal", "thinkingLevel", "low"),
        ("Google", "gemini-4-argon", "none", "thinkingLevel", "low"),
        ("Google", "gemini-4-argon", "max", "thinkingLevel", "high"),
        ("Google", "gemini-3.6-flash", "none", "thinkingLevel", "minimal"),
        ("Google", "gemini-3.6-flash", "xhigh", "thinkingLevel", "high"),
        ("Google", "gemma-4-31b-it", "low", "thinkingLevel", "minimal"),
        ("Google", "gemma-4-31b-it", "medium", "thinkingLevel", "high"),
        # OpenRouter maps efforts itself, so they go out as given
        (
            "OpenRouter",
            "google/gemini-4-argon",
            "minimal",
            "reasoning_effort",
            "minimal",
        ),
        ("OpenRouter", "openai/gpt-6-astra", "none", "reasoning_effort", "none"),
        ("Google", "gemini-3.6-flash", "minimal", "thinkingLevel", "minimal"),
    ],
)
def test_reasoning_effort_becomes_one_the_model_accepts(
    provider, model, effort, key, expected
):
    config = TranslationConfig(reasoning_effort=effort, max_tokens=4096)
    generation_config = _build_generation_config(provider, model, config)
    # Gemini nests its level one deeper than the other providers' top-level keys
    sent = generation_config.get("thinkingConfig", generation_config)
    assert sent.get(key) == expected


@pytest.mark.parametrize(
    ("provider", "model", "effort", "expected"),
    [
        ("Anthropic", "claude-opus-4-5", "max", "high"),
        ("Anthropic", "claude-opus-4-5", "xhigh", "high"),
        ("Anthropic", "claude-sonnet-4-6", "xhigh", "max"),
        ("Anthropic", "claude-opus-4-8", "xhigh", "xhigh"),
        ("Anthropic", "claude-opus-4-5", "low", "low"),
        # OpenRouter maps efforts itself, so they go out as given
        ("OpenRouter", "anthropic/claude-opus-4-5", "max", "max"),
    ],
)
def test_claude_effort_becomes_one_the_model_accepts(provider, model, effort, expected):
    config = TranslationConfig(effort=effort, max_tokens=4096)
    assert _build_generation_config(provider, model, config)["effort"] == expected
