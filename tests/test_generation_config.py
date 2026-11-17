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
        ("Moonshot AI", "kimi-k3.1-pro", "none", "reasoning_effort", "low"),
        ("Moonshot AI", "kimi-k3.1-pro", "xhigh", "reasoning_effort", "max"),
        ("Z.ai", "glm-5.3", "auto", "reasoning_effort", "high"),
        ("Google", "gemini-4-argon", "minimal", "thinkingLevel", "low"),
        ("Google", "gemini-4-argon", "none", "thinkingLevel", "low"),
        ("Google", "gemini-4-argon", "max", "thinkingLevel", "high"),
        ("Google", "gemini-4-argon", "xhigh", "thinkingLevel", "high"),
        ("Google", "gemini-4-hydrogen", "xhigh", "thinkingLevel", "xhigh"),
        ("Google", "gemini-4-hydrogen", "max", "thinkingLevel", "xhigh"),
        ("Google", "gemini-4-hydrogen", "minimal", "thinkingLevel", "low"),
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
        ("Google", "gemini-2.5-flash-lite", "none", "thinkingBudget", 0),
        ("QwenCloud", "qwen4-max", "minimal", "reasoning_effort", "low"),
        ("QwenCloud", "qwen4-max", "medium", "reasoning_effort", "high"),
        ("QwenCloud", "qwen4-max", "high", "reasoning_effort", "high"),
        ("QwenCloud", "qwen4-max", "xhigh", "reasoning_effort", "xhigh"),
        ("QwenCloud", "qwen4-max", "max", "reasoning_effort", "xhigh"),
        ("QwenCloud", "qwen4-max", "none", "reasoning_effort", None),
        ("QwenCloud", "qwen3.8-max", "high", "reasoning_effort", "xhigh"),
        ("OpenAI", "gpt-6.2-sol", "none", "reasoning_effort", "low"),
        ("OpenAI", "gpt-6.2-sol", "minimal", "reasoning_effort", "low"),
        ("OpenAI", "gpt-6.2-sol", "max", "reasoning_effort", "max"),
        ("OpenAI", "gpt-6.2-astra", "none", "reasoning_effort", "low"),
        ("OpenAI", "gpt-6.2-astra", "minimal", "reasoning_effort", "low"),
        ("OpenAI", "gpt-6.2-astra", "xhigh", "reasoning_effort", "xhigh"),
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


@pytest.mark.parametrize(
    ("provider", "model", "key", "expected"),
    [
        ("Google", "gemini-4-argon", "thinkingLevel", "medium"),
        ("Google", "gemini-4-hydrogen", "thinkingLevel", "high"),
        ("Google", "gemma-4-31b-it", "thinkingLevel", "high"),
        ("OpenAI", "gpt-6.1-sol", "reasoning_effort", "medium"),
        ("OpenAI", "gpt-6.2-sol", "reasoning_effort", "medium"),
        ("OpenAI", "gpt-6.2-astra", "reasoning_effort", "medium"),
        ("OpenAI", "gpt-6.2-sol-pro", "reasoning_mode", "pro"),
        ("OpenAI", "gpt-6.2-astra-pro", "reasoning_mode", "pro"),
        ("SpaceXAI", "grok-4.7", "reasoning_effort", "medium"),
        ("OpenRouter", "openai/gpt-6.1-sol", "reasoning_effort", "medium"),
        ("OpenRouter", "openai/gpt-6.2-sol", "reasoning_effort", "medium"),
        ("OpenRouter", "openai/gpt-6.2-astra", "reasoning_effort", "medium"),
        ("Anthropic", "claude-opus-5-5", "effort", "medium"),
        ("QwenCloud", "qwen4-max", "reasoning_effort", "high"),
        ("QwenCloud", "qwen3.8-max", "reasoning_effort", "xhigh"),
    ],
)
def test_default_effort_where_supported(provider, model, key, expected):
    config = TranslationConfig(max_tokens=4096)
    generation_config = _build_generation_config(provider, model, config)
    sent = generation_config.get("thinkingConfig", generation_config)
    assert sent.get(key) == expected
