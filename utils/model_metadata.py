from typing import Optional


def get_max_tokens_cap(provider: str, model_name: Optional[str]) -> Optional[int]:
    """
    Get the maximum allowed max_tokens value for a specific provider/model combination.

    Returns:
        - 32768 for OpenAI GPT 4.1 models
        - 16384 for OpenAI GPT 4o models and models with "chat" in the name
        - 31744 for Anthropic Claude Opus 4/4.1 models
        - 29696 for xAI Grok fast models
        - 8192 for DeepSeek "deepseek-chat" model (not including via OpenRouter)
        - 23552 for Z.ai "glm-4.6v" model
        - 16384 for Z.ai "glm-4.5v" model
        - None for all other models (no cap, use existing 63488 max)
    """
    if not model_name:
        return None

    model_lower = model_name.lower()

    if provider == "OpenAI":
        if "gpt-4.1" in model_lower:
            return 32768
        if "gpt-4o" in model_lower:
            return 16384
        if "chat" in model_lower:
            return 16384
    elif provider == "Anthropic":
        if "claude-opus-4" in model_lower and "claude-opus-4-5" not in model_lower:
            return 31744
    elif provider == "xAI":
        if "grok" in model_lower and "fast" in model_lower:
            return 29696
    elif provider == "OpenRouter":
        is_openai_model = "openai/" in model_lower or model_lower.startswith("gpt-")
        is_anthropic_model = "anthropic/" in model_lower or model_lower.startswith(
            "claude-"
        )
        is_grok_model = "grok" in model_lower

        if is_openai_model:
            if "gpt-4.1" in model_lower:
                return 32768
            if "gpt-4o" in model_lower:
                return 16384
            if "chat" in model_lower:
                return 16384
        if is_anthropic_model:
            if "claude-opus-4" in model_lower and "claude-opus-4.5" not in model_lower:
                return 31744
        if is_grok_model and "fast" in model_lower:
            return 29696
        if "glm-4.6v" in model_lower:
            return 23552
        if "glm-4.5v" in model_lower:
            return 16384
    elif provider == "DeepSeek":
        if model_lower == "deepseek-chat":
            return 8192
    elif provider == "Z.ai":
        if model_lower == "glm-4.6v":
            return 23552
        if model_lower == "glm-4.5v":
            return 16384

    return None


def is_openai_compatible_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if an OpenAI-Compatible model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "thinking" in lm or "reasoning" in lm


def is_deepseek_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if a DeepSeek model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm == "deepseek-reasoner"


def is_zai_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if a Z.ai model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm.startswith("glm-4.")


def is_xai_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if an xAI model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "non-reasoning" in lm:
        return False
    return "reasoning" in lm or "grok-4-0709" in lm


def is_opus_45_model(model_name: Optional[str]) -> bool:
    """Check if an Anthropic model is Claude Opus 4.5 (supports effort parameter)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm.startswith("claude-opus-4-5")
