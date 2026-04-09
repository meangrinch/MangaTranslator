import re
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
        if (
            "claude-opus-4" in model_lower
            and "claude-opus-4-5" not in model_lower
            and "claude-opus-4-6" not in model_lower
        ):
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
            if (
                "claude-opus-4" in model_lower
                and "claude-opus-4.5" not in model_lower
                and "claude-opus-4.6" not in model_lower
            ):
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


def is_gpt5_series(model_name: Optional[str]) -> bool:
    """Check if a model is any GPT-5 variant (5, 5.1, 5.2, 5.3, 5.4, etc.)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm.startswith("gpt-5") or "/gpt-5" in lm


def is_gpt5_chat_variant(model_name: Optional[str]) -> bool:
    """Check if a GPT-5 model is a chat variant (non-reasoning)."""
    return is_gpt5_series(model_name) and "chat" in (model_name or "").lower()


def is_gpt5_pro(model_name: Optional[str]) -> bool:
    """Check if a GPT-5 model is a pro variant."""
    return is_gpt5_series(model_name) and "-pro" in (model_name or "").lower()


_GPT5_GEN_RE = re.compile(r"gpt-(5(?:\.\d+)?)", re.IGNORECASE)


def get_gpt5_generation(model_name: Optional[str]) -> Optional[str]:
    """Extract the GPT-5 generation string.

    Returns '5', '5.1', '5.2', '5.3', '5.4', etc. or None if not a GPT-5 model.
    """
    if not model_name:
        return None
    m = _GPT5_GEN_RE.search(model_name)
    return m.group(1) if m else None


def is_openai_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if an OpenAI model is reasoning-capable (GPT-5 series, o1, o3, o4-mini)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return (
        lm.startswith("gpt-5")
        or "/gpt-5" in lm
        or lm.startswith("o1")
        or "/o1" in lm
        or lm.startswith("o3")
        or "/o3" in lm
        or lm.startswith("o4-mini")
        or "/o4-mini" in lm
    )


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
    return lm.startswith("glm-4.") or lm.startswith("glm-5")


def is_xai_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if an xAI model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "non-reasoning" in lm:
        return False
    return "reasoning" in lm or "multi-agent" in lm


def is_anthropic_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if an Anthropic model is reasoning-capable.

    Uses `in` (not `startswith`) so OpenRouter-prefixed names also match.
    Checks both hyphen and dot variants for haiku (4-5 vs 4.5).
    """
    if not model_name:
        return False
    lm = model_name.lower()
    return (
        "claude-opus-4" in lm
        or "claude-sonnet-4" in lm
        or "claude-haiku-4-5" in lm
        or "claude-haiku-4.5" in lm
    )


def is_moonshot_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if a Moonshot model is reasoning-capable.

    Kimi-k2-thinking/turbo and kimi-k2.5 are reasoning models,
    though their reasoning cannot be controlled via reasoning_effort.
    """
    if not model_name:
        return False
    lm = model_name.lower()
    return "thinking" in lm or "kimi-k2.5" in lm


def is_opus_45_model(model_name: Optional[str]) -> bool:
    """Check if a model is Claude Opus 4.5 (supports effort parameter)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.5" in lm) or ("4-5" in lm)


def is_opus_46_model(model_name: Optional[str]) -> bool:
    """Check if a model is Claude Opus 4.6 (adaptive thinking, max effort)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.6" in lm) or ("4-6" in lm)


def is_sonnet_46_model(model_name: Optional[str]) -> bool:
    """Check if a model is Claude Sonnet 4.6 (adaptive thinking, max effort)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "sonnet" not in lm:
        return False
    return ("4.6" in lm) or ("4-6" in lm)


def is_46_model(model_name: Optional[str]) -> bool:
    """Check if a model is any Claude 4.6 variant (Opus or Sonnet)."""
    return is_opus_46_model(model_name) or is_sonnet_46_model(model_name)


def is_gemma_model(model_name: Optional[str]) -> bool:
    """Check if a model is a Gemma model (e.g., gemma-4-31b-it)."""
    if not model_name:
        return False
    return "gemma-" in model_name.lower()


def is_gemini_3_model(model_name: Optional[str]) -> bool:
    """Check if a model is any Gemini 3 variant (3, 3.1, etc.)."""
    if not model_name:
        return False
    return "gemini-3" in model_name.lower()


def is_gemini_3_flash_model(model_name: Optional[str]) -> bool:
    """Check if a model is Gemini 3 Flash (not Flash Lite)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-3" in lm and "flash" in lm and "flash-lite" not in lm


def is_gemini_25_flash_model(model_name: Optional[str]) -> bool:
    """Check if a model is Gemini 2.5 Flash."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-2.5" in lm and "flash" in lm


def is_gemini_25_pro_model(model_name: Optional[str]) -> bool:
    """Check if a model is Gemini 2.5 Pro."""
    if not model_name:
        return False
    return "gemini-2.5-pro" in model_name.lower()


def is_google_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if a Google model is reasoning-capable (Gemini 2.5, 3 series, or Gemma)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-2.5" in lm or "gemini-3" in lm or "gemma-" in lm


def is_rosetta_model(model_name: Optional[str]) -> bool:
    """Check if a model is a YanoljaNEXT Rosetta translation model.

    Requires both 'rosetta' and 'yanoljanext' in the name to avoid false positives.
    Matches e.g. 'yanoljanext-rosetta-4b-2511'.
    """
    if not model_name:
        return False
    lm = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    return "rosetta" in lm and "yanoljanext" in lm
