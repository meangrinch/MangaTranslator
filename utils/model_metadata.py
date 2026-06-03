import re
from typing import Optional

FLUX_SDCPP_QUANT_FILES = {
    "flux_klein_4b": {
        "diffusion_model": {
            "repo_id": "unsloth/FLUX.2-klein-4B-GGUF",
            "default": "Q4_K_M",
            "filenames": {
                "Q8_0": "flux-2-klein-4b-Q8_0.gguf",
                "Q6_K": "flux-2-klein-4b-Q6_K.gguf",
                "Q5_K_M": "flux-2-klein-4b-Q5_K_M.gguf",
                "Q4_K_M": "flux-2-klein-4b-Q4_K_M.gguf",
                "Q3_K_M": "flux-2-klein-4b-Q3_K_M.gguf",
            },
        },
        "llm": {
            "repo_id": "unsloth/Qwen3-4B-GGUF",
            "default": "Q4_K_XL",
            "filenames": {
                "Q8_0": "Qwen3-4B-Q8_0.gguf",
                "Q6_K_XL": "Qwen3-4B-UD-Q6_K_XL.gguf",
                "Q5_K_XL": "Qwen3-4B-UD-Q5_K_XL.gguf",
                "Q4_K_XL": "Qwen3-4B-UD-Q4_K_XL.gguf",
                "Q3_K_XL": "Qwen3-4B-UD-Q3_K_XL.gguf",
            },
        },
    },
    "flux_klein_9b": {
        "diffusion_model": {
            "repo_id": "unsloth/FLUX.2-klein-9B-GGUF",
            "default": "Q4_K_M",
            "filenames": {
                "Q8_0": "flux-2-klein-9b-Q8_0.gguf",
                "Q6_K": "flux-2-klein-9b-Q6_K.gguf",
                "Q5_K_M": "flux-2-klein-9b-Q5_K_M.gguf",
                "Q4_K_M": "flux-2-klein-9b-Q4_K_M.gguf",
                "Q3_K_M": "flux-2-klein-9b-Q3_K_M.gguf",
            },
        },
        "llm": {
            "repo_id": "unsloth/Qwen3-8B-GGUF",
            "default": "Q4_K_XL",
            "filenames": {
                "Q8_0": "Qwen3-8B-Q8_0.gguf",
                "Q6_K_XL": "Qwen3-8B-UD-Q6_K_XL.gguf",
                "Q5_K_XL": "Qwen3-8B-UD-Q5_K_XL.gguf",
                "Q4_K_XL": "Qwen3-8B-UD-Q4_K_XL.gguf",
                "Q3_K_XL": "Qwen3-8B-UD-Q3_K_XL.gguf",
            },
        },
    },
    "flux_kontext": {
        "diffusion_model": {
            "repo_id": "unsloth/FLUX.1-Kontext-dev-GGUF",
            "default": "Q4_K_M",
            "subdir": "kontext",
            "filenames": {
                "Q8_0": "flux1-kontext-dev-Q8_0.gguf",
                "Q6_K": "flux1-kontext-dev-Q6_K.gguf",
                "Q5_K_M": "flux1-kontext-dev-Q5_K_M.gguf",
                "Q4_K_M": "flux1-kontext-dev-Q4_K_M.gguf",
                "Q3_K_M": "flux1-kontext-dev-Q3_K_M.gguf",
            },
        },
        "t5xxl": {
            "repo_id": "city96/t5-v1_1-xxl-encoder-gguf",
            "default": "Q4_K_M",
            "subdir": "kontext",
            "filenames": {
                "Q8_0": "t5-v1_1-xxl-encoder-Q8_0.gguf",
                "Q6_K": "t5-v1_1-xxl-encoder-Q6_K.gguf",
                "Q5_K_M": "t5-v1_1-xxl-encoder-Q5_K_M.gguf",
                "Q4_K_M": "t5-v1_1-xxl-encoder-Q4_K_M.gguf",
                "Q3_K_M": "t5-v1_1-xxl-encoder-Q3_K_M.gguf",
            },
        },
    },
}

FLUX_SDCPP_DIFFUSION_QUANTS = tuple(
    FLUX_SDCPP_QUANT_FILES["flux_klein_4b"]["diffusion_model"]["filenames"]
)
FLUX_SDCPP_QWEN_QUANTS = tuple(
    FLUX_SDCPP_QUANT_FILES["flux_klein_4b"]["llm"]["filenames"]
)
FLUX_SDCPP_T5_QUANTS = tuple(
    FLUX_SDCPP_QUANT_FILES["flux_kontext"]["t5xxl"]["filenames"]
)
FLUX_SDCPP_TEXT_ENCODER_QUANTS = FLUX_SDCPP_QWEN_QUANTS + tuple(
    quant for quant in FLUX_SDCPP_T5_QUANTS if quant not in FLUX_SDCPP_QWEN_QUANTS
)
FLUX_BACKENDS = ("sdcpp", "sdnq", "nunchaku")


def flux_sdcpp_text_encoder_asset_key(model_key: str) -> Optional[str]:
    """Map a Flux model key to its sd.cpp text-encoder asset key."""
    if model_key == "flux_kontext":
        return "t5xxl"
    if model_key in ("flux_klein_4b", "flux_klein_9b"):
        return "llm"
    return None


def flux_sdcpp_quant_default(model_key: str, asset_key: str) -> str:
    """Default quant for a model/asset, sourced from FLUX_SDCPP_QUANT_FILES."""
    spec = FLUX_SDCPP_QUANT_FILES.get(model_key, {}).get(asset_key)
    return spec["default"] if spec else ""


def flux_sdcpp_text_encoder_quants(model_key: str) -> tuple[str, ...]:
    asset_key = flux_sdcpp_text_encoder_asset_key(model_key)
    if asset_key is None:
        return ()
    return tuple(FLUX_SDCPP_QUANT_FILES[model_key][asset_key]["filenames"])


def flux_sdcpp_text_encoder_default(model_key: str) -> str:
    asset_key = flux_sdcpp_text_encoder_asset_key(model_key)
    if asset_key is None:
        return flux_sdcpp_quant_default("flux_klein_4b", "llm")
    return flux_sdcpp_quant_default(model_key, asset_key)


def flux_sdcpp_valid_text_encoder_quant(model_key: str, quant: str) -> str:
    """Return a text-encoder quant that is valid for the selected Flux model."""
    quants = flux_sdcpp_text_encoder_quants(model_key) or FLUX_SDCPP_TEXT_ENCODER_QUANTS
    default = flux_sdcpp_text_encoder_default(model_key)
    if quant in quants:
        return quant
    if default in quants:
        return default
    return quants[0] if quants else ""


def flux_valid_backend(model_key: str, backend: str) -> str:
    if model_key in ("flux_klein_4b", "flux_klein_9b"):
        return backend if backend in ("sdcpp", "sdnq") else "sdnq"
    if model_key == "flux_kontext":
        return backend if backend in FLUX_BACKENDS else "sdnq"
    return backend if backend in FLUX_BACKENDS else "sdnq"


def get_max_tokens_cap(provider: str, model_name: Optional[str]) -> Optional[int]:
    """
    Get the maximum allowed max_tokens value for a specific provider/model combination.

    Returns:
        - 32768 for OpenAI GPT 4.1 models
        - 16384 for OpenAI GPT 4o models and models with "chat" in the name
        - 31744 for Anthropic Claude Opus 4/4.1 models (not 4.5+)
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
            and "claude-opus-4-7" not in model_lower
            and "claude-opus-4-8" not in model_lower
        ):
            return 31744
    elif provider == "OpenRouter":
        is_openai_model = "openai/" in model_lower or model_lower.startswith("gpt-")
        is_anthropic_model = "anthropic/" in model_lower or model_lower.startswith(
            "claude-"
        )

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
                and "claude-opus-4.7" not in model_lower
                and "claude-opus-4.8" not in model_lower
            ):
                return 31744
        if "glm-4.6v" in model_lower:
            return 23552
        if "glm-4.5v" in model_lower:
            return 16384
    elif provider == "Z.ai":
        if model_lower == "glm-4.6v":
            return 23552
        if model_lower == "glm-4.5v":
            return 16384
    elif provider == "Moonshot AI":
        if "kimi-k2." in model_lower:
            return 32768

    return None


def is_gpt5_series(model_name: Optional[str]) -> bool:
    """Check if a model is any GPT-5 variant (5, 5.1, 5.2, 5.3, 5.4, 5.5, etc.)."""
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


def is_openai_model_family(model_name: Optional[str]) -> bool:
    """Check if a model name is OpenAI-family, including OpenRouter-prefixed IDs."""
    if not model_name:
        return False
    lm = model_name.lower()
    return (
        "openai/" in lm
        or lm.startswith("gpt-")
        or lm.startswith("o1")
        or lm.startswith("o3")
        or lm.startswith("o4-mini")
        or "/gpt-" in lm
        or "/o1" in lm
        or "/o3" in lm
        or "/o4-mini" in lm
    )


_GPT5_GEN_RE = re.compile(r"gpt-(5(?:\.\d+)?)", re.IGNORECASE)


def supports_openai_original_image_detail(model_name: Optional[str]) -> bool:
    """Whether an OpenAI model supports image detail='original'."""
    if not is_gpt5_series(model_name):
        return False

    lm = (model_name or "").lower()
    if any(token in lm for token in ("mini", "nano", "chat")):
        return False

    gen = get_gpt5_generation(model_name)
    if not gen or gen == "5":
        return False

    try:
        major, minor = (int(part) for part in gen.split(".", 1))
    except ValueError:
        return False

    return (major, minor) >= (5, 4)


def get_gpt5_generation(model_name: Optional[str]) -> Optional[str]:
    """Extract the GPT-5 generation string.

    Returns '5', '5.1', '5.2', '5.3', '5.4', '5.5', etc. or None if not a GPT-5 model.
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
    return lm in ("deepseek-v4-pro", "deepseek-v4-flash")


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
    return (
        lm.startswith("grok-4.3")
        or "grok-4.20" in lm
        or "reasoning" in lm
        or "multi-agent" in lm
    )


def supports_xai_reasoning_parameter(model_name: Optional[str]) -> bool:
    """Whether xAI accepts the Responses API `reasoning` parameter."""
    if not model_name:
        return False
    lm = model_name.lower()
    return (
        lm.startswith("grok-4.3") or "multi-agent" in lm
    ) and "non-reasoning" not in lm


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

    All kimi-k2.X models default to thinking enabled.
    """
    if not model_name:
        return False
    return "kimi-k2." in model_name.lower()


def is_mimo_multimodal_model(model_name: Optional[str]) -> bool:
    """Check if a MiMo model supports multimodal (image) input."""
    if not model_name:
        return False
    return model_name.lower() == "mimo-v2.5"


def is_mimo_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if a MiMo model is reasoning-capable (hybrid thinking)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm in ("mimo-v2.5-pro", "mimo-v2.5")


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


def is_opus_47_model(model_name: Optional[str]) -> bool:
    """Check if a model is Claude Opus 4.7 (adaptive thinking, xhigh effort, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.7" in lm) or ("4-7" in lm)


def is_opus_48_model(model_name: Optional[str]) -> bool:
    """Check if a model is Claude Opus 4.8 (same API constraints as Opus 4.7)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.8" in lm) or ("4-8" in lm)


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
