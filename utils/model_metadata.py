import re

FLUX_SDCPP_INT8_CONVROT = "INT8-ConvRot"

_QWEN3_VL_4B_GGUF = {
    "repo_id": "unsloth/Qwen3-VL-4B-Instruct-GGUF",
    "default": "Q4_K_XL",
    "filenames": {
        "Q8_0": "Qwen3-VL-4B-Instruct-Q8_0.gguf",
        "Q6_K_XL": "Qwen3-VL-4B-Instruct-UD-Q6_K_XL.gguf",
        "Q5_K_XL": "Qwen3-VL-4B-Instruct-UD-Q5_K_XL.gguf",
        "Q4_K_XL": "Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf",
        "Q3_K_XL": "Qwen3-VL-4B-Instruct-UD-Q3_K_XL.gguf",
    },
}

# "quant_repos" overrides "repo_id" for single quants. "vision" is the projector a GGUF text
# encoder needs for reference-image editing; the INT8-ConvRot text encoders contain theirs.
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
    "flux3_klein": {
        "diffusion_model": {
            "repo_id": "unsloth/flux-3-image-klein-GGUF",
            "default": "Q4_K_M",
            # unsloth's INT8 file is a torchao checkpoint, which sd.cpp can't read
            "quant_repos": {
                FLUX_SDCPP_INT8_CONVROT: "ml-resources/flux-3-image-klein-comfy"
            },
            "filenames": {
                FLUX_SDCPP_INT8_CONVROT: (
                    "diffusion_models/flux3-image-klein-int8-convrot.safetensors"
                ),
                "Q8_0": "flux-3-image-klein-Q8_0.gguf",
                "Q6_K": "flux-3-image-klein-Q6_K.gguf",
                "Q5_K_M": "flux-3-image-klein-Q5_K_M.gguf",
                "Q4_K_M": "flux-3-image-klein-Q4_K_M.gguf",
                "Q3_K_M": "flux-3-image-klein-Q3_K_M.gguf",
            },
        },
        "llm": {
            **_QWEN3_VL_4B_GGUF,
            "quant_repos": {FLUX_SDCPP_INT8_CONVROT: "unsloth/flux-3-image-klein-FP8"},
            "filenames": {
                FLUX_SDCPP_INT8_CONVROT: (
                    "flux-3-image-klein-text_encoder-INT8-ConvRot.safetensors"
                ),
                **_QWEN3_VL_4B_GGUF["filenames"],
            },
        },
    },
    "fibo2_edit_turbo": {
        "diffusion_model": {
            "repo_id": "Comfy-Org/Fibo-2.0",
            "default": FLUX_SDCPP_INT8_CONVROT,
            "filenames": {
                FLUX_SDCPP_INT8_CONVROT: (
                    "diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors"
                ),
            },
        },
        "llm": {
            **_QWEN3_VL_4B_GGUF,
            "vision": {
                "repo_id": "unsloth/Qwen3-VL-4B-Instruct-GGUF",
                "filename": "mmproj-F16.gguf",
                "local_name": "Qwen3-VL-4B-Instruct-mmproj-F16.gguf",
            },
        },
    },
    "qwen_image_2_1_turbo": {
        "diffusion_model": {
            "repo_id": "unsloth/Qwen-Image-2.1-Turbo-GGUF",
            "default": "Q4_K_M",
            # unsloth's Turbo INT8-ConvRot is a torchao checkpoint, which sd.cpp can't read
            "quant_repos": {FLUX_SDCPP_INT8_CONVROT: "Comfy-Org/Qwen-Image-2.1"},
            "filenames": {
                FLUX_SDCPP_INT8_CONVROT: (
                    "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"
                ),
                "Q8_0": "qwen-image-2.1-turbo-Q8_0.gguf",
                "Q6_K": "qwen-image-2.1-turbo-Q6_K.gguf",
                "Q5_K_M": "qwen-image-2.1-turbo-Q5_K_M.gguf",
                "Q4_K_M": "qwen-image-2.1-turbo-Q4_K_M.gguf",
                "Q3_K_M": "qwen-image-2.1-turbo-Q3_K_M.gguf",
            },
        },
        "llm": {
            "repo_id": "unsloth/Qwen3-VL-8B-Instruct-GGUF",
            "default": "Q4_K_XL",
            # Turbo shares Qwen-Image-2.1's text encoder byte for byte
            "quant_repos": {FLUX_SDCPP_INT8_CONVROT: "unsloth/Qwen-Image-2.1-FP8"},
            "filenames": {
                FLUX_SDCPP_INT8_CONVROT: (
                    "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors"
                ),
                "Q8_0": "Qwen3-VL-8B-Instruct-Q8_0.gguf",
                "Q6_K_XL": "Qwen3-VL-8B-Instruct-UD-Q6_K_XL.gguf",
                "Q5_K_XL": "Qwen3-VL-8B-Instruct-UD-Q5_K_XL.gguf",
                "Q4_K_XL": "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
                "Q3_K_XL": "Qwen3-VL-8B-Instruct-UD-Q3_K_XL.gguf",
            },
            "vision": {
                "repo_id": "unsloth/Qwen3-VL-8B-Instruct-GGUF",
                "filename": "mmproj-F16.gguf",
                "local_name": "Qwen3-VL-8B-Instruct-mmproj-F16.gguf",
            },
        },
    },
}


def _sdcpp_quant_union(*asset_keys: str) -> tuple[str, ...]:
    """Every quant offered for the asset keys across models, INT8-ConvRot first, then
    in first-seen order (matching the per-model lists)."""
    quants = dict.fromkeys(
        quant
        for asset_key in asset_keys
        for assets in FLUX_SDCPP_QUANT_FILES.values()
        for quant in assets.get(asset_key, {}).get("filenames", ())
    )
    return tuple(sorted(quants, key=lambda quant: quant != FLUX_SDCPP_INT8_CONVROT))


FLUX_SDCPP_DIFFUSION_QUANTS = _sdcpp_quant_union("diffusion_model")
FLUX_SDCPP_TEXT_ENCODER_QUANTS = _sdcpp_quant_union("llm", "t5xxl")
FLUX_BACKENDS = ("sdcpp", "sdnq", "nunchaku")

# Backends each diffusion inpainting method supports
FLUX_METHOD_BACKENDS = {
    "flux_klein_9b": ("sdcpp", "sdnq"),
    "flux_klein_4b": ("sdcpp", "sdnq"),
    "flux_kontext": ("sdcpp", "sdnq", "nunchaku"),
    "flux3_klein": ("sdcpp", "sdnq"),
    "fibo2_edit_turbo": ("sdcpp",),
    "qwen_image_2_1_turbo": ("sdcpp", "sdnq"),
}

# (default, maximum) denoising steps per diffusion inpainting method
FLUX_METHOD_STEPS = {
    "flux_klein_9b": (4, 12),
    "flux_klein_4b": (4, 12),
    "flux_kontext": (8, 30),
    "flux3_klein": (16, 16),
    "fibo2_edit_turbo": (4, 12),
    "qwen_image_2_1_turbo": (8, 8),
}

# Sampled on the checkpoint's fixed sigma schedule, so the step count can't change
FLUX_FIXED_STEP_METHODS = ("qwen_image_2_1_turbo",)

# Inpainted through FluxKleinInpainter's crop pipeline (~1MP scaling, luminance correction)
KLEIN_PIPELINE_METHODS = (
    "flux_klein_9b",
    "flux_klein_4b",
    "flux3_klein",
    "fibo2_edit_turbo",
    "qwen_image_2_1_turbo",
)

# Providers whose API has no built-in web search tool this app can use
NO_WEB_SEARCH_PROVIDERS = ("OpenAI-Compatible", "DeepSeek", "Moonshot AI", "OpenCode")


def flux_sdcpp_text_encoder_asset_key(model_key: str) -> str | None:
    """Map a Flux model key to its sd.cpp text-encoder asset key."""
    if model_key == "flux_kontext":
        return "t5xxl"
    if model_key in FLUX_SDCPP_QUANT_FILES:
        return "llm"
    return None


def flux_sdcpp_quant_default(model_key: str, asset_key: str) -> str:
    """Default quant for a model/asset, sourced from FLUX_SDCPP_QUANT_FILES."""
    spec = FLUX_SDCPP_QUANT_FILES.get(model_key, {}).get(asset_key)
    return spec["default"] if spec else ""


def _valid_quant(quants: tuple[str, ...], quant: str, default: str) -> str:
    if quant in quants:
        return quant
    if default in quants:
        return default
    return quants[0] if quants else ""


def flux_sdcpp_diffusion_quants(model_key: str) -> tuple[str, ...]:
    spec = FLUX_SDCPP_QUANT_FILES.get(model_key, {}).get("diffusion_model")
    return tuple(spec["filenames"]) if spec else ()


def flux_sdcpp_valid_diffusion_quant(model_key: str, quant: str) -> str:
    """Return a diffusion-model quant that is valid for the selected Flux model."""
    quants = flux_sdcpp_diffusion_quants(model_key) or FLUX_SDCPP_DIFFUSION_QUANTS
    default = flux_sdcpp_quant_default(
        model_key, "diffusion_model"
    ) or flux_sdcpp_quant_default("flux_klein_4b", "diffusion_model")
    return _valid_quant(quants, quant, default)


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
    return _valid_quant(quants, quant, flux_sdcpp_text_encoder_default(model_key))


def flux_method_backends(model_key: str) -> tuple[str, ...]:
    """Backends of a diffusion inpainting method; empty for LaMa, OpenCV, and none."""
    return FLUX_METHOD_BACKENDS.get(model_key, ())


def flux_valid_backend(model_key: str, backend: str) -> str:
    backends = flux_method_backends(model_key) or FLUX_BACKENDS
    if backend in backends:
        return backend
    return "sdnq" if "sdnq" in backends else backends[0]


def flux_steps_range(model_key: str) -> tuple[int, int]:
    """(default, maximum) denoising steps; Klein's range for non-diffusion methods."""
    return FLUX_METHOD_STEPS.get(model_key, FLUX_METHOD_STEPS["flux_klein_4b"])


def flux_steps_fixed(model_key: str) -> bool:
    return model_key in FLUX_FIXED_STEP_METHODS


def get_max_tokens_cap(provider: str, model_name: str | None) -> int | None:
    """
    Get the maximum allowed max_tokens value for a specific provider/model combination.

    Returns:
        - 16384 for OpenAI models with "chat" in the name
        - 32768 for Moonshot Kimi K2.x models
        - None for all other models (no cap, use existing 63488 max)
    """
    if not model_name:
        return None

    model_lower = model_name.lower()

    is_openai_model = provider == "OpenAI" or (
        provider == "OpenRouter"
        and ("openai/" in model_lower or model_lower.startswith("gpt-"))
    )
    if is_openai_model and "chat" in model_lower:
        return 16384
    if provider == "Moonshot AI" and "kimi-k2." in model_lower:
        return 32768
    return None


def is_gpt5_series(model_name: str | None) -> bool:
    """Check if a model is any GPT-5 variant (5, 5.2, 5.4, etc.)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm.startswith("gpt-5") or "/gpt-5" in lm


def is_gpt5_chat_variant(model_name: str | None) -> bool:
    """Check if a GPT-5 model is a chat variant (non-reasoning)."""
    return is_gpt5_series(model_name) and "chat" in (model_name or "").lower()


def is_gpt5_pro(model_name: str | None) -> bool:
    """Check if a GPT-5 model is a pro variant (real slug or virtual GPT-5.6 pro)."""
    return is_gpt5_series(model_name) and "-pro" in (model_name or "").lower()


def is_openai_model_family(model_name: str | None) -> bool:
    """Check if a model name is OpenAI-family, including OpenRouter-prefixed IDs."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "openai/" in lm or lm.startswith("gpt-") or "/gpt-" in lm


def is_google_model_family(model_name: str | None) -> bool:
    """Check if a model name is Google/Gemini-family, including OpenRouter-prefixed IDs."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "google/" in lm or "gemini" in lm or "gemma" in lm


def is_anthropic_model_family(model_name: str | None) -> bool:
    """Check if a model name is Anthropic/Claude-family, including OpenRouter-prefixed IDs."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "anthropic/" in lm or lm.startswith("claude-")


_GPT5_GEN_RE = re.compile(r"gpt-(5(?:\.\d+)?)", re.IGNORECASE)


def get_gpt5_generation(model_name: str | None) -> str | None:
    """Extract the GPT-5 generation string.

    Returns '5.2', '5.3', '5.4', '5.5', '5.6', etc. or None if not a GPT-5 model.
    """
    if not model_name:
        return None
    m = _GPT5_GEN_RE.search(model_name)
    return m.group(1) if m else None


def _parse_gpt5_gen_parts(gen: str | None) -> tuple[int, int] | None:
    if not gen:
        return None
    try:
        if "." in gen:
            major, minor = (int(part) for part in gen.split(".", 1))
            return major, minor
        return int(gen), 0
    except ValueError:
        return None


def supports_gpt5_xhigh_effort(model_name: str | None) -> bool:
    """Whether a GPT-5 model accepts reasoning.effort='xhigh'."""
    parts = _parse_gpt5_gen_parts(get_gpt5_generation(model_name))
    if parts is None:
        return False
    return parts >= (5, 2)


def supports_gpt5_max_effort(model_name: str | None) -> bool:
    """Whether a GPT-5 model accepts reasoning.effort='max' (GPT-5.6+)."""
    parts = _parse_gpt5_gen_parts(get_gpt5_generation(model_name))
    if parts is None:
        return False
    return parts >= (5, 6)


def is_gpt56_virtual_pro(model_name: str | None) -> bool:
    """GPT-5.6 pro is not a separate API slug; UI uses *-pro entries mapped to reasoning.mode."""
    return get_gpt5_generation(model_name) == "5.6" and is_gpt5_pro(model_name)


def is_gpt6_series(model_name: str | None) -> bool:
    """Check if a model is any GPT-6 variant (Astra, Sol, Luna)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gpt-6" in lm or "/gpt-6" in lm


def is_gpt6_astra(model_name: str | None) -> bool:
    """Check if a model is GPT-6 Astra."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gpt-6-astra" in lm


def is_gpt6_sol(model_name: str | None) -> bool:
    """Check if a model is GPT-6 Sol."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gpt-6-sol" in lm


def is_gpt61_sol(model_name: str | None) -> bool:
    """Check if a model is GPT-6.1 Sol."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gpt-6.1-sol" in lm


def is_gpt6_luna(model_name: str | None) -> bool:
    """Check if a model is GPT-6 Luna."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gpt-6-luna" in lm


def is_gpt6_pro(model_name: str | None) -> bool:
    """GPT-6 pro entries are not separate API slugs; UI uses *-pro entries mapped to reasoning.mode."""
    if not model_name:
        return False
    return is_gpt6_series(model_name) and model_name.lower().endswith("-pro")


def is_gpt6_astra_pro(model_name: str | None) -> bool:
    """GPT-6 Astra pro is not a separate API slug; UI uses *-pro entries mapped to reasoning.mode."""
    if not model_name:
        return False
    return is_gpt6_astra(model_name) and model_name.lower().endswith("-pro")


def is_openai_virtual_pro(model_name: str | None) -> bool:
    """Whether an OpenAI model uses a virtual -pro UI entry mapped to reasoning.mode=pro."""
    if not model_name:
        return False
    return is_gpt56_virtual_pro(model_name) or is_gpt6_pro(model_name)


def resolve_openai_api_model_name(model_name: str | None) -> str | None:
    """Map UI model names to the slug sent to the OpenAI API.

    Virtual GPT-5.6 pro and GPT-6 pro entries (e.g. gpt-5.6-sol-pro, gpt-6-astra-pro, gpt-6-sol-pro)
    strip the trailing -pro suffix. Historical pro slugs (gpt-5.5-pro-...) are left unchanged.
    """
    if not model_name or not is_openai_virtual_pro(model_name):
        return model_name
    if model_name.lower().endswith("-pro"):
        return model_name[: -len("-pro")]
    return model_name


def supports_openai_original_image_detail(model_name: str | None) -> bool:
    """Whether an OpenAI model supports image detail='original'."""
    if is_gpt6_series(model_name):
        return True

    if is_gpt5_series(model_name):
        lm = (model_name or "").lower()
        if any(token in lm for token in ("mini", "chat")):
            return False

        parts = _parse_gpt5_gen_parts(get_gpt5_generation(model_name))
        if parts is None:
            return False

        return parts >= (5, 4)

    return False


def supports_openai_verbosity(model_name: str | None) -> bool:
    """Whether an OpenAI model supports the text.verbosity parameter (GPT-5 series, GPT-6 series)."""
    if not model_name or is_gpt5_chat_variant(model_name):
        return False
    return is_gpt5_series(model_name) or is_gpt6_series(model_name)


def is_openai_reasoning_model(model_name: str | None) -> bool:
    """Check if an OpenAI model is reasoning-capable (GPT-5 series, GPT-6 series)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return (
        lm.startswith(("gpt-5", "gpt-6"))
        or "/gpt-5" in lm
        or "/gpt-6" in lm
        or is_gpt6_series(model_name)
    )


def clamp_gpt6_reasoning_effort(model_name: str | None, effort: str) -> str:
    """GPT-6 has no "minimal" effort, and Astra / 6.1 Sol have no "none"; both become "low"."""
    if effort == "minimal" or (
        effort == "none" and (is_gpt6_astra(model_name) or is_gpt61_sol(model_name))
    ):
        return "low"
    return effort


def clamp_openai_reasoning_effort(model_name: str | None, effort: str) -> str:
    """Replace a reasoning effort an OpenAI model lacks with the nearest one it accepts."""
    if is_gpt6_series(model_name):
        return clamp_gpt6_reasoning_effort(model_name, effort)
    # GPT-5: "max" needs 5.6+ and "xhigh" needs 5.2+
    if effort == "max" and not supports_gpt5_max_effort(model_name):
        effort = "xhigh"
    if effort == "xhigh" and not supports_gpt5_xhigh_effort(model_name):
        effort = "high"
    return effort


def is_azure_url(url: str | None) -> bool:
    """Check if a URL is an Azure OpenAI or Azure AI Services endpoint."""
    if not url:
        return False
    url_lower = url.lower()
    azure_domains = (
        ".azure.com",
        ".azure-api.net",
        ".azure.us",
        ".azure.cn",
        ".microsoftazure.de",
    )
    return any(domain in url_lower for domain in azure_domains)


def is_openai_compatible_reasoning_model(model_name: str | None) -> bool:
    """Check if an OpenAI-Compatible model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    return (
        "thinking" in lm
        or "reasoning" in lm
        or is_openai_reasoning_model(model_name)
        or is_anthropic_reasoning_model(model_name)
    )


def is_deepseek_reasoning_model(model_name: str | None) -> bool:
    """Check if a DeepSeek model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "deepseek-v4" in lm or "deepseek-flash" in lm


def is_zai_reasoning_model(model_name: str | None) -> bool:
    """Check if a Z.ai model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm.startswith(("glm-4.", "glm-5"))


def supports_zai_reasoning_effort(model_name: str | None) -> bool:
    """Check if a Z.ai model supports the reasoning_effort API parameter."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm == "glm-5.2" or lm.startswith(("glm-5.2-", "glm-5.3-")) or lm == "glm-5.3"


_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def _nearest_effort(effort: str, accepted: tuple[str, ...]) -> str:
    """Pick the accepted reasoning effort nearest to `effort`, the stronger one on a tie.

    An effort outside the scale, such as "auto", counts as "high". With nothing
    accepted, the effort is returned as it is.
    """
    if not accepted or effort in accepted:
        return effort
    rank = _EFFORT_ORDER.index
    target = rank(effort if effort in _EFFORT_ORDER else "high")
    return max(accepted, key=lambda e: (-abs(rank(e) - target), rank(e)))


def clamp_zai_reasoning_effort(model_name: str | None, effort: str) -> str:
    """Replace a reasoning effort GLM-5.3 lacks with the nearest one it accepts."""
    if "glm-5.3" in (model_name or "").lower():
        return _nearest_effort(effort, ("low", "high", "max"))
    return effort


def is_zai_vision_model(model_name: str | None) -> bool:
    """Check if a Z.ai model supports vision/multimodal input."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "glm-5.3-flash" in lm or lm.endswith("v") or "v-" in lm


def is_deepseek_vision_model(model_name: str | None) -> bool:
    """Check if a DeepSeek model supports vision/multimodal input."""
    if not model_name:
        return False
    lm = model_name.lower()
    return ("deepseek" in lm and "vision" in lm) or "deepseek-flash" in lm


def is_xai_reasoning_model(model_name: str | None) -> bool:
    """Check if a SpaceXAI model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "non-reasoning" in lm:
        return False
    return (
        lm.startswith(("grok-4.3", "grok-4.5", "grok-4.6", "grok-4.7"))
        or "grok-4.20" in lm
        or "reasoning" in lm
        or "multi-agent" in lm
    )


def xai_reasoning_efforts(model_name: str | None) -> tuple[str, ...]:
    """Reasoning efforts a SpaceXAI model accepts, judged by its Grok generation alone.

    Empty when the Responses API `reasoning` parameter does not apply.
    """
    lm = (model_name or "").lower()
    if "multi-agent" in lm or lm.startswith(("grok-4.6", "grok-4.7")):
        return ("low", "medium", "high", "xhigh")
    if lm.startswith(("grok-4.3", "grok-4.5")):
        return ("none", "low", "medium", "high")
    return ()


def clamp_xai_reasoning_effort(model_name: str | None, effort: str) -> str:
    """Replace a reasoning effort a SpaceXAI model lacks with the nearest one it accepts."""
    return _nearest_effort(effort, xai_reasoning_efforts(model_name))


def supports_xai_reasoning_parameter(model_name: str | None) -> bool:
    """Whether SpaceXAI accepts the Responses API `reasoning` parameter."""
    return (
        bool(xai_reasoning_efforts(model_name))
        and "non-reasoning" not in model_name.lower()
    )


def is_anthropic_reasoning_model(model_name: str | None) -> bool:
    """Check if an Anthropic model is reasoning-capable.

    Uses `in` (not `startswith`) so OpenRouter-prefixed names also match.
    """
    if not model_name:
        return False
    lm = model_name.lower()
    return (
        "claude-opus-4" in lm
        or is_sonnet_46_model(model_name)
        or is_sonnet_5_model(model_name)
        or is_sonnet_55_model(model_name)
        or is_opus_5_model(model_name)
        or is_opus_55_model(model_name)
        or is_fable_5_model(model_name)
        or is_haiku_55_model(model_name)
    )


def is_moonshot_k3_model(model_name: str | None) -> bool:
    """Check if a model is in the Kimi K3 family (kimi-k3, kimi-k3.1-pro, and other variants)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return lm == "kimi-k3" or lm.startswith(("kimi-k3-", "kimi-k3."))


def is_moonshot_reasoning_model(model_name: str | None) -> bool:
    """Check if a Moonshot model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "kimi-k2." in lm or is_moonshot_k3_model(model_name)


def supports_moonshot_reasoning_effort(model_name: str | None) -> bool:
    """Check if a Moonshot model supports the reasoning_effort API parameter."""
    return is_moonshot_k3_model(model_name)


def clamp_moonshot_reasoning_effort(model_name: str | None, effort: str) -> str:
    """Replace a reasoning effort Kimi K3 lacks with the nearest one it accepts."""
    if supports_moonshot_reasoning_effort(model_name):
        return _nearest_effort(effort, ("low", "high", "max"))
    return effort


def is_mimo_multimodal_model(model_name: str | None) -> bool:
    """Check if a MiMo model supports multimodal (image) input."""
    if not model_name:
        return False
    return "mimo-v2.6" in model_name.lower()


def is_opencode_chat_completions_model(
    model_name: str | None, tier: str = "zen"
) -> bool:
    """Check if OpenCode serves a model on chat/completions, the only route this app calls.

    The rest are served on /responses, /messages, or a Gemini-style path.
    """
    if not model_name:
        return False
    base_name = model_name.strip().lower().split("/")[-1]
    # Name prefixes copied from OpenCode's endpoint tables (2026-10) because its
    # /models list has no endpoint field; revisit when a listed model 404s.
    if base_name.startswith(("gpt-", "claude-", "gemini-", "grok-", "muse-", "jev-")):
        return False
    if tier == "go":
        return not base_name.startswith(("qwen", "minimax-"))
    return not base_name.startswith("qwen") or base_name.startswith("qwen3.8-max")


def is_opencode_multimodal_model(model_name: str | None) -> bool:
    """Check if an OpenCode model is multimodal/vision-capable for LLM OCR mode."""
    if not model_name:
        return False
    lm = model_name.strip().lower()
    base_name = lm.split("/")[-1]

    if base_name in (
        "big-pickle",
        "deepseek-v4-pro",
        "minimax-m2.7",
        "glm-5.2",
        "glm-5.3",
        "mimo-v2.5-pro",
    ) or base_name.startswith(("jev-", "hy3", "hy4", "nemotron-", "longcat-2.0")):
        return False

    if (
        "starcoder" in base_name
        or "codegeex" in base_name
        or "embed" in base_name
        or (
            ("coder" in base_name or "-code" in base_name or "code-" in base_name)
            and not base_name.startswith("kimi-")
        )
    ):
        return False

    if is_deepseek_reasoning_model(model_name) and not (
        is_deepseek_vision_model(model_name) or ("deepseek-v4" in lm and "flash" in lm)
    ):
        return False

    # Scoped provider checks to avoid false positives (e.g., Z.ai "v-" pattern matching "jev")
    if (
        is_openai_model_family(model_name)
        or is_google_model_family(model_name)
        or is_anthropic_model_family(model_name)
        or "grok" in lm
        or base_name.startswith(("mimo-", "kimi-"))
        or is_mimo_multimodal_model(model_name)
        or is_moonshot_k3_model(model_name)
        or is_meta_reasoning_model(model_name)
        or is_deepseek_vision_model(model_name)
        or ("deepseek-v4" in lm and "flash" in lm)
        or (lm.startswith("glm-") and is_zai_vision_model(model_name))
    ):
        return True

    if (
        "qwen" in lm
        or base_name == "minimax-m3"
        or base_name.startswith("longcat-2.5")
        or "vision" in lm
        or "multimodal" in lm
        or "omni" in lm
        or "vl" in lm
    ):
        return True

    # Unlisted or temporary models default to vision-capable
    return True


def is_opencode_reasoning_model(model_name: str | None) -> bool:
    """Check if an OpenCode model is reasoning-capable (OpenCode serves other vendors' models)."""
    return (
        is_deepseek_reasoning_model(model_name)
        or is_qwencloud_reasoning_model(model_name)
        or is_zai_reasoning_model(model_name)
        or is_openai_reasoning_model(model_name)
        or is_moonshot_reasoning_model(model_name)
        or is_mimo_reasoning_model(model_name)
        or is_xai_reasoning_model(model_name)
        or is_meta_reasoning_model(model_name)
        or is_anthropic_reasoning_model(model_name)
    )


def is_mimo_reasoning_model(model_name: str | None) -> bool:
    """Check if a MiMo model is reasoning-capable (hybrid thinking)."""
    if not model_name:
        return False
    return "mimo-v2.6" in model_name.lower()


def is_qwencloud_reasoning_model(model_name: str | None) -> bool:
    """Check if a QwenCloud model is reasoning-capable (toggleable thinking mode)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "qwen" in lm


def supports_qwencloud_reasoning_effort(model_name: str | None) -> bool:
    """Check if a QwenCloud model supports the reasoning_effort API parameter (Qwen 3.8+)."""
    if not model_name:
        return False
    return "qwen3.8" in model_name.lower()


def is_meta_reasoning_model(model_name: str | None) -> bool:
    """Check if a Meta Model API model is reasoning-capable (muse-spark and muse-wonder series)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "muse-spark" in lm or "muse-wonder" in lm


def supports_meta_reasoning_effort(model_name: str | None) -> bool:
    """Check if a Meta Model API model supports reasoning effort configuration."""
    return is_meta_reasoning_model(model_name)


def supports_meta_max_effort(model_name: str | None) -> bool:
    """Check if a Meta model supports "max" effort (muse-wonder, standard-tier muse-spark-1.3)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "muse-wonder" in lm or ("muse-spark-1.3" in lm and "contributor" not in lm)


def is_opus_45_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 4.5 (supports effort parameter)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.5" in lm) or ("4-5" in lm)


def is_opus_46_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 4.6 (adaptive thinking, max effort)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.6" in lm) or ("4-6" in lm)


def is_opus_47_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 4.7 (adaptive thinking, xhigh effort, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.7" in lm) or ("4-7" in lm)


def is_opus_48_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 4.8 (same API constraints as Opus 4.7)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("4.8" in lm) or ("4-8" in lm)


def is_opus_5_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 5 (adaptive thinking on by default, max effort support, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return "claude-opus-5" in lm


def is_opus_55_model(model_name: str | None) -> bool:
    """Check if a model is Claude Opus 5.5 (always-on adaptive thinking, max effort support, default effort medium, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "opus" not in lm:
        return False
    return ("5.5" in lm) or ("5-5" in lm)


def is_fable_5_model(model_name: str | None) -> bool:
    """Check if a model is Claude Fable 5+ (always-on adaptive thinking, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "claude-fable-5" in lm


def is_sonnet_5_model(model_name: str | None) -> bool:
    """Check if a model is Claude Sonnet 5 (adaptive thinking on by default, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "sonnet" not in lm:
        return False
    if is_sonnet_55_model(model_name):
        return False
    return "claude-sonnet-5" in lm


def is_sonnet_55_model(model_name: str | None) -> bool:
    """Check if a model is Claude Sonnet 5.5 (adaptive thinking on by default, effort parameter with max effort, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "sonnet" not in lm:
        return False
    return ("5.5" in lm) or ("5-5" in lm)


def is_haiku_55_model(model_name: str | None) -> bool:
    """Check if a model is Claude Haiku 5.5 (adaptive thinking on by default, no sampling params)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "claude-haiku-5-5" in lm or "claude-haiku-5.5" in lm


def anthropic_model_flags(model_name: str | None) -> dict[str, bool]:
    """Cumulative Claude API capability flags for generation_config / OpenRouter metadata.

    Tiers (each includes the capabilities of tiers below):
    - Opus 4.5: is_claude_effort
    - Sonnet/Opus 4.6: + is_claude_effort_max
    - Sonnet/Opus 4.7/4.8/5: + is_claude_effort_xhigh (also strips sampling params)
    - Sonnet/Opus 5, Haiku 5.5: + is_claude_adaptive_default
    - Fable 5/5.1, Opus 5.5: same as 4.7+ plus is_claude_omit_thinking
    """
    if not model_name:
        return {}
    if is_fable_5_model(model_name) or is_opus_55_model(model_name):
        return {
            "is_claude_effort": True,
            "is_claude_effort_max": True,
            "is_claude_effort_xhigh": True,
            "is_claude_omit_thinking": True,
        }
    if is_sonnet_55_model(model_name):
        return {
            "is_claude_effort": True,
            "is_claude_effort_max": True,
            "is_claude_effort_xhigh": True,
            "is_claude_adaptive_default": True,
            "is_claude_between_tools": True,
        }
    if (
        is_sonnet_5_model(model_name)
        or is_opus_5_model(model_name)
        or is_haiku_55_model(model_name)
    ):
        return {
            "is_claude_effort": True,
            "is_claude_effort_max": True,
            "is_claude_effort_xhigh": True,
            "is_claude_adaptive_default": True,
        }
    if is_opus_47_model(model_name) or is_opus_48_model(model_name):
        return {
            "is_claude_effort": True,
            "is_claude_effort_max": True,
            "is_claude_effort_xhigh": True,
        }
    if is_46_model(model_name):
        return {
            "is_claude_effort": True,
            "is_claude_effort_max": True,
        }
    if is_opus_45_model(model_name):
        return {"is_claude_effort": True}
    return {}


def is_anthropic_no_sampling_model(model_name: str | None) -> bool:
    """Anthropic models that reject temperature/top_k at the API."""
    flags = anthropic_model_flags(model_name)
    return flags.get("is_claude_effort_xhigh", False) or flags.get(
        "is_claude_no_sampling", False
    )


def anthropic_omits_thinking_config(model_name: str | None) -> bool:
    """Models with always-on thinking that omit reasoning-effort API config (e.g. Fable 5+)."""
    return anthropic_model_flags(model_name).get("is_claude_omit_thinking", False)


def anthropic_uses_adaptive_thinking_info(model_name: str | None) -> bool:
    """Whether reasoning-effort UI should offer/describe adaptive (auto/none) thinking."""
    flags = anthropic_model_flags(model_name)
    return bool(
        flags.get("is_claude_effort_max") and not flags.get("is_claude_omit_thinking")
    )


def anthropic_reasoning_effort_config(
    model_name: str | None,
) -> tuple[bool, list[str], str | None]:
    """Reasoning-effort dropdown config for Anthropic and OpenRouter Claude models."""
    if anthropic_omits_thinking_config(model_name):
        return False, [], None
    if not is_anthropic_reasoning_model(model_name):
        return False, [], None
    flags = anthropic_model_flags(model_name)
    if flags.get("is_claude_effort_max"):
        return True, ["auto", "none"], "auto"
    return True, ["high", "medium", "low", "none"], "low"


def anthropic_effort_config(
    model_name: str | None,
) -> tuple[bool, list[str], str | None]:
    """Effort dropdown config for Anthropic and OpenRouter Claude models."""
    flags = anthropic_model_flags(model_name)
    if not flags.get("is_claude_effort"):
        return False, [], None
    if is_opus_55_model(model_name):
        return True, ["max", "xhigh", "high", "medium", "low"], "high"
    if flags.get("is_claude_effort_xhigh"):
        return True, ["max", "xhigh", "high", "medium", "low"], "high"
    if flags.get("is_claude_effort_max"):
        return True, ["max", "high", "medium", "low"], "high"
    return True, ["high", "medium", "low"], "high"


def clamp_claude_effort(model_name: str | None, effort: str) -> str:
    """Replace an effort a Claude model lacks with the nearest one it accepts."""
    return _nearest_effort(effort, tuple(anthropic_effort_config(model_name)[1]))


def is_sonnet_46_model(model_name: str | None) -> bool:
    """Check if a model is Claude Sonnet 4.6 (adaptive thinking, max effort)."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "claude" not in lm or "sonnet" not in lm:
        return False
    return ("4.6" in lm) or ("4-6" in lm)


def is_46_model(model_name: str | None) -> bool:
    """Check if a model is any Claude 4.6 variant (Opus or Sonnet)."""
    return is_opus_46_model(model_name) or is_sonnet_46_model(model_name)


def is_gemma_model(model_name: str | None) -> bool:
    """Check if a model is a Gemma model (e.g., gemma-4-31b-it)."""
    if not model_name:
        return False
    return "gemma-" in model_name.lower()


def is_gemini_4_model(model_name: str | None) -> bool:
    """Check if a model is any Gemini 4 variant (e.g., gemini-4-argon)."""
    if not model_name:
        return False
    return "gemini-4" in model_name.lower()


def is_gemini_3_plus_model(model_name: str | None) -> bool:
    """Check if a model is Gemini 3 or newer (3, 3.1, 4, etc.).

    Gemini 4 shares the Gemini 3 API surface: thinking levels and per-part media resolution.
    """
    if not model_name:
        return False
    return "gemini-3" in model_name.lower() or is_gemini_4_model(model_name)


def supports_gemini_code_execution(model_name: str | None) -> bool:
    """Check if a Google model gets the code execution tool (Gemini 3+, not 2.5 or Gemma)."""
    return is_gemini_3_plus_model(model_name)


def is_gemini_38_flash_model(model_name: str | None) -> bool:
    """Check if a model is Gemini 3.8 Flash."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-3.8" in lm and "flash" in lm


def is_gemini_36_flash_model(model_name: str | None) -> bool:
    """Check if a model is Gemini 3.6 Flash."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-3.6" in lm and "flash" in lm


def is_gemini_35_flash_lite_model(model_name: str | None) -> bool:
    """Check if a model is Gemini 3.5 Flash Lite."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-3.5" in lm and "flash" in lm and "lite" in lm


def is_gemini_no_sampling_model(model_name: str | None) -> bool:
    """Check if a Gemini model does not support custom sampling parameters."""
    return (
        is_gemini_38_flash_model(model_name)
        or is_gemini_36_flash_model(model_name)
        or is_gemini_35_flash_lite_model(model_name)
        or is_gemini_4_model(model_name)
    )


def is_gemini_no_minimal_thinking_model(model_name: str | None) -> bool:
    """Check if a Gemini model has no "minimal" thinking level (3.8 Flash and Gemini 4)."""
    return is_gemini_38_flash_model(model_name) or is_gemini_4_model(model_name)


def gemini_thinking_levels(model_name: str | None) -> tuple[str, ...]:
    """Thinking levels a Gemini 3+ or Gemma model accepts.

    Empty for models without thinking levels, such as Gemini 2.5 with its token budget.
    """
    if is_gemma_model(model_name):
        return ("minimal", "high")
    if not is_gemini_3_plus_model(model_name):
        return ()
    if "flash" in model_name.lower() and not is_gemini_no_minimal_thinking_model(
        model_name
    ):
        return ("minimal", "low", "medium", "high")
    return ("low", "medium", "high")


def clamp_gemini_reasoning_effort(model_name: str | None, effort: str) -> str:
    """Replace a thinking level a Gemini model lacks with the nearest one it accepts."""
    return _nearest_effort(effort, gemini_thinking_levels(model_name))


def is_gemini_25_flash_model(model_name: str | None) -> bool:
    """Check if a model is Gemini 2.5 Flash."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-2.5" in lm and "flash" in lm


def is_gemini_25_pro_model(model_name: str | None) -> bool:
    """Check if a model is Gemini 2.5 Pro."""
    if not model_name:
        return False
    return "gemini-2.5-pro" in model_name.lower()


def is_google_reasoning_model(model_name: str | None) -> bool:
    """Check if a Google model is reasoning-capable (Gemini 2.5, 3/4 series, or Gemma)."""
    if not model_name:
        return False
    lm = model_name.lower()
    return "gemini-2.5" in lm or is_gemini_3_plus_model(lm) or "gemma-" in lm


def is_rosetta_model(model_name: str | None) -> bool:
    """Check if a model is a YanoljaNEXT Rosetta translation model.

    Requires both 'rosetta' and 'yanoljanext' in the name to avoid false positives.
    Matches e.g. 'yanoljanext-rosetta-4b-2511'.
    """
    if not model_name:
        return False
    lm = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    return "rosetta" in lm and "yanoljanext" in lm


def is_hy_mt2_model(model_name: str | None) -> bool:
    """Check if a model is a Tencent Hy-MT2 translation model."""
    if not model_name:
        return False
    return "hy-mt2" in model_name.lower()


def get_hy_mt2_sampling_defaults(
    model_name: str | None,
) -> dict[str, float | int | None]:
    """Model-card sampling defaults for Hy-MT2 (1.8B/7B vs 30B-A3B)."""
    lm = (model_name or "").lower()
    if "30b" in lm or "a3b" in lm:
        return {
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": None,
            "max_tokens": 4096,
        }
    return {
        "temperature": 0.7,
        "top_p": 0.6,
        "top_k": 20,
        "max_tokens": 4096,
    }


# provider -> check for whether a model reasons. OpenRouter is absent because its
# model list answers that, not the model name (see openrouter_is_reasoning_model).
REASONING_MODEL_CHECKS = {
    "Google": is_google_reasoning_model,
    "OpenAI": is_openai_reasoning_model,
    "Anthropic": is_anthropic_reasoning_model,
    "SpaceXAI": is_xai_reasoning_model,
    "Meta Model": is_meta_reasoning_model,
    "OpenAI-Compatible": is_openai_compatible_reasoning_model,
    "DeepSeek": is_deepseek_reasoning_model,
    "Z.ai": is_zai_reasoning_model,
    "Moonshot AI": is_moonshot_reasoning_model,
    "Xiaomi MiMo": is_mimo_reasoning_model,
    "QwenCloud": is_qwencloud_reasoning_model,
    "OpenCode": is_opencode_reasoning_model,
}
