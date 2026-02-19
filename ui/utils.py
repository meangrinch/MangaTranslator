import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import gradio as gr
import requests
from PIL import Image

from core.llm_defaults import get_provider_sampling_defaults
from utils.endpoints import openrouter_is_reasoning_model
from utils.exceptions import ValidationError
from utils.logging import log_message
from utils.model_metadata import (
    get_max_tokens_cap,
    is_46_model,
    is_deepseek_reasoning_model,
    is_openai_compatible_reasoning_model,
    is_opus_45_model,
    is_xai_reasoning_model,
    is_zai_reasoning_model,
)

from .settings_manager import DEFAULT_SETTINGS, PROVIDER_MODELS, get_saved_settings


def get_available_providers(ocr_method: str) -> List[str]:
    """Get list of available providers based on OCR method."""
    all_providers = list(PROVIDER_MODELS.keys())

    if ocr_method == "manga-ocr":
        return all_providers
    else:
        # For LLM OCR, exclude text-only providers (DeepSeek)
        return [p for p in all_providers if p not in ("DeepSeek",)]


ERROR_PREFIX = "❌ Error: "
SUCCESS_PREFIX = "✅ "

# Global caches for API models (Session-based)
OPENROUTER_MODEL_CACHE = {}
COMPATIBLE_MODEL_CACHE = {"url": None, "models": None}


def get_available_font_packs(fonts_base_dir: Path) -> Tuple[List[str], Optional[str]]:
    """Get list of available font packs (subdirectories) in the fonts directory"""
    if not fonts_base_dir.exists():
        return [], None
    font_dirs = [d.name for d in fonts_base_dir.iterdir() if d.is_dir()]
    font_dirs.sort()

    if font_dirs:
        default_font = font_dirs[0]
    else:
        default_font = None

    return font_dirs, default_font


def validate_api_key(api_key: str, provider: str) -> tuple[bool, str]:
    """Validate API key format based on provider."""
    if not api_key and provider != "OpenAI-Compatible":
        return False, f"{provider} API key is required"
    elif not api_key and provider == "OpenAI-Compatible":
        return True, f"{provider} API key is optional and not provided."  # Valid state

    if provider == "Google" and not (api_key.startswith("AI") and len(api_key) == 39):
        return (
            False,
            "Invalid Google API key format (should start with 'AI' and be 39 chars)",
        )
    if provider == "OpenAI" and not (api_key.startswith("sk-") and len(api_key) >= 48):
        return False, "Invalid OpenAI API key format (should start with 'sk-')"
    if provider == "Anthropic" and not (
        api_key.startswith("sk-ant-") and len(api_key) >= 100
    ):
        return False, "Invalid Anthropic API key format (should start with 'sk-ant-')"
    if provider == "xAI" and not api_key.startswith("xai-"):
        return False, "Invalid xAI API key format (should start with 'xai-')"
    if provider == "OpenRouter" and not (
        api_key.startswith("sk-or-") and len(api_key) >= 48
    ):
        return False, "Invalid OpenRouter API key format (should start with 'sk-or-')"
    if provider == "DeepSeek" and not api_key.startswith("sk-"):
        return False, "Invalid DeepSeek API key format (should start with 'sk-')"
    if provider == "Moonshot AI" and not api_key.startswith("sk-"):
        return False, "Invalid Moonshot AI API key format (should start with 'sk-')"
    # No specific format check for Z.ai or OpenAI-Compatible keys

    return True, f"{provider} API key format looks valid"


def validate_huggingface_token(token: str) -> tuple[bool, str]:
    """Validate HuggingFace token format."""
    if not token:
        return True, "HuggingFace token is optional and not provided."

    if not token.startswith("hf_"):
        return False, "Invalid HuggingFace token format (should start with 'hf_')"

    return True, "HuggingFace token format looks valid"


def validate_image(image: Any) -> tuple[bool, str]:
    """Validate uploaded image (accepts path or PIL Image)"""
    if image is None:
        return False, "Please upload an image"

    try:
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            filepath = Path(image)
            if not filepath.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                return False, "Unsupported image format. Please use JPEG, PNG or WEBP."
        elif isinstance(image, Image.Image):
            img = image
        else:
            return False, f"Unexpected image type: {type(image)}"

        width, height = img.size
        if width < 600 or height < 600:
            return (
                False,
                f"Image dimensions too small ({width}x{height}). Min recommended size is 600x600.",
            )
        if width > 8000 or height > 8000:
            return (
                False,
                f"Image dimensions too large ({width}x{height}). Max allowed size is 8000x8000.",
            )

        return True, "Image is valid"
    except FileNotFoundError:
        return False, f"Invalid image path: {image}"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def validate_font_directory(font_dir: Path) -> tuple[bool, str]:
    """Validate that the font directory contains at least one font file"""
    if not font_dir.exists():
        return (
            False,
            f"Font directory '{font_dir.name}' not found at {font_dir.resolve()}",
        )
    if not font_dir.is_dir():
        return False, f"Path '{font_dir.name}' is not a directory."

    font_files = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
    if not font_files:
        return (
            False,
            f"No font files (.ttf or .otf) found in '{font_dir.name}' directory",
        )

    return True, f"Found {len(font_files)} font files in directory"


def update_font_dropdown(fonts_base_dir: Path):
    """Update the font pack dropdown list"""
    try:
        font_packs, _ = get_available_font_packs(fonts_base_dir)
        saved_settings = get_saved_settings()
        current_font = saved_settings.get("font_pack")
        selected_font_val = (
            current_font
            if current_font in font_packs
            else (font_packs[0] if font_packs else None)
        )

        current_batch_font = saved_settings.get("batch_font_pack")
        selected_batch_font_val = (
            current_batch_font
            if current_batch_font in font_packs
            else (font_packs[0] if font_packs else None)
        )

        return (
            gr.update(choices=font_packs, value=selected_font_val),
            gr.update(choices=font_packs, value=selected_batch_font_val),
            f"{SUCCESS_PREFIX}Found {len(font_packs)} font packs",
        )
    except Exception as e:
        return gr.update(choices=[]), gr.update(choices=[]), f"{ERROR_PREFIX}{str(e)}"


def refresh_models_and_fonts(fonts_base_dir: Path):
    """Update font dropdown lists; YOLO is auto-detected so no model dropdown update."""
    try:
        font_packs, _ = get_available_font_packs(fonts_base_dir)
        saved_settings = get_saved_settings()
        current_font = saved_settings.get("font_pack")
        selected_font_val = (
            current_font
            if current_font in font_packs
            else (font_packs[0] if font_packs else None)
        )
        single_font_result = gr.update(choices=font_packs, value=selected_font_val)

        current_batch_font = saved_settings.get("batch_font_pack")
        selected_batch_font_val = (
            current_batch_font
            if current_batch_font in font_packs
            else (font_packs[0] if font_packs else None)
        )
        batch_font_result = gr.update(choices=font_packs, value=selected_batch_font_val)

        current_osb_font = saved_settings.get("outside_text_osb_font_pack")
        selected_osb_font_val = (
            current_osb_font
            if current_osb_font in font_packs
            else (font_packs[0] if font_packs else None)
        )
        osb_font_result = gr.update(
            choices=[""] + font_packs, value=selected_osb_font_val
        )

        font_count = len(font_packs)
        font_text = "1 font pack" if font_count == 1 else f"{font_count} font packs"
        gr.Info(f"YOLO model auto-detected. Found {font_text}")

        return single_font_result, batch_font_result, osb_font_result
    except Exception as e:
        gr.Error(f"Error refreshing resources: {str(e)}")
        font_packs, _ = get_available_font_packs(fonts_base_dir)
        return (
            gr.update(choices=font_packs),
            gr.update(choices=font_packs),
            gr.update(choices=[""] + font_packs),
        )


def _is_google_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if a Google model is reasoning-capable."""
    if not model_name:
        return False
    name = (model_name or "").lower()
    return "gemini-2.5" in name or "gemini-3" in name


def _is_moonshot_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if a Moonshot model is reasoning-capable.

    Moonshot kimi-k2-thinking and kimi-k2-thinking-turbo are reasoning models,
    but their reasoning cannot be controlled via reasoning_effort.
    """
    if not model_name:
        return False
    lm = model_name.lower()
    return "thinking" in lm or "kimi-k2.5" in lm


def is_reasoning_model(provider: str, model_name: Optional[str]) -> bool:
    """Check if a model is reasoning-capable based on provider and model name."""
    if not model_name:
        return False

    if provider == "Google":
        return _is_google_reasoning_model(model_name)
    elif provider == "OpenAI":
        return _is_openai_reasoning_model(model_name)
    elif provider == "Anthropic":
        return _is_anthropic_reasoning_model(model_name, provider="Anthropic")
    elif provider == "xAI":
        return is_xai_reasoning_model(model_name)
    elif provider == "DeepSeek":
        return is_deepseek_reasoning_model(model_name)
    elif provider == "Z.ai":
        return is_zai_reasoning_model(model_name)
    elif provider == "Moonshot AI":
        return _is_moonshot_reasoning_model(model_name)
    elif provider == "OpenRouter":
        try:
            return openrouter_is_reasoning_model(model_name, debug=False)
        except Exception:
            # Fallback to False if detection fails
            return False
    elif provider == "OpenAI-Compatible":
        return is_openai_compatible_reasoning_model(model_name)
    else:
        return False


def get_enable_web_search_label_and_info(provider: str) -> Tuple[str, str]:
    """
    Returns the label and info text for the enable_web_search checkbox based on provider.

    Args:
        provider: The provider name

    Returns:
        Tuple of (label, info) strings
    """
    # All providers use the same label
    label = "Enable Web Search"

    # Provider-specific info text
    info_map = {
        "Google": (
            "Use Gemini's web search for up-to-date information. "
            "Might improve translation quality. Can be used with 'special instructions' to discover more information."
        ),
        "OpenRouter": (
            "Use OpenRouter's web search (Exa) for up-to-date information. "
            "Might improve translation quality. Can be used with 'special instructions' to discover more information."
        ),
        "OpenAI": (
            "Use OpenAI's web search tool for up-to-date information. "
            "Might improve translation quality. Can be used with 'special instructions' to discover more information."
        ),
        "Anthropic": (
            "Use Anthropic's web search tool for up-to-date information. "
            "Might improve translation quality. Can be used with 'special instructions' to discover more information."
        ),
        "xAI": (
            "Use xAI's web search tool for up-to-date information. "
            "Might improve translation quality. Can be used with 'special instructions' to discover more information."
        ),
        "Z.ai": (
            "Use Z.ai's web search tool for up-to-date information. "
            "Might improve translation quality. Can be used with 'special instructions' to discover more information."
        ),
        "Moonshot AI": (
            "Use Moonshot AI's web search tool for up-to-date information. "
            "Might improve translation quality. Can be used with 'special instructions' to discover more information."
        ),
    }

    info = info_map.get(
        provider,
        "Enable web search for up-to-date information. Might improve translation quality.",
    )
    return (label, info)


def get_reasoning_effort_label(provider: str, model_name: Optional[str] = None) -> str:
    """
    Returns the label for the reasoning_effort dropdown based on provider/model.

    Args:
        provider: The provider name
        model_name: The model name (optional)

    Returns:
        Label string:
        - "Thinking Level" for Gemini 3 models
        - "Thinking Budget" for Google reasoning models (non-Gemini 3)
        - "Extended Thinking" for Anthropic reasoning models
        - "Reasoning Effort" otherwise
    """
    if not model_name:
        return "Reasoning Effort"

    lm = model_name.lower()
    is_gemini_3 = "gemini-3" in lm

    if (provider == "Google" or provider == "OpenRouter") and is_gemini_3:
        return "Thinking Level"
    elif (
        provider == "Google"
        and is_reasoning_model(provider, model_name)
        and not is_gemini_3
    ):
        return "Thinking Budget"
    elif provider == "OpenRouter":
        is_google_model = "google/" in lm or "gemini" in lm
        if (
            is_google_model
            and is_reasoning_model(provider, model_name)
            and not is_gemini_3
        ):
            return "Thinking Budget"
        elif _is_anthropic_reasoning_model(model_name, "OpenRouter"):
            return "Extended Thinking"
    elif provider == "Anthropic" and _is_anthropic_reasoning_model(
        model_name, provider
    ):
        return "Extended Thinking"
    else:
        return "Reasoning Effort"


def get_reasoning_effort_info_text(
    provider: str, model_name: Optional[str] = None, choices: Optional[List[str]] = None
) -> str:
    """Get provider-specific info text for reasoning effort dropdown."""
    if choices is None:
        choices = []

    # Determine base description text based on provider/model
    if provider == "Google":
        if model_name and "gemini-3" in model_name.lower():
            return "Controls model's internal reasoning effort."
        base_text = "Controls reasoning token allocation relative to 'max_tokens'"
    elif provider == "OpenAI":
        return "Controls model's internal reasoning effort."
    elif provider == "xAI":
        return "Controls model's internal reasoning effort."
    elif provider == "Moonshot AI":
        return "Enables or disables model thinking (high=enabled, none=disabled)."
    elif provider == "Z.ai":
        return "Enables or disables model thinking (auto=enabled, none=disabled)."
    elif provider == "OpenRouter" and model_name:
        lm = model_name.lower()
        is_openai_reasoning = (
            "gpt-5" in lm or "o1" in lm or "o3" in lm or "o4-mini" in lm
        )
        is_grok_reasoning = "grok-4" in lm
        is_gemini_3 = "gemini-3" in lm
        if is_openai_reasoning or is_grok_reasoning or is_gemini_3:
            return "Controls model's internal reasoning effort."
        else:
            base_text = (
                "Controls reasoning token budget allocation relative to 'max_tokens'"
            )
    else:
        base_text = "Controls reasoning token budget allocation relative to max_tokens"

    options = []
    if "auto" in choices:
        options.append("auto=model decides")
    if "xhigh" in choices:
        options.append("xhigh=95%")
    if "high" in choices:
        options.append("high=80%")
    if "medium" in choices:
        options.append("medium=50%")
    if "low" in choices:
        options.append("low=20%")
    if "minimal" in choices:
        options.append("minimal=10%")
    if "none" in choices:
        if model_name and "gemini-2.5-pro" in model_name.lower():
            options.append("none=128 tokens - minimum allowed")
        else:
            options.append("none=disabled")

    if options:
        return f"{base_text} ({', '.join(options)})."
    else:
        return base_text + "."


def _is_openai_reasoning_model(model_name: Optional[str]) -> bool:
    """Check if an OpenAI model is reasoning-capable."""
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


def _is_anthropic_reasoning_model(
    model_name: Optional[str], provider: str = "Anthropic"
) -> bool:
    """Check if an Anthropic model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    if provider == "OpenRouter":
        # OpenRouter uses dots (4.5) not hyphens (4-5)
        return (
            "claude-opus-4" in lm or "claude-sonnet-4" in lm or "claude-haiku-4.5" in lm
        )
    else:
        return (
            lm.startswith("claude-opus-4")
            or lm.startswith("claude-sonnet-4")
            or lm.startswith("claude-haiku-4-5")
        )


def _is_grok_reasoning_model(model_name: Optional[str], provider: str = "xAI") -> bool:
    """Check if a Grok model is reasoning-capable."""
    if not model_name:
        return False
    lm = model_name.lower()
    if "non-reasoning" in lm:
        return False
    if provider == "OpenRouter":
        return "grok-4" in lm
    else:
        return "grok-4" in lm and "reasoning" in lm


def get_reasoning_effort_config(
    provider: str, model_name: Optional[str]
) -> Tuple[bool, List[str], Optional[str]]:
    """
    Get reasoning effort configuration for a provider/model combination.

    Returns:
        Tuple of (visible, choices, default_value)
    """
    if not model_name:
        return False, [], None

    lm = model_name.lower()

    if provider == "Google":
        is_reasoning = is_reasoning_model(provider, model_name)
        if not is_reasoning:
            return False, [], None

        is_gemini_3 = "gemini-3" in lm
        if is_gemini_3:
            if "flash" in lm:
                return True, ["high", "medium", "low", "minimal"], "high"
            if "gemini-3.1" in lm:
                return True, ["high", "medium", "low"], "high"
            return True, ["high", "low"], "high"

        is_flash = "gemini-2.5-flash" in lm
        is_pro = "gemini-2.5-pro" in lm
        if is_flash or is_pro:
            return True, ["auto", "high", "medium", "low", "minimal", "none"], "auto"
        else:
            return True, ["auto", "high", "medium", "low", "minimal"], "auto"

    elif provider == "OpenAI":
        is_reasoning = _is_openai_reasoning_model(model_name)
        if not is_reasoning:
            return False, [], None

        if "chat" in lm:
            return False, [], None

        is_gpt5_1 = lm.startswith("gpt-5.1")
        is_gpt5_2 = lm.startswith("gpt-5.2")
        is_gpt5 = lm.startswith("gpt-5") and not (is_gpt5_1 or is_gpt5_2)
        if is_gpt5_1:
            return True, ["high", "medium", "low", "none"], "medium"
        elif is_gpt5_2:
            return True, ["xhigh", "high", "medium", "low", "none"], "medium"
        elif is_gpt5:
            return True, ["high", "medium", "low", "minimal"], "medium"
        else:
            return True, ["high", "medium", "low"], "medium"

    elif provider == "Anthropic":
        is_reasoning = _is_anthropic_reasoning_model(model_name, provider)
        if not is_reasoning:
            return False, [], None
        # Claude 4.6 models use adaptive thinking
        if is_46_model(model_name):
            return True, ["auto", "none"], "auto"
        # Older models use budget-based thinking
        return True, ["high", "medium", "low", "none"], "low"

    elif provider == "xAI":
        is_reasoning = is_xai_reasoning_model(model_name)
        if not is_reasoning:
            return False, [], None
        return True, ["high", "low"], "high"

    elif provider == "DeepSeek":
        is_reasoning = is_deepseek_reasoning_model(model_name)
        if not is_reasoning:
            return False, [], None
        return True, ["high", "medium", "low"], "high"

    elif provider == "Z.ai":
        is_reasoning = is_zai_reasoning_model(model_name)
        if not is_reasoning:
            return False, [], None
        # Z.ai supports enabled/disabled thinking, mapped to auto/none
        return True, ["auto", "none"], "auto"

    elif provider == "Moonshot AI":
        # Moonshot AI kimi-k2.5 supports reasoning control via the 'thinking' parameter.
        # Older models have always-on reasoning.
        if "kimi-k2.5" in lm:
            return True, ["high", "none"], "high"
        return False, [], None

    elif provider == "OpenRouter":
        is_google_model = "google/" in lm or "gemini" in lm

        if is_google_model:
            is_reasoning = is_reasoning_model(provider, model_name)
            if not is_reasoning:
                return False, [], None

            return True, ["xhigh", "high", "medium", "low", "minimal", "none"], "low"

        try:
            is_reasoning = openrouter_is_reasoning_model(model_name, debug=False)
        except Exception:
            is_reasoning = False

        if is_reasoning:
            return True, ["xhigh", "high", "medium", "low", "minimal", "none"], "low"

        is_anthropic_model = "anthropic/" in lm or lm.startswith("claude-")
        if is_anthropic_model:
            return False, [], "none"
        else:
            return False, [], None

    elif provider == "OpenAI-Compatible":
        return False, [], None

    return False, [], None


def get_effort_config(
    provider: str, model_name: Optional[str]
) -> Tuple[bool, List[str], Optional[str]]:
    """
    Get effort configuration for Claude Opus/Sonnet 4.6 and Opus 4.5 models (Anthropic/OpenRouter).

    Returns:
        Tuple of (visible, choices, default_value)
    """
    if provider not in ("Anthropic", "OpenRouter"):
        return False, [], None

    if is_46_model(model_name):
        # Claude 4.6 models support "max" effort level
        return True, ["max", "high", "medium", "low"], "medium"
    elif is_opus_45_model(model_name):
        return True, ["high", "medium", "low"], "medium"
    else:
        return False, [], None


def get_media_resolution_config(
    provider: str, model_name: Optional[str]
) -> Tuple[bool, List[str], str]:
    """
    Get media resolution configuration for a provider/model combination.

    Returns:
        Tuple of (visible, choices, info_text)
    """
    is_gemini_3 = (
        provider == "Google" and model_name and "gemini-3" in model_name.lower()
    )

    if is_gemini_3:
        return (
            True,
            ["auto", "high", "medium", "low"],
            "Resolution for Gemini 3 to process images.",
        )
    elif provider == "xAI":
        return (True, ["auto", "high", "low"], "Resolution for Grok to process images.")

    return False, ["auto"], ""


def update_translation_ui(provider: str, _current_temp: float, ocr_method: str = "LLM"):
    """Updates API key/URL visibility, model dropdown, temp slider max, and top_k interactivity.

    Args:
        provider: The selected translation provider
        _current_temp: Current temperature value (unused but kept for compatibility)
        ocr_method: OCR method ("LLM" or "manga-ocr"). Used to filter vision-only models.
    """
    saved_settings = get_saved_settings()
    provider_models_dict = saved_settings.get(
        "provider_models", DEFAULT_SETTINGS["provider_models"]
    )
    remembered_model = provider_models_dict.get(provider)

    models = PROVIDER_MODELS.get(provider, [])

    if provider == "Z.ai" and ocr_method == "LLM":
        models = [m for m in models if m in ("glm-4.5v", "glm-4.6v")]
    elif provider == "Moonshot AI" and ocr_method == "LLM":
        models = [m for m in models if "kimi-k2.5" in m.lower()]

    selected_model = (
        remembered_model
        if remembered_model in models
        else (models[0] if models else None)
    )
    model_update = gr.update(choices=models, value=selected_model)

    google_visible_update = gr.update(visible=(provider == "Google"))
    openai_visible_update = gr.update(visible=(provider == "OpenAI"))
    anthropic_visible_update = gr.update(visible=(provider == "Anthropic"))
    xai_visible_update = gr.update(visible=(provider == "xAI"))
    deepseek_visible_update = gr.update(visible=(provider == "DeepSeek"))
    zai_visible_update = gr.update(visible=(provider == "Z.ai"))
    moonshot_visible_update = gr.update(visible=(provider == "Moonshot AI"))
    openrouter_visible_update = gr.update(visible=(provider == "OpenRouter"))
    openai_compatible_url_visible_update = gr.update(
        visible=(provider == "OpenAI-Compatible")
    )
    openai_compatible_key_visible_update = gr.update(
        visible=(provider == "OpenAI-Compatible")
    )
    if provider == "OpenRouter" or provider == "OpenAI-Compatible":
        model_update = gr.update(
            value=remembered_model,
            choices=[remembered_model] if remembered_model else [],
        )
    else:
        model_update = gr.update(choices=models, value=selected_model)

    sampling_defaults = get_provider_sampling_defaults(provider)
    default_temp = float(sampling_defaults["temperature"])
    default_top_p = float(sampling_defaults["top_p"])
    default_top_k = int(sampling_defaults["top_k"])

    temp_max = 1.0 if provider == "Anthropic" else 2.0
    new_temp_value = min(default_temp, temp_max)
    temp_update = gr.update(maximum=temp_max, value=new_temp_value)

    top_k_interactive = provider not in (
        "OpenAI",
        "Anthropic",
        "xAI",
        "DeepSeek",
        "Z.ai",
        "Moonshot AI",
    )
    top_k_update = gr.update(interactive=top_k_interactive, value=default_top_k)

    top_p_interactive = provider != "Anthropic"
    top_p_update = gr.update(value=default_top_p, interactive=top_p_interactive)

    if remembered_model:
        is_reasoning = is_reasoning_model(provider, remembered_model)
        max_tokens_value = 16384 if is_reasoning else 4096
    else:
        max_tokens_value = 4096

    max_tokens_cap = get_max_tokens_cap(provider, remembered_model)
    max_tokens_maximum = max_tokens_cap if max_tokens_cap is not None else 63488
    max_tokens_update = gr.update(value=max_tokens_value, maximum=max_tokens_maximum)

    def _is_gemini_3_model(name: Optional[str]) -> bool:
        if not name:
            return False
        return "gemini-3" in name.lower()

    def _is_gemini_3_flash_model(name: Optional[str]) -> bool:
        if not name:
            return False
        lm = name.lower()
        return "gemini-3" in lm and "flash" in lm

    is_gemini_3_google = (
        provider == "Google"
        and remembered_model
        and _is_gemini_3_model(remembered_model)
    )
    is_gemini_3_flash_google = (
        provider == "Google"
        and remembered_model
        and _is_gemini_3_flash_model(remembered_model)
    )
    is_gemini_3_openrouter = (
        provider == "OpenRouter"
        and remembered_model
        and _is_gemini_3_model(remembered_model)
    )

    enable_web_search_visible = provider not in ("OpenAI-Compatible", "DeepSeek")
    enable_web_search_label, enable_web_search_info = (
        get_enable_web_search_label_and_info(provider)
    )

    enable_web_search_update = gr.update(
        visible=enable_web_search_visible,
        label=enable_web_search_label,
        info=enable_web_search_info,
    )

    is_gemini_3 = is_gemini_3_google or is_gemini_3_openrouter
    media_resolution_visible = provider == "Google" and not is_gemini_3
    media_resolution_update = gr.update(visible=media_resolution_visible)

    # Gemini 3 and xAI specific dropdowns
    mr_visible, mr_choices, mr_info_base = get_media_resolution_config(
        provider, remembered_model
    )

    mr_bubbles_info = mr_info_base.replace("process images", "process bubble images")

    mr_update_kwargs = {
        "visible": mr_visible,
        "choices": mr_choices,
    }
    if not mr_visible:
        mr_update_kwargs["value"] = "auto"

    media_resolution_bubbles_update = gr.update(
        info=mr_bubbles_info, **mr_update_kwargs
    )

    mr_context_info = mr_info_base.replace(
        "process images", "process context (full page) images"
    )
    media_resolution_context_update = gr.update(
        info=mr_context_info, **mr_update_kwargs
    )

    # Moonshot AI: reasoning effort is visible for kimi-k2.5 models
    reasoning_effort_visible, reasoning_choices, reasoning_default_value = (
        get_reasoning_effort_config(provider, remembered_model)
    )

    reasoning_info_text = get_reasoning_effort_info_text(
        provider, remembered_model, reasoning_choices
    )

    if reasoning_choices and reasoning_default_value not in reasoning_choices:
        reasoning_default_value = reasoning_choices[0] if reasoning_choices else None
    elif not reasoning_choices:
        reasoning_default_value = None

    reasoning_effort_label = get_reasoning_effort_label(provider, remembered_model)

    reasoning_effort_visible_update = gr.update(
        visible=reasoning_effort_visible,
        choices=reasoning_choices,
        value=reasoning_default_value,
        label=reasoning_effort_label,
        info=reasoning_info_text,
    )

    # Effort dropdown (Claude Opus 4.5 only)
    effort_visible, effort_choices, effort_default_value = get_effort_config(
        provider, remembered_model
    )
    effort_update = gr.update(
        visible=effort_visible,
        choices=effort_choices,
        value=effort_default_value,
    )

    enable_code_execution_update = gr.update(visible=is_gemini_3_flash_google)

    return (
        google_visible_update,
        openai_visible_update,
        anthropic_visible_update,
        xai_visible_update,
        deepseek_visible_update,
        zai_visible_update,
        moonshot_visible_update,
        openrouter_visible_update,
        openai_compatible_url_visible_update,
        openai_compatible_key_visible_update,
        model_update,
        temp_update,
        top_p_update,
        top_k_update,
        max_tokens_update,
        enable_web_search_update,
        enable_code_execution_update,
        media_resolution_update,
        media_resolution_bubbles_update,
        media_resolution_context_update,
        reasoning_effort_visible_update,
        effort_update,
    )


def update_params_for_model(
    provider: str, model_name: Optional[str], current_temp: float
):
    """Adjusts temp/top_k sliders and visibility toggles based on selected provider/model."""
    if not provider:
        return gr.update(), gr.update()

    temp_max = 2.0
    top_k_interactive = True
    top_p_interactive = True

    if provider == "Anthropic":
        temp_max = 1.0
        top_k_interactive = False
        top_p_interactive = False
    elif provider in ("OpenAI", "xAI", "DeepSeek", "Moonshot AI"):
        top_k_interactive = False
    elif provider == "OpenRouter":
        is_openai_model = model_name and (
            "openai/" in model_name or model_name.startswith("gpt-")
        )
        is_anthropic_model = model_name and (
            "anthropic/" in model_name or model_name.startswith("claude-")
        )
        if is_anthropic_model:
            temp_max = 1.0
            top_p_interactive = False
        if is_openai_model or is_anthropic_model:
            top_k_interactive = False
    elif provider == "OpenAI-Compatible":
        pass

    new_temp_value = min(current_temp, temp_max)
    temp_update = gr.update(maximum=temp_max, value=new_temp_value)

    top_k_update = gr.update(interactive=top_k_interactive)
    top_p_update = gr.update(interactive=top_p_interactive)

    def _is_gemini_3_model(name: Optional[str]) -> bool:
        if not name:
            return False
        return "gemini-3" in name.lower()

    def _is_gemini_3_flash_model(name: Optional[str]) -> bool:
        if not name:
            return False
        lm = name.lower()
        return "gemini-3" in lm and "flash" in lm

    is_gemini_3_google = (
        provider == "Google" and model_name and _is_gemini_3_model(model_name)
    )
    is_gemini_3_flash_google = (
        provider == "Google" and model_name and _is_gemini_3_flash_model(model_name)
    )
    is_gemini_3_openrouter = (
        provider == "OpenRouter" and model_name and _is_gemini_3_model(model_name)
    )

    reasoning_visible, reasoning_choices, default_val = get_reasoning_effort_config(
        provider, model_name
    )

    if default_val in reasoning_choices:
        value = default_val
    elif reasoning_choices:
        value = reasoning_choices[0]
    else:
        value = None

    info_text = get_reasoning_effort_info_text(provider, model_name, reasoning_choices)
    reasoning_effort_label = get_reasoning_effort_label(provider, model_name)

    reasoning_effort_update = gr.update(
        visible=reasoning_visible,
        choices=reasoning_choices,
        value=value,
        label=reasoning_effort_label,
        info=info_text,
    )

    # Web search checkbox is visible for Google, OpenRouter, OpenAI, Anthropic, and xAI providers
    enable_web_search_visible = provider not in ("OpenAI-Compatible", "DeepSeek")

    enable_web_search_label, enable_web_search_info = (
        get_enable_web_search_label_and_info(provider)
    )

    enable_web_search_update = gr.update(
        visible=enable_web_search_visible,
        label=enable_web_search_label,
        info=enable_web_search_info,
    )

    is_gemini_3_google = (
        provider == "Google" and model_name and _is_gemini_3_model(model_name)
    )
    is_gemini_3_openrouter = (
        provider == "OpenRouter" and model_name and _is_gemini_3_model(model_name)
    )
    is_gemini_3 = is_gemini_3_google or is_gemini_3_openrouter
    media_resolution_visible = provider == "Google" and not is_gemini_3
    media_resolution_update = gr.update(visible=media_resolution_visible)

    # Gemini 3 and xAI specific dropdowns
    mr_visible, mr_choices, mr_info_base = get_media_resolution_config(
        provider, model_name
    )

    mr_bubbles_info = mr_info_base.replace("process images", "process bubble images")
    media_resolution_bubbles_update = gr.update(
        visible=mr_visible,
        choices=mr_choices,
        info=mr_bubbles_info,
    )

    mr_context_info = mr_info_base.replace(
        "process images", "process context (full page) images"
    )
    media_resolution_context_update = gr.update(
        visible=mr_visible,
        choices=mr_choices,
        info=mr_context_info,
    )

    is_reasoning = is_reasoning_model(provider, model_name)

    max_tokens_value = 16384 if is_reasoning else 4096
    max_tokens_cap = get_max_tokens_cap(provider, model_name)
    max_tokens_maximum = max_tokens_cap if max_tokens_cap is not None else 63488
    max_tokens_update = gr.update(value=max_tokens_value, maximum=max_tokens_maximum)

    # Effort dropdown (Claude Opus 4.5 only)
    effort_visible, effort_choices, effort_default_value = get_effort_config(
        provider, model_name
    )
    effort_update = gr.update(
        visible=effort_visible,
        choices=effort_choices,
        value=effort_default_value,
    )

    enable_code_execution_update = gr.update(visible=is_gemini_3_flash_google)

    return (
        temp_update,
        top_p_update,
        top_k_update,
        max_tokens_update,
        enable_web_search_update,
        enable_code_execution_update,
        media_resolution_update,
        media_resolution_bubbles_update,
        media_resolution_context_update,
        reasoning_effort_update,
        effort_update,
    )


def switch_settings_view(
    selected_group_index: int,
    setting_groups: List[gr.Group],
    nav_buttons: List[gr.Button],
):
    """Handles switching visibility of setting groups and styling nav buttons."""
    updates = []
    for i, _ in enumerate(setting_groups):
        updates.append(gr.update(visible=(i == selected_group_index)))
    base_class = "nav-button"
    selected_class = "nav-button-selected"
    for i, _ in enumerate(nav_buttons):
        new_classes = [base_class]
        if i == selected_group_index:
            new_classes.append(selected_class)
        updates.append(gr.update(elem_classes=new_classes))
    return updates


def fetch_and_update_openrouter_models(
    ocr_method: str = "LLM", current_model: Optional[str] = None
):
    """Fetches models from OpenRouter API and updates dropdown.

    Args:
        ocr_method: "LLM" for vision-capable models, "manga-ocr" for text-only models
    """
    global OPENROUTER_MODEL_CACHE
    verbose = get_saved_settings().get("verbose", False)

    # Check if we have cached raw response
    raw_models = OPENROUTER_MODEL_CACHE.get("raw_response")
    if raw_models is None:
        log_message("Fetching OpenRouter models from API...", verbose=verbose)
        try:
            response = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
            response.raise_for_status()
            data = response.json()
            raw_models = data.get("data", [])
            OPENROUTER_MODEL_CACHE["raw_response"] = raw_models
            log_message(
                f"Fetched {len(raw_models)} models from OpenRouter API", verbose=verbose
            )
        except requests.exceptions.RequestException as e:
            gr.Warning(f"Failed to fetch OpenRouter models: {e}")
            return gr.update(choices=[])
        except Exception as e:  # Catch other potential errors like JSON parsing
            gr.Warning(f"Unexpected error fetching OpenRouter models: {e}")
            return gr.update(choices=[])
    else:
        log_message(
            f"Using cached OpenRouter models (filtering for OCR method: {ocr_method})",
            verbose=verbose,
        )

    # Filter models in-memory based on OCR method
    filtered_models = []
    for model in raw_models:
        arch = model.get("architecture", {}) or {}
        input_modalities = arch.get("input_modalities", []) or []
        if not isinstance(input_modalities, list):
            input_modalities = []
        input_modalities_lc = [str(m).lower() for m in input_modalities]

        output_modalities = arch.get("output_modalities", []) or []
        if not isinstance(output_modalities, list):
            output_modalities = []
        output_modalities_lc = [str(m).lower() for m in output_modalities]

        if ocr_method == "manga-ocr":
            # For manga-ocr: require text input and text output
            if "text" in input_modalities_lc and "text" in output_modalities_lc:
                filtered_models.append(model["id"])
        else:
            # For LLM OCR: require vision capability (image input + text output)
            if "image" in input_modalities_lc and "text" in output_modalities_lc:
                filtered_models.append(model["id"])

    filtered_models.sort()

    log_message(
        f"Filtered to {len(filtered_models)} OpenRouter models (OCR method: {ocr_method})",
        verbose=verbose,
    )

    saved_settings = get_saved_settings()
    provider_models_dict = saved_settings.get(
        "provider_models", DEFAULT_SETTINGS["provider_models"]
    )
    remembered_or_model = provider_models_dict.get("OpenRouter")

    preferred_model = current_model if current_model in filtered_models else None
    if preferred_model is None and remembered_or_model in filtered_models:
        preferred_model = remembered_or_model

    selected_or_model = preferred_model or (
        filtered_models[0] if filtered_models else None
    )
    return gr.update(choices=filtered_models, value=selected_or_model)


def fetch_and_update_compatible_models(
    url: str,
    api_key: Optional[str],
    current_model: Optional[str] = None,
    force_refresh: bool = False,
):
    """Fetches models from a generic OpenAI-Compatible endpoint and updates dropdown.

    Args:
        url: Base URL of the OpenAI-Compatible endpoint.
        api_key: Optional API key for the endpoint.
        current_model: Currently selected model in the dropdown to preserve when available.
        force_refresh: If True, bypass cache and re-fetch models from the endpoint.
    """
    global COMPATIBLE_MODEL_CACHE
    verbose = get_saved_settings().get("verbose", False)
    if not url or not url.startswith(("http://", "https://")):
        gr.Warning(
            "Please enter a valid URL (starting with http:// or https://) for the OpenAI-Compatible endpoint."
        )
        return gr.update(choices=[], value=None)

    if (
        not force_refresh
        and COMPATIBLE_MODEL_CACHE.get("url") == url
        and COMPATIBLE_MODEL_CACHE.get("models") is not None
    ):
        log_message(f"Using cached models from {url}", verbose=verbose)
        cached_models = COMPATIBLE_MODEL_CACHE["models"]
        saved_settings = get_saved_settings()
        provider_models_dict = saved_settings.get(
            "provider_models", DEFAULT_SETTINGS["provider_models"]
        )
        remembered_comp_model = provider_models_dict.get("OpenAI-Compatible")
        selected_comp_model = (
            current_model
            if current_model in cached_models
            else (
                remembered_comp_model
                if remembered_comp_model in cached_models
                else (cached_models[0] if cached_models else None)
            )
        )
        return gr.update(choices=cached_models, value=selected_comp_model)

    log_message(f"Fetching models from {url}", verbose=verbose)
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        fetch_url = f"{url.rstrip('/')}/models"
        response = requests.get(fetch_url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        all_models_data = data.get("data", [])
        if not isinstance(all_models_data, list):
            # Some endpoints (like Ollama) return a list of models directly under a 'models' key
            if isinstance(data.get("models"), list):
                all_models_data = data["models"]
            else:
                raise ValidationError(
                    "Invalid response format: 'data' or 'models' key not found or not a list."
                )

        fetched_models = [
            model.get(
                "id", model.get("name")
            )  # Handle different key names ('id' or 'name')
            for model in all_models_data
            if isinstance(model, dict) and (model.get("id") or model.get("name"))
        ]
        fetched_models = [m for m in fetched_models if m]
        # Filter out embedding models (case-insensitive)
        fetched_models = [m for m in fetched_models if "embedding" not in m.lower()]
        fetched_models.sort()

        COMPATIBLE_MODEL_CACHE["url"] = url
        COMPATIBLE_MODEL_CACHE["models"] = fetched_models
        log_message(f"Fetched {len(fetched_models)} models from {url}", verbose=verbose)
        if not fetched_models:
            gr.Warning(
                f"No models found at {fetch_url}. Check the URL and API key (if required)."
            )

        saved_settings = get_saved_settings()
        provider_models_dict = saved_settings.get(
            "provider_models", DEFAULT_SETTINGS["provider_models"]
        )
        remembered_comp_model = provider_models_dict.get("OpenAI-Compatible")
        selected_comp_model = (
            current_model
            if current_model in fetched_models
            else (
                remembered_comp_model
                if remembered_comp_model in fetched_models
                else (fetched_models[0] if fetched_models else None)
            )
        )
        return gr.update(choices=fetched_models, value=selected_comp_model)

    except requests.exceptions.RequestException as e:
        gr.Error(f"Error fetching models from {url}: {e}")
        return gr.update(choices=[], value=None)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        error_detail = (
            "Check if the URL points to a valid OpenAI-Compatible '/v1' "
            "or '/api/tags' (Ollama) endpoint."
        )
        gr.Error(f"Error parsing response from {url}: {e}. {error_detail}")
        return gr.update(choices=[], value=None)
    except Exception as e:
        gr.Error(f"Unexpected error fetching models from {url}: {e}")
        return gr.update(choices=[], value=None)


def initial_dynamic_fetch(provider: str, url: str, key: Optional[str]):
    """Handle initial model fetching for dynamic providers on app load."""
    if provider == "OpenRouter":
        return fetch_and_update_openrouter_models()
    elif provider == "OpenAI-Compatible":
        return fetch_and_update_compatible_models(url, key)
    return gr.update()


def format_thinking_status(
    provider: str,
    model_name: Optional[str],
    reasoning_effort: Optional[str],
) -> str:
    """
    Format the thinking status string for success messages.

    Args:
        provider: The provider name (e.g., "Google", "Anthropic", "OpenRouter")
        model_name: The model name
        reasoning_effort: The reasoning effort level (e.g., "high", "medium", "low", "auto", "none")

    Returns:
        str: Formatted thinking status string (e.g., " (thinking: high)" or " (no thinking)")
    """
    if not model_name:
        return ""

    thinking_status_str = ""
    if provider == "Google" and model_name:
        if "gemini-3" in model_name.lower():
            effort = reasoning_effort or "high"
            thinking_status_str = f" (thinking: {effort})"
        elif "gemini-2.5-flash" in model_name:
            effort = reasoning_effort or "auto"
            if effort == "none":
                thinking_status_str = " (no thinking)"
            else:
                thinking_status_str = f" (thinking: {effort})"
    elif provider == "OpenRouter" and model_name and "gemini-3" in model_name.lower():
        effort = reasoning_effort or "high"
        thinking_status_str = f" (thinking: {effort})"
    elif provider == "Anthropic" and model_name:
        lm = model_name.lower()
        if (
            lm.startswith("claude-opus-4")
            or lm.startswith("claude-sonnet-4")
            or lm.startswith("claude-haiku-4-5")
        ):
            effort = reasoning_effort or "none"
            if effort == "none":
                thinking_status_str = " (no thinking)"
            else:
                thinking_status_str = f" (thinking: {effort})"

    return thinking_status_str
