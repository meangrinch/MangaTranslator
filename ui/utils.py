import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import gradio as gr
import requests
from PIL import Image

from utils.logging import log_message

from .settings_manager import DEFAULT_SETTINGS, PROVIDER_MODELS, get_saved_settings

ERROR_PREFIX = "❌ Error: "
SUCCESS_PREFIX = "✅ "

# Global caches for API models (Session-based)
OPENROUTER_MODEL_CACHE = {"models": None}
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
    # No specific format check for OpenAI-Compatible keys

    return True, f"{provider} API key format looks valid"


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
            img = image  # Already a PIL image
        else:
            # Assume filepath or PIL
            return False, f"Unexpected image type: {type(image)}"

        width, height = img.size
        if width < 600 or height < 600:
            return (
                False,
                f"Image dimensions too small ({width}x{height}). Min recommended size is 600x600.",
            )

        return True, "Image is valid"
    except FileNotFoundError:
        return False, f"Invalid image path: {image}"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def validate_font_directory(font_dir: Path) -> tuple[bool, str]:
    """Validate that the font directory contains at least one font file"""
    if not font_dir:
        return False, "Font directory not specified"
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


def refresh_models_and_fonts(models_dir: Path, fonts_base_dir: Path):
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

        font_count = len(font_packs)
        font_text = "1 font pack" if font_count == 1 else f"{font_count} font packs"
        gr.Info(f"YOLO model auto-detected. Found {font_text}")

        return single_font_result, batch_font_result
    except Exception as e:
        gr.Error(f"Error refreshing resources: {str(e)}")
        font_packs, _ = get_available_font_packs(fonts_base_dir)
        return gr.update(choices=font_packs), gr.update(choices=font_packs)


def update_translation_ui(provider: str, current_temp: float):
    """Updates API key/URL visibility, model dropdown, temp slider max, and top_k interactivity."""
    saved_settings = get_saved_settings()
    provider_models_dict = saved_settings.get(
        "provider_models", DEFAULT_SETTINGS["provider_models"]
    )
    remembered_model = provider_models_dict.get(provider)

    models = PROVIDER_MODELS.get(provider, [])
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

    temp_max = 1.0 if provider == "Anthropic" else 2.0
    new_temp_value = min(current_temp, temp_max)
    temp_update = gr.update(maximum=temp_max, value=new_temp_value)

    top_k_interactive = provider != "OpenAI" and provider != "xAI"
    top_k_update = gr.update(interactive=top_k_interactive)

    enable_thinking_visible = (
        provider == "Google"
        or provider == "Anthropic"
        or provider == "xAI"
        or provider == "OpenRouter"
    )
    enable_thinking_update = gr.update(visible=enable_thinking_visible)

    reasoning_effort_visible = False
    if provider == "OpenAI":
        reasoning_effort_visible = True
    elif provider == "OpenRouter" and remembered_model:
        lm = remembered_model.lower()
        is_reasoning_capable = (
            "gpt-5" in lm or "o1" in lm or "o3" in lm or "o4-mini" in lm
        )
        reasoning_effort_visible = is_reasoning_capable
    reasoning_effort_visible_update = gr.update(visible=reasoning_effort_visible)

    # Visibility for OpenRouter override
    openrouter_override_visible_update = gr.update(visible=(provider == "OpenRouter"))

    return (
        google_visible_update,
        openai_visible_update,
        anthropic_visible_update,
        xai_visible_update,
        openrouter_visible_update,
        openai_compatible_url_visible_update,
        openai_compatible_key_visible_update,
        model_update,
        temp_update,
        top_k_update,
        enable_thinking_update,
        reasoning_effort_visible_update,
        openrouter_override_visible_update,
    )


def update_params_for_model(
    provider: str, model_name: Optional[str], current_temp: float
):
    """Adjusts temp/top_k sliders and visibility toggles based on selected provider/model."""
    if not provider:  # Should not happen, but safeguard
        return gr.update(), gr.update()

    temp_max = 2.0
    top_k_interactive = True

    if provider == "Anthropic":
        temp_max = 1.0
    elif provider == "OpenAI" or provider == "xAI":
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
        if is_openai_model or is_anthropic_model:
            top_k_interactive = False
    elif provider == "OpenAI-Compatible":
        pass

    new_temp_value = min(current_temp, temp_max)
    temp_update = gr.update(maximum=temp_max, value=new_temp_value)

    top_k_update = gr.update(interactive=top_k_interactive)

    # Determine visibility for the enable_thinking checkbox
    is_flash_model = (
        provider == "Google" and model_name and "gemini-2.5-flash" in model_name
    )
    # Anthropic reasoning-capable models for thinking toggle

    def _is_anthropic_reasoning_model(name: Optional[str]) -> bool:
        if not name:
            return False
        lower = name.lower()
        return (
            lower.startswith("claude-opus-4-1")
            or lower.startswith("claude-opus-4")
            or lower.startswith("claude-sonnet-4")
            or lower.startswith("claude-3-7-sonnet")
        )

    def _is_openrouter_thinking_model(name: Optional[str]) -> bool:
        if not name:
            return False
        lower = name.lower()

        is_gemini_reasoning = "gemini-2.5-flash" in lower
        is_anthropic_reasoning = (
            "claude-opus-4.1" in lower
            or "claude-opus-4" in lower
            or "claude-sonnet-4" in lower
        )
        is_grok_reasoning = "grok" in lower and "fast" in lower

        return is_gemini_reasoning or is_anthropic_reasoning or is_grok_reasoning

    is_anthropic_thinking_model = (
        provider == "Anthropic" and _is_anthropic_reasoning_model(model_name)
    )
    is_openrouter_thinking_model = (
        provider == "OpenRouter" and _is_openrouter_thinking_model(model_name)
    )
    enable_thinking_update = gr.update(
        visible=is_flash_model
        or is_anthropic_thinking_model
        or is_openrouter_thinking_model
    )

    # Determine reasoning-effort dropdown visibility and choices
    reasoning_visible = False
    reasoning_choices = ["low", "medium", "high"]

    def _is_openai_reasoning_model(name: Optional[str]) -> bool:
        if not name:
            return False
        lm = name.lower()
        is_reasoning_capable = (
            "gpt-5" in lm or "o1" in lm or "o3" in lm or "o4-mini" in lm
        )
        return is_reasoning_capable

    if provider == "OpenAI" and model_name:
        lm = model_name.lower()
        is_gpt5 = lm.startswith("gpt-5")
        is_reasoning_capable = (
            is_gpt5
            or lm.startswith("o1")
            or lm.startswith("o3")
            or lm.startswith("o4-mini")
        )
        reasoning_visible = is_reasoning_capable
        if is_gpt5:
            reasoning_choices = ["minimal", "low", "medium", "high"]
        else:
            reasoning_choices = ["low", "medium", "high"]
    elif (
        provider == "OpenRouter"
        and model_name
        and _is_openai_reasoning_model(model_name)
    ):
        lm = model_name.lower()
        reasoning_visible = True
        reasoning_choices = ["low", "medium", "high"]

    # Pick a safe value if saved value not allowed in current choices
    saved = get_saved_settings()
    current_val = saved.get("reasoning_effort", "medium")
    if current_val in reasoning_choices:
        value = current_val
    else:
        value = "low" if "low" in reasoning_choices else reasoning_choices[0]

    reasoning_effort_update = gr.update(
        visible=reasoning_visible, choices=reasoning_choices, value=value
    )

    return temp_update, top_k_update, enable_thinking_update, reasoning_effort_update


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


def fetch_and_update_openrouter_models():
    """Fetches vision-capable models from OpenRouter API and updates dropdown."""
    global OPENROUTER_MODEL_CACHE
    verbose = get_saved_settings().get("verbose", False)
    # Check if cache is already populated for this session
    if OPENROUTER_MODEL_CACHE["models"] is not None:
        log_message("Using cached OpenRouter models", verbose=verbose)
        cached_models = OPENROUTER_MODEL_CACHE["models"]
        saved_settings = get_saved_settings()
        provider_models_dict = saved_settings.get(
            "provider_models", DEFAULT_SETTINGS["provider_models"]
        )
        remembered_or_model = provider_models_dict.get("OpenRouter")
        selected_or_model = (
            remembered_or_model
            if remembered_or_model in cached_models
            else (cached_models[0] if cached_models else None)
        )
        return gr.update(choices=cached_models, value=selected_or_model)

    # If not cached, proceed with fetching
    log_message("Fetching OpenRouter models", verbose=verbose)
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        response.raise_for_status()
        data = response.json()
        all_models = data.get("data", [])

        # Vision keywords for the model id
        id_keywords = ["omni", "vision", "vl", "multimodal", "gemini", "claude"]
        # Vision keywords for description checks
        desc_keywords = ["omni", "vision", "vl", "multimodal"]
        filtered_models = []
        for model in all_models:
            model_id = model.get("id", "").lower()
            description = model.get("description", "").lower()
            arch = model.get("architecture", {}) or {}

            # Determine multimodality from architecture fields
            modality_field = str(arch.get("modality", "")).lower()
            input_modalities = arch.get("input_modalities", []) or []
            if not isinstance(input_modalities, list):
                input_modalities = []
            input_modalities_lc = [str(m).lower() for m in input_modalities]

            has_image_in_modality = (
                any(k in modality_field for k in ["image", "vision", "multimodal"])
                if modality_field
                else False
            )
            has_image_input = any(
                ("image" in m) or (m == "video") for m in input_modalities_lc
            )

            # Check if model supports image input
            supports_image_input = (
                arch.get("instruct_type") == "multimodal"
                or has_image_in_modality
                or has_image_input
                or any(keyword in model_id for keyword in id_keywords)
                or any(keyword in description for keyword in desc_keywords)
            )

            if supports_image_input:
                filtered_models.append(model["id"])

        filtered_models.sort()

        OPENROUTER_MODEL_CACHE["models"] = filtered_models
        log_message(
            f"Fetched {len(filtered_models)} OpenRouter models", verbose=verbose
        )

        saved_settings = get_saved_settings()
        provider_models_dict = saved_settings.get(
            "provider_models", DEFAULT_SETTINGS["provider_models"]
        )
        remembered_or_model = provider_models_dict.get("OpenRouter")
        selected_or_model = (
            remembered_or_model
            if remembered_or_model in filtered_models
            else (filtered_models[0] if filtered_models else None)
        )
        return gr.update(choices=filtered_models, value=selected_or_model)

    except requests.exceptions.RequestException as e:
        gr.Warning(f"Failed to fetch OpenRouter models: {e}")
        return gr.update(choices=[])  # Return empty choices on error
    except Exception as e:  # Catch other potential errors like JSON parsing
        gr.Warning(f"Unexpected error fetching OpenRouter models: {e}")
        return gr.update(choices=[])


def fetch_and_update_compatible_models(url: str, api_key: Optional[str]):
    """Fetches models from a generic OpenAI-Compatible endpoint and updates dropdown."""
    global COMPATIBLE_MODEL_CACHE
    verbose = get_saved_settings().get("verbose", False)
    if not url or not url.startswith(("http://", "https://")):
        gr.Warning(
            "Please enter a valid URL (starting with http:// or https://) for the OpenAI-Compatible endpoint."
        )
        return gr.update(choices=[], value=None)

    # Check if cache is already populated for this URL in this session
    if (
        COMPATIBLE_MODEL_CACHE.get("url") == url
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
            remembered_comp_model
            if remembered_comp_model in cached_models
            else (cached_models[0] if cached_models else None)
        )
        return gr.update(choices=cached_models, value=selected_comp_model)

    # If not cached, proceed with fetching
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
                raise ValueError(
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
            remembered_comp_model
            if remembered_comp_model in fetched_models
            else (fetched_models[0] if fetched_models else None)
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


def update_openrouter_override_visibility(provider: str):
    """Return a gr.update controlling the OpenRouter override checkbox visibility."""
    return gr.update(visible=(provider == "OpenRouter"))
