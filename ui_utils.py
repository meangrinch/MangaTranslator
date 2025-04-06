import requests
import json
from pathlib import Path
from typing import Optional, Any, List, Tuple

import gradio as gr
from PIL import Image

from config_utils import get_saved_settings, PROVIDER_MODELS, DEFAULT_SETTINGS

# --- Constants ---
ERROR_PREFIX = "❌ Error: "
SUCCESS_PREFIX = "✅ "

# Global caches for API models (Session-based)
OPENROUTER_MODEL_CACHE = {'models': None}
COMPATIBLE_MODEL_CACHE = {'url': None, 'models': None}

# --- Resource Discovery ---
def get_available_models(models_directory: Path) -> List[str]:
    """Get list of available YOLO models in the models directory"""
    model_files = list(models_directory.glob("*.pt"))
    models = [p.name for p in model_files]
    models.sort()
    return models

def get_available_font_packs(fonts_base_dir: Path) -> Tuple[List[str], Optional[str]]:
    """Get list of available font packs (subdirectories) in the fonts directory"""
    if not fonts_base_dir.exists():
        print(f"Warning: Font base directory not found at {fonts_base_dir}")
        return [], None
    font_dirs = [d.name for d in fonts_base_dir.iterdir() if d.is_dir()]
    font_dirs.sort()

    if font_dirs:
        default_font = font_dirs[0]
    else:
        default_font = None

    return font_dirs, default_font

# --- UI Input Validation ---
def validate_api_key(api_key: str, provider: str) -> tuple[bool, str]:
    """Validate API key format based on provider."""
    if not api_key and provider != "OpenAI-compatible":
        return False, f"{provider} API key is required"
    elif not api_key and provider == "OpenAI-compatible":
         return True, f"{provider} API key is optional and not provided." # Valid state

    if provider == "Gemini" and not (api_key.startswith('AI') and len(api_key) == 39):
        return False, "Invalid Gemini API key format (should start with 'AI' and be 39 chars)"
    if provider == "OpenAI" and not (api_key.startswith('sk-') and len(api_key) > 50):
         return False, "Invalid OpenAI API key format (should start with 'sk-')"
    if provider == "Anthropic" and not (api_key.startswith('sk-ant-') and len(api_key) > 100):
         return False, "Invalid Anthropic API key format (should start with 'sk-ant-')"
    if provider == "OpenRouter" and not (api_key.startswith('sk-or-') and len(api_key) > 50):
         return False, "Invalid OpenRouter API key format (should start with 'sk-or-')"
    # No specific format check for OpenAI-compatible keys

    return True, f"{provider} API key format looks valid"

def validate_image(image: Any) -> tuple[bool, str]:
    """Validate uploaded image (accepts path or PIL Image)"""
    if image is None:
        return False, "Please upload an image"

    try:
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            filepath = Path(image)
            if not filepath.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                 return False, "Unsupported image format. Please use JPEG, PNG or WEBP."
        elif isinstance(image, Image.Image):
             img = image # Already a PIL image
        else:
            # Assume filepath or PIL
             return False, f"Unexpected image type: {type(image)}"

        width, height = img.size
        if width < 600 or height < 600:
            return False, f"Image dimensions too small ({width}x{height}). Min recommended size is 600x600."

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
        return False, f"Font directory '{font_dir.name}' not found at {font_dir.resolve()}"
    if not font_dir.is_dir():
        return False, f"Path '{font_dir.name}' is not a directory."

    font_files = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
    if not font_files:
        return False, f"No font files (.ttf or .otf) found in '{font_dir.name}' directory"

    return True, f"Found {len(font_files)} font files in directory"

# --- UI Update Functions ---
def update_model_dropdown(models_dir: Path):
    """Update the YOLO model dropdown list"""
    try:
        models = get_available_models(models_dir)
        saved_settings = get_saved_settings()
        current_model = saved_settings.get('yolo_model')
        selected_model_val = current_model if current_model in models else (models[0] if models else None)
        return gr.update(choices=models, value=selected_model_val), f"{SUCCESS_PREFIX}Found {len(models)} models"
    except Exception as e:
        print(f"Error updating model dropdown: {e}")
        return gr.update(choices=[]), f"{ERROR_PREFIX}{str(e)}"

def update_font_dropdown(fonts_base_dir: Path):
    """Update the font pack dropdown list"""
    try:
        font_packs, _ = get_available_font_packs(fonts_base_dir)
        saved_settings = get_saved_settings()
        current_font = saved_settings.get('font_pack')
        selected_font_val = current_font if current_font in font_packs else (font_packs[0] if font_packs else None)

        current_batch_font = saved_settings.get('batch_font_pack')
        selected_batch_font_val = current_batch_font if current_batch_font in font_packs else (font_packs[0] if font_packs else None)

        return gr.update(choices=font_packs, value=selected_font_val), \
               gr.update(choices=font_packs, value=selected_batch_font_val), \
               f"{SUCCESS_PREFIX}Found {len(font_packs)} font packs"
    except Exception as e:
        print(f"Error updating font dropdown: {e}")
        return gr.update(choices=[]), gr.update(choices=[]), f"{ERROR_PREFIX}{str(e)}"

def refresh_models_and_fonts(models_dir: Path, fonts_base_dir: Path):
    """Update both model dropdown and font dropdown lists"""
    try:
        models = get_available_models(models_dir)
        saved_settings = get_saved_settings()
        current_model = saved_settings.get('yolo_model')
        selected_model_val = current_model if current_model in models else (models[0] if models else None)
        model_result = gr.update(choices=models, value=selected_model_val)

        font_packs, _ = get_available_font_packs(fonts_base_dir)
        current_font = saved_settings.get('font_pack')
        selected_font_val = current_font if current_font in font_packs else (font_packs[0] if font_packs else None)
        single_font_result = gr.update(choices=font_packs, value=selected_font_val)

        current_batch_font = saved_settings.get('batch_font_pack')
        selected_batch_font_val = current_batch_font if current_batch_font in font_packs else (font_packs[0] if font_packs else None)
        batch_font_result = gr.update(choices=font_packs, value=selected_batch_font_val)


        model_count = len(models)
        model_text = "1 model" if model_count == 1 else f"{model_count} models"

        font_count = len(font_packs)
        font_text = "1 font pack" if font_count == 1 else f"{font_count} font packs"

        if models and font_packs:
            gr.Info(f"Detected {model_text} and {font_text}")
        elif not models and not font_packs:
            gr.Warning("No models or font packs found")
        elif not models:
            gr.Warning(f"No models found. Found {font_text}")
        else: # Not fonts
            gr.Warning(f"No font packs found. Found {model_text}")

        return model_result, single_font_result, batch_font_result
    except Exception as e:
        gr.Error(f"Error refreshing resources: {str(e)}")
        # Fallback to current state if error occurs during refresh
        models = get_available_models(models_dir)
        saved_settings = get_saved_settings()
        current_model = saved_settings.get('yolo_model')
        selected_model_val = current_model if current_model in models else None

        font_packs, _ = get_available_font_packs(fonts_base_dir)
        current_font = saved_settings.get('font_pack')
        selected_font_val = current_font if current_font in font_packs else None
        current_batch_font = saved_settings.get('batch_font_pack')
        selected_batch_font_val = current_batch_font if current_batch_font in font_packs else None

        return gr.update(choices=models, value=selected_model_val), \
               gr.update(choices=font_packs, value=selected_font_val), \
               gr.update(choices=font_packs, value=selected_batch_font_val)

def update_translation_ui(provider: str, current_temp: float):
    """Updates API key/URL visibility, model dropdown, temp slider max, and top_k interactivity."""
    saved_settings = get_saved_settings()
    provider_models_dict = saved_settings.get('provider_models', DEFAULT_SETTINGS['provider_models'])
    remembered_model = provider_models_dict.get(provider)

    models = PROVIDER_MODELS.get(provider, [])
    selected_model = remembered_model if remembered_model in models else (models[0] if models else None)
    model_update = gr.update(choices=models, value=selected_model)

    gemini_visible_update = gr.update(visible=(provider == "Gemini"))
    openai_visible_update = gr.update(visible=(provider == "OpenAI"))
    anthropic_visible_update = gr.update(visible=(provider == "Anthropic"))
    openrouter_visible_update = gr.update(visible=(provider == "OpenRouter"))
    # Visibility updates for the new fields
    openai_compatible_url_visible_update = gr.update(visible=(provider == "OpenAI-compatible"))
    openai_compatible_key_visible_update = gr.update(visible=(provider == "OpenAI-compatible"))
    if provider == "OpenRouter" or provider == "OpenAI-compatible":
        model_update = gr.update(value=remembered_model, choices=[remembered_model] if remembered_model else [])
    else:
        model_update = gr.update(choices=models, value=selected_model)

    # Adjust temperature slider max and potentially clamp current value
    temp_max = 1.0 if provider == "Anthropic" else 2.0
    new_temp_value = min(current_temp, temp_max)
    temp_update = gr.update(maximum=temp_max, value=new_temp_value)

    # Adjust top_k interactivity
    top_k_interactive = provider != "OpenAI"
    top_k_update = gr.update(interactive=top_k_interactive)

    return (
        gemini_visible_update, openai_visible_update, anthropic_visible_update, openrouter_visible_update,
        openai_compatible_url_visible_update,
        openai_compatible_key_visible_update,
        model_update, temp_update, top_k_update
    )

def update_params_for_model(provider: str, model_name: Optional[str], current_temp: float):
    """Adjusts temp/top_k sliders based on selected provider/model."""
    if not provider: # Should not happen, but safeguard
        return gr.update(), gr.update()

    temp_max = 2.0
    top_k_interactive = True

    if provider == "Anthropic":
        temp_max = 1.0
    elif provider == "OpenAI":
        top_k_interactive = False
    elif provider == "OpenRouter":
        # Assume OpenAI/Anthropic models on OpenRouter have same limitations
        is_openai_model = model_name and ("openai/" in model_name or model_name.startswith("gpt-"))
        is_anthropic_model = model_name and ("anthropic/" in model_name or model_name.startswith("claude-"))
        if is_anthropic_model:
            temp_max = 1.0
        if is_openai_model or is_anthropic_model:
            top_k_interactive = False
    elif provider == "OpenAI-compatible":
         pass

    new_temp_value = min(current_temp, temp_max)
    temp_update = gr.update(maximum=temp_max, value=new_temp_value)

    top_k_update = gr.update(interactive=top_k_interactive)

    return temp_update, top_k_update

def switch_settings_view(selected_group_index: int, setting_groups: List[gr.Group], nav_buttons: List[gr.Button]):
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

# --- API Fetching for UI ---
def fetch_and_update_openrouter_models():
    """Fetches vision-capable models from OpenRouter API and updates dropdown."""
    global OPENROUTER_MODEL_CACHE
    verbose = get_saved_settings().get("verbose", False)
    # Check if cache is already populated for this session
    if OPENROUTER_MODEL_CACHE['models'] is not None:
        if verbose: print("Using session-cached OpenRouter models.")
        cached_models = OPENROUTER_MODEL_CACHE['models']
        saved_settings = get_saved_settings()
        provider_models_dict = saved_settings.get('provider_models', DEFAULT_SETTINGS['provider_models'])
        remembered_or_model = provider_models_dict.get("OpenRouter")
        selected_or_model = remembered_or_model if remembered_or_model in cached_models else (cached_models[0] if cached_models else None)
        return gr.update(choices=cached_models, value=selected_or_model)

    # If not cached, proceed with fetching
    if verbose: print("Fetching OpenRouter models...")
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        response.raise_for_status()
        data = response.json()
        all_models = data.get('data', [])

        vision_keywords = ["omni", "vision", "vl", "multimodal"]
        filtered_models = []
        for model in all_models:
            model_id = model.get('id', '').lower()
            description = model.get('description', '').lower()
            # Check if model supports image input (basic check)
            supports_image_input = model.get('architecture', {}).get('instruct_type') == 'multimodal' or \
                                   any(keyword in model_id for keyword in vision_keywords) or \
                                   any(keyword in description for keyword in vision_keywords)

            if supports_image_input:
                filtered_models.append(model['id'])

        filtered_models.sort()

        OPENROUTER_MODEL_CACHE['models'] = filtered_models
        if verbose: print(f"Fetched and filtered {len(filtered_models)} vision-capable OpenRouter models.")

        saved_settings = get_saved_settings()
        provider_models_dict = saved_settings.get('provider_models', DEFAULT_SETTINGS['provider_models'])
        remembered_or_model = provider_models_dict.get("OpenRouter")
        selected_or_model = remembered_or_model if remembered_or_model in filtered_models else (filtered_models[0] if filtered_models else None)
        return gr.update(choices=filtered_models, value=selected_or_model)

    except requests.exceptions.RequestException as e:
        if verbose: print(f"Error fetching OpenRouter models: {e}")
        gr.Warning(f"Failed to fetch OpenRouter models: {e}")
        return gr.update(choices=[]) # Return empty choices on error
    except Exception as e: # Catch other potential errors like JSON parsing
        if verbose: print(f"Unexpected error fetching OpenRouter models: {e}")
        gr.Warning(f"Unexpected error fetching OpenRouter models: {e}")
        return gr.update(choices=[])

def fetch_and_update_compatible_models(url: str, api_key: Optional[str]):
    """Fetches models from a generic OpenAI-compatible endpoint and updates dropdown."""
    global COMPATIBLE_MODEL_CACHE
    verbose = get_saved_settings().get("verbose", False)
    if not url or not url.startswith(("http://", "https://")):
        if verbose: print("Invalid or missing URL for OpenAI-Compatible endpoint.")
        gr.Warning("Please enter a valid URL (starting with http:// or https://) for the OpenAI-Compatible endpoint.")
        return gr.update(choices=[], value=None)

    # Check if cache is already populated for this URL in this session
    if COMPATIBLE_MODEL_CACHE.get('url') == url and COMPATIBLE_MODEL_CACHE.get('models') is not None:
        if verbose: print(f"Using session-cached OpenAI-Compatible models for URL: {url}")
        cached_models = COMPATIBLE_MODEL_CACHE['models']
        saved_settings = get_saved_settings()
        provider_models_dict = saved_settings.get('provider_models', DEFAULT_SETTINGS['provider_models'])
        remembered_comp_model = provider_models_dict.get("OpenAI-compatible")
        selected_comp_model = remembered_comp_model if remembered_comp_model in cached_models else (cached_models[0] if cached_models else None)
        return gr.update(choices=cached_models, value=selected_comp_model)

    # If not cached, proceed with fetching
    if verbose: print(f"Fetching OpenAI-Compatible models from URL: {url}")
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        fetch_url = f"{url.rstrip('/')}/models"
        response = requests.get(fetch_url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        all_models_data = data.get('data', [])
        if not isinstance(all_models_data, list):
             # Some endpoints (like Ollama) might just return a list of models directly under a 'models' key
             if isinstance(data.get('models'), list):
                 all_models_data = data['models']
             else:
                 raise ValueError("Invalid response format: 'data' or 'models' key not found or not a list.")

        fetched_models = [model.get('id', model.get('name')) # Handle different key names ('id' or 'name')
                          for model in all_models_data if isinstance(model, dict) and (model.get('id') or model.get('name'))]
        fetched_models = [m for m in fetched_models if m]
        fetched_models.sort()

        COMPATIBLE_MODEL_CACHE['url'] = url
        COMPATIBLE_MODEL_CACHE['models'] = fetched_models
        # Timestamp removed for session caching
        if verbose: print(f"Fetched {len(fetched_models)} OpenAI-Compatible models from {url}.")
        if not fetched_models:
             gr.Warning(f"No models found at {fetch_url}. Check the URL and API key (if required).")

        saved_settings = get_saved_settings()
        provider_models_dict = saved_settings.get('provider_models', DEFAULT_SETTINGS['provider_models'])
        remembered_comp_model = provider_models_dict.get("OpenAI-compatible")
        selected_comp_model = remembered_comp_model if remembered_comp_model in fetched_models else (fetched_models[0] if fetched_models else None)
        return gr.update(choices=fetched_models, value=selected_comp_model)

    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching OpenAI-Compatible models from {url}: {e}"
        if verbose: print(error_msg)
        gr.Error(error_msg)
        return gr.update(choices=[], value=None)
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        error_msg = f"Error parsing response from {url}: {e}. Check if the URL points to a valid OpenAI-compatible '/v1' or '/api/tags' (Ollama) endpoint."
        if verbose: print(error_msg)
        gr.Error(error_msg)
        return gr.update(choices=[], value=None)
    except Exception as e:
        error_msg = f"Unexpected error fetching/processing OpenAI-Compatible models: {e}"
        if verbose: print(error_msg)
        gr.Error(error_msg)
        return gr.update(choices=[], value=None)

def initial_dynamic_fetch(provider: str, url: str, key: Optional[str]):
    """Handle initial model fetching for dynamic providers on app load."""
    verbose_load = False # Default verbose to False for initial load

    if provider == "OpenRouter":
        if verbose_load: print("Initial load: OpenRouter selected, fetching models...")
        return fetch_and_update_openrouter_models()
    elif provider == "OpenAI-compatible":
         if verbose_load: print("Initial load: OpenAI-Compatible selected, fetching models...")
         return fetch_and_update_compatible_models(url, key)
    return gr.update() # No update needed for static providers
