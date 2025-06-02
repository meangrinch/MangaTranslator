import json
import os
from pathlib import Path
from typing import Dict, Any, List
from utils.logging import log_message

CONFIG_FILE = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "MangaTranslator" / "config.json"

PROVIDER_MODELS: Dict[str, List[str]] = {
    "Gemini": [
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.5-pro-exp-03-25",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ],
    "OpenAI": [
        "o4-mini",
        "o3",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "Anthropic": [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
    ],
    "OpenRouter": [],
    "OpenAI-Compatible": [],
}

DEFAULT_SETTINGS = {
    "provider": "Gemini",
    "gemini_api_key": "",
    "openai_api_key": "",
    "anthropic_api_key": "",
    "openrouter_api_key": "",
    "openai_compatible_url": "http://localhost:11434/v1",
    "openai_compatible_api_key": "",
    "model_name": PROVIDER_MODELS["Gemini"][0] if PROVIDER_MODELS["Gemini"] else None,  # Default active model
    "provider_models": {
        "Gemini": PROVIDER_MODELS["Gemini"][0] if PROVIDER_MODELS["Gemini"] else None,
        "OpenAI": PROVIDER_MODELS["OpenAI"][0] if PROVIDER_MODELS["OpenAI"] else None,
        "Anthropic": PROVIDER_MODELS["Anthropic"][0] if PROVIDER_MODELS["Anthropic"] else None,
        "OpenRouter": None,
        "OpenAI-Compatible": None,
    },
    "input_language": "Japanese",
    "output_language": "English",
    "reading_direction": "rtl",
    "translation_mode": "one-step",
    "confidence": 0.35,
    "dilation_kernel_size": 7,
    "dilation_iterations": 1,
    "use_otsu_threshold": False,
    "min_contour_area": 50,
    "closing_kernel_size": 7,
    "closing_iterations": 1,
    "constraint_erosion_kernel_size": 5,
    "constraint_erosion_iterations": 1,
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_font_size": 14,
    "min_font_size": 8,
    "line_spacing": 1.0,
    "use_subpixel_rendering": True,
    "font_hinting": "none",
    "use_ligatures": False,
    "font_pack": None,
    "verbose": False,
    "jpeg_quality": 95,
    "png_compression": 6,
    "output_format": "auto",
    "cleaning_only": False,
    "enable_thinking": True,  # Specific to Gemini 2.5 Flash
}

DEFAULT_BATCH_SETTINGS = {
    "batch_input_language": "Japanese",
    "batch_output_language": "English",
    "batch_font_pack": None,
}


def save_config(incoming_settings: Dict[str, Any]):
    """Save all settings to config file, updating provider_models and cleaning old keys."""
    try:
        current_config_on_disk = {}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    current_config_on_disk = json.load(f)
            except json.JSONDecodeError:
                log_message(
                    f"Warning: Could not decode existing config file at {CONFIG_FILE}. Overwriting with new settings.",
                    always_print=True,
                )
            except Exception as e:
                log_message(
                    f"Warning: Error reading config file {CONFIG_FILE}: {e}. Overwriting with new settings.",
                    always_print=True,
                )

        known_keys = set(DEFAULT_SETTINGS.keys()) | set(DEFAULT_BATCH_SETTINGS.keys())
        known_keys.add("provider_models")
        known_keys.add("yolo_model")
        known_keys.add("cleaning_only")
        all_defaults = {**DEFAULT_SETTINGS, **DEFAULT_BATCH_SETTINGS}
        known_keys.add("openai_compatible_url")
        known_keys.add("openai_compatible_api_key")
        known_keys.add("translation_mode")

        config_to_write = {}
        changed_setting_keys = []

        provider_models_to_save = current_config_on_disk.get(
            "provider_models", DEFAULT_SETTINGS["provider_models"].copy()
        )
        if not isinstance(provider_models_to_save, dict):  # Handle potential corruption
            provider_models_to_save = DEFAULT_SETTINGS["provider_models"].copy()

        selected_provider = incoming_settings.get("provider")
        selected_model = incoming_settings.get("model_name")
        if selected_provider and selected_model:
            provider_models_to_save[selected_provider] = selected_model

        old_provider_models = current_config_on_disk.get("provider_models", {})
        if old_provider_models != provider_models_to_save:
            changed_setting_keys.append("provider_models")
        config_to_write["provider_models"] = provider_models_to_save

        for key in known_keys:
            if key == "provider_models":
                continue

            incoming_value = incoming_settings.get(key)
            current_value_on_disk = current_config_on_disk.get(key)
            default_value = all_defaults.get(key)

            value_to_write = incoming_value if incoming_value is not None else default_value
            config_to_write[key] = value_to_write

            changed = False
            if key in current_config_on_disk:
                if current_value_on_disk != incoming_value:
                    changed = True
            elif incoming_value != default_value:
                changed = True

            if changed:
                changed_setting_keys.append(key)

        os.makedirs(CONFIG_FILE.parent, exist_ok=True)

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_to_write, f, indent=2)

        if changed_setting_keys:
            changed_setting_keys = [k for k in changed_setting_keys if k is not None]
            changed_setting_keys.sort()
            return f"Saved changes: {', '.join(changed_setting_keys)}"
        else:
            return "Saved changes: none"

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"Failed to save settings: {str(e)}"


def get_saved_settings() -> Dict[str, Any]:
    """Get all saved settings from config file, falling back to defaults."""
    settings = {}
    settings.update(DEFAULT_SETTINGS)
    settings.update(DEFAULT_BATCH_SETTINGS)

    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved_config = json.load(f)

            for key in settings.keys():
                if key in saved_config:
                    settings[key] = saved_config[key]

            # Special handling for potentially missing keys or nested structures from older configs
            if "yolo_model" in saved_config:
                settings["yolo_model"] = saved_config["yolo_model"]
            if "provider_models" not in settings:
                settings["provider_models"] = DEFAULT_SETTINGS["provider_models"].copy()
            elif not isinstance(settings["provider_models"], dict):
                log_message("Warning: 'provider_models' in config is not a dictionary. Resetting.", always_print=True)
                settings["provider_models"] = DEFAULT_SETTINGS["provider_models"].copy()

            loaded_provider = settings.get("provider", DEFAULT_SETTINGS["provider"])
            provider_models_dict = settings.get("provider_models", DEFAULT_SETTINGS["provider_models"])
            saved_model_for_provider = provider_models_dict.get(loaded_provider)

            if loaded_provider == "OpenRouter" or loaded_provider == "OpenAI-Compatible":
                settings["model_name"] = saved_model_for_provider
            else:
                valid_models = PROVIDER_MODELS.get(loaded_provider, [])
                if saved_model_for_provider and saved_model_for_provider in valid_models:
                    settings["model_name"] = saved_model_for_provider
                else:
                    default_model_for_provider = DEFAULT_SETTINGS["provider_models"].get(loaded_provider)
                    if default_model_for_provider and default_model_for_provider in valid_models:
                        settings["model_name"] = default_model_for_provider
                    elif valid_models:
                        settings["model_name"] = valid_models[0]
                    else:
                        settings["model_name"] = None
                    if (
                        saved_model_for_provider and saved_model_for_provider != settings["model_name"]
                    ):  # Only warn if there *was* a saved value that's now invalid/different
                        log_message(
                            f"Warning: Saved model '{saved_model_for_provider}' not valid "
                            f"or available for provider '{loaded_provider}'. "
                            f"Using '{settings['model_name']}'.",
                            always_print=True,
                        )

    except json.JSONDecodeError:
        log_message(f"Warning: Could not decode config file at {CONFIG_FILE}. Using defaults.", always_print=True)
        default_provider = DEFAULT_SETTINGS["provider"]
        if default_provider != "OpenRouter" and default_provider != "OpenAI-Compatible":
            valid_models = PROVIDER_MODELS.get(default_provider, [])
            settings["model_name"] = valid_models[0] if valid_models else None
        else:
            settings["model_name"] = None
    except Exception as e:
        log_message(f"Warning: Error reading config file {CONFIG_FILE}: {e}. Using defaults.", always_print=True)
        default_provider = DEFAULT_SETTINGS["provider"]
        if default_provider != "OpenRouter" and default_provider != "OpenAI-Compatible":
            valid_models = PROVIDER_MODELS.get(default_provider, [])
            settings["model_name"] = valid_models[0] if valid_models else None
        else:
            settings["model_name"] = None

    return settings


def reset_to_defaults() -> Dict[str, Any]:
    """Reset all settings to default values, preserving API keys and YOLO model if they exist."""
    settings = {}
    settings.update(DEFAULT_SETTINGS)
    settings.update(DEFAULT_BATCH_SETTINGS)

    # Preserve existing keys if they exist in the current saved config
    current_saved = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                current_saved = json.load(f)
        except Exception:
            pass

        preserved_keys = [
            "gemini_api_key",
            "openai_api_key",
            "anthropic_api_key",
            "openrouter_api_key",
            "openai_compatible_api_key",
            "yolo_model",
        ]
        for key in preserved_keys:
            if key in current_saved:
                settings[key] = current_saved[key]

    settings["provider_models"] = DEFAULT_SETTINGS["provider_models"].copy()
    default_provider = DEFAULT_SETTINGS["provider"]
    settings["provider"] = default_provider
    settings["model_name"] = settings["provider_models"].get(default_provider)
    settings["cleaning_only"] = DEFAULT_SETTINGS["cleaning_only"]
    settings["translation_mode"] = DEFAULT_SETTINGS["translation_mode"]

    return settings
