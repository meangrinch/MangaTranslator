import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from utils.logging import log_message

CONFIG_FILE = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "MangaTranslator" / "config.json"

PROVIDER_MODELS: Dict[str, List[str]] = {
    "Google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview-09-2025",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ],
    "OpenAI": [
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
        "gpt-5-chat-latest",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "chatgpt-4o-latest",
        "o4-mini-2025-04-16",
        "o3-2025-04-16",
        "o1-2024-12-17",
        "gpt-4-turbo-2024-04-09",
        "o3-pro-2025-06-10",
        "o1-pro-2025-03-19",
    ],
    "Anthropic": [
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-latest",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ],
    "xAI": [
        "grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning",
        "grok-4-0709",
        "grok-2-vision-1212",
    ],
    "OpenRouter": [],
    "OpenAI-Compatible": [],
}

DEFAULT_SETTINGS = {
    "provider": "Google",
    "google_api_key": "",
    "openai_api_key": "",
    "anthropic_api_key": "",
    "xai_api_key": "",
    "openrouter_api_key": "",
    "openai_compatible_url": "http://localhost:11434/v1",
    "openai_compatible_api_key": "",
    "model_name": PROVIDER_MODELS["Google"][0] if PROVIDER_MODELS["Google"] else None,  # Default active model
    "provider_models": {
        "Google": PROVIDER_MODELS["Google"][0] if PROVIDER_MODELS["Google"] else None,
        "OpenAI": PROVIDER_MODELS["OpenAI"][0] if PROVIDER_MODELS["OpenAI"] else None,
        "Anthropic": PROVIDER_MODELS["Anthropic"][0] if PROVIDER_MODELS["Anthropic"] else None,
        "xAI": PROVIDER_MODELS["xAI"][0] if PROVIDER_MODELS["xAI"] else None,
        "OpenRouter": None,
        "OpenAI-Compatible": None,
    },
    "input_language": "Japanese",
    "output_language": "English",
    "reading_direction": "rtl",
    "translation_mode": "one-step",
    "confidence": 0.35,
    "use_sam2": True,
    "use_otsu_threshold": False,
    "thresholding_value": 190,
    "roi_shrink_px": 4,
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_font_size": 14,
    "min_font_size": 8,
    "line_spacing": 1.0,
    "use_subpixel_rendering": True,
    "font_hinting": "none",
    "use_ligatures": False,
    "hyphenate_before_scaling": True,
    "hyphen_penalty": 1000.0,
    "hyphenation_min_word_length": 8,
    "badness_exponent": 3.0,
    "padding_pixels": 8.0,
    "font_pack": None,
    "verbose": False,
    "jpeg_quality": 95,
    "png_compression": 6,
    "output_format": "auto",
    "cleaning_only": False,
    "test_mode": False,
    "enable_thinking": True,  # Gemini 2.5 Flash & Claude reasoning models
    "reasoning_effort": "medium",  # OpenAI reasoning models; gpt-5 also supports 'minimal'
    "send_full_page_context": True,
    "openrouter_reasoning_override": False,  # Forces max output tokens to 8192
}

DEFAULT_BATCH_SETTINGS = {
    "batch_input_language": "Japanese",
    "batch_output_language": "English",
    "batch_font_pack": None,
}


# Canonical save order for config.json (unknown keys appended alphabetically at the end)
CANONICAL_CONFIG_KEY_ORDER: List[str] = [
    # Provider and model selection
    "provider_models",
    "provider",
    "model_name",
    "google_api_key",
    "openai_api_key",
    "anthropic_api_key",
    "openrouter_api_key",
    "openai_compatible_url",
    "openai_compatible_api_key",

    # Translation behavior / LLM options
    "input_language",
    "output_language",
    "reading_direction",
    "translation_mode",
    "temperature",
    "top_p",
    "top_k",
    "send_full_page_context",
    "enable_thinking",
    "reasoning_effort",
    "openrouter_reasoning_override",

    # Rendering
    "font_pack",
    "max_font_size",
    "min_font_size",
    "line_spacing",
    "use_subpixel_rendering",
    "font_hinting",
    "use_ligatures",
    "hyphenate_before_scaling",
    "hyphen_penalty",
    "hyphenation_min_word_length",
    "badness_exponent",

    # Models / Detection
    "confidence",
    "use_sam2",

    # Cleaning
    "thresholding_value",
    "use_otsu_threshold",
    "roi_shrink_px",

    # Output
    "output_format",
    "jpeg_quality",
    "png_compression",

    # General
    "verbose",
    "cleaning_only",
    "test_mode",

    # Batch
    "batch_input_language",
    "batch_output_language",
    "batch_font_pack",
]


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
        known_keys.add("cleaning_only")
        known_keys.add("use_sam2")
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

        # Reorder keys according to canonical order, then append unknown keys alphabetically
        known_in_order = [k for k in CANONICAL_CONFIG_KEY_ORDER if k in config_to_write]
        unknown_keys = sorted([k for k in config_to_write.keys() if k not in CANONICAL_CONFIG_KEY_ORDER])
        ordered_keys = known_in_order + unknown_keys
        ordered_config: Dict[str, Any] = OrderedDict((k, config_to_write[k]) for k in ordered_keys)

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(ordered_config, f, indent=2)

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
            if "use_sam2" in saved_config:
                settings["use_sam2"] = bool(saved_config["use_sam2"])
            if "provider_models" not in settings:
                settings["provider_models"] = DEFAULT_SETTINGS["provider_models"].copy()
            elif not isinstance(settings["provider_models"], dict):
                log_message("Warning: 'provider_models' in config is not a dictionary. Resetting.", always_print=True)
                settings["provider_models"] = DEFAULT_SETTINGS["provider_models"].copy()

            # Back-compat: migrate older configs
            try:
                # 1) provider string: Gemini -> Google
                if saved_config.get("provider") == "Gemini":
                    settings["provider"] = "Google"

                # 2) API key: 'gemini_api_key' -> 'google_api_key'
                if ("google_api_key" in settings and not settings["google_api_key"]) and (
                    "gemini_api_key" in saved_config and saved_config.get("gemini_api_key")
                ):
                    settings["google_api_key"] = saved_config.get("gemini_api_key", "")

                # 3) provider_models: move key 'Gemini' -> 'Google' if present
                pm = settings.get("provider_models") or {}
                if isinstance(pm, dict) and "Gemini" in pm:
                    pm.setdefault("Google", pm.get("Gemini"))
                    try:
                        del pm["Gemini"]
                    except Exception:
                        pass
                    settings["provider_models"] = pm
            except Exception:
                pass

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
            "google_api_key",
            "openai_api_key",
            "anthropic_api_key",
            "openrouter_api_key",
            "openai_compatible_api_key",
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
