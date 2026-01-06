import functools
from pathlib import Path
from typing import Any

import gradio as gr

from . import callbacks, settings_manager, utils

_ALPHABETICAL_LANGUAGES = [
    "Afrikaans",
    "Albanian",
    "Arabic",
    "Belarusian",
    "Bengali",
    "Bosnian",
    "Bulgarian",
    "Chinese (Simplified)",
    "Chinese (Traditional)",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Persian (Farsi)",
    "French",
    "German",
    "Hindi",
    "Hungarian",
    "Icelandic",
    "Indonesian",
    "Irish",
    "Italian",
    "Japanese",
    "Korean",
    "Latvian",
    "Lithuanian",
    "Malay",
    "Maltese",
    "Marathi",
    "Nepali",
    "Norwegian",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Serbian (cyrillic)",
    "Serbian (latin)",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tamil",
    "Telugu",
    "Thai",
    "Tagalog",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Uzbek",
    "Vietnamese",
    "Welsh",
]

SOURCE_LANGUAGES = [
    "Japanese",
    "Korean",
    "Simplified Chinese",
    "Traditional Chinese",
] + [
    lang
    for lang in _ALPHABETICAL_LANGUAGES
    if lang
    not in ["Japanese", "Korean", "Chinese (Simplified)", "Chinese (Traditional)"]
]

TARGET_LANGUAGES = ["English"] + [
    lang for lang in _ALPHABETICAL_LANGUAGES if lang != "English"
]

js_credits = """
function() {
    const footer = document.querySelector('footer');
    if (footer) {
        // Check if credits already exist
        if (footer.parentNode.querySelector('.mangatl-credits')) {
            return;
        }
        const newContent = document.createElement('div');
        newContent.className = 'mangatl-credits'; // Add a class for identification
        newContent.innerHTML = 'made by <a href="https://github.com/meangrinch">grinnch</a> with ❤️'; // credits

        newContent.style.textAlign = 'center';
        newContent.style.paddingTop = '50px';
        newContent.style.color = 'lightgray';

        // Style the hyperlink
        const link = newContent.querySelector('a');
        if (link) {
            link.style.color = 'gray';
            link.style.textDecoration = 'underline';
        }

        footer.parentNode.insertBefore(newContent, footer);
    }
}
"""

js_status_fade = """
() => {
    // Find the specific config status element by its ID
    const statusElement = document.getElementById('config_status_message');  // Config status

    // Apply fade logic only to the config status element
    if (statusElement) {
        if (statusElement && statusElement.textContent.trim() !== "") {
            clearTimeout(statusElement.fadeTimer);
            clearTimeout(statusElement.resetTimer);

            statusElement.style.display = 'block';
            statusElement.style.transition = 'none';
            statusElement.style.opacity = '1';

            const fadeDelay = 3000;
            const fadeDuration = 1000;

            statusElement.fadeTimer = setTimeout(() => {
                statusElement.style.transition = `opacity ${fadeDuration}ms ease-out`;
                statusElement.style.opacity = '0';

                statusElement.resetTimer = setTimeout(() => {
                    statusElement.style.display = 'none';
                    statusElement.style.opacity = '1';
                    statusElement.style.transition = 'none';
                }, fadeDuration);

            }, fadeDelay);
        } else {
            // Ensure hidden if empty
            statusElement.style.display = 'none';
        }
    }
}
"""

js_refresh_button_reset = """
() => {
    setTimeout(() => {
        const refreshButton = document.querySelector('.config-refresh-button button');
         if (refreshButton) {
            refreshButton.textContent = 'Refresh Models / Fonts';
            refreshButton.disabled = false;
        }
    }, 100); // Small delay to ensure Gradio update cycle completes
}
"""

js_refresh_button_processing = """
() => {
    const refreshButton = document.querySelector('.config-refresh-button button');
    if (refreshButton) {
        refreshButton.textContent = 'Refreshing...';
        refreshButton.disabled = true;
    }
    return []; // Required for JS function input/output
}
"""


def create_layout(
    models_dir: Path, fonts_base_dir: Path, target_device: Any
) -> gr.Blocks:
    """Creates the Gradio UI layout and connects callbacks."""

    with gr.Blocks(
        title="MangaTranslator", js=js_credits, css_paths="style.css"
    ) as app:

        gr.Markdown("# MangaTranslator")

        font_choices, initial_default_font = utils.get_available_font_packs(
            fonts_base_dir
        )
        saved_settings = settings_manager.get_saved_settings()

        saved_font_pack = saved_settings.get("font_pack")
        default_font = (
            saved_font_pack
            if saved_font_pack in font_choices
            else (initial_default_font if initial_default_font else None)
        )
        batch_saved_font_pack = saved_settings.get("batch_font_pack")
        batch_default_font = (
            batch_saved_font_pack
            if batch_saved_font_pack in font_choices
            else (initial_default_font if initial_default_font else None)
        )

        saved_osb_font_pack = saved_settings.get("outside_text_osb_font_pack", "")
        if saved_osb_font_pack not in ([""] + font_choices):
            saved_osb_font_pack = ""

        initial_provider = saved_settings.get(
            "provider", settings_manager.DEFAULT_SETTINGS["provider"]
        )
        initial_model_name = saved_settings.get("model_name")

        if initial_provider == "OpenRouter" or initial_provider == "OpenAI-Compatible":
            initial_models_choices = [initial_model_name] if initial_model_name else []
        else:
            initial_models_choices = settings_manager.PROVIDER_MODELS.get(
                initial_provider, []
            )

        saved_max_tokens = saved_settings.get("max_tokens")
        if saved_max_tokens is not None:
            initial_max_tokens = saved_max_tokens
        else:
            is_reasoning = utils.is_reasoning_model(
                initial_provider, initial_model_name
            )
            initial_max_tokens = 16384 if is_reasoning else 4096

        # Calculate initial max_tokens maximum based on provider/model
        initial_max_tokens_cap = utils.get_max_tokens_cap(
            initial_provider, initial_model_name
        )
        initial_max_tokens_maximum = (
            initial_max_tokens_cap if initial_max_tokens_cap is not None else 63488
        )

        # --- Define UI Components ---
        with gr.Tabs():
            with gr.TabItem("Translator"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="filepath",
                            label="Upload Image",
                            show_download_button=False,
                            image_mode=None,
                            elem_id="translator_input_image",
                        )
                        font_dropdown = gr.Dropdown(
                            choices=font_choices,
                            label="Text Font",
                            value=default_font,
                            filterable=False,
                        )
                        with gr.Accordion("Translation Settings", open=True):
                            # Hidden state to store original language selection before manga-ocr forces Japanese
                            original_language_state = gr.State(
                                value=saved_settings.get("input_language", "Japanese")
                            )
                            input_language = gr.Dropdown(
                                SOURCE_LANGUAGES,
                                label="Source Language",
                                value=saved_settings.get("input_language", "Japanese"),
                                allow_custom_value=True,
                            )
                            output_language = gr.Dropdown(
                                TARGET_LANGUAGES,
                                label="Target Language",
                                value=saved_settings.get("output_language", "English"),
                                allow_custom_value=True,
                            )
                        special_instructions = gr.Textbox(
                            label="Special Instructions",
                            placeholder="Give the LLM optional context, formatting instructions, etc.",
                            value=saved_settings.get("special_instructions", ""),
                            lines=1,
                            max_lines=10,
                        )
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            type="pil",
                            label="Translated Image",
                            interactive=False,
                            elem_id="translator_output_image",
                        )
                        # Assign specific ID for JS targeting
                        status_message = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_id="translator_status_message",
                        )
                        with gr.Row():
                            translate_button = gr.Button("Translate", variant="primary")
                            clear_button = gr.Button("Clear")
                            cancel_button = gr.Button(
                                "Cancel", variant="stop", visible=False
                            )

            with gr.TabItem("Batch"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_files = gr.File(
                            label="Upload Images or Folder",
                            file_count="directory",
                            file_types=["image"],
                            type="filepath",
                        )
                        input_zip = gr.File(
                            label="Upload ZIP Archive (preserves directory structure)",
                            file_count="single",
                            file_types=[".zip"],
                            type="filepath",
                        )
                        batch_font_dropdown = gr.Dropdown(
                            choices=font_choices,
                            label="Text Font",
                            value=batch_default_font,
                            filterable=False,
                        )
                        with gr.Accordion("Translation Settings", open=True):
                            # Hidden state to store original language selection before manga-ocr forces Japanese
                            batch_original_language_state = gr.State(
                                value=saved_settings.get(
                                    "batch_input_language", "Japanese"
                                )
                            )
                            batch_input_language = gr.Dropdown(
                                SOURCE_LANGUAGES,
                                label="Source Language",
                                value=saved_settings.get(
                                    "batch_input_language", "Japanese"
                                ),
                                allow_custom_value=True,
                            )
                            batch_output_language = gr.Dropdown(
                                TARGET_LANGUAGES,
                                label="Target Language",
                                value=saved_settings.get(
                                    "batch_output_language", "English"
                                ),
                                allow_custom_value=True,
                            )
                        batch_special_instructions = gr.Textbox(
                            label="Special Instructions",
                            placeholder="Give the LLM optional context, formatting instructions, etc.",
                            value=saved_settings.get("batch_special_instructions", ""),
                            lines=1,
                            max_lines=10,
                        )
                    with gr.Column(scale=1):
                        batch_output_gallery = gr.Gallery(
                            label="Translated Images",
                            show_label=True,
                            columns=4,
                            rows=2,
                            height="auto",
                            object_fit="contain",
                        )
                        # Assign specific ID for JS targeting
                        batch_status_message = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_id="batch_status_message",
                        )
                        with gr.Row():
                            batch_process_button = gr.Button(
                                "Start Batch Translating", variant="primary"
                            )
                            batch_clear_button = gr.Button("Clear")
                            batch_cancel_button = gr.Button(
                                "Cancel", variant="stop", visible=False
                            )

            with gr.TabItem("Config", elem_id="settings-tab-container"):
                config_initial_provider = initial_provider
                config_initial_model_name = initial_model_name
                config_initial_models_choices = initial_models_choices

                with gr.Row(elem_id="config-button-row"):
                    save_config_btn = gr.Button(
                        "Save Config", variant="primary", scale=3
                    )
                    reset_defaults_btn = gr.Button(
                        "Reset Defaults", variant="secondary", scale=1
                    )

                # Assign specific ID for JS targeting
                config_status = gr.Markdown(elem_id="config_status_message")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, elem_id="settings-nav"):
                        nav_buttons = []
                        setting_groups = []
                        nav_button_detection = gr.Button(
                            "Detection",
                            elem_classes=["nav-button", "nav-button-selected"],
                        )
                        nav_buttons.append(nav_button_detection)
                        nav_button_cleaning = gr.Button(
                            "Cleaning", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_cleaning)
                        nav_button_translation = gr.Button(
                            "Translation", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_translation)
                        nav_button_rendering = gr.Button(
                            "Rendering", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_rendering)
                        nav_button_outside_text = gr.Button(
                            "OSB Text", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_outside_text)
                        nav_button_output = gr.Button(
                            "Output", elem_classes="nav-button"
                        )
                        nav_buttons.append(nav_button_output)
                        nav_button_other = gr.Button("Other", elem_classes="nav-button")
                        nav_buttons.append(nav_button_other)

                    with gr.Column(scale=4, elem_id="config-content-area"):
                        # --- Detection Settings ---
                        with gr.Group(
                            visible=True, elem_classes="settings-group"
                        ) as group_detection:
                            gr.Markdown("### Speech Bubble Detection")
                            confidence = gr.Slider(
                                0.1,
                                1.0,
                                value=saved_settings.get("confidence", 0.6),
                                step=0.05,
                                label="Bubble Confidence Threshold",
                                info="Lower values detect more bubbles, but potentially include false positives.",
                            )
                            conjoined_detection_checkbox = gr.Checkbox(
                                value=saved_settings.get("conjoined_detection", True),
                                label="Enable Conjoined Bubble Detection",
                                info=(
                                    "Uses a secondary YOLO model to detect and split "
                                    "conjoined speech bubbles into separate bubbles."
                                ),
                            )
                            conjoined_confidence = gr.Slider(
                                0.1,
                                1.0,
                                value=saved_settings.get("conjoined_confidence", 0.35),
                                step=0.05,
                                label="Conjoined Bubble Confidence Threshold",
                                info="Increase to filter out false positives, but may miss some conjoined bubbles.",
                                interactive=saved_settings.get(
                                    "conjoined_detection", True
                                ),
                            )
                            use_panel_sorting_checkbox = gr.Checkbox(
                                value=saved_settings.get("use_panel_sorting", True),
                                label="Use Panel-aware Sorting",
                                info=(
                                    "Use a panel detection YOLO model to group and sort speech bubbles "
                                    "within each panel for better reading order accuracy."
                                ),
                            )
                            panel_confidence = gr.Slider(
                                0.05,
                                1.0,
                                value=saved_settings.get("panel_confidence", 0.25),
                                step=0.05,
                                label="Panel Confidence Threshold",
                                info="Increase to filter out false positives, but may miss some panels.",
                                interactive=saved_settings.get(
                                    "use_panel_sorting", True
                                ),
                            )
                            use_sam2_checkbox = gr.Checkbox(
                                value=saved_settings.get("use_sam2", True),
                                label="Use SAM 2.1 for Segmentation",
                                info=(
                                    "Greatly enhances bubble segmentation quality, especially for oddly shaped bubbles."
                                    " Uses YOLO segmentation if disabled."
                                ),
                            )
                            osb_text_verification_checkbox = gr.Checkbox(
                                value=saved_settings.get(
                                    "use_osb_text_verification", True
                                ),
                                label="Use OSB Text model for Bubble Verification",
                                info=(
                                    "Use the OSB text YOLO model to confirm bubble detections fully cover text. "
                                    "Requires a Hugging Face token."
                                    "(hf_token is shared with the 'OSB Text' section)."
                                ),
                            )
                            config_reading_direction = gr.Radio(
                                choices=["rtl", "ltr"],
                                label="Reading Direction",
                                value=saved_settings.get("reading_direction", "rtl"),
                                info="Order for sorting bubbles (rtl=Manga, ltr=Comic).",
                                elem_id="config_reading_direction",
                            )
                        setting_groups.append(group_detection)

                        # --- Cleaning Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_cleaning:
                            gr.Markdown("### Mask Cleaning & Refinement")
                            thresholding_value = gr.Slider(
                                0,
                                255,
                                value=saved_settings.get("thresholding_value", 210),
                                step=1,
                                label="Fixed Threshold Value",
                                info="Brightness threshold for text detection. Lower helps clean edge-hugging text.",
                                interactive=not saved_settings.get(
                                    "use_otsu_threshold", False
                                ),
                            )
                            use_otsu_threshold = gr.Checkbox(
                                value=saved_settings.get("use_otsu_threshold", False),
                                label="Force Automatic Thresholding (Otsu)",
                                info=(
                                    "Force Otsu's method for thresholding instead of the fixed value (on all bubbles). "
                                    "Recommended for varied lighting. Used as fallback when the fixed "
                                    "value fails, regardless of set value."
                                ),
                            )
                            roi_shrink_px = gr.Slider(
                                0,
                                8,
                                value=saved_settings.get("roi_shrink_px", 4),
                                step=1,
                                label="Shrink Threshold ROI (px)",
                                info=(
                                    "Shrink the threshold ROI inward by N pixels before fill. "
                                    "Lower helps clean edge-hugging text; higher preserves outlines."
                                ),
                            )
                            inpaint_colored_bubbles = gr.Checkbox(
                                value=saved_settings.get(
                                    "inpaint_colored_bubbles", True
                                ),
                                label="Use Flux Kontext to Inpaint Colored Bubbles",
                                info=(
                                    "Use Flux Kontext for bubble cleaning when the interior is not pure white/black "
                                    "(e.g., colored/complex). Requires a Hugging Face token "
                                    "(hf_token and Flux settings are shared with the 'OSB Text' section)."
                                ),
                            )
                        setting_groups.append(group_cleaning)

                        # --- Translation Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_translation:
                            gr.Markdown("### OCR & Translation")
                            config_translation_mode = gr.Radio(
                                choices=["one-step", "two-step"],
                                label="Translation Mode",
                                value=saved_settings.get(
                                    "translation_mode",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "translation_mode"
                                    ],
                                ),
                                info=(
                                    "Determines whether to perform OCR and translation together or separately. "
                                    "'two-step' might improve translation quality for less-capable LLMs."
                                ),
                                elem_id="config_translation_mode",
                            )
                            initial_ocr_method = saved_settings.get(
                                "ocr_method",
                                settings_manager.DEFAULT_SETTINGS.get(
                                    "ocr_method", "LLM"
                                ),
                            )
                            ocr_method_radio = gr.Radio(
                                choices=["LLM", "manga-ocr"],
                                label="OCR Method",
                                value=initial_ocr_method,
                                info=(
                                    "Determines whether to use a vision-capable LLM or a local OCR model for OCR. "
                                    "'manga-ocr' only supports Japanese, enables text-only LLMs for translation, "
                                    "and must be used in 'two-step' translation mode."
                                ),
                                elem_id="ocr_method_radio",
                                interactive=saved_settings.get(
                                    "translation_mode",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "translation_mode"
                                    ],
                                )
                                != "one-step",
                            )

                            gr.Markdown("### LLM Settings")
                            available_providers = utils.get_available_providers(
                                initial_ocr_method
                            )
                            initial_provider_value = (
                                config_initial_provider
                                if config_initial_provider in available_providers
                                else (
                                    available_providers[0]
                                    if available_providers
                                    else "Google"
                                )
                            )
                            if initial_provider_value != config_initial_provider:
                                config_initial_provider = initial_provider_value
                            provider_selector = gr.Radio(
                                choices=available_providers,
                                label="Translation Provider",
                                value=initial_provider_value,
                                elem_id="provider_selector",
                            )
                            google_api_key = gr.Textbox(
                                label="Google AI Studio API Key",
                                placeholder="Enter Google AI Studio API key (starts with AI...)",
                                type="password",
                                value=saved_settings.get("google_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Google"),
                                elem_id="google_api_key",
                                info="Stored locally. Or set via GOOGLE_API_KEY env var.",
                            )
                            openai_api_key = gr.Textbox(
                                label="OpenAI API Key",
                                placeholder="Enter OpenAI API key (starts with sk-...)",
                                type="password",
                                value=saved_settings.get("openai_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenAI"),
                                elem_id="openai_api_key",
                                info="Stored locally. Or set via OPENAI_API_KEY env var.",
                            )
                            anthropic_api_key = gr.Textbox(
                                label="Anthropic API Key",
                                placeholder="Enter Anthropic API key (starts with sk-ant-...)",
                                type="password",
                                value=saved_settings.get("anthropic_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Anthropic"),
                                elem_id="anthropic_api_key",
                                info="Stored locally. Or set via ANTHROPIC_API_KEY env var.",
                            )
                            xai_api_key = gr.Textbox(
                                label="xAI API Key",
                                placeholder="Enter xAI API key (starts with xai-...)",
                                type="password",
                                value=saved_settings.get("xai_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "xAI"),
                                elem_id="xai_api_key",
                                info="Stored locally. Or set via XAI_API_KEY env var.",
                            )
                            deepseek_api_key = gr.Textbox(
                                label="DeepSeek API Key",
                                placeholder="Enter DeepSeek API key (starts with sk-...)",
                                type="password",
                                value=saved_settings.get("deepseek_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "DeepSeek"),
                                elem_id="deepseek_api_key",
                                info="Stored locally. Or set via DEEPSEEK_API_KEY env var.",
                            )
                            zai_api_key = gr.Textbox(
                                label="Z.ai API Key",
                                placeholder="Enter Z.ai API key",
                                type="password",
                                value=saved_settings.get("zai_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Z.ai"),
                                elem_id="zai_api_key",
                                info="Stored locally. Or set via ZAI_API_KEY env var.",
                            )
                            moonshot_api_key = gr.Textbox(
                                label="Moonshot API Key",
                                placeholder="Enter Moonshot API key (starts with sk-...)",
                                type="password",
                                value=saved_settings.get("moonshot_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Moonshot AI"),
                                elem_id="moonshot_api_key",
                                info="Stored locally. Or set via MOONSHOT_API_KEY env var.",
                            )
                            openrouter_api_key = gr.Textbox(
                                label="OpenRouter API Key",
                                placeholder="Enter OpenRouter API key (starts with sk-or-...)",
                                type="password",
                                value=saved_settings.get("openrouter_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenRouter"),
                                elem_id="openrouter_api_key",
                                info="Stored locally. Or set via OPENROUTER_API_KEY env var.",
                            )
                            openai_compatible_url_input = gr.Textbox(
                                label="OpenAI-Compatible URL",
                                placeholder="Enter Base URL (e.g., http://localhost:1234/v1)",
                                type="text",
                                value=saved_settings.get(
                                    "openai_compatible_url",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "openai_compatible_url"
                                    ],
                                ),
                                show_copy_button=False,
                                visible=(
                                    config_initial_provider == "OpenAI-Compatible"
                                ),
                                elem_id="openai_compatible_url_input",
                                info="Base URL of your OpenAI-Compatible API endpoint.",
                            )
                            openai_compatible_api_key_input = gr.Textbox(
                                label="OpenAI-Compatible API Key (Optional)",
                                placeholder="Enter API key if required",
                                type="password",
                                value=saved_settings.get(
                                    "openai_compatible_api_key", ""
                                ),
                                show_copy_button=False,
                                visible=(
                                    config_initial_provider == "OpenAI-Compatible"
                                ),
                                elem_id="openai_compatible_api_key_input",
                                info="Stored locally. Or set via OPENAI_COMPATIBLE_API_KEY env var.",
                            )
                            config_model_name = gr.Dropdown(
                                choices=config_initial_models_choices,
                                label="Model",
                                value=config_initial_model_name,
                                info="Select the specific model for the chosen provider.",
                                elem_id="config_model_name",
                                allow_custom_value=True,
                            )
                            (
                                _initial_reasoning_effort_visible,
                                _initial_reasoning_effort_choices,
                                _initial_reasoning_effort_default,
                            ) = utils.get_reasoning_effort_config(
                                config_initial_provider, config_initial_model_name
                            )

                            _initial_reasoning_effort_value = saved_settings.get(
                                "reasoning_effort"
                            )
                            if _initial_reasoning_effort_value is None:
                                _initial_reasoning_effort_value = (
                                    _initial_reasoning_effort_default
                                )
                            elif (
                                _initial_reasoning_effort_choices
                                and _initial_reasoning_effort_value
                                not in _initial_reasoning_effort_choices
                            ):
                                _initial_reasoning_effort_value = (
                                    _initial_reasoning_effort_default
                                )
                            elif not _initial_reasoning_effort_choices:
                                _initial_reasoning_effort_value = None

                            _initial_reasoning_effort_info = (
                                utils.get_reasoning_effort_info_text(
                                    config_initial_provider,
                                    config_initial_model_name,
                                    _initial_reasoning_effort_choices,
                                )
                            )

                            _initial_reasoning_effort_label = (
                                utils.get_reasoning_effort_label(
                                    config_initial_provider,
                                    config_initial_model_name,
                                )
                            )

                            reasoning_effort_dropdown = gr.Radio(
                                choices=_initial_reasoning_effort_choices,
                                label=_initial_reasoning_effort_label,
                                value=_initial_reasoning_effort_value,
                                info=_initial_reasoning_effort_info,
                                visible=_initial_reasoning_effort_visible,
                                elem_id="reasoning_effort_dropdown",
                            )

                            # Effort dropdown (Claude Opus 4.5 only)
                            (
                                _initial_effort_visible,
                                _initial_effort_choices,
                                _initial_effort_default,
                            ) = utils.get_effort_config(
                                config_initial_provider, config_initial_model_name
                            )
                            _initial_effort_value = saved_settings.get("effort")
                            if _initial_effort_value is None:
                                _initial_effort_value = _initial_effort_default
                            elif (
                                _initial_effort_choices
                                and _initial_effort_value not in _initial_effort_choices
                            ):
                                _initial_effort_value = _initial_effort_default
                            elif not _initial_effort_choices:
                                _initial_effort_value = None

                            effort_dropdown = gr.Radio(
                                choices=_initial_effort_choices,
                                label="Effort",
                                value=_initial_effort_value,
                                info="Controls token spending eagerness. Claude Opus 4.5 only.",
                                visible=_initial_effort_visible,
                                elem_id="effort_dropdown",
                            )

                            _initial_enable_web_search_visible = (
                                config_initial_provider
                                not in ("OpenAI-Compatible", "DeepSeek")
                            )
                            (
                                _initial_enable_web_search_label,
                                _initial_enable_web_search_info,
                            ) = utils.get_enable_web_search_label_and_info(
                                config_initial_provider
                                if _initial_enable_web_search_visible
                                else "Google"
                            )

                            enable_web_search_checkbox = gr.Checkbox(
                                label=_initial_enable_web_search_label,
                                value=saved_settings.get("enable_web_search", False),
                                info=_initial_enable_web_search_info,
                                visible=_initial_enable_web_search_visible,
                                elem_id="enable_web_search_checkbox",
                            )

                            # Compute initial visibility for media_resolution (Google provider only, but NOT Gemini 3)
                            _initial_media_resolution_visible = False
                            try:
                                if config_initial_provider == "Google":
                                    if (
                                        config_initial_model_name
                                        and "gemini-3"
                                        in config_initial_model_name.lower()
                                    ):
                                        _initial_media_resolution_visible = False
                                    else:
                                        _initial_media_resolution_visible = True
                            except Exception:
                                _initial_media_resolution_visible = False
                            initial_media_resolution_value = saved_settings.get(
                                "media_resolution", "auto"
                            )

                            media_resolution_dropdown = gr.Radio(
                                label="Media Resolution",
                                choices=["auto", "high", "medium", "low"],
                                value=initial_media_resolution_value,
                                info="Resolution for Gemini to process bubble/context images.",
                                visible=_initial_media_resolution_visible,
                                elem_id="media_resolution_dropdown",
                            )

                            # Compute initial visibility for Gemini 3 specific media resolution options
                            _initial_media_resolution_bubbles_visible = False
                            _initial_media_resolution_context_visible = False
                            try:
                                if (
                                    config_initial_provider == "Google"
                                    and config_initial_model_name
                                ):
                                    if "gemini-3" in config_initial_model_name.lower():
                                        _initial_media_resolution_bubbles_visible = True
                                        _initial_media_resolution_context_visible = True
                            except Exception:
                                pass
                            initial_media_resolution_bubbles_value = saved_settings.get(
                                "media_resolution_bubbles", "auto"
                            )
                            initial_media_resolution_context_value = saved_settings.get(
                                "media_resolution_context", "auto"
                            )

                            media_resolution_bubbles_dropdown = gr.Radio(
                                label="Media Resolution (Bubbles)",
                                choices=["auto", "high", "medium", "low"],
                                value=initial_media_resolution_bubbles_value,
                                info="Resolution for Gemini 3 to process bubble images.",
                                visible=_initial_media_resolution_bubbles_visible,
                                elem_id="media_resolution_bubbles_dropdown",
                            )

                            media_resolution_context_dropdown = gr.Radio(
                                label="Media Resolution (Context)",
                                choices=["auto", "high", "medium", "low"],
                                value=initial_media_resolution_context_value,
                                info="Resolution for Gemini 3 to process context (full page) images.",
                                visible=_initial_media_resolution_context_visible,
                                elem_id="media_resolution_context_dropdown",
                            )

                            temperature = gr.Slider(
                                0,
                                2.0,
                                value=saved_settings.get("temperature", 0.1),
                                step=0.05,
                                label="Temperature",
                                info="Controls creativity. Lower = deterministic; higher = random.",
                                elem_id="config_temperature",
                            )
                            top_p = gr.Slider(
                                0,
                                1,
                                value=saved_settings.get("top_p", 0.95),
                                step=0.05,
                                label="Top P",
                                info="Controls diversity. Lower = focused; higher = random.",
                                elem_id="config_top_p",
                            )
                            top_k = gr.Slider(
                                0,
                                64,
                                value=saved_settings.get("top_k", 64),
                                step=1,
                                label="Top K",
                                info="Limits sampling pool to top K tokens.",
                                interactive=(config_initial_provider != "OpenAI"),
                                elem_id="config_top_k",
                            )
                            max_tokens = gr.Slider(
                                2048,
                                initial_max_tokens_maximum,
                                value=initial_max_tokens,
                                step=1024,
                                label="Max Tokens",
                                info="Maximum number of tokens in the response.",
                                elem_id="config_max_tokens",
                            )

                            gr.Markdown("### Context & Upscaling")
                            send_full_page_context = gr.Checkbox(
                                value=saved_settings.get(
                                    "send_full_page_context", True
                                ),
                                label="Send Full Page to LLM",
                                info=(
                                    "Include full page image as context. Might improve translation quality. "
                                    "Disable if refusals/using less-capable models or to reduce token usage."
                                ),
                                interactive=initial_ocr_method != "manga-ocr",
                            )
                            upscale_method = gr.Radio(
                                choices=[
                                    ("Model", "model"),
                                    ("Model (Lite)", "model_lite"),
                                    ("LANCZOS", "lanczos"),
                                    ("None", "none"),
                                ],
                                value=saved_settings.get(
                                    "upscale_method", "model_lite"
                                ),
                                label="Bubble/Context Resizing Method",
                                info=(
                                    "Method to resize cropped bubble images/full page before sending to LLM. "
                                    "Model is best quality, Model (Lite) is slightly worse quality but faster/less "
                                    "memory, LANCZOS is worst quality but fastest/least memory."
                                ),
                            )
                            initial_upscale_method = saved_settings.get(
                                "upscale_method", "model_lite"
                            )
                            sliders_interactive = initial_upscale_method != "none"
                            bubble_min_side_pixels = gr.Slider(
                                64,
                                512,
                                value=saved_settings.get("bubble_min_side_pixels", 128),
                                step=16,
                                label="Bubble Min Side Pixels",
                                info=(
                                    "Target minimum side length for speech bubble resizing. "
                                    "Increase for better OCR quality, but may increase token usage."
                                ),
                                elem_id="config_bubble_min_side_pixels",
                                interactive=sliders_interactive,
                            )
                            context_image_max_side_pixels = gr.Slider(
                                512,
                                2560,
                                value=saved_settings.get(
                                    "context_image_max_side_pixels", 1024
                                ),
                                step=128,
                                label="Context Image Max Side Pixels",
                                info=(
                                    "Target maximum side length for full page image resizing. "
                                    "Increase for better OCR quality, but may increase token usage."
                                ),
                                elem_id="config_context_image_max_side_pixels",
                                interactive=sliders_interactive,
                            )
                            osb_min_side_pixels = gr.Slider(
                                64,
                                512,
                                value=saved_settings.get("osb_min_side_pixels", 128),
                                step=16,
                                label="OSB Text Min Side Pixels",
                                info=(
                                    "Target minimum side length for outside speech bubble resizing. "
                                    "Increase for better OCR quality, but may increase token usage."
                                ),
                                elem_id="config_osb_min_side_pixels",
                                interactive=sliders_interactive,
                            )
                        setting_groups.append(group_translation)

                        # --- Rendering Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_rendering:
                            gr.Markdown("### Font Rendering")
                            max_font_size = gr.Slider(
                                5,
                                50,
                                value=saved_settings.get("max_font_size", 16),
                                step=1,
                                label="Max Font Size (px)",
                                info="The largest font size the renderer will attempt to use.",
                            )
                            min_font_size = gr.Slider(
                                5,
                                50,
                                value=saved_settings.get("min_font_size", 8),
                                step=1,
                                label="Min Font Size (px)",
                                info="The smallest font size the renderer will attempt to use before giving up.",
                            )
                            line_spacing_mult = gr.Slider(
                                0.5,
                                2.0,
                                value=saved_settings.get("line_spacing_mult", 1.0),
                                step=0.05,
                                label="Line Spacing Multiplier",
                                info="Adjusts the vertical space between lines of text (1.0 = standard).",
                            )
                            use_subpixel_rendering = gr.Checkbox(
                                value=saved_settings.get(
                                    "use_subpixel_rendering", False
                                ),
                                label="Use Subpixel Rendering",
                                info=(
                                    "Improves text clarity on RGB-based displays. "
                                    "Disable if using a PenTile-based display (i.e., an OLED screen)"
                                ),
                            )
                            font_hinting = gr.Radio(
                                choices=["none", "slight", "normal", "full"],
                                value=saved_settings.get("font_hinting", "none"),
                                label="Font Hinting",
                                info="Adjusts glyph outlines to fit pixel grid. 'None' is often best for "
                                "high-res displays.",
                            )
                            use_ligatures = gr.Checkbox(
                                value=saved_settings.get("use_ligatures", False),
                                label="Use Standard Ligatures (e.g., fi, fl)",
                                info="Enables common letter combinations to be rendered as single glyphs "
                                "(must be supported by the font).",
                            )
                            gr.Markdown("### Text Layout")
                            hyphenate_before_scaling = gr.Checkbox(
                                value=saved_settings.get(
                                    "hyphenate_before_scaling", True
                                ),
                                label="Hyphenate Long Words",
                                info="Try inserting hyphens when wrapping before reducing font size.",
                            )
                            hyphen_penalty = gr.Slider(
                                100,
                                2000,
                                value=saved_settings.get("hyphen_penalty", 1000.0),
                                step=100,
                                label="Hyphen Penalty",
                                info="Penalty for hyphenated line breaks in text layout. "
                                "Increase to discourage hyphenation.",
                                interactive=saved_settings.get(
                                    "hyphenate_before_scaling", True
                                ),
                            )
                            hyphenation_min_word_length = gr.Slider(
                                6,
                                10,
                                value=saved_settings.get(
                                    "hyphenation_min_word_length", 8
                                ),
                                step=1,
                                label="Min Word Length for Hyphenation",
                                info="Minimum word length required for hyphenation.",
                                interactive=saved_settings.get(
                                    "hyphenate_before_scaling", True
                                ),
                            )
                            badness_exponent = gr.Slider(
                                2.0,
                                4.0,
                                value=saved_settings.get("badness_exponent", 3.0),
                                step=0.5,
                                label="Badness Exponent",
                                info="Exponent for line badness calculation in text layout. "
                                "Increase to avoid loose lines.",
                            )
                            padding_pixels = gr.Slider(
                                2,
                                12,
                                value=saved_settings.get("padding_pixels", 5.0),
                                step=1,
                                label="Padding Pixels",
                                info="Padding between text and the edge of the speech bubble. "
                                "Increase for more space between text and bubble boundaries.",
                            )
                            supersampling_factor = gr.Slider(
                                1,
                                16,
                                value=saved_settings.get("supersampling_factor", 4),
                                step=1,
                                label="Supersampling Factor",
                                info="Render text at Nx resolution then downscale for smoother edges. "
                                "Higher values improve quality but use slightly more memory. 1 = disabled.",
                            )
                        setting_groups.append(group_rendering)

                        # --- Outside Text Removal Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_outside_text:
                            gr.Markdown("### Outside Speech Bubble Text")
                            outside_text_enabled = gr.Checkbox(
                                value=saved_settings.get("outside_text_enabled", False),
                                label="Enable OSB Text Detection",
                                info="Detect, inpaint, and translate text outside speech bubbles.",
                            )
                            outside_text_huggingface_token = gr.Textbox(
                                value=saved_settings.get(
                                    "outside_text_huggingface_token", ""
                                ),
                                label="HuggingFace Token (Required for certain features)",
                                type="password",
                                info=(
                                    "Required for downloading OSB Text Detection (YOLO) and Flux Kontext models "
                                    "from HuggingFace Hub."
                                ),
                            )

                            # Wrap all settings except the enable checkbox and token in a Column with visibility control
                            with gr.Column(
                                visible=saved_settings.get(
                                    "outside_text_enabled", False
                                )
                            ) as outside_text_settings_wrapper:
                                gr.Markdown("### Detection")
                                outside_text_osb_confidence = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=saved_settings.get(
                                        "outside_text_osb_confidence", 0.6
                                    ),
                                    step=0.05,
                                    label="OSB Text Detection Confidence",
                                    info="Lower values detect more text, but potentially include false positives.",
                                )
                                outside_text_bbox_expansion_percent = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=saved_settings.get(
                                        "outside_text_bbox_expansion_percent", 0.1
                                    ),
                                    step=0.05,
                                    label="Bounding Box Expansion",
                                    info=(
                                        "Percentage to expand bounding boxes for text detection. "
                                        "Higher values capture more context around text."
                                    ),
                                )
                                outside_text_text_box_proximity_ratio = gr.Slider(
                                    0.01,
                                    0.1,
                                    value=saved_settings.get(
                                        "outside_text_text_box_proximity_ratio", 0.02
                                    ),
                                    step=0.01,
                                    label="Text Box Proximity Ratio",
                                    info=(
                                        "Ratio for grouping nearby text boxes (as fraction of image dimension). "
                                        "Increase to group more distant boxes together."
                                    ),
                                )
                                outside_text_enable_page_number_filtering = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                    label="Filter Page Numbers",
                                    info=(
                                        "Use manga-ocr on margin detections to drop likely page numbers. "
                                        "Slightly slower and may detect false positives."
                                    ),
                                )
                                outside_text_page_filter_margin_threshold = gr.Slider(
                                    0.0,
                                    0.3,
                                    value=saved_settings.get(
                                        "outside_text_page_filter_margin_threshold",
                                        0.1,
                                    ),
                                    step=0.01,
                                    label="Page Number Margin Ratio",
                                    info=(
                                        "Maximum vertical margin (ratio of height) for page-number filtering."
                                    ),
                                    interactive=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                )
                                outside_text_page_filter_min_area_ratio = gr.Slider(
                                    0.0,
                                    0.2,
                                    value=saved_settings.get(
                                        "outside_text_page_filter_min_area_ratio",
                                        0.05,
                                    ),
                                    step=0.01,
                                    label="Page Number Min Area Ratio",
                                    info=(
                                        "Minimum area ratio for page-number filtering."
                                    ),
                                    interactive=saved_settings.get(
                                        "outside_text_enable_page_number_filtering",
                                        False,
                                    ),
                                )
                                gr.Markdown("### Inpainting")
                                outside_text_flux_num_inference_steps = gr.Slider(
                                    1,
                                    30,
                                    value=saved_settings.get(
                                        "outside_text_flux_num_inference_steps", 8
                                    ),
                                    step=1,
                                    label="Steps",
                                    info=(
                                        "Number of denoising steps for Flux. "
                                        "15 is best for quality (diminishing returns beyond); "
                                        "below 6 shows noticeable degradation."
                                    ),
                                    interactive=not saved_settings.get(
                                        "outside_text_force_cv2_inpainting", False
                                    ),
                                )
                                outside_text_flux_residual_diff_threshold = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=saved_settings.get(
                                        "outside_text_flux_residual_diff_threshold",
                                        0.15,
                                    ),
                                    step=0.01,
                                    label="Residual Diff Threshold",
                                    info=(
                                        "First Block Caching threshold for Flux. "
                                        "Higher = faster, but lower quality."
                                    ),
                                    interactive=not saved_settings.get(
                                        "outside_text_force_cv2_inpainting", False
                                    ),
                                )
                                outside_text_seed = gr.Number(
                                    value=saved_settings.get("outside_text_seed", 1),
                                    label="Seed",
                                    info="Seed for reproducible inpainting (-1 = random)",
                                    precision=0,
                                    interactive=not saved_settings.get(
                                        "outside_text_force_cv2_inpainting", False
                                    ),
                                )
                                outside_text_force_cv2_inpainting = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_force_cv2_inpainting", False
                                    ),
                                    label="Force OpenCV Inpainting Instead of Flux",
                                    info=(
                                        "Bypasses Flux and generates a simple white/black background for maximum speed "
                                        "and readability. Useful if you do not value background preservation."
                                    ),
                                )

                                gr.Markdown("### Font Rendering")
                                outside_text_osb_font_pack = gr.Dropdown(
                                    value=saved_osb_font_pack,
                                    choices=[""] + font_choices,
                                    label="Text Font",
                                    info="Font for rendering OSB text translations (leave empty to use main font)",
                                )
                                outside_text_osb_max_font_size = gr.Slider(
                                    5,
                                    96,
                                    value=saved_settings.get(
                                        "outside_text_osb_max_font_size", 64
                                    ),
                                    step=1,
                                    label="Max Font Size (px)",
                                    info="The largest font size the renderer will attempt to use for OSB text.",
                                )
                                outside_text_osb_min_font_size = gr.Slider(
                                    5,
                                    96,
                                    value=saved_settings.get(
                                        "outside_text_osb_min_font_size", 12
                                    ),
                                    step=1,
                                    label="Min Font Size (px)",
                                    info="The smallest font size the renderer will attempt to use for OSB text.",
                                )
                                outside_text_osb_line_spacing = gr.Slider(
                                    0.5,
                                    2.0,
                                    value=saved_settings.get(
                                        "outside_text_osb_line_spacing", 1.0
                                    ),
                                    step=0.05,
                                    label="Line Spacing Multiplier",
                                    info=(
                                        "Adjusts the vertical space between lines of text (1.0 = standard). "
                                        "Also affects vertically stacked text. Decrease for tighter text (e.g., 0.9)."
                                    ),
                                )
                                outside_text_osb_use_subpixel_rendering = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_osb_use_subpixel_rendering", True
                                    ),
                                    label="Use Subpixel Rendering",
                                    info=(
                                        "Improves text clarity on RGB-based displays. "
                                        "Disable if using a PenTile-based display (i.e., an OLED screen)"
                                    ),
                                )
                                outside_text_osb_font_hinting = gr.Radio(
                                    choices=["none", "slight", "normal", "full"],
                                    value=saved_settings.get(
                                        "outside_text_osb_font_hinting", "none"
                                    ),
                                    label="Font Hinting",
                                    info="Adjusts glyph outlines to fit pixel grid. 'None' is often best for "
                                    "high-res displays.",
                                )
                                outside_text_osb_use_ligatures = gr.Checkbox(
                                    value=saved_settings.get(
                                        "outside_text_osb_use_ligatures", False
                                    ),
                                    label="Use Standard Ligatures (e.g., fi, fl)",
                                    info="Enables common letter combinations to be rendered as single glyphs "
                                    "(must be supported by the font).",
                                )
                                outside_text_osb_outline_width = gr.Slider(
                                    0,
                                    10,
                                    value=saved_settings.get(
                                        "outside_text_osb_outline_width", 3.0
                                    ),
                                    step=0.5,
                                    label="Outline Width (px)",
                                    info="Width of text outline for OSB text.",
                                )
                        setting_groups.append(group_outside_text)

                        # --- Output Settings ---
                        image_upscale_mode_default = saved_settings.get(
                            "image_upscale_mode", "off"
                        )
                        if image_upscale_mode_default not in {
                            "off",
                            "initial",
                            "final",
                        }:
                            image_upscale_mode_default = "off"
                        image_upscale_factor_default = float(
                            saved_settings.get("image_upscale_factor", 2.0)
                        )

                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_output:
                            gr.Markdown("### Output Format")
                            output_format = gr.Radio(
                                choices=["auto", "png", "jpeg"],
                                label="Image Output Format",
                                value=saved_settings.get("output_format", "auto"),
                                info="'auto' uses the same format as the input image (defaults to PNG if unknown).",
                            )
                            jpeg_quality = gr.Slider(
                                1,
                                100,
                                value=saved_settings.get("jpeg_quality", 95),
                                step=1,
                                label="JPEG Quality",
                                info="Higher levels result in better quality, but larger file sizes.",
                                interactive=saved_settings.get("output_format", "auto")
                                != "png",
                            )
                            png_compression = gr.Slider(
                                0,
                                6,
                                value=saved_settings.get("png_compression", 2),
                                step=1,
                                label="PNG Compression Level",
                                info=(
                                    "Uses OxiPNG. Higher levels result in smaller file sizes, "
                                    "but slower processing times."
                                ),
                                interactive=saved_settings.get("output_format", "auto")
                                != "jpeg",
                            )
                            gr.Markdown("### Upscaling")
                            image_upscale_mode = gr.Radio(
                                choices=["off", "initial", "final"],
                                value=image_upscale_mode_default,
                                label="Image Upscaling Method",
                                info=(
                                    "Determines whether to upscale the initial untranslated image or the final "
                                    "translated image. 'Initial' may result in cleaner text, but cause inconsistent "
                                    "results compare to 'final'."
                                ),
                            )
                            image_upscale_model = gr.Radio(
                                choices=[
                                    ("Model", "model"),
                                    ("Model (Lite)", "model_lite"),
                                ],
                                value=saved_settings.get(
                                    "image_upscale_model", "model_lite"
                                ),
                                label="Upscaling Model",
                                info=(
                                    "Model to use for image upscaling. Model is best quality, "
                                    "Model (Lite) is slightly worse quality but faster/less memory."
                                ),
                                interactive=image_upscale_mode_default != "off",
                            )
                            image_upscale_factor = gr.Slider(
                                1.0,
                                8.0,
                                value=image_upscale_factor_default,
                                step=0.1,
                                label="Upscale Factor",
                                info=(
                                    "Factor for the selected upscaling mode. "
                                    "The selected model will perform an upscaling pass every 2x, "
                                    "downscaling to meet the target if needed."
                                ),
                                interactive=image_upscale_mode_default != "off",
                            )
                            auto_scale = gr.Checkbox(
                                value=saved_settings.get("auto_scale", True),
                                label="Auto-Scale to Image Size",
                                info=(
                                    "Automatically scale pipeline parameters (fonts, kernels, etc.) "
                                    "to input image size, treating it as 1MP. Ensures consistent behavior "
                                    "across different image resolutions."
                                ),
                            )
                        setting_groups.append(group_output)

                        # --- Other Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_other:
                            gr.Markdown("### Other")
                            refresh_resources_button = gr.Button(
                                "Refresh Models / Fonts",
                                variant="secondary",
                                elem_classes="config-button",
                            )
                            unload_models_button = gr.Button(
                                "Force Unload Models",
                                variant="secondary",
                                elem_classes="config-button",
                            )
                            verbose = gr.Checkbox(
                                value=saved_settings.get("verbose", False),
                                label="Verbose Logging",
                                info="Enable verbose logging in console.",
                            )
                            cleaning_only_toggle = gr.Checkbox(
                                value=saved_settings.get("cleaning_only", False),
                                label="Cleaning-only Mode",
                                info="Skip translation and text rendering, output only the cleaned speech bubbles.",
                                interactive=not saved_settings.get("test_mode", False),
                            )
                            test_mode_toggle = gr.Checkbox(
                                value=saved_settings.get("test_mode", False),
                                label="Test Mode",
                                info=(
                                    "Skip translation and render placeholder text (lorem ipsum)."
                                ),
                                interactive=not saved_settings.get(
                                    "cleaning_only", False
                                ),
                            )
                        setting_groups.append(group_other)

        # --- Define Event Handlers ---
        save_config_inputs = [
            confidence,
            conjoined_confidence,
            panel_confidence,
            use_sam2_checkbox,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            config_reading_direction,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_translation_mode,
            ocr_method_radio,
            max_font_size,
            min_font_size,
            line_spacing_mult,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            output_format,
            jpeg_quality,
            png_compression,
            verbose,
            cleaning_only_toggle,
            test_mode_toggle,
            input_language,
            output_language,
            font_dropdown,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            enable_web_search_checkbox,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            send_full_page_context,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            special_instructions,
            batch_special_instructions,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            padding_pixels,
            supersampling_factor,
            outside_text_enabled,
            outside_text_seed,
            outside_text_force_cv2_inpainting,
            outside_text_flux_num_inference_steps,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
        ]

        reset_outputs = [
            confidence,
            conjoined_confidence,
            panel_confidence,
            use_sam2_checkbox,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            config_reading_direction,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_translation_mode,
            ocr_method_radio,
            max_font_size,
            min_font_size,
            line_spacing_mult,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            output_format,
            jpeg_quality,
            png_compression,
            verbose,
            cleaning_only_toggle,
            test_mode_toggle,
            input_language,
            output_language,
            font_dropdown,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            enable_web_search_checkbox,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            config_status,
            send_full_page_context,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            special_instructions,
            batch_special_instructions,
            outside_text_enabled,
            outside_text_seed,
            outside_text_force_cv2_inpainting,
            outside_text_flux_num_inference_steps,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
        ]

        translate_inputs = [
            input_image,
            confidence,
            conjoined_confidence,
            panel_confidence,
            use_sam2_checkbox,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_reading_direction,
            config_translation_mode,
            ocr_method_radio,
            input_language,
            output_language,
            font_dropdown,
            max_font_size,
            min_font_size,
            line_spacing_mult,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            output_format,
            jpeg_quality,
            png_compression,
            verbose,
            cleaning_only_toggle,
            test_mode_toggle,
            enable_web_search_checkbox,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            send_full_page_context,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            padding_pixels,
            supersampling_factor,
            outside_text_enabled,
            outside_text_seed,
            outside_text_force_cv2_inpainting,
            outside_text_flux_num_inference_steps,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            special_instructions,
            batch_special_instructions,
        ]

        batch_inputs = [
            input_files,
            input_zip,
            confidence,
            conjoined_confidence,
            panel_confidence,
            use_sam2_checkbox,
            conjoined_detection_checkbox,
            osb_text_verification_checkbox,
            use_panel_sorting_checkbox,
            thresholding_value,
            use_otsu_threshold,
            inpaint_colored_bubbles,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            deepseek_api_key,
            zai_api_key,
            moonshot_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_reading_direction,
            config_translation_mode,
            ocr_method_radio,
            input_language,
            output_language,
            font_dropdown,
            max_font_size,
            min_font_size,
            line_spacing_mult,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            output_format,
            jpeg_quality,
            png_compression,
            verbose,
            cleaning_only_toggle,
            test_mode_toggle,
            enable_web_search_checkbox,
            media_resolution_dropdown,
            media_resolution_bubbles_dropdown,
            media_resolution_context_dropdown,
            reasoning_effort_dropdown,
            effort_dropdown,
            send_full_page_context,
            upscale_method,
            bubble_min_side_pixels,
            context_image_max_side_pixels,
            osb_min_side_pixels,
            hyphenate_before_scaling,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            padding_pixels,
            supersampling_factor,
            outside_text_enabled,
            outside_text_seed,
            outside_text_force_cv2_inpainting,
            outside_text_flux_num_inference_steps,
            outside_text_flux_residual_diff_threshold,
            outside_text_osb_confidence,
            outside_text_enable_page_number_filtering,
            outside_text_page_filter_margin_threshold,
            outside_text_page_filter_min_area_ratio,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_bbox_expansion_percent,
            outside_text_text_box_proximity_ratio,
            image_upscale_mode,
            image_upscale_factor,
            image_upscale_model,
            auto_scale,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            special_instructions,
            batch_special_instructions,
        ]

        # Config Tab Navigation & Updates
        output_components_for_switch = setting_groups + nav_buttons
        nav_button_detection.click(
            fn=lambda idx=0: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_cleaning.click(
            fn=lambda idx=1: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_translation.click(
            fn=lambda idx=2: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_rendering.click(
            fn=lambda idx=3: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_outside_text.click(
            fn=lambda idx=4: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_output.click(
            fn=lambda idx=5: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_other.click(
            fn=lambda idx=6: utils.switch_settings_view(
                idx, setting_groups, nav_buttons
            ),
            outputs=output_components_for_switch,
            queue=False,
        )

        output_format.change(
            fn=callbacks.handle_output_format_change,
            inputs=output_format,
            outputs=[jpeg_quality, png_compression],
            queue=False,
        )

        image_upscale_mode.change(
            fn=lambda mode: (
                gr.update(interactive=mode != "off"),
                gr.update(interactive=mode != "off"),
            ),
            inputs=image_upscale_mode,
            outputs=[image_upscale_factor, image_upscale_model],
            queue=False,
        )

        provider_selector.change(
            fn=callbacks.handle_provider_change,
            inputs=[provider_selector, temperature, ocr_method_radio],
            outputs=[
                google_api_key,
                openai_api_key,
                anthropic_api_key,
                xai_api_key,
                deepseek_api_key,
                zai_api_key,
                moonshot_api_key,
                openrouter_api_key,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
                config_model_name,
                temperature,
                top_p,
                top_k,
                max_tokens,
                enable_web_search_checkbox,
                media_resolution_dropdown,
                media_resolution_bubbles_dropdown,
                media_resolution_context_dropdown,
                reasoning_effort_dropdown,
                effort_dropdown,
            ],
            queue=False,
        ).then(  # Trigger model fetch *after* provider change updates visibility etc.
            fn=lambda prov, url, key, ocr_method: (
                utils.fetch_and_update_compatible_models(url, key)
                if prov == "OpenAI-Compatible"
                else (
                    utils.fetch_and_update_openrouter_models(ocr_method=ocr_method)
                    if prov == "OpenRouter"
                    else gr.update()
                )
            ),
            inputs=[
                provider_selector,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
                ocr_method_radio,
            ],
            outputs=[config_model_name],
            queue=True,  # Allow fetching to happen in the background
        )

        config_model_name.change(
            fn=callbacks.handle_model_change,
            inputs=[provider_selector, config_model_name, temperature],
            outputs=[
                temperature,
                top_k,
                max_tokens,
                enable_web_search_checkbox,
                media_resolution_dropdown,
                media_resolution_bubbles_dropdown,
                media_resolution_context_dropdown,
                reasoning_effort_dropdown,
                effort_dropdown,
            ],
            queue=False,
        )

        # Thresholding checkbox change handler
        use_otsu_threshold.change(
            fn=callbacks.handle_thresholding_change,
            inputs=use_otsu_threshold,
            outputs=thresholding_value,
            queue=False,
        )

        # Hyphenation checkbox change handler
        hyphenate_before_scaling.change(
            fn=callbacks.handle_hyphenation_change,
            inputs=hyphenate_before_scaling,
            outputs=[hyphen_penalty, hyphenation_min_word_length],
            queue=False,
        )

        # Cleaning-only and Test mode mutual exclusivity handlers
        cleaning_only_toggle.change(
            fn=callbacks.handle_cleaning_only_change,
            inputs=cleaning_only_toggle,
            outputs=test_mode_toggle,
            queue=False,
        )

        # Test mode toggle change handler
        test_mode_toggle.change(
            fn=callbacks.handle_test_mode_change,
            inputs=test_mode_toggle,
            outputs=cleaning_only_toggle,
            queue=False,
        )

        # OSB enable/disable handler
        outside_text_enabled.change(
            fn=lambda x: gr.update(visible=x),
            inputs=outside_text_enabled,
            outputs=outside_text_settings_wrapper,
            queue=False,
        )

        # Page-number filtering toggle -> enable/disable related sliders
        outside_text_enable_page_number_filtering.change(
            fn=lambda enabled: (
                gr.update(interactive=enabled),
                gr.update(interactive=enabled),
            ),
            inputs=outside_text_enable_page_number_filtering,
            outputs=[
                outside_text_page_filter_margin_threshold,
                outside_text_page_filter_min_area_ratio,
            ],
            queue=False,
        )

        # Force cv2 inpainting toggle -> disable Flux controls
        outside_text_force_cv2_inpainting.change(
            fn=lambda forced: (
                gr.update(interactive=not forced),
                gr.update(interactive=not forced),
                gr.update(interactive=not forced),
            ),
            inputs=outside_text_force_cv2_inpainting,
            outputs=[
                outside_text_flux_num_inference_steps,
                outside_text_flux_residual_diff_threshold,
                outside_text_seed,
            ],
            queue=False,
        )

        # Conjoined detection change handler - clears SAM cache
        conjoined_detection_checkbox.change(
            fn=callbacks.handle_conjoined_detection_change,
            inputs=conjoined_detection_checkbox,
            outputs=conjoined_confidence,
            queue=False,
        )

        # Panel sorting change handler
        use_panel_sorting_checkbox.change(
            fn=lambda enabled: gr.update(interactive=enabled),
            inputs=use_panel_sorting_checkbox,
            outputs=panel_confidence,
            queue=False,
        )

        # Confidence threshold change handlers - clear YOLO cache
        confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=confidence,
            outputs=None,
            queue=False,
        )

        conjoined_confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=conjoined_confidence,
            outputs=None,
            queue=False,
        )

        panel_confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=panel_confidence,
            outputs=None,
            queue=False,
        )

        outside_text_osb_confidence.change(
            fn=callbacks.handle_confidence_threshold_change,
            inputs=outside_text_osb_confidence,
            outputs=None,
            queue=False,
        )

        # Translation mode change handler - disable OCR selection when one-step
        config_translation_mode.change(
            fn=callbacks.handle_translation_mode_change,
            inputs=[config_translation_mode, ocr_method_radio],
            outputs=ocr_method_radio,
            queue=False,
        )

        # OCR method change handler
        ocr_method_radio.change(
            fn=callbacks.handle_ocr_method_change,
            inputs=[
                ocr_method_radio,
                input_language,
                original_language_state,
                batch_input_language,
                batch_original_language_state,
                provider_selector,
                config_model_name,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
            ],
            outputs=[
                provider_selector,
                input_language,
                original_language_state,
                batch_input_language,
                batch_original_language_state,
                send_full_page_context,
                config_model_name,
            ],
            queue=False,
        )

        # Upscale method change handler
        upscale_method.change(
            fn=lambda x: [
                gr.update(interactive=x != "none"),
                gr.update(interactive=x != "none"),
                gr.update(interactive=x != "none"),
            ],
            inputs=upscale_method,
            outputs=[
                bubble_min_side_pixels,
                context_image_max_side_pixels,
                osb_min_side_pixels,
            ],
            queue=False,
        )

        # Config Save/Reset Buttons
        save_config_btn.click(
            fn=callbacks.handle_save_config_click,
            inputs=save_config_inputs,
            outputs=[config_status],
            queue=False,
        ).then(fn=None, inputs=None, outputs=None, js=js_status_fade, queue=False)

        reset_defaults_btn.click(
            fn=functools.partial(
                callbacks.handle_reset_defaults_click,
                fonts_base_dir=fonts_base_dir,
            ),
            inputs=[],
            outputs=reset_outputs,
            queue=False,
        ).then(fn=None, inputs=None, outputs=None, js=js_status_fade, queue=False)

        # Refresh Button
        refresh_outputs = [
            font_dropdown,
            batch_font_dropdown,
            outside_text_osb_font_pack,
        ]
        refresh_resources_button.click(
            fn=functools.partial(
                callbacks.handle_refresh_resources_click,
                fonts_base_dir=fonts_base_dir,
            ),
            inputs=[],
            outputs=refresh_outputs,
            js=js_refresh_button_processing,
        ).then(fn=None, inputs=None, outputs=None, js=js_refresh_button_reset)

        # Unload Models Button
        unload_models_button.click(
            fn=callbacks.handle_unload_models_click,
            inputs=[],
            outputs=[],
        )

        # Translator Tab Button
        clear_button.click(
            fn=lambda: (None, None, ""),
            outputs=[input_image, output_image, status_message],
            queue=False,
        )
        batch_clear_button.click(
            fn=lambda: (None, None, None, ""),
            outputs=[
                input_files,
                input_zip,
                batch_output_gallery,
                batch_status_message,
            ],
            queue=False,
        )
        translate_event = translate_button.click(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=True,
                button_text_processing="Translating...",
                button_text_idle="Translate",
            ),
            outputs=[
                translate_button,
                clear_button,
                cancel_button,
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
            ],
            queue=False,
        ).then(
            fn=functools.partial(
                callbacks.handle_translate_click,
                models_dir=models_dir,
                fonts_base_dir=fonts_base_dir,
                target_device=target_device,
            ),
            inputs=translate_inputs,
            outputs=[output_image, status_message],
        )
        translate_event.then(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=False,
                button_text_processing="Translating...",
                button_text_idle="Translate",
            ),
            outputs=[
                translate_button,
                clear_button,
                cancel_button,
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
            ],
            queue=False,
        ).then(fn=None, inputs=None, outputs=None, queue=False)

        cancel_button.click(
            fn=callbacks.cancel_process,
            cancels=translate_event,
            queue=False,
        )

        # Batch Tab Button
        batch_event = batch_process_button.click(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=True,
                button_text_processing="Processing...",
                button_text_idle="Start Batch Translating",
            ),
            outputs=[
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
                translate_button,
                clear_button,
                cancel_button,
            ],
            queue=False,
        ).then(
            fn=functools.partial(
                callbacks.handle_batch_click,
                models_dir=models_dir,
                fonts_base_dir=fonts_base_dir,
                target_device=target_device,
            ),
            inputs=batch_inputs,
            outputs=[batch_output_gallery, batch_status_message],
        )
        batch_event.then(
            fn=functools.partial(
                callbacks.update_process_buttons,
                processing=False,
                button_text_processing="Processing...",
                button_text_idle="Start Batch Translating",
            ),
            outputs=[
                batch_process_button,
                batch_clear_button,
                batch_cancel_button,
                translate_button,
                clear_button,
                cancel_button,
            ],
            queue=False,
        ).then(fn=None, inputs=None, outputs=None, queue=False)

        batch_cancel_button.click(
            fn=callbacks.cancel_process,
            cancels=batch_event,
            queue=False,
        )

        app.load(
            fn=callbacks.handle_app_load,
            inputs=[
                provider_selector,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
            ],
            outputs=[config_model_name],
            queue=False,
        )

    return app
