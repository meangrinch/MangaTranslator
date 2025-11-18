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
            is_reasoning = utils._is_reasoning_model(
                initial_provider, initial_model_name
            )
            initial_max_tokens = 16384 if is_reasoning else 4096

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
                            label="Upload ZIP Archive",
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
                                value=saved_settings.get("confidence", 0.35),
                                step=0.05,
                                label="Bubble Detection Confidence",
                                info="Lower values detect more bubbles, potentially including false positives.",
                            )
                            use_sam2_checkbox = gr.Checkbox(
                                value=saved_settings.get("use_sam2", True),
                                label="Use SAM 2.1 for Segmentation",
                                info="Enhances bubble segmentation quality, especially for oddly shaped bubbles.",
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
                                label="Use Automatic Thresholding (Otsu)",
                                info="Automatically determine the brightness threshold instead of using the fixed "
                                "value. Recommended for varied lighting.",
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
                        setting_groups.append(group_cleaning)

                        # --- Translation Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_translation:
                            gr.Markdown("### OCR & Translation")
                            config_translation_mode = gr.Radio(
                                choices=["one-step", "two-step"],
                                label="LLM Translation Mode",
                                value=saved_settings.get(
                                    "translation_mode",
                                    settings_manager.DEFAULT_SETTINGS[
                                        "translation_mode"
                                    ],
                                ),
                                info=(
                                    "Determines whether to run OCR and Translation in one step or two. "
                                    "'two-step' might improve translation quality for less-capable LLMs."
                                ),
                                elem_id="config_translation_mode",
                            )

                            gr.Markdown("### LLM Settings")
                            provider_selector = gr.Radio(
                                choices=list(settings_manager.PROVIDER_MODELS.keys()),
                                label="Translation Provider",
                                value=config_initial_provider,
                                elem_id="provider_selector",
                            )
                            google_api_key = gr.Textbox(
                                label="Google API Key",
                                placeholder="Enter Google API key (starts with AI...)",
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
                                placeholder="Enter Base URL (e.g., http://localhost:11434/v1)",
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
                            # Compute initial visibility for enable thinking (Google, Anthropic, OpenRouter)
                            _initial_enable_thinking_visible = False
                            try:
                                if (
                                    config_initial_provider == "Google"
                                    and config_initial_model_name
                                ):
                                    _initial_enable_thinking_visible = (
                                        "gemini-2.5-flash" in config_initial_model_name
                                    )
                                elif (
                                    config_initial_provider == "Anthropic"
                                    and config_initial_model_name
                                ):
                                    lm = config_initial_model_name.lower()
                                    _initial_enable_thinking_visible = (
                                        lm.startswith("claude-opus-4-1")
                                        or lm.startswith("claude-opus-4")
                                        or lm.startswith("claude-sonnet-4")
                                        or lm.startswith("claude-3-7-sonnet")
                                    )
                                elif (
                                    config_initial_provider == "OpenRouter"
                                    and config_initial_model_name
                                ):
                                    lm = config_initial_model_name.lower()
                                    is_gemini_reasoning = "gemini-2.5-flash" in lm
                                    is_anthropic_reasoning = (
                                        "claude-opus-4.1" in lm
                                        or "claude-opus-4" in lm
                                        or "claude-sonnet-4" in lm
                                    )
                                    is_grok_reasoning = "grok" in lm and "fast" in lm
                                    _initial_enable_thinking_visible = (
                                        is_gemini_reasoning
                                        or is_anthropic_reasoning
                                        or is_grok_reasoning
                                    )
                            except Exception:
                                _initial_enable_thinking_visible = False

                            enable_thinking_checkbox = gr.Checkbox(
                                label="Enable Thinking",
                                value=saved_settings.get("enable_thinking", True),
                                info=(
                                    "Enable reasoning for Gemini 2.5 Flash, Claude, or Grok (via OpenRouter)."
                                ),
                                visible=_initial_enable_thinking_visible,
                                elem_id="enable_thinking_checkbox",
                            )

                            _initial_enable_grounding_visible = False
                            try:
                                if config_initial_provider == "Google":
                                    _initial_enable_grounding_visible = True
                                elif (
                                    config_initial_provider == "OpenRouter"
                                    and config_initial_model_name
                                    and "gemini" in config_initial_model_name.lower()
                                ):
                                    _initial_enable_grounding_visible = True
                            except Exception:
                                _initial_enable_grounding_visible = False

                            enable_grounding_checkbox = gr.Checkbox(
                                label="Enable Grounding",
                                value=saved_settings.get("enable_grounding", False),
                                info=(
                                    "Allow Gemini to use Google Search for up-to-date info. "
                                    "Might improve translation quality."
                                ),
                                visible=_initial_enable_grounding_visible,
                                elem_id="enable_grounding_checkbox",
                            )
                            # Compute initial visibility for reasoning effort (OpenAI and OpenRouter OpenAI models)
                            _initial_reasoning_effort_visible = False
                            try:
                                if config_initial_provider == "OpenAI":
                                    _initial_reasoning_effort_visible = True
                                elif (
                                    config_initial_provider == "OpenRouter"
                                    and config_initial_model_name
                                ):
                                    lm = config_initial_model_name.lower()
                                    is_reasoning_capable = (
                                        "gpt-5" in lm
                                        or "o1" in lm
                                        or "o3" in lm
                                        or "o4-mini" in lm
                                    )
                                    _initial_reasoning_effort_visible = (
                                        is_reasoning_capable
                                    )
                            except Exception:
                                _initial_reasoning_effort_visible = False

                            reasoning_effort_dropdown = gr.Dropdown(
                                choices=["low", "medium", "high"],
                                label="Reasoning Effort (OpenAI/OpenRouter)",
                                value=saved_settings.get("reasoning_effort", "medium"),
                                info=(
                                    "Controls internal effort for OpenAI reasoning and "
                                    "OpenRouter OpenAI models."
                                ),
                                visible=_initial_reasoning_effort_visible,
                                elem_id="reasoning_effort_dropdown",
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
                                32768,
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
                                    "Include full page image as context. Disable if refusals/using less-capable models."
                                ),
                            )
                            upscale_method = gr.Radio(
                                choices=[
                                    ("Model", "model"),
                                    ("LANCZOS", "lanczos"),
                                    ("None", "none"),
                                ],
                                value=saved_settings.get("upscale_method", "model"),
                                label="Bubble/Context Upscaling Method",
                                info=(
                                    "Determines how to upscale cropped bubble images/full page before sending to LLM. "
                                    "Improves OCR; does not affect output image quality."
                                ),
                            )
                            initial_upscale_method = saved_settings.get(
                                "upscale_method", "model"
                            )
                            sliders_interactive = initial_upscale_method != "none"
                            bubble_min_side_pixels = gr.Slider(
                                64,
                                512,
                                value=saved_settings.get("bubble_min_side_pixels", 128),
                                step=16,
                                label="Bubble Min Side Pixels",
                                info="Target minimum side length for speech bubble upscaling",
                                elem_id="config_bubble_min_side_pixels",
                                interactive=sliders_interactive,
                            )
                            context_image_max_side_pixels = gr.Slider(
                                512,
                                2560,
                                value=saved_settings.get(
                                    "context_image_max_side_pixels", 1536
                                ),
                                step=128,
                                label="Context Image Max Side Pixels",
                                info="Target maximum side length for full page image",
                                elem_id="config_context_image_max_side_pixels",
                                interactive=sliders_interactive,
                            )
                            osb_min_side_pixels = gr.Slider(
                                64,
                                512,
                                value=saved_settings.get("osb_min_side_pixels", 128),
                                step=16,
                                label="OSB Text Min Side Pixels",
                                info="Target minimum side length for outside speech bubble upscaling",
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
                                value=saved_settings.get("max_font_size", 14),
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
                            line_spacing = gr.Slider(
                                0.5,
                                2.0,
                                value=saved_settings.get("line_spacing", 1.0),
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
                        setting_groups.append(group_rendering)

                        # --- Outside Text Removal Settings ---
                        with gr.Group(
                            visible=False, elem_classes="settings-group"
                        ) as group_outside_text:
                            gr.Markdown("### Outside Speech Bubble Text")
                            outside_text_enabled = gr.Checkbox(
                                value=saved_settings.get("outside_text_enabled", False),
                                label="Enable OSB Detection",
                                info="Detect, inpaint, and translate text outside speech bubbles.",
                            )

                            # Wrap all settings except the enable checkbox in a Column with visibility control
                            with gr.Column(
                                visible=saved_settings.get(
                                    "outside_text_enabled", False
                                )
                            ) as outside_text_settings_wrapper:
                                gr.Markdown("### Detection")
                                outside_text_easyocr_min_size = gr.Slider(
                                    10,
                                    500,
                                    value=saved_settings.get(
                                        "outside_text_easyocr_min_size", 200
                                    ),
                                    step=10,
                                    label="Min Text Size (px)",
                                    info="Minimum text region size in pixels for OSB text detection. "
                                    "Smaller text may render awkwardly.",
                                )
                                gr.Markdown("### Inpainting")
                                outside_text_huggingface_token = gr.Textbox(
                                    value=saved_settings.get(
                                        "outside_text_huggingface_token", ""
                                    ),
                                    label="HuggingFace Token (Required)",
                                    type="password",
                                    info="Required for downloading Flux Kontext files from HuggingFace Hub.",
                                )
                                outside_text_flux_num_inference_steps = gr.Slider(
                                    1,
                                    30,
                                    value=saved_settings.get(
                                        "outside_text_flux_num_inference_steps", 8
                                    ),
                                    step=1,
                                    label="Steps",
                                    info=(
                                        "Number of denoising steps for Flux Kontext. "
                                        "15 is best for quality (diminishing returns beyond); "
                                        "below 6 shows noticeable degradation."
                                    ),
                                )
                                outside_text_flux_residual_diff_threshold = gr.Slider(
                                    0.0,
                                    1.0,
                                    value=saved_settings.get(
                                        "outside_text_flux_residual_diff_threshold",
                                        0.12,
                                    ),
                                    step=0.01,
                                    label="Residual Diff Threshold",
                                    info=(
                                        "First Block Caching threshold for inference "
                                        "(higher = faster, lower quality)."
                                    ),
                                )
                                outside_text_seed = gr.Number(
                                    value=saved_settings.get("outside_text_seed", 1),
                                    label="Seed",
                                    info="Seed for reproducible inpainting (-1 = random)",
                                    precision=0,
                                )

                                gr.Markdown("### Font Rendering")
                                outside_text_osb_font_pack = gr.Dropdown(
                                    value=saved_settings.get(
                                        "outside_text_osb_font_pack", ""
                                    ),
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
                                    50,
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
                                    info="Adjusts the vertical space between lines of text (1.0 = standard).",
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
                            gr.Markdown("### Image Output")
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
                                label="JPEG Quality (higher = better quality)",
                                interactive=saved_settings.get("output_format", "auto")
                                != "png",
                            )
                            png_compression = gr.Slider(
                                0,
                                9,
                                value=saved_settings.get("png_compression", 6),
                                step=1,
                                label="PNG Compression Level (higher = smaller file)",
                                interactive=saved_settings.get("output_format", "auto")
                                != "jpeg",
                            )
                            image_upscale_mode = gr.Radio(
                                choices=["off", "initial", "final"],
                                value=image_upscale_mode_default,
                                label="Image Upscaling Method",
                                info=(
                                    "Determines whether to upscale the initial untranslated image or the final "
                                    "translated image. 'Initial' results in cleaner text, but consumes more memory."
                                ),
                            )
                            image_upscale_factor = gr.Slider(
                                1.0,
                                8.0,
                                value=image_upscale_factor_default,
                                step=0.1,
                                label="Upscale Factor",
                                info="Factor for the selected upscaling mode.",
                                interactive=image_upscale_mode_default != "off",
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
            use_sam2_checkbox,
            config_reading_direction,
            thresholding_value,
            use_otsu_threshold,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_translation_mode,
            max_font_size,
            min_font_size,
            line_spacing,
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
            enable_thinking_checkbox,
            enable_grounding_checkbox,
            reasoning_effort_dropdown,
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
            outside_text_enabled,
            outside_text_seed,
            outside_text_flux_num_inference_steps,
            outside_text_flux_residual_diff_threshold,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_easyocr_min_size,
            image_upscale_mode,
            image_upscale_factor,
        ]

        reset_outputs = [
            confidence,
            use_sam2_checkbox,
            config_reading_direction,
            thresholding_value,
            use_otsu_threshold,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            max_tokens,
            config_translation_mode,
            max_font_size,
            min_font_size,
            line_spacing,
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
            enable_thinking_checkbox,
            enable_grounding_checkbox,
            reasoning_effort_dropdown,
            config_status,
            send_full_page_context,
            upscale_method,
            hyphenate_before_scaling,
            special_instructions,
            batch_special_instructions,
            outside_text_enabled,
            outside_text_seed,
            outside_text_flux_num_inference_steps,
            outside_text_flux_residual_diff_threshold,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_easyocr_min_size,
            image_upscale_mode,
            image_upscale_factor,
        ]

        translate_inputs = [
            input_image,
            confidence,
            use_sam2_checkbox,
            thresholding_value,
            use_otsu_threshold,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
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
            input_language,
            output_language,
            font_dropdown,
            max_font_size,
            min_font_size,
            line_spacing,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            output_format,
            jpeg_quality,
            png_compression,
            verbose,
            cleaning_only_toggle,
            test_mode_toggle,
            enable_thinking_checkbox,
            enable_grounding_checkbox,
            reasoning_effort_dropdown,
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
            outside_text_enabled,
            outside_text_seed,
            outside_text_flux_num_inference_steps,
            outside_text_flux_residual_diff_threshold,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_easyocr_min_size,
            image_upscale_mode,
            image_upscale_factor,
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
            use_sam2_checkbox,
            thresholding_value,
            use_otsu_threshold,
            roi_shrink_px,
            provider_selector,
            google_api_key,
            openai_api_key,
            anthropic_api_key,
            xai_api_key,
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
            input_language,
            output_language,
            font_dropdown,
            max_font_size,
            min_font_size,
            line_spacing,
            use_subpixel_rendering,
            font_hinting,
            use_ligatures,
            output_format,
            jpeg_quality,
            png_compression,
            verbose,
            cleaning_only_toggle,
            test_mode_toggle,
            enable_thinking_checkbox,
            enable_grounding_checkbox,
            reasoning_effort_dropdown,
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
            outside_text_enabled,
            outside_text_seed,
            outside_text_flux_num_inference_steps,
            outside_text_flux_residual_diff_threshold,
            outside_text_huggingface_token,
            outside_text_osb_font_pack,
            outside_text_osb_max_font_size,
            outside_text_osb_min_font_size,
            outside_text_osb_use_ligatures,
            outside_text_osb_outline_width,
            outside_text_osb_line_spacing,
            outside_text_osb_use_subpixel_rendering,
            outside_text_osb_font_hinting,
            outside_text_easyocr_min_size,
            image_upscale_mode,
            image_upscale_factor,
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
            fn=lambda mode: gr.update(interactive=mode != "off"),
            inputs=image_upscale_mode,
            outputs=image_upscale_factor,
            queue=False,
        )

        provider_selector.change(
            fn=callbacks.handle_provider_change,
            inputs=[provider_selector, temperature],
            outputs=[
                google_api_key,
                openai_api_key,
                anthropic_api_key,
                xai_api_key,
                openrouter_api_key,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
                config_model_name,
                temperature,
                top_k,
                max_tokens,
                enable_thinking_checkbox,
                enable_grounding_checkbox,
                reasoning_effort_dropdown,
            ],
            queue=False,
        ).then(  # Trigger model fetch *after* provider change updates visibility etc.
            fn=lambda prov, url, key: (
                utils.fetch_and_update_compatible_models(url, key)
                if prov == "OpenAI-Compatible"
                else (
                    utils.fetch_and_update_openrouter_models()
                    if prov == "OpenRouter"
                    else gr.update()
                )
            ),
            inputs=[
                provider_selector,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
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
                enable_thinking_checkbox,
                enable_grounding_checkbox,
                reasoning_effort_dropdown,
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
