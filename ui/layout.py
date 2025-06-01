import functools
from pathlib import Path
from typing import Any

import gradio as gr

from . import settings_manager
from . import utils
from . import callbacks

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
        newContent.innerHTML = 'Made by <a href="https://github.com/meangrinch">grinnch</a> with ❤️'; // credits

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


def create_layout(models_dir: Path, fonts_base_dir: Path, target_device: Any) -> gr.Blocks:
    """Creates the Gradio UI layout and connects callbacks."""

    with gr.Blocks(title="MangaTranslator", js=js_credits, css_paths="style.css") as app:

        gr.Markdown("# MangaTranslator")

        model_choices = utils.get_available_models(models_dir)
        font_choices, initial_default_font = utils.get_available_font_packs(fonts_base_dir)
        saved_settings = settings_manager.get_saved_settings()

        saved_yolo_model = saved_settings.get("yolo_model")
        default_yolo_model = (
            saved_yolo_model if saved_yolo_model in model_choices else (model_choices[0] if model_choices else None)
        )

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

        initial_provider = saved_settings.get("provider", settings_manager.DEFAULT_SETTINGS["provider"])
        initial_model_name = saved_settings.get("model_name")

        if initial_provider == "OpenRouter" or initial_provider == "OpenAI-compatible":
            initial_models_choices = [initial_model_name] if initial_model_name else []
        else:
            initial_models_choices = settings_manager.PROVIDER_MODELS.get(initial_provider, [])

        # --- Define UI Components ---
        with gr.Tabs():
            with gr.TabItem("Translator"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="filepath",
                            label="Upload Image",
                            show_download_button=False,
                            image_mode="RGBA",
                            elem_id="translator_input_image",
                        )
                        font_dropdown = gr.Dropdown(
                            choices=font_choices, label="Text Font", value=default_font, filterable=False
                        )
                        with gr.Accordion("Translation Settings", open=True):
                            input_language = gr.Dropdown(
                                [
                                    "Japanese",
                                    "Korean",
                                    "Chinese",
                                    "Spanish",
                                    "French",
                                    "German",
                                    "Italian",
                                    "Portuguese",
                                    "Russian",
                                ],
                                label="Source Language",
                                value=saved_settings.get("input_language", "Japanese"),
                                allow_custom_value=True,
                            )
                            output_language = gr.Dropdown(
                                [
                                    "English",
                                    "Spanish",
                                    "French",
                                    "German",
                                    "Italian",
                                    "Portuguese",
                                    "Russian",
                                    "Chinese",
                                    "Korean",
                                    "Japanese",
                                ],
                                label="Target Language",
                                value=saved_settings.get("output_language", "English"),
                                allow_custom_value=True,
                            )
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            type="pil", label="Translated Image", interactive=False, elem_id="translator_output_image"
                        )
                        # Assign specific ID for JS targeting
                        status_message = gr.Textbox(
                            label="Status", interactive=False, elem_id="translator_status_message"
                        )
                        with gr.Row():
                            translate_button = gr.Button("Translate", variant="primary")
                            clear_button = gr.ClearButton([input_image, output_image, status_message], value="Clear")  # noqa

            with gr.TabItem("Batch"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_files = gr.Files(
                            label="Upload Images or Folder",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath",
                        )
                        batch_font_dropdown = gr.Dropdown(
                            choices=font_choices, label="Text Font", value=batch_default_font, filterable=False
                        )
                        with gr.Accordion("Translation Settings", open=True):
                            batch_input_language = gr.Dropdown(
                                [
                                    "Japanese",
                                    "Chinese",
                                    "Korean",
                                    "Spanish",
                                    "French",
                                    "German",
                                    "Italian",
                                    "Portuguese",
                                    "Russian",
                                ],
                                label="Source Language",
                                value=saved_settings.get("batch_input_language", "Japanese"),
                                allow_custom_value=True,
                            )
                            batch_output_language = gr.Dropdown(
                                [
                                    "English",
                                    "Spanish",
                                    "French",
                                    "German",
                                    "Italian",
                                    "Portuguese",
                                    "Russian",
                                    "Japanese",
                                    "Chinese",
                                    "Korean",
                                ],
                                label="Target Language",
                                value=saved_settings.get("batch_output_language", "English"),
                                allow_custom_value=True,
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
                            label="Status", interactive=False, elem_id="batch_status_message"
                        )
                        with gr.Row():
                            batch_process_button = gr.Button("Start Batch Translating", variant="primary")
                            batch_clear_button = gr.ClearButton(  # noqa
                                [input_files, batch_output_gallery, batch_status_message], value="Clear"
                            )

            with gr.TabItem("Config", elem_id="settings-tab-container"):
                config_initial_provider = initial_provider
                config_initial_model_name = initial_model_name
                config_initial_models_choices = initial_models_choices
                config_default_yolo = default_yolo_model

                with gr.Row(elem_id="config-button-row"):
                    save_config_btn = gr.Button("Save Config", variant="primary", scale=3)
                    reset_defaults_btn = gr.Button("Reset Defaults", variant="secondary", scale=1)

                # Assign specific ID for JS targeting
                config_status = gr.Markdown(elem_id="config_status_message")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, elem_id="settings-nav"):
                        nav_buttons = []
                        setting_groups = []
                        nav_button_detection = gr.Button(
                            "Detection", elem_classes=["nav-button", "nav-button-selected"]
                        )
                        nav_buttons.append(nav_button_detection)
                        nav_button_cleaning = gr.Button("Cleaning", elem_classes="nav-button")
                        nav_buttons.append(nav_button_cleaning)
                        nav_button_translation = gr.Button("Translation", elem_classes="nav-button")
                        nav_buttons.append(nav_button_translation)
                        nav_button_rendering = gr.Button("Rendering", elem_classes="nav-button")
                        nav_buttons.append(nav_button_rendering)
                        nav_button_output = gr.Button("Output", elem_classes="nav-button")
                        nav_buttons.append(nav_button_output)
                        nav_button_other = gr.Button("Other", elem_classes="nav-button")
                        nav_buttons.append(nav_button_other)

                    with gr.Column(scale=4, elem_id="config-content-area"):
                        # --- Detection Settings ---
                        with gr.Group(visible=True, elem_classes="settings-group") as group_detection:
                            gr.Markdown("### YOLO Model")
                            config_yolo_model_dropdown = gr.Dropdown(
                                choices=model_choices,
                                label="YOLO Model",
                                value=config_default_yolo,
                                filterable=False,
                                info="Model used for detecting speech bubbles (requires mask segmentation "
                                     "capabilities).",
                            )
                            confidence = gr.Slider(
                                0.1,
                                1.0,
                                value=saved_settings.get("confidence", 0.35),
                                step=0.05,
                                label="Bubble Detection Confidence",
                                info="Lower values detect more bubbles, potentially including false positives.",
                            )
                            config_reading_direction = gr.Radio(
                                choices=["rtl", "ltr"],
                                label="Reading Direction",
                                value=saved_settings.get("reading_direction", "rtl"),
                                info="Order for sorting bubbles (rtl=Manga, ltr=Comics).",
                                elem_id="config_reading_direction",
                            )
                        setting_groups.append(group_detection)

                        # --- Cleaning Settings ---
                        with gr.Group(visible=False, elem_classes="settings-group") as group_cleaning:
                            gr.Markdown("### Mask Cleaning & Refinement")
                            gr.Markdown("#### ROI Expansion")
                            dilation_kernel_size = gr.Slider(
                                1,
                                15,
                                value=saved_settings.get("dilation_kernel_size", 7),
                                step=2,
                                label="Dilation Kernel Size",
                                info="Controls how much the initial bubble detection grows outwards. "
                                     "Increase if blocky artifacts are present.",
                            )
                            dilation_iterations = gr.Slider(
                                1,
                                5,
                                value=saved_settings.get("dilation_iterations", 1),
                                step=1,
                                label="Dilation Iterations",
                                info="Controls how many times to apply the above bubble detection growth.",
                            )
                            gr.Markdown("#### Bubble Interior Isolation")
                            use_otsu_threshold = gr.Checkbox(
                                value=saved_settings.get("use_otsu_threshold", False),
                                label="Use Automatic Thresholding (Otsu)",
                                info="Automatically determine the brightness threshold instead of using the fixed "
                                     "value (210). Recommended for varied lighting.",
                            )
                            min_contour_area = gr.Slider(
                                0,
                                500,
                                value=saved_settings.get("min_contour_area", 50),
                                step=10,
                                label="Min Bubble Area",
                                info="Removes tiny detected regions. Increase to ignore small specks or dots "
                                     "inside the bubble area.",
                            )
                            gr.Markdown("#### Mask Smoothing")
                            closing_kernel_size = gr.Slider(
                                1,
                                15,
                                value=saved_settings.get("closing_kernel_size", 7),
                                step=2,
                                label="Closing Kernel Size",
                                info="Controls the size of gaps between outline and text to fill. "
                                     "Increase to capture text very close to the bubble outline.",
                            )
                            closing_iterations = gr.Slider(
                                1,
                                5,
                                value=saved_settings.get("closing_iterations", 1),
                                step=1,
                                label="Closing Iterations",
                                info="Controls how many times to apply the above gap filling.",
                            )
                            gr.Markdown("#### Edge Constraint")
                            constraint_erosion_kernel_size = gr.Slider(
                                1,
                                15,
                                value=saved_settings.get("constraint_erosion_kernel_size", 9),
                                step=2,
                                label="Constraint Erosion Kernel Size",
                                info="Controls how much to shrink the cleaned mask inwards. "
                                     "Increase if the bubble's outline is being erased.",
                            )
                            constraint_erosion_iterations = gr.Slider(
                                1,
                                3,
                                value=saved_settings.get("constraint_erosion_iterations", 1),
                                step=1,
                                label="Constraint Erosion Iterations",
                                info="Controls how many times to apply the above shrinkage.",
                            )
                        setting_groups.append(group_cleaning)

                        # --- Translation Settings ---
                        with gr.Group(visible=False, elem_classes="settings-group") as group_translation:
                            gr.Markdown("### LLM Settings")
                            provider_selector = gr.Radio(
                                choices=list(settings_manager.PROVIDER_MODELS.keys()),
                                label="Translation Provider",
                                value=config_initial_provider,
                                elem_id="provider_selector",
                            )
                            gemini_api_key = gr.Textbox(
                                label="Gemini API Key",
                                placeholder="Enter Gemini API key (starts with AI...)",
                                type="password",
                                value=saved_settings.get("gemini_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Gemini"),
                                elem_id="gemini_api_key",
                                info="Stored locally in config file. Can also be set via GOOGLE_API_KEY "
                                     "environment variable.",
                            )
                            openai_api_key = gr.Textbox(
                                label="OpenAI API Key",
                                placeholder="Enter OpenAI API key (starts with sk-...)",
                                type="password",
                                value=saved_settings.get("openai_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenAI"),
                                elem_id="openai_api_key",
                                info="Stored locally in config file. Can also be set via OPENAI_API_KEY "
                                     "environment variable.",
                            )
                            anthropic_api_key = gr.Textbox(
                                label="Anthropic API Key",
                                placeholder="Enter Anthropic API key (starts with sk-ant-...)",
                                type="password",
                                value=saved_settings.get("anthropic_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "Anthropic"),
                                elem_id="anthropic_api_key",
                                info="Stored locally in config file. Can also be set via ANTHROPIC_API_KEY "
                                     "environment variable.",
                            )
                            openrouter_api_key = gr.Textbox(
                                label="OpenRouter API Key",
                                placeholder="Enter OpenRouter API key (starts with sk-or-...)",
                                type="password",
                                value=saved_settings.get("openrouter_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenRouter"),
                                elem_id="openrouter_api_key",
                                info="Stored locally in config file. Can also be set via OPENROUTER_API_KEY "
                                     "environment variable.",
                            )
                            openai_compatible_url_input = gr.Textbox(
                                label="OpenAI-compatible URL",
                                placeholder="Enter Base URL (e.g., http://localhost:11434/v1)",
                                type="text",
                                value=saved_settings.get(
                                    "openai_compatible_url", settings_manager.DEFAULT_SETTINGS["openai_compatible_url"]
                                ),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenAI-compatible"),
                                elem_id="openai_compatible_url_input",
                                info="The base URL of your OpenAI-compatible API endpoint.",
                            )
                            openai_compatible_api_key_input = gr.Textbox(
                                label="OpenAI-compatible API Key (Optional)",
                                placeholder="Enter API key if required",
                                type="password",
                                value=saved_settings.get("openai_compatible_api_key", ""),
                                show_copy_button=False,
                                visible=(config_initial_provider == "OpenAI-compatible"),
                                elem_id="openai_compatible_api_key_input",
                                info="Stored locally in config file. Can also be set via "
                                     "OPENAI_COMPATIBLE_API_KEY environment variable.",
                            )
                            config_model_name = gr.Dropdown(
                                choices=config_initial_models_choices,
                                label="Model",
                                value=config_initial_model_name,
                                info="Select the specific model for the chosen provider.",
                                elem_id="config_model_name",
                                allow_custom_value=True,
                            )
                            enable_thinking_checkbox = gr.Checkbox(
                                label="Enable Thinking",
                                value=saved_settings.get("enable_thinking", True),
                                info="Enables 'thinking' capabilities for Gemini 2.5 Flash models.",
                                visible=(
                                    config_initial_provider == "Gemini"
                                    and "gemini-2.5-flash" in config_initial_model_name
                                ),
                                elem_id="enable_thinking_checkbox",
                            )
                            temperature = gr.Slider(
                                0,
                                2.0,
                                value=saved_settings.get("temperature", 0.1),
                                step=0.05,
                                label="Temperature",
                                info="Controls randomness. Lower is more deterministic.",
                                elem_id="config_temperature",
                            )
                            top_p = gr.Slider(
                                0,
                                1,
                                value=saved_settings.get("top_p", 0.95),
                                step=0.05,
                                label="Top P",
                                info="Nucleus sampling parameter.",
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
                            config_translation_mode = gr.Radio(
                                choices=["one-step", "two-step"],
                                label="Translation Mode",
                                value=saved_settings.get(
                                    "translation_mode", settings_manager.DEFAULT_SETTINGS["translation_mode"]
                                ),
                                info=("Method for translation ('one-step' combines OCR/Translate, 'two-step' "
                                      "separates them). 'two-step' might improve translation quality for "
                                      "less-capable LLMs."),
                                elem_id="config_translation_mode",
                            )
                        setting_groups.append(group_translation)

                        # --- Rendering Settings ---
                        with gr.Group(visible=False, elem_classes="settings-group") as group_rendering:
                            gr.Markdown("### Text Rendering")
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
                                value=saved_settings.get("use_subpixel_rendering", False),
                                label="Use Subpixel Rendering",
                                info="Improves text clarity on LCD screens, but can cause color fringing.",
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
                        setting_groups.append(group_rendering)

                        # --- Output Settings ---
                        with gr.Group(visible=False, elem_classes="settings-group") as group_output:
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
                                interactive=saved_settings.get("output_format", "auto") != "png",
                            )
                            png_compression = gr.Slider(
                                0,
                                9,
                                value=saved_settings.get("png_compression", 6),
                                step=1,
                                label="PNG Compression Level (higher = smaller file)",
                                interactive=saved_settings.get("output_format", "auto") != "jpeg",
                            )
                        setting_groups.append(group_output)

                        # --- Other Settings ---
                        with gr.Group(visible=False, elem_classes="settings-group") as group_other:
                            gr.Markdown("### Other")
                            refresh_resources_button = gr.Button(
                                "Refresh Models / Fonts",
                                variant="secondary",
                                elem_classes="config-refresh-button",
                            )
                            verbose = gr.Checkbox(
                                value=saved_settings.get("verbose", False),
                                label="Verbose Logging",
                                info="Enable verbose logging in console.",
                            )
                            cleaning_only_toggle = gr.Checkbox(
                                value=saved_settings.get("cleaning_only", False),
                                label="Cleaning Only Mode",
                                info="Skip translation and text rendering, output only the cleaned speech bubbles.",
                            )
                        setting_groups.append(group_other)

        # --- Define Event Handlers ---
        save_config_inputs = [
            config_yolo_model_dropdown,
            confidence,
            config_reading_direction,
            dilation_kernel_size,
            dilation_iterations,
            use_otsu_threshold,
            min_contour_area,
            closing_kernel_size,
            closing_iterations,
            constraint_erosion_kernel_size,
            constraint_erosion_iterations,
            provider_selector,
            gemini_api_key,
            openai_api_key,
            anthropic_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
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
            input_language,
            output_language,
            font_dropdown,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            enable_thinking_checkbox,
        ]

        reset_outputs = [
            config_yolo_model_dropdown,
            confidence,
            config_reading_direction,
            dilation_kernel_size,
            dilation_iterations,
            use_otsu_threshold,
            min_contour_area,
            closing_kernel_size,
            closing_iterations,
            constraint_erosion_kernel_size,
            constraint_erosion_iterations,
            provider_selector,
            gemini_api_key,
            openai_api_key,
            anthropic_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
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
            input_language,
            output_language,
            font_dropdown,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
            enable_thinking_checkbox,
            config_status,
        ]

        translate_inputs = [
            input_image,
            config_yolo_model_dropdown,
            confidence,
            dilation_kernel_size,
            dilation_iterations,
            use_otsu_threshold,
            min_contour_area,
            closing_kernel_size,
            closing_iterations,
            constraint_erosion_kernel_size,
            constraint_erosion_iterations,
            provider_selector,
            gemini_api_key,
            openai_api_key,
            anthropic_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
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
            enable_thinking_checkbox,
        ]

        batch_inputs = [
            input_files,
            config_yolo_model_dropdown,
            confidence,
            dilation_kernel_size,
            dilation_iterations,
            use_otsu_threshold,
            min_contour_area,
            closing_kernel_size,
            closing_iterations,
            constraint_erosion_kernel_size,
            constraint_erosion_iterations,
            provider_selector,
            gemini_api_key,
            openai_api_key,
            anthropic_api_key,
            openrouter_api_key,
            openai_compatible_url_input,
            openai_compatible_api_key_input,
            config_model_name,
            temperature,
            top_p,
            top_k,
            config_reading_direction,
            config_translation_mode,
            batch_input_language,
            batch_output_language,
            batch_font_dropdown,
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
            enable_thinking_checkbox,
        ]

        # Config Tab Navigation & Updates
        output_components_for_switch = setting_groups + nav_buttons
        nav_button_detection.click(
            fn=lambda idx=0: utils.switch_settings_view(idx, setting_groups, nav_buttons),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_cleaning.click(
            fn=lambda idx=1: utils.switch_settings_view(idx, setting_groups, nav_buttons),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_translation.click(
            fn=lambda idx=2: utils.switch_settings_view(idx, setting_groups, nav_buttons),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_rendering.click(
            fn=lambda idx=3: utils.switch_settings_view(idx, setting_groups, nav_buttons),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_output.click(
            fn=lambda idx=4: utils.switch_settings_view(idx, setting_groups, nav_buttons),
            outputs=output_components_for_switch,
            queue=False,
        )
        nav_button_other.click(
            fn=lambda idx=5: utils.switch_settings_view(idx, setting_groups, nav_buttons),
            outputs=output_components_for_switch,
            queue=False,
        )

        output_format.change(
            fn=callbacks.handle_output_format_change,
            inputs=output_format,
            outputs=[jpeg_quality, png_compression],
            queue=False,
        )

        provider_selector.change(
            fn=callbacks.handle_provider_change,
            inputs=[provider_selector, temperature],
            outputs=[
                gemini_api_key,
                openai_api_key,
                anthropic_api_key,
                openrouter_api_key,
                openai_compatible_url_input,
                openai_compatible_api_key_input,
                config_model_name,
                temperature,
                top_k,
            ],
            queue=False,
        ).then(  # Trigger model fetch *after* provider change updates visibility etc.
            fn=lambda prov, url, key: (
                utils.fetch_and_update_compatible_models(url, key)
                if prov == "OpenAI-compatible"
                else (utils.fetch_and_update_openrouter_models() if prov == "OpenRouter" else gr.update())
            ),
            inputs=[provider_selector, openai_compatible_url_input, openai_compatible_api_key_input],
            outputs=[config_model_name],
            queue=True,  # Allow fetching to happen in the background
        )

        config_model_name.change(
            fn=callbacks.handle_model_change,
            inputs=[provider_selector, config_model_name, temperature],
            outputs=[temperature, top_k, enable_thinking_checkbox],
            queue=False,
        )

        # Config Save/Reset Buttons
        save_config_btn.click(
            fn=callbacks.handle_save_config_click, inputs=save_config_inputs, outputs=[config_status], queue=False
        ).then(fn=None, inputs=None, outputs=None, js=js_status_fade, queue=False)

        reset_defaults_btn.click(
            fn=functools.partial(
                callbacks.handle_reset_defaults_click, models_dir=models_dir, fonts_base_dir=fonts_base_dir
            ),
            inputs=[],
            outputs=reset_outputs,
            queue=False,
        ).then(fn=None, inputs=None, outputs=None, js=js_status_fade, queue=False)

        # Refresh Button
        refresh_outputs = [config_yolo_model_dropdown, font_dropdown, batch_font_dropdown]
        refresh_resources_button.click(
            fn=functools.partial(
                callbacks.handle_refresh_resources_click, models_dir=models_dir, fonts_base_dir=fonts_base_dir
            ),
            inputs=[],
            outputs=refresh_outputs,
            js=js_refresh_button_processing,
        ).then(fn=None, inputs=None, outputs=None, js=js_refresh_button_reset)

        # Translator Tab Button
        translate_button.click(
            fn=functools.partial(
                callbacks.update_button_state,
                processing=True,
                button_text_processing="Translating...",
                button_text_idle="Translate",
            ),
            outputs=translate_button,
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
        ).then(
            fn=functools.partial(
                callbacks.update_button_state,
                processing=False,
                button_text_processing="Translating...",
                button_text_idle="Translate",
            ),
            outputs=translate_button,
            queue=False,
        ).then(
            fn=None, inputs=None, outputs=None, queue=False
        )

        # Batch Tab Button
        batch_process_button.click(
            fn=functools.partial(
                callbacks.update_button_state,
                processing=True,
                button_text_processing="Processing...",
                button_text_idle="Start Batch Translating",
            ),
            outputs=batch_process_button,
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
        ).then(
            fn=functools.partial(
                callbacks.update_button_state,
                processing=False,
                button_text_processing="Processing...",
                button_text_idle="Start Batch Translating",
            ),
            outputs=batch_process_button,
            queue=False,
        ).then(
            fn=None, inputs=None, outputs=None, queue=False
        )

        app.load(
            fn=callbacks.handle_app_load,
            inputs=[provider_selector, openai_compatible_url_input, openai_compatible_api_key_input],
            outputs=[config_model_name],
            queue=False,
        )

    return app
