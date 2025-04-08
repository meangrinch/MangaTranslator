import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
from typing import Optional, Union
from PIL import Image
import torch

import core
from core.config import MangaTranslatorConfig, DetectionConfig, CleaningConfig, TranslationConfig, RenderingConfig, OutputConfig
from core.generate_manga import translate_and_render, batch_translate_images
from core.common_utils import validate_core_inputs, ValidationError
import config_utils
import ui_utils
from ui_utils import switch_settings_view

def custom_except_hook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, gr.Error) or issubclass(exc_type, gr.CancelledError):
        print(f"Gradio error: {exc_value}")
    else:
        import traceback
        print("Non-Gradio Error Caught by Custom Hook:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = custom_except_hook

# Helps prevent fragmentation OOM errors on some GPUs
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Constants moved to config_utils.py
ERROR_PREFIX = "❌ Error: "
SUCCESS_PREFIX = "✅ "

parser = argparse.ArgumentParser(description='Manga Translator')
parser.add_argument('--models', type=str, default='./models', help='Directory containing YOLO model files')
parser.add_argument('--fonts', type=str, default='./fonts', help='Base directory containing font pack subdirectories')
parser.add_argument('--open-browser', action='store_true', help='Automatically open the default web browser')
parser.add_argument('--port', type=int, default=7676, help='Port number for the web UI')
parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if CUDA is available')
args = parser.parse_args()

MODELS_DIR = Path(args.models)
FONTS_BASE_DIR = Path(args.fonts)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FONTS_BASE_DIR, exist_ok=True)

target_device = torch.device('cpu') if args.cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device_info_str = "CPU"
if target_device.type == 'cuda':
    try:
        gpu_name = torch.cuda.get_device_name(0)
        device_info_str = f"CUDA ({gpu_name}, ID: 0)"
    except Exception as e:
        device_info_str = "CUDA (Unknown GPU Name)"
print(f"Using device: {device_info_str.upper()}")
print(f"PyTorch version: {torch.__version__}")
print(f"MangaTranslator version: {core.__version__}")

def translate_manga(
    image: Union[str, Path, Image.Image],
    selected_yolo_model: str,
    detection_cfg: DetectionConfig,
    cleaning_cfg: CleaningConfig,
    translation_cfg: TranslationConfig,
    rendering_cfg: RenderingConfig,
    output_cfg: OutputConfig,
    cleaning_only: bool = False, # Added for cleaning only mode
    verbose: bool = False,
    device: Optional[torch.device] = None
):
    """
    Process a single manga image with translation.
    Handles UI validation, core validation, calls the core pipeline, and generates status.
    """
    start_time = time.time()
    translated_image = None

    # --- UI Validation ---
    img_valid, img_msg = ui_utils.validate_image(image)
    if not img_valid:
        raise gr.Error(f"{ERROR_PREFIX}{img_msg}", print_exception=False)

    api_valid, api_msg = ui_utils.validate_api_key(
        translation_cfg.gemini_api_key if translation_cfg.provider == "Gemini" else
        translation_cfg.openai_api_key if translation_cfg.provider == "OpenAI" else
        translation_cfg.anthropic_api_key if translation_cfg.provider == "Anthropic" else
        translation_cfg.openrouter_api_key if translation_cfg.provider == "OpenRouter" else
        translation_cfg.openai_compatible_api_key if translation_cfg.provider == "OpenAI-compatible" else "",
        translation_cfg.provider
    )
    if not api_valid:
        raise gr.Error(f"{ERROR_PREFIX}{api_msg}", print_exception=False)
    if translation_cfg.provider == "OpenAI-compatible":
        if not translation_cfg.openai_compatible_url:
             raise gr.Error(f"{ERROR_PREFIX}OpenAI-Compatible URL is required.", print_exception=False)
        if not translation_cfg.openai_compatible_url.startswith(("http://", "https://")):
             raise gr.Error(f"{ERROR_PREFIX}Invalid OpenAI-Compatible URL format. Must start with http:// or https://", print_exception=False)

    # --- Core Validation ---
    try:
        # Pass font pack name to core validation
        rendering_cfg_for_val = RenderingConfig(
            font_dir=rendering_cfg.font_dir,
            max_font_size=rendering_cfg.max_font_size,
            min_font_size=rendering_cfg.min_font_size,
            line_spacing=rendering_cfg.line_spacing,
            font_hinting=rendering_cfg.font_hinting,
        )
        yolo_model_path, font_dir_path = validate_core_inputs(
            translation_cfg=translation_cfg,
            rendering_cfg=rendering_cfg_for_val,
            selected_yolo_model_name=selected_yolo_model,
            models_dir=MODELS_DIR,
            fonts_base_dir=FONTS_BASE_DIR
        )
    except (FileNotFoundError, ValidationError, ValueError) as e:
        raise gr.Error(f"{ERROR_PREFIX}Configuration Error: {str(e)}", print_exception=False)

    temp_image_path = None
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if isinstance(image, (str, Path)):
            input_path = Path(image)
            original_ext = input_path.suffix.lower()
            image_path_for_processing = str(input_path)
        else:
            original_ext = '.png'
            temp_image_path = Path(tempfile.mktemp(suffix='.png'))
            image.save(temp_image_path)
            image_path_for_processing = str(temp_image_path)

        output_format = output_cfg.output_format
        if output_format == "auto":
            output_ext = original_ext if original_ext in ['.jpg', '.jpeg', '.png', '.webp'] else '.png' # Default to png if auto fails
        elif output_format == "png":
            output_ext = '.png'
        elif output_format == "jpeg":
            output_ext = '.jpg'
        else:
            output_ext = original_ext if original_ext in ['.jpg', '.jpeg', '.png', '.webp'] else '.png'

        output_dir = Path("./output")
        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir / f"MangaTranslator_{timestamp}{output_ext}"

        if output_ext in ['.jpg', '.jpeg']:
            image_mode = "RGB"
        else:
            image_mode = "RGBA" # Default to RGBA for PNG/WEBP

        output_cfg_final = OutputConfig(
            jpeg_quality=output_cfg.jpeg_quality,
            png_compression=output_cfg.png_compression,
            image_mode=image_mode,
            output_format=output_format
        )

        config = MangaTranslatorConfig(
            yolo_model_path=str(yolo_model_path),
            verbose=verbose,
            device=device or target_device,
            detection=detection_cfg,
            cleaning=cleaning_cfg,
            translation=translation_cfg,
            rendering=RenderingConfig(
                 font_dir=str(font_dir_path),
                 max_font_size=rendering_cfg.max_font_size,
                 min_font_size=rendering_cfg.min_font_size,
                 line_spacing=rendering_cfg.line_spacing,
                 use_subpixel_rendering=rendering_cfg.use_subpixel_rendering,
                 font_hinting=rendering_cfg.font_hinting,
                 use_ligatures=rendering_cfg.use_ligatures
            ),
            output=output_cfg_final,
            cleaning_only=cleaning_only # Pass the flag here
        )

        translated_image = translate_and_render(
            image_path=image_path_for_processing,
            config=config,
            output_path=save_path
        )

        if not translated_image:
             raise gr.Error("Translation process failed to return an image.", print_exception=False)

        width, height = translated_image.size
        processing_time = time.time() - start_time

        provider = config.translation.provider
        model_name = config.translation.model_name
        temp_val = config.translation.temperature
        top_p_val = config.translation.top_p
        top_k_val = config.translation.top_k

        llm_params_str = f"• LLM Params: Temp={temp_val:.2f}, Top-P={top_p_val:.2f}"
        param_notes = ""
        if provider == "Gemini":
            llm_params_str += f", Top-K={top_k_val}"
        elif provider == "Anthropic":
            llm_params_str += f", Top-K={top_k_val}"
            param_notes = " (Temp clamped <= 1.0)"
        elif provider == "OpenAI":
            param_notes = " (Top-K N/A)"
        elif provider == "OpenRouter":
            is_openai_model = "openai/" in model_name or model_name.startswith("gpt-")
            is_anthropic_model = "anthropic/" in model_name or model_name.startswith("claude-")
            if is_openai_model or is_anthropic_model:
                param_notes = " (Temp clamped <= 1.0, Top-K N/A)"
            else:
                llm_params_str += f", Top-K={top_k_val}"
        llm_params_str += param_notes

        processing_mode_str = "Cleaning Only" if config.cleaning_only else "Translation"
        msg_parts = [
            f"{SUCCESS_PREFIX}{processing_mode_str} completed!\n",
            f"• Image size: {width}x{height} pixels\n",
            f"• Provider: {provider}\n",
            f"• Model: {model_name}\n",
            f"• Source language: {config.translation.input_language}\n",
            f"• Target language: {config.translation.output_language}\n",
            f"• Reading Direction: {config.translation.reading_direction.upper()}\n",
            f"• Font pack: {font_dir_path.name}\n",
            f"• YOLO Model: {selected_yolo_model}\n",
            f"• Cleaning Params: DKS:{config.cleaning.dilation_kernel_size}, DI:{config.cleaning.dilation_iterations}, Otsu:{config.cleaning.use_otsu_threshold}, ",
            f"MCA:{config.cleaning.min_contour_area}, CKS:{config.cleaning.closing_kernel_size}, CI:{config.cleaning.closing_iterations}, ",
            f"CEKS:{config.cleaning.constraint_erosion_kernel_size}, CEI:{config.cleaning.constraint_erosion_iterations}\n"
        ]
        if not config.cleaning_only:
            msg_parts.append(f"{llm_params_str}\n")
        msg_parts.append(f"• Output format: {output_cfg_final.output_format}\n")
        if output_ext in ['.jpg', '.jpeg']:
            msg_parts.append(f"• JPEG quality: {config.output.jpeg_quality}\n")
        if output_ext == '.png':
             msg_parts.append(f"• PNG compression: {config.output.png_compression}\n")
        msg_parts.extend([
            f"• Processing time: {processing_time:.2f} seconds\n",
            f"• Saved to: {save_path}"
        ])
        translator_status_msg = "".join(msg_parts)

        result = translated_image.copy()
        return result, translator_status_msg

    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"{ERROR_PREFIX}An unexpected error occurred: {str(e)}", print_exception=False)
    finally:
        # Clean up temporary file if created
        if temp_image_path and temp_image_path.exists():
            try:
                os.remove(temp_image_path)
            except Exception as e_clean:
                print(f"Warning: Could not remove temporary image file {temp_image_path}: {e_clean}")

def process_batch(
    input_dir_or_files: Union[str, list],
    selected_yolo_model: str,
    detection_cfg: DetectionConfig,
    cleaning_cfg: CleaningConfig,
    translation_cfg: TranslationConfig,
    rendering_cfg: RenderingConfig,
    output_cfg: OutputConfig,
    cleaning_only: bool = False, # Added for cleaning only mode
    verbose: bool = False,
    progress=gr.Progress(),
    device: Optional[torch.device] = None
):
    """
    Process a batch of manga images with translation.
    Handles UI validation, core validation, calls the core batch pipeline, and generates status.
    """
    start_time = time.time()

    # --- UI Validation (API Key / URL) ---
    api_valid, api_msg = ui_utils.validate_api_key(
        translation_cfg.gemini_api_key if translation_cfg.provider == "Gemini" else
        translation_cfg.openai_api_key if translation_cfg.provider == "OpenAI" else
        translation_cfg.anthropic_api_key if translation_cfg.provider == "Anthropic" else
        translation_cfg.openrouter_api_key if translation_cfg.provider == "OpenRouter" else
        translation_cfg.openai_compatible_api_key if translation_cfg.provider == "OpenAI-compatible" else "",
        translation_cfg.provider
    )
    if not api_valid:
        raise gr.Error(f"{ERROR_PREFIX}{api_msg}", print_exception=False)
    if translation_cfg.provider == "OpenAI-compatible":
        if not translation_cfg.openai_compatible_url:
             raise gr.Error(f"{ERROR_PREFIX}OpenAI-Compatible URL is required.", print_exception=False)
        if not translation_cfg.openai_compatible_url.startswith(("http://", "https://")):
             raise gr.Error(f"{ERROR_PREFIX}Invalid OpenAI-Compatible URL format. Must start with http:// or https://", print_exception=False)

    # --- Core Validation ---
    try:
        # Pass font pack name to core validation
        rendering_cfg_for_val = RenderingConfig(
            font_dir=rendering_cfg.font_dir,
            max_font_size=rendering_cfg.max_font_size,
            min_font_size=rendering_cfg.min_font_size,
            line_spacing=rendering_cfg.line_spacing,
            font_hinting=rendering_cfg.font_hinting,
        )
        yolo_model_path, font_dir_path = validate_core_inputs(
            translation_cfg=translation_cfg,
            rendering_cfg=rendering_cfg_for_val,
            selected_yolo_model_name=selected_yolo_model,
            models_dir=MODELS_DIR,
            fonts_base_dir=FONTS_BASE_DIR
        )
    except (FileNotFoundError, ValidationError, ValueError) as e:
        raise gr.Error(f"{ERROR_PREFIX}Configuration Error: {str(e)}", print_exception=False)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path("./output") / timestamp

    def update_progress(value, desc="Processing..."):
        progress(value, desc=desc)

    temp_dir_path_obj = None # For cleaning up temporary directory if needed
    try:
        process_dir = None
        if isinstance(input_dir_or_files, list):
            if not input_dir_or_files:
                raise gr.Error(f"{ERROR_PREFIX}No files provided for batch processing.", print_exception=False)

            temp_dir_path_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir_path_obj.name)

            image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            image_files_to_copy = []
            for f_path in input_dir_or_files:
                p = Path(f_path)
                if p.is_file() and p.suffix.lower() in image_extensions:
                    # Basic validation before copying
                    img_valid, img_msg = ui_utils.validate_image(p)
                    if img_valid:
                        image_files_to_copy.append(p)
                    else:
                        print(f"Warning: Skipping invalid file '{p.name}': {img_msg}")


            if not image_files_to_copy:
                 raise gr.Error(f"{ERROR_PREFIX}No valid image files found in the selection.", print_exception=False)

            progress(0.1, desc=f"Preparing {len(image_files_to_copy)} files...")
            for img_file in image_files_to_copy:
                try:
                    shutil.copy2(img_file, temp_dir_path / img_file.name)
                except Exception as copy_err:
                    print(f"Warning: Could not copy file {img_file.name}: {copy_err}")
            process_dir = temp_dir_path

        elif isinstance(input_dir_or_files, str):
             input_path = Path(input_dir_or_files)
             if not input_path.exists() or not input_path.is_dir():
                 raise gr.Error(f"{ERROR_PREFIX}Input directory '{input_dir_or_files}' does not exist or is not a directory.", print_exception=False)
             process_dir = input_path
        else:
             raise gr.Error(f"{ERROR_PREFIX}Invalid input type for batch processing.", print_exception=False)

        if not process_dir:
             raise gr.Error(f"{ERROR_PREFIX}Could not determine processing directory.", print_exception=False)

        config = MangaTranslatorConfig(
            yolo_model_path=str(yolo_model_path),
            verbose=verbose,
            device=device or target_device,
            detection=detection_cfg,
            cleaning=cleaning_cfg,
            translation=translation_cfg,
            rendering=RenderingConfig(
                 font_dir=str(font_dir_path),
                 max_font_size=rendering_cfg.max_font_size,
                 min_font_size=rendering_cfg.min_font_size,
                 line_spacing=rendering_cfg.line_spacing,
                 use_subpixel_rendering=rendering_cfg.use_subpixel_rendering,
                 font_hinting=rendering_cfg.font_hinting,
                 use_ligatures=rendering_cfg.use_ligatures
            ),
            output=output_cfg,
            cleaning_only=cleaning_only # Pass the flag here
        )

        progress(0.2, desc="Starting batch processing...")

        results = batch_translate_images(
            input_dir=process_dir,
            config=config,
            output_dir=output_path,
            progress_callback=update_progress
        )

        progress(0.95, desc="Processing complete, preparing results...")

        success_count = results["success_count"]
        error_count = results["error_count"]
        errors = results["errors"]
        total_images = success_count + error_count

        gallery_images = []
        if output_path.exists():
            processed_files = list(output_path.glob("*.*"))
            # Sort naturally (e.g., page_1, page_2, page_10)
            processed_files.sort(key=lambda x: tuple(int(part) if part.isdigit() else part for part in re.split(r'(\d+)', x.stem)))
            gallery_images = [str(file_path) for file_path in processed_files]

        processing_time = time.time() - start_time
        seconds_per_image = processing_time / success_count if success_count > 0 else 0

        provider = config.translation.provider
        model_name = config.translation.model_name
        temp_val = config.translation.temperature
        top_p_val = config.translation.top_p
        top_k_val = config.translation.top_k

        llm_params_str = f"• LLM Params: Temp={temp_val:.2f}, Top-P={top_p_val:.2f}"
        param_notes = ""
        if provider == "Gemini":
            llm_params_str += f", Top-K={top_k_val}"
        elif provider == "Anthropic":
            llm_params_str += f", Top-K={top_k_val}"
            param_notes = " (Temp clamped <= 1.0)"
        elif provider == "OpenAI":
            param_notes = " (Top-K N/A)"
        elif provider == "OpenRouter":
            is_openai_model = "openai/" in model_name or model_name.startswith("gpt-")
            is_anthropic_model = "anthropic/" in model_name or model_name.startswith("claude-")
            if is_openai_model or is_anthropic_model:
                param_notes = " (Temp clamped <= 1.0, Top-K N/A)"
            else:
                llm_params_str += f", Top-K={top_k_val}"
        llm_params_str += param_notes

        error_summary = ""
        if error_count > 0:
            error_summary = f"\n• Warning: {error_count} image(s) failed to process."
            if verbose and errors:
                for filename, error_msg in errors.items():
                    print(f"Error processing {filename}: {error_msg}")

        processing_mode_str = "Cleaning Only" if config.cleaning_only else "Translation"
        msg_parts = [
            f"{SUCCESS_PREFIX}Batch {processing_mode_str.lower()} completed!\n",
            f"• Provider: {provider}\n",
            f"• Model: {model_name}\n",
            f"• Source language: {config.translation.input_language}\n",
            f"• Target language: {config.translation.output_language}\n",
            f"• Reading Direction: {config.translation.reading_direction.upper()}\n",
            f"• Font pack: {font_dir_path.name}\n",
            f"• YOLO Model: {selected_yolo_model}\n",
            f"• Cleaning Params: DKS:{config.cleaning.dilation_kernel_size}, DI:{config.cleaning.dilation_iterations}, Otsu:{config.cleaning.use_otsu_threshold}, ",
            f"MCA:{config.cleaning.min_contour_area}, CKS:{config.cleaning.closing_kernel_size}, CI:{config.cleaning.closing_iterations}, ",
            f"CEKS:{config.cleaning.constraint_erosion_kernel_size}, CEI:{config.cleaning.constraint_erosion_iterations}\n"
        ]
        if not config.cleaning_only:
            msg_parts.append(f"{llm_params_str}\n")
        msg_parts.append(f"• Output format: {config.output.output_format}\n")
        if config.output.output_format == 'jpeg':
             msg_parts.append(f"• JPEG quality: {config.output.jpeg_quality}\n")
        if config.output.output_format == 'png':
             msg_parts.append(f"• PNG compression: {config.output.png_compression}\n")
        msg_parts.extend([
            f"• Successful translations: {success_count}/{total_images}{error_summary}\n",
            f"• Total processing time: {processing_time:.2f} seconds ({seconds_per_image:.2f} seconds/image)\n",
            f"• Saved to: {output_path}"
        ])
        batch_status_msg = "".join(msg_parts)

        progress(1.0, desc="Batch processing complete")
        return gallery_images, batch_status_msg

    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"{ERROR_PREFIX}An unexpected error occurred during batch processing: {str(e)}", print_exception=False)
    finally:
        # Clean up temporary directory if created
        if temp_dir_path_obj:
            try:
                temp_dir_path_obj.cleanup()
            except Exception as e_clean:
                print(f"Warning: Could not clean up temporary directory {temp_dir_path_obj.name}: {e_clean}")

js_credits = """
function() {
    const footer = document.querySelector('footer');
    if (footer) {
        const newContent = document.createElement('div');
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

with gr.Blocks(title="Manga Translator", js=js_credits, css_paths="style.css") as app:

    gr.Markdown("# Manga Translator")

    model_choices = ui_utils.get_available_models(MODELS_DIR)
    font_choices, initial_default_font = ui_utils.get_available_font_packs(FONTS_BASE_DIR)
    saved_settings = config_utils.get_saved_settings()

    saved_yolo_model = saved_settings.get('yolo_model')
    default_yolo_model = saved_yolo_model if saved_yolo_model in model_choices else (model_choices[0] if model_choices else None)

    saved_font_pack = saved_settings.get('font_pack')
    default_font = saved_font_pack if saved_font_pack in font_choices else (initial_default_font if initial_default_font else None)
    batch_saved_font_pack = saved_settings.get('batch_font_pack')
    batch_default_font = batch_saved_font_pack if batch_saved_font_pack in font_choices else (initial_default_font if initial_default_font else None)

    initial_provider = saved_settings.get("provider", config_utils.DEFAULT_SETTINGS["provider"])
    initial_model_name = saved_settings.get("model_name")

    if initial_provider == "OpenRouter" or initial_provider == "OpenAI-compatible":
         initial_models_choices = [initial_model_name] if initial_model_name else []
    else:
         initial_models_choices = config_utils.PROVIDER_MODELS.get(initial_provider, [])

    with gr.Tabs():
        with gr.TabItem("Translator"):

            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        type="filepath",
                        label="Upload Image",
                        show_download_button=False,
                        image_mode="RGBA",
                        elem_id="translator_input_image"
                    )

                    font_dropdown = gr.Dropdown(
                        choices=font_choices,
                        label="Text Font",
                        value=default_font,
                        filterable=False
                    )

                    with gr.Accordion("Translation Settings", open=True):
                        input_language = gr.Dropdown(
                            ["Japanese", "Chinese", "Korean", "Spanish", "French", "German", "Italian", "Portuguese", "Russian"],
                            label="Source Language",
                            value=saved_settings.get("input_language", "Japanese")
                        )
                        output_language = gr.Dropdown(
                            ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Japanese", "Chinese", "Korean"],
                            label="Target Language",
                            value=saved_settings.get("output_language", "English")
                        )

                with gr.Column(scale=1):
                    output_image = gr.Image(
                        type="pil",
                        label="Translated Image",
                        interactive=False,
                        elem_id="translator_output_image"
                    )
                    status_message = gr.Textbox(label="Status", interactive=False)

                    with gr.Row():
                        translate_button = gr.Button("Translate", variant="primary")
                        clear_button = gr.ClearButton(
                            [input_image, output_image, status_message],
                            value="Clear"
                        )

        with gr.TabItem("Batch"):

            with gr.Row():
                with gr.Column(scale=1):
                    input_files = gr.Files(
                        label="Upload Images or Folder",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath"
                    )

                    batch_font_dropdown = gr.Dropdown(
                        choices=font_choices,
                        label="Text Font",
                        value=batch_default_font,
                        filterable=False
                    )

                    with gr.Accordion("Translation Settings", open=True):
                         batch_input_language = gr.Dropdown(
                             ["Japanese", "Chinese", "Korean", "Spanish", "French", "German", "Italian", "Portuguese", "Russian"],
                             label="Source Language",
                             value=saved_settings.get("batch_input_language", "Japanese")
                         )
                         batch_output_language = gr.Dropdown(
                             ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Japanese", "Chinese", "Korean"],
                             label="Target Language",
                             value=saved_settings.get("batch_output_language", "English")
                         )

                with gr.Column(scale=1):
                    batch_output_gallery = gr.Gallery(
                        label="Translated Images",
                        show_label=True,
                        columns=4,
                        rows=2,
                        height="auto",
                        object_fit="contain"
                    )
                    batch_status_message = gr.Textbox(label="Status", interactive=False)

                    with gr.Row():
                        batch_process_button = gr.Button("Start Batch Translating", variant="primary")
                        batch_clear_button = gr.ClearButton(
                            [input_files, batch_output_gallery, batch_status_message],
                            value="Clear"
                        )

        with gr.TabItem("Config", elem_id="settings-tab-container"):
            # Load settings again here to ensure components get the latest values
            saved_settings_config = config_utils.get_saved_settings()
            config_initial_provider = saved_settings_config.get("provider", config_utils.DEFAULT_SETTINGS["provider"])
            config_initial_model_name = saved_settings_config.get("model_name")

            if config_initial_provider == "OpenRouter" or config_initial_provider == "OpenAI-compatible":
                 config_initial_models_choices = [config_initial_model_name] if config_initial_model_name else []
            else:
                 config_initial_models_choices = config_utils.PROVIDER_MODELS.get(config_initial_provider, [])

            config_initial_yolo = saved_settings_config.get('yolo_model')
            config_default_yolo = config_initial_yolo if config_initial_yolo in model_choices else (model_choices[0] if model_choices else None)

            config_initial_font = saved_settings_config.get('font_pack')
            config_default_font = config_initial_font if config_initial_font in font_choices else (initial_default_font if initial_default_font else None)


            with gr.Row(elem_id="config-button-row"):
                save_config_btn = gr.Button("Save Config", variant="primary", scale=3)
                reset_defaults_btn = gr.Button("Reset Defaults", variant="secondary", scale=1)

            config_status = gr.Markdown(elem_id="config_status")

            with gr.Row(equal_height=False):

                with gr.Column(scale=1, elem_id="settings-nav"):
                    nav_buttons = []
                    setting_groups = []

                    nav_button_detection = gr.Button("Detection", elem_classes=["nav-button", "nav-button-selected"])
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

                    with gr.Group(visible=True, elem_classes="settings-group") as group_detection:
                        gr.Markdown("### YOLO Model")
                        config_yolo_model_dropdown = gr.Dropdown(
                            choices=model_choices, label="YOLO Model",
                            value=config_default_yolo,
                            filterable=False,
                            info="Model used for detecting speech bubbles (requires mask segmentation capabilities)."
                        )
                        confidence = gr.Slider(
                            0.1, 1.0, value=saved_settings_config.get("confidence", 0.35), step=0.05,
                            label="Bubble Detection Confidence",
                            info="Lower values detect more bubbles, potentially including false positives."
                        )
                        config_reading_direction = gr.Radio(
                            choices=["rtl", "ltr"],
                            label="Reading Direction",
                            value=saved_settings_config.get("reading_direction", "rtl"),
                            info="Order for sorting bubbles (rtl=Manga, ltr=Comics).",
                            elem_id="config_reading_direction"
                        )
                    setting_groups.append(group_detection)

                    with gr.Group(visible=False, elem_classes="settings-group") as group_cleaning:
                        gr.Markdown("### Mask Cleaning & Refinement")
                        gr.Markdown("#### ROI Expansion")
                        dilation_kernel_size = gr.Slider(
                            1, 15, value=saved_settings_config.get("dilation_kernel_size", 7), step=2,
                            label="Dilation Kernel Size", info="Controls how much the initial bubble detection grows outwards. Increase if blocky artifacts are present."
                        )
                        dilation_iterations = gr.Slider(
                            1, 5, value=saved_settings_config.get("dilation_iterations", 1), step=1,
                            label="Dilation Iterations", info="Controls how many times to apply the above bubble detection growth."
                        )
                        gr.Markdown("#### Bubble Interior Isolation")
                        use_otsu_threshold = gr.Checkbox(
                            value=saved_settings_config.get("use_otsu_threshold", False),
                            label="Use Automatic Thresholding (Otsu)", info="Automatically determine the brightness threshold instead of using the fixed value (210). Recommended for varied lighting."
                        )
                        min_contour_area = gr.Slider(
                            0, 500, value=saved_settings_config.get("min_contour_area", 50), step=10,
                            label="Min Bubble Area", info="Removes tiny detected regions. Increase to ignore small specks or dots inside the bubble area."
                        )
                        gr.Markdown("#### Mask Smoothing")
                        closing_kernel_size = gr.Slider(
                            1, 15, value=saved_settings_config.get("closing_kernel_size", 7), step=2,
                            label="Closing Kernel Size", info="Controls the size of gaps between outline and text to fill. Increase to capture text very close to the bubble outline."
                        )
                        closing_iterations = gr.Slider(
                            1, 5, value=saved_settings_config.get("closing_iterations", 1), step=1,
                            label="Closing Iterations", info="Controls how many times to apply the above gap filling."
                        )
                        gr.Markdown("#### Edge Constraint")
                        constraint_erosion_kernel_size = gr.Slider(
                            1, 15, value=saved_settings_config.get("constraint_erosion_kernel_size", 9), step=2,
                            label="Constraint Erosion Kernel Size", info="Controls how much to shrink the cleaned mask inwards. Increase if the bubble's outline is being erased."
                        )
                        constraint_erosion_iterations = gr.Slider(
                            1, 3, value=saved_settings_config.get("constraint_erosion_iterations", 1), step=1,
                            label="Constraint Erosion Iterations", info="Controls how many times to apply the above shrinkage."
                        )
                    setting_groups.append(group_cleaning)

                    with gr.Group(visible=False, elem_classes="settings-group") as group_translation:
                        gr.Markdown("### LLM Settings")
                        provider_selector = gr.Radio(
                            choices=list(config_utils.PROVIDER_MODELS.keys()),
                            label="Translation Provider",
                            value=config_initial_provider,
                            elem_id="provider_selector"
                        )
                        gemini_api_key = gr.Textbox(
                            label="Gemini API Key", placeholder="Enter Gemini API key (starts with AI...)", type="password",
                            value=saved_settings_config.get('gemini_api_key', ''), show_copy_button=False,
                            visible=(config_initial_provider == "Gemini"), elem_id="gemini_api_key",
                            info="Stored locally in config file. Can also be set via GOOGLE_API_KEY environment variable."
                        )
                        openai_api_key = gr.Textbox(
                            label="OpenAI API Key", placeholder="Enter OpenAI API key (starts with sk-...)", type="password",
                            value=saved_settings_config.get('openai_api_key', ''), show_copy_button=False,
                            visible=(config_initial_provider == "OpenAI"), elem_id="openai_api_key",
                            info="Stored locally in config file. Can also be set via OPENAI_API_KEY environment variable."
                        )
                        anthropic_api_key = gr.Textbox(
                            label="Anthropic API Key", placeholder="Enter Anthropic API key (starts with sk-ant-...)", type="password",
                            value=saved_settings_config.get('anthropic_api_key', ''), show_copy_button=False,
                            visible=(config_initial_provider == "Anthropic"), elem_id="anthropic_api_key",
                            info="Stored locally in config file. Can also be set via ANTHROPIC_API_KEY environment variable."
                        )
                        openrouter_api_key = gr.Textbox(
                            label="OpenRouter API Key", placeholder="Enter OpenRouter API key (starts with sk-or-...)", type="password",
                            value=saved_settings_config.get('openrouter_api_key', ''), show_copy_button=False,
                            visible=(config_initial_provider == "OpenRouter"), elem_id="openrouter_api_key",
                            info="Stored locally in config file. Can also be set via OPENROUTER_API_KEY environment variable."
                        )
                        openai_compatible_url_input = gr.Textbox(
                           label="OpenAI-compatible URL", placeholder="Enter Base URL (e.g., http://localhost:11434/v1)", type="text",
                           value=saved_settings_config.get('openai_compatible_url', config_utils.DEFAULT_SETTINGS['openai_compatible_url']), show_copy_button=False,
                           visible=(config_initial_provider == "OpenAI-compatible"), elem_id="openai_compatible_url_input",
                           info="The base URL of your OpenAI-compatible API endpoint."
                        )
                        openai_compatible_api_key_input = gr.Textbox(
                           label="OpenAI-compatible API Key (Optional)", placeholder="Enter API key if required", type="password",
                           value=saved_settings_config.get('openai_compatible_api_key', ''), show_copy_button=False,
                           visible=(config_initial_provider == "OpenAI-compatible"), elem_id="openai_compatible_api_key_input",
                           info="Stored locally in config file. Can also be set via OPENAI_COMPATIBLE_API_KEY environment variable."
                        )
                        config_model_name = gr.Dropdown(
                            choices=config_initial_models_choices,
                            label="Model",
                            value=config_initial_model_name,
                            info="Select the specific model for the chosen provider.",
                            elem_id="config_model_name",
                            allow_custom_value=True
                        )
                        temperature = gr.Slider(
                            0, 2.0, value=saved_settings_config.get("temperature", 0.1), step=0.05,
                            label="Temperature", info="Controls randomness. Lower is more deterministic.",
                            elem_id="config_temperature"
                        )
                        top_p = gr.Slider(
                            0, 1, value=saved_settings_config.get("top_p", 0.95), step=0.05,
                            label="Top P", info="Nucleus sampling parameter.",
                            elem_id="config_top_p"
                        )
                        top_k = gr.Slider(
                            0, 64, value=saved_settings_config.get("top_k", 1), step=1,
                            label="Top K", info="Limits sampling pool to top K tokens.",
                            interactive=(config_initial_provider != "OpenAI"),
                            elem_id="config_top_k"
                        )
                    setting_groups.append(group_translation)

                    with gr.Group(visible=False, elem_classes="settings-group") as group_rendering:
                        gr.Markdown("### Text Rendering")
                        max_font_size = gr.Slider(
                            5, 50, value=saved_settings_config.get("max_font_size", 14), step=1,
                            label="Max Font Size (px)", info="The largest font size the renderer will attempt to use."
                        )
                        min_font_size = gr.Slider(
                             5, 50, value=saved_settings_config.get("min_font_size", 8), step=1,
                             label="Min Font Size (px)", info="The smallest font size the renderer will attempt to use before giving up."
                        )
                        line_spacing = gr.Slider(
                            0.5, 2.0, value=saved_settings_config.get("line_spacing", 1.0), step=0.05,
                            label="Line Spacing Multiplier", info="Adjusts the vertical space between lines of text (1.0 = standard)."
                        )
                        use_subpixel_rendering = gr.Checkbox(
                            value=saved_settings_config.get("use_subpixel_rendering", False),
                            label="Use Subpixel Rendering", info="Improves text clarity on LCD screens, but can cause color fringing."
                        )
                        font_hinting = gr.Radio(
                            choices=["none", "slight", "normal", "full"],
                            value=saved_settings_config.get("font_hinting", "none"),
                            label="Font Hinting", info="Adjusts glyph outlines to fit pixel grid. 'None' is often best for high-res displays."
                        )
                        use_ligatures = gr.Checkbox(
                            value=saved_settings_config.get("use_ligatures", False),
                            label="Use Standard Ligatures (e.g., fi, fl)", info="Enables common letter combinations to be rendered as single glyphs (must be supported by the font)."
                        )
                    setting_groups.append(group_rendering)

                    with gr.Group(visible=False, elem_classes="settings-group") as group_output:
                        gr.Markdown("### Image Output")
                        output_format = gr.Radio(
                            choices=["auto", "png", "jpeg"], label="Image Output Format",
                            value=saved_settings_config.get("output_format", "auto"),
                            info="'auto' uses the same format as the input image (defaults to PNG if unknown)."
                        )
                        jpeg_quality = gr.Slider(
                            1, 100, value=saved_settings_config.get("jpeg_quality", 95), step=1,
                            label="JPEG Quality (higher = better quality)",
                            interactive=saved_settings_config.get("output_format", "auto") != "png"
                        )
                        png_compression = gr.Slider(
                            0, 9, value=saved_settings_config.get("png_compression", 6), step=1,
                            label="PNG Compression Level (higher = smaller file)",
                            interactive=saved_settings_config.get("output_format", "auto") != "jpeg"
                        )
                    setting_groups.append(group_output)

                    with gr.Group(visible=False, elem_classes="settings-group") as group_other:
                        gr.Markdown("### Other")

                        refresh_resources_button = gr.Button(
                            "Refresh Models / Fonts",
                            variant="secondary",
                            elem_classes="config-refresh-button"
                        )
                        verbose = gr.Checkbox(
                            value=saved_settings_config.get("verbose", False),
                            label="Verbose Logging (console)"
                        )
                        cleaning_only_toggle = gr.Checkbox(
                            value=saved_settings_config.get("cleaning_only", False),
                            label="Cleaning Only Mode",
                            info="Skip translation and text rendering, output only the cleaned speech bubbles."
                        )
                    setting_groups.append(group_other)

            output_components_for_switch = setting_groups + nav_buttons

            nav_button_detection.click(fn=lambda idx=0: switch_settings_view(idx, setting_groups, nav_buttons), outputs=output_components_for_switch, queue=False)
            nav_button_cleaning.click(fn=lambda idx=1: switch_settings_view(idx, setting_groups, nav_buttons), outputs=output_components_for_switch, queue=False)
            nav_button_translation.click(fn=lambda idx=2: switch_settings_view(idx, setting_groups, nav_buttons), outputs=output_components_for_switch, queue=False)
            nav_button_rendering.click(fn=lambda idx=3: switch_settings_view(idx, setting_groups, nav_buttons), outputs=output_components_for_switch, queue=False)
            nav_button_output.click(fn=lambda idx=4: switch_settings_view(idx, setting_groups, nav_buttons), outputs=output_components_for_switch, queue=False)
            nav_button_other.click(fn=lambda idx=5: switch_settings_view(idx, setting_groups, nav_buttons), outputs=output_components_for_switch, queue=False)

            # Update JPEG/PNG interactivity based on output format
            output_format.change(
                fn=lambda fmt: [gr.update(interactive=fmt != "png"), gr.update(interactive=fmt != "jpeg")],
                inputs=output_format, outputs=[jpeg_quality, png_compression], queue=False
            )

            # Update Translation UI based on provider selection
            provider_selector.change(
                fn=ui_utils.update_translation_ui,
                inputs=[provider_selector, temperature],
                outputs=[
                    gemini_api_key, openai_api_key, anthropic_api_key, openrouter_api_key,
                    openai_compatible_url_input, openai_compatible_api_key_input,
                    config_model_name,
                    temperature,
                    top_k
                ],
                queue=False
            ).then( # Trigger model fetch *after* provider change updates visibility etc.
                fn=lambda prov, url, key: ui_utils.fetch_and_update_compatible_models(url, key) if prov == "OpenAI-compatible" else (ui_utils.fetch_and_update_openrouter_models() if prov == "OpenRouter" else gr.update()),
                inputs=[provider_selector, openai_compatible_url_input, openai_compatible_api_key_input],
                outputs=[config_model_name],
                queue=True # Allow fetching to happen in the background
            )

    save_config_inputs = [
        config_yolo_model_dropdown, confidence, config_reading_direction,
        dilation_kernel_size, dilation_iterations, use_otsu_threshold, min_contour_area,
        closing_kernel_size, closing_iterations, constraint_erosion_kernel_size, constraint_erosion_iterations,
        provider_selector, gemini_api_key, openai_api_key, anthropic_api_key, openrouter_api_key, openai_compatible_url_input, openai_compatible_api_key_input, config_model_name,
        temperature, top_p, top_k,
        max_font_size, min_font_size, line_spacing, use_subpixel_rendering, font_hinting, use_ligatures,
        output_format, jpeg_quality, png_compression,
        verbose, cleaning_only_toggle, # Added cleaning_only_toggle
        input_language, output_language, font_dropdown,
        batch_input_language, batch_output_language, batch_font_dropdown
    ]
    save_config_btn.click(
        fn=lambda yolo, conf, rd, dks, di, otsu, mca, cks, ci, ceks, cei,
                 prov, gem_key, oai_key, ant_key, or_key, comp_url, comp_key, model, temp, tp, tk,
                 max_fs, min_fs, ls, subpix, hint, liga,
                 out_fmt, jq, pngc, verb, cleaning_only_val, # Added cleaning_only_val
                 s_in_lang, s_out_lang, s_font,
                 b_in_lang, b_out_lang, b_font:
            config_utils.save_config({
                'yolo_model': yolo, 'confidence': conf, 'reading_direction': rd,
                'dilation_kernel_size': dks, 'dilation_iterations': di, 'use_otsu_threshold': otsu, 'min_contour_area': mca,
                'closing_kernel_size': cks, 'closing_iterations': ci, 'constraint_erosion_kernel_size': ceks, 'constraint_erosion_iterations': cei,
                'provider': prov, 'gemini_api_key': gem_key, 'openai_api_key': oai_key, 'anthropic_api_key': ant_key, 'openrouter_api_key': or_key,
                'openai_compatible_url': comp_url, 'openai_compatible_api_key': comp_key, 'model_name': model,
                'temperature': temp, 'top_p': tp, 'top_k': tk,
                'font_pack': s_font,
                'max_font_size': max_fs, 'min_font_size': min_fs, 'line_spacing': ls, 'use_subpixel_rendering': subpix, 'font_hinting': hint, 'use_ligatures': liga,
                'output_format': out_fmt, 'jpeg_quality': jq, 'png_compression': pngc,
                'verbose': verb,
                'cleaning_only': cleaning_only_val, # Save cleaning_only value
                'input_language': s_in_lang, 'output_language': s_out_lang,
                'batch_input_language': b_in_lang, 'batch_output_language': b_out_lang, 'batch_font_pack': b_font
            })[1],
        inputs=save_config_inputs,
        outputs=[config_status],
        queue=False
    ).then(
        fn=lambda: None, inputs=None, outputs=None, js="""
        () => {
            const statusElement = document.getElementById('config_status');
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
            } else if (statusElement) {
                statusElement.style.display = 'none';
            }
        }
        """, queue=False
    )

    reset_outputs = [
        config_yolo_model_dropdown, confidence, config_reading_direction,
        dilation_kernel_size, dilation_iterations, use_otsu_threshold, min_contour_area,
        closing_kernel_size, closing_iterations, constraint_erosion_kernel_size, constraint_erosion_iterations,
        provider_selector, gemini_api_key, openai_api_key, anthropic_api_key, openrouter_api_key, openai_compatible_url_input, openai_compatible_api_key_input, config_model_name,
        temperature, top_p, top_k,
        max_font_size, min_font_size, line_spacing, use_subpixel_rendering, font_hinting, use_ligatures,
        output_format, jpeg_quality, png_compression,
        verbose, cleaning_only_toggle, # Added cleaning_only_toggle
        input_language, output_language, font_dropdown,
        batch_input_language, batch_output_language, batch_font_dropdown,
        config_status
    ]

    def apply_defaults():
        defaults = config_utils.reset_to_defaults()
        available_yolo_models = ui_utils.get_available_models(MODELS_DIR)
        reset_yolo_model = defaults.get('yolo_model') if defaults.get('yolo_model') in available_yolo_models else (available_yolo_models[0] if available_yolo_models else None)

        available_fonts, _ = ui_utils.get_available_font_packs(FONTS_BASE_DIR)
        reset_single_font = defaults.get('font_pack') if defaults.get('font_pack') in available_fonts else (available_fonts[0] if available_fonts else None)
        batch_reset_font = defaults.get('batch_font_pack') if defaults.get('batch_font_pack') in available_fonts else (available_fonts[0] if available_fonts else None)

        default_provider = defaults.get('provider', config_utils.DEFAULT_SETTINGS["provider"])
        default_provider_models = defaults.get('provider_models', config_utils.DEFAULT_SETTINGS["provider_models"])
        default_model_name = default_provider_models.get(default_provider)

        if default_provider == "OpenRouter":
             default_models_choices = [default_model_name] if default_model_name else []
        else:
             default_models_choices = config_utils.PROVIDER_MODELS.get(default_provider, [])

        gemini_visible = default_provider == "Gemini"
        openai_visible = default_provider == "OpenAI"
        anthropic_visible = default_provider == "Anthropic"
        gemini_visible = default_provider == "Gemini"
        openai_visible = default_provider == "OpenAI"
        anthropic_visible = default_provider == "Anthropic"
        openrouter_visible = default_provider == "OpenRouter"
        compatible_visible = default_provider == "OpenAI-compatible"
        temp_max = 1.0 if default_provider == "Anthropic" else 2.0
        temp_val = min(defaults['temperature'], temp_max)
        top_k_interactive = default_provider != "OpenAI"

        return [
            gr.update(value=reset_yolo_model), defaults['confidence'], defaults['reading_direction'],
            defaults['dilation_kernel_size'], defaults['dilation_iterations'], defaults['use_otsu_threshold'], defaults['min_contour_area'],
            defaults['closing_kernel_size'], defaults['closing_iterations'], defaults['constraint_erosion_kernel_size'], defaults['constraint_erosion_iterations'],
            gr.update(value=default_provider),
            gr.update(value=defaults.get('gemini_api_key', ''), visible=gemini_visible),
            gr.update(value=defaults.get('openai_api_key', ''), visible=openai_visible),
            gr.update(value=defaults.get('anthropic_api_key', ''), visible=anthropic_visible),
            gr.update(value=defaults.get('openrouter_api_key', ''), visible=openrouter_visible),
            gr.update(value=defaults.get('openai_compatible_url', config_utils.DEFAULT_SETTINGS['openai_compatible_url']), visible=compatible_visible),
            gr.update(value=defaults.get('openai_compatible_api_key', ''), visible=compatible_visible),
            gr.update(choices=default_models_choices, value=default_model_name),
            gr.update(value=temp_val, maximum=temp_max),
            defaults['top_p'],
            gr.update(value=defaults['top_k'], interactive=top_k_interactive),
            defaults['max_font_size'], defaults['min_font_size'], defaults['line_spacing'], defaults['use_subpixel_rendering'], defaults['font_hinting'], defaults['use_ligatures'],
            defaults['output_format'], defaults['jpeg_quality'], defaults['png_compression'],
            defaults['verbose'],
            defaults.get('cleaning_only', False), # Reset cleaning_only
            defaults['input_language'], defaults['output_language'], gr.update(value=reset_single_font),
            defaults['batch_input_language'], defaults['batch_output_language'], gr.update(value=batch_reset_font),
            f"Settings reset to defaults (API keys preserved)."
        ]
    reset_defaults_btn.click(
        fn=apply_defaults, inputs=[], outputs=reset_outputs, queue=False
    ).then(
        fn=lambda: None, inputs=None, outputs=None, js="""
        () => {
            const statusElement = document.getElementById('config_status');
            if (statusElement && statusElement.textContent.trim() !== "") {
                 clearTimeout(statusElement.fadeTimer); clearTimeout(statusElement.resetTimer);
                 statusElement.style.display = 'block'; statusElement.style.transition = 'none'; statusElement.style.opacity = '1';
                 const fadeDelay = 3000; const fadeDuration = 1000;
                 statusElement.fadeTimer = setTimeout(() => {
                     statusElement.style.transition = `opacity ${fadeDuration}ms ease-out`; statusElement.style.opacity = '0';
                     statusElement.resetTimer = setTimeout(() => {
                         statusElement.style.display = 'none'; statusElement.style.opacity = '1'; statusElement.style.transition = 'none';
                     }, fadeDuration);
                 }, fadeDelay);
             } else if (statusElement) { statusElement.style.display = 'none'; }
        }
        """, queue=False
    )

    translate_inputs = [
        input_image,
        config_yolo_model_dropdown, confidence,
        dilation_kernel_size, dilation_iterations, use_otsu_threshold, min_contour_area,
        closing_kernel_size, closing_iterations, constraint_erosion_kernel_size, constraint_erosion_iterations,
        provider_selector, gemini_api_key, openai_api_key, anthropic_api_key, openrouter_api_key, openai_compatible_url_input, openai_compatible_api_key_input, config_model_name,
        temperature, top_p, top_k, config_reading_direction,
        font_dropdown,
        max_font_size, min_font_size, line_spacing, use_subpixel_rendering, font_hinting, use_ligatures,
        output_format, jpeg_quality, png_compression,
        verbose, cleaning_only_toggle, # Added cleaning_only_toggle
        input_language, output_language
    ]
    translate_button.click(
        fn=lambda: gr.update(interactive=False, value="Translating..."), outputs=translate_button, queue=False
    ).then(
        lambda img, yolo_mdl, conf, dks, di, otsu, mca, cks, ci, ceks, cei,
               prov, gem_key, oai_key, ant_key, or_key, comp_url, comp_key, model, temp, tp, tk, rd,
               s_font, max_fs, min_fs, ls, subpix, hint, liga,
               out_fmt, jq, pngc, verb, cleaning_only_val, # Added cleaning_only_val
               s_in_lang, s_out_lang:

            translate_manga(
                image=img,
                selected_yolo_model=yolo_mdl,
                detection_cfg=DetectionConfig(confidence=conf),
                cleaning_cfg=CleaningConfig(
                    dilation_kernel_size=dks, dilation_iterations=di, use_otsu_threshold=otsu,
                    min_contour_area=mca, closing_kernel_size=cks, closing_iterations=ci,
                    constraint_erosion_kernel_size=ceks, constraint_erosion_iterations=cei
                ),
                translation_cfg=TranslationConfig(
                    provider=prov, gemini_api_key=gem_key, openai_api_key=oai_key, anthropic_api_key=ant_key, openrouter_api_key=or_key,
                    openai_compatible_url=comp_url, openai_compatible_api_key=comp_key,
                    model_name=model, temperature=temp, top_p=tp, top_k=tk,
                    input_language=s_in_lang, output_language=s_out_lang,
                    reading_direction=rd
                ),
                rendering_cfg=RenderingConfig(
                    font_dir=s_font,
                    max_font_size=max_fs, min_font_size=min_fs, line_spacing=ls,
                    use_subpixel_rendering=subpix, font_hinting=hint, use_ligatures=liga
                ),
                output_cfg=OutputConfig(
                    output_format=out_fmt, jpeg_quality=jq, png_compression=pngc
                ),
                cleaning_only=cleaning_only_val, # Pass cleaning_only value
                verbose=verb,
                device=target_device
            ),
        inputs=translate_inputs,
        outputs=[output_image, status_message]
    ).then(
        fn=lambda: gr.update(interactive=True, value="Translate"), outputs=translate_button, queue=False
    )

    batch_inputs = [
        input_files,
        config_yolo_model_dropdown, confidence,
        dilation_kernel_size, dilation_iterations, use_otsu_threshold, min_contour_area,
        closing_kernel_size, closing_iterations, constraint_erosion_kernel_size, constraint_erosion_iterations,
        provider_selector, gemini_api_key, openai_api_key, anthropic_api_key, openrouter_api_key, openai_compatible_url_input, openai_compatible_api_key_input, config_model_name,
        temperature, top_p, top_k, config_reading_direction,
        batch_font_dropdown,
        max_font_size, min_font_size, line_spacing, use_subpixel_rendering, font_hinting, use_ligatures,
        output_format, jpeg_quality, png_compression,
        verbose, cleaning_only_toggle, # Added cleaning_only_toggle
        batch_input_language, batch_output_language
    ]
    batch_process_button.click(
        fn=lambda: gr.update(interactive=False, value="Processing..."), outputs=batch_process_button, queue=False
    ).then(
        lambda files, yolo_mdl, conf, dks, di, otsu, mca, cks, ci, ceks, cei,
               prov, gem_key, oai_key, ant_key, or_key, comp_url, comp_key, model, temp, tp, tk, rd,
               b_font, max_fs, min_fs, ls, subpix, hint, liga,
               out_fmt, jq, pngc, verb, cleaning_only_val, # Added cleaning_only_val
               b_in_lang, b_out_lang:
            process_batch(
                input_dir_or_files=files,
                selected_yolo_model=yolo_mdl,
                detection_cfg=DetectionConfig(confidence=conf),
                cleaning_cfg=CleaningConfig(
                    dilation_kernel_size=dks, dilation_iterations=di, use_otsu_threshold=otsu,
                    min_contour_area=mca, closing_kernel_size=cks, closing_iterations=ci,
                    constraint_erosion_kernel_size=ceks, constraint_erosion_iterations=cei
                ),
                translation_cfg=TranslationConfig(
                    provider=prov, gemini_api_key=gem_key, openai_api_key=oai_key, anthropic_api_key=ant_key, openrouter_api_key=or_key,
                    openai_compatible_url=comp_url, openai_compatible_api_key=comp_key,
                    model_name=model, temperature=temp, top_p=tp, top_k=tk,
                    input_language=b_in_lang, output_language=b_out_lang,
                    reading_direction=rd
                ),
                rendering_cfg=RenderingConfig(
                    font_dir=b_font,
                    max_font_size=max_fs, min_font_size=min_fs, line_spacing=ls,
                    use_subpixel_rendering=subpix, font_hinting=hint, use_ligatures=liga
                ),
                output_cfg=OutputConfig(
                    output_format=out_fmt, jpeg_quality=jq, png_compression=pngc
                ),
                cleaning_only=cleaning_only_val, # Pass cleaning_only value
                verbose=verb,
                device=target_device
            ),
        inputs=batch_inputs,
        outputs=[batch_output_gallery, batch_status_message]
    ).then(
        fn=lambda: gr.update(interactive=True, value="Start Batch Translating"), outputs=batch_process_button, queue=False
    )

    refresh_outputs = [
        config_yolo_model_dropdown,
        font_dropdown,
        batch_font_dropdown
    ]
    refresh_resources_button.click(
        fn=lambda: ui_utils.refresh_models_and_fonts(MODELS_DIR, FONTS_BASE_DIR),
        inputs=[],
        outputs=refresh_outputs,
        js="""
            function() {
                const refreshButton = document.querySelector('.config-refresh-button button');
                if (refreshButton) {
                    refreshButton.textContent = 'Refreshing...';
                    refreshButton.disabled = true;
                }
                return [];
            }
        """
    ).then(
        fn=None, inputs=None, outputs=None, js="""
            function() {
                setTimeout(() => {
                    const refreshButton = document.querySelector('.config-refresh-button button');
                     if (refreshButton) {
                        refreshButton.textContent = 'Refresh Models / Fonts';
                        refreshButton.disabled = false;
                    }
                }, 100);
            }
        """, queue=False
    )

    config_model_name.change(
        fn=ui_utils.update_params_for_model,
        inputs=[provider_selector, config_model_name, temperature],
        outputs=[temperature, top_k],
        queue=False
    )

    app.load(
        fn=ui_utils.initial_dynamic_fetch,
        inputs=[provider_selector, openai_compatible_url_input, openai_compatible_api_key_input],
        outputs=[config_model_name],
        queue=False
    )

    app.queue()

if __name__ == "__main__":
    app.launch(inbrowser=args.open_browser, server_port=args.port, show_error=True)
