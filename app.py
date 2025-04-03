import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import gradio as gr
from PIL import Image
import torch

import core
from core.config import MangaTranslatorConfig, DetectionConfig, CleaningConfig, TranslationConfig, RenderingConfig, OutputConfig
from core.generate_manga import translate_and_render, batch_translate_images
from typing import Union

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

CONFIG_FILE = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser("~"))) / "MangaTranslator" / "config.json"

ERROR_PREFIX = "❌ Error: "
SUCCESS_PREFIX = "✅ "

DEFAULT_SETTINGS = {
    "input_language": "Japanese",
    "output_language": "English",
    "reading_direction": "rtl",
    "gemini_model": "gemini-2.0-flash",
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
    "top_k": 1,
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
}

DEFAULT_BATCH_SETTINGS = {
    "batch_input_language": "Japanese",
    "batch_output_language": "English",
    "batch_reading_direction": "rtl",
    "batch_gemini_model": "gemini-2.0-flash",
    "batch_font_pack": None
}

# --- Argument Parsing and Global Setup ---
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
os.makedirs(CONFIG_FILE.parent, exist_ok=True)

target_device = torch.device('cpu') if args.cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Print Detailed Device Info ---
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

def get_available_models(models_directory):
    """Get list of available YOLO models in the models directory"""
    model_files = list(Path(models_directory).glob("*.pt"))
    models = [p.name for p in model_files]
    
    return models

def get_available_font_packs():
    """Get list of available font packs (subdirectories) in the fonts directory"""
    font_dirs = [d.name for d in FONTS_BASE_DIR.iterdir() if d.is_dir()]
    font_dirs.sort()

    if font_dirs:
        default_font = font_dirs[0]
    else:
        default_font = None

    return font_dirs, default_font

def save_api_key(api_key):
    """Save API key to config file"""
    try:
        config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
        
        config['gemini_api_key'] = api_key.strip()
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        
        return True, f"{SUCCESS_PREFIX}API key saved successfully"
    except Exception as e:
        return False, f"{ERROR_PREFIX}Failed to save API key: {str(e)}"

def validate_api_key(api_key):
    """Validate Gemini API key format"""
    if not api_key:
        return False, "API key is required"
    
    if not (api_key.startswith('AI') and len(api_key) == 39):
        return False, "Invalid API key format. Gemini API keys usually start with 'AI'"
    
    return True, "API key format valid"

def validate_image(image):
    """Validate uploaded image"""
    if image is None:
        return False, "Please upload an image"

    try:
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image
            
        width, height = img.size
        if width < 600 or height < 600:
            return False, f"Image dimensions too small ({width}x{height}). Min recommended size is 600x600."
        
        if isinstance(image, str):
            if not image.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                return False, "Unsupported image format. Please use JPEG, PNG or WEBP."
        
        return True, "Image is valid"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

def validate_font_directory(font_dir):
    """
    Validate that the font directory contains at least one font file
    
    Args:
        font_dir (str or Path): Path to the font directory
        
    Returns:
        tuple: (is_valid, message)
    """
    font_dir = Path(font_dir)
    if not font_dir.exists():
        return False, f"Font directory '{font_dir.name}' not found"
    
    font_files = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
    if not font_files:
        return False, f"No font files (.ttf or .otf) found in '{font_dir.name}' directory"
    
    return True, f"Found {len(font_files)} font files in directory"

def save_config(settings):
    """Save all settings to config file"""
    try:
        config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
        
        changed_settings = []
        for key, value in settings.items():
            if key in config and config[key] != value:
                changed_settings.append(key)
            elif key not in config:
                changed_settings.append(key)
        
        for key, value in settings.items():
            config[key] = value
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        if changed_settings:
            return True, f"Saved changes: {', '.join(changed_settings)}"
        else:
            return True, "Saved changes: none"
            
    except Exception as e:
        return False, f"Failed to save settings: {str(e)}"

def get_saved_settings():
    """Get all saved settings from config file"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

            settings = {}
            settings.update(DEFAULT_SETTINGS)
            settings.update(DEFAULT_BATCH_SETTINGS)

            for key in settings.keys():
                if key in config:
                    settings[key] = config[key]

            if 'gemini_api_key' in config:
                settings['gemini_api_key'] = config['gemini_api_key']

            if 'yolo_model' in config:
                 settings['yolo_model'] = config['yolo_model']

            return settings
        return DEFAULT_SETTINGS.copy()
    except Exception:
        return DEFAULT_SETTINGS.copy()

def reset_to_defaults():
    """Reset all settings to default values"""
    settings = DEFAULT_SETTINGS.copy()
    settings.update(DEFAULT_BATCH_SETTINGS.copy())
    
    saved_settings = get_saved_settings()
    if 'gemini_api_key' in saved_settings:
        settings['gemini_api_key'] = saved_settings['gemini_api_key']
    
    if 'yolo_model' in saved_settings:
        settings['yolo_model'] = saved_settings['yolo_model']
    
    return settings


# --- Consolidated Validation ---

def _validate_common_inputs(api_key: str, selected_model: str, font_pack: str,
                            max_font_size: int, min_font_size: int, line_spacing: float,
                            font_hinting: str) -> tuple[Path, Path]: # Added font_hinting
    """
    Validates common inputs used by both single and batch translation.

    Args:
        api_key (str): Gemini API key.
        selected_model (str): Filename of the selected YOLO model.
        font_pack (str): Name of the selected font pack directory.
        max_font_size (int): Maximum font size for rendering.
        min_font_size (int): Minimum font size for rendering.
        line_spacing (float): Line spacing multiplier.
        font_hinting (str): Font hinting setting.

    Returns:
        tuple[Path, Path]: Validated absolute path to the YOLO model and font directory.

    Raises:
        gr.Error: If any validation fails.
    """

    api_valid, api_msg = validate_api_key(api_key)
    if not api_valid:
        raise gr.Error(api_msg, print_exception=False)

    available_models = get_available_models(MODELS_DIR)
    if not available_models:
        raise gr.Error(
            f"{ERROR_PREFIX}No YOLO models found. Please download a model file "
            f"and place it in the 'models' directory, then refresh the models list.",
            print_exception=False
        )
    if not selected_model:
        raise gr.Error(f"{ERROR_PREFIX}Please select a YOLO model.", print_exception=False)
    if selected_model not in available_models:
        raise gr.Error(
            f"{ERROR_PREFIX}Selected model '{selected_model}' is not available. Please choose from: {', '.join(available_models)}",
            print_exception=False
        )
    yolo_model_path = MODELS_DIR / selected_model
    if not yolo_model_path.exists():
        raise gr.Error(f"{ERROR_PREFIX}YOLO model not found at {yolo_model_path}.", print_exception=False)

    # Font Pack
    if not font_pack:
        raise gr.Error(f"{ERROR_PREFIX}Please select a font pack.", print_exception=False)
    font_dir = FONTS_BASE_DIR / font_pack
    font_valid, font_msg = validate_font_directory(font_dir)
    if not font_valid:
        raise gr.Error(f"{ERROR_PREFIX}{font_msg}.", print_exception=False)

    # Rendering Params
    if not (isinstance(max_font_size, int) and max_font_size > 0):
        raise gr.Error(f"{ERROR_PREFIX}Max Font Size must be a positive integer.", print_exception=False)
    if not (isinstance(min_font_size, int) and min_font_size > 0):
        raise gr.Error(f"{ERROR_PREFIX}Min Font Size must be a positive integer.", print_exception=False)
    if not (isinstance(line_spacing, (int, float)) and float(line_spacing) > 0):
         raise gr.Error(f"{ERROR_PREFIX}Line Spacing must be a positive number.", print_exception=False)
    if min_font_size > max_font_size:
         raise gr.Error(f"{ERROR_PREFIX}Min Font Size cannot be larger than Max Font Size.", print_exception=False)
    if font_hinting not in ["none", "slight", "normal", "full"]:
        raise gr.Error(f"{ERROR_PREFIX}Invalid Font Hinting value. Must be one of: none, slight, normal, full.", print_exception=False)

    return yolo_model_path, font_dir


# --- Translation Functions ---
def translate_manga(image: Union[str, Path, Image.Image],
                    api_key: str, input_language="Japanese", output_language="English",
                    gemini_model="gemini-2.0-flash", confidence=0.35, dilation_kernel_size=7, dilation_iterations=1,
                    use_otsu_threshold=False, min_contour_area=50, closing_kernel_size=7, closing_iterations=1, # Changed threshold_value
                    constraint_erosion_kernel_size=5, constraint_erosion_iterations=1, temperature=0.1, top_p=0.95,
                    top_k=1, selected_model="", font_pack=None,
                    max_font_size=14, min_font_size=8, line_spacing=1.0,
                    use_subpixel_rendering=False, font_hinting="none", use_ligatures=False,
                    verbose=False, jpeg_quality=95, png_compression=6,
                    output_format="auto", device=None):
    """
    Process a single manga image with translation.
    
    This function handles validation of inputs, translates text in the provided image
    through the translation pipeline, and generates a status report.
    
    Args:
        image (str or PIL.Image): Image to translate, either as a file path or PIL image
        api_key (str): Gemini API key for translation
        input_language (str, optional): Source language. Defaults to "Japanese".
        output_language (str, optional): Target language. Defaults to "English".
        gemini_model (str, optional): Gemini model to use. Defaults to "gemini-2.0-flash".
        confidence (float, optional): Bubble detection confidence threshold. Defaults to 0.35.
        dilation_kernel_size (int): Kernel size for dilating initial YOLO mask for ROI.
        dilation_iterations (int): Iterations for dilation.
        use_otsu_threshold (bool): If True, use Otsu's method for thresholding instead of the fixed value (210).
        min_contour_area (int): Minimum area for a contour to be considered part of the bubble interior.
        closing_kernel_size (int): Kernel size for morphological closing to smooth mask and include text.
        closing_iterations (int): Iterations for closing.
        constraint_erosion_kernel_size (int): Kernel size for eroding the original YOLO mask for edge constraint.
        constraint_erosion_iterations (int): Iterations for constraint erosion.
        temperature (float, optional): Temperature for Gemini model. Defaults to 0.1.
        top_p (float, optional): Top-p parameter for Gemini model. Defaults to 0.95.
        top_k (int, optional): Top-k parameter for Gemini model. Defaults to 1.
        selected_model (str, optional): YOLO model filename to use. Defaults to "".
        font_pack (str, optional): Font pack to use. Defaults to None.
        max_font_size (int, optional): Max font size for rendering. Defaults to 14.
        min_font_size (int, optional): Min font size for rendering. Defaults to 8.
        line_spacing (float, optional): Line spacing multiplier. Defaults to 1.0.
        use_subpixel_rendering (bool, optional): Enable subpixel rendering. Defaults to False.
        font_hinting (str, optional): Font hinting level ('none', 'slight', 'normal', 'full'). Defaults to "none".
        use_ligatures (bool, optional): Enable standard ligatures. Defaults to False.
        verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
        jpeg_quality (int, optional): JPEG quality setting (1-100). Defaults to 95.
        png_compression (int, optional): PNG compression level (0-9). Defaults to 6.
        output_format (str, optional): Output image format ('auto', 'png', 'jpeg'). Defaults to "auto".
        device (torch.device, optional): The target device to run models on.
    
    Returns:
        tuple: 
            - translated_image (PIL.Image): The translated image
            - translator_status_msg (str): Detailed status message with processing statistics
    
    Raises:
        gr.Error: If inputs are invalid or processing fails
    """
    start_time = time.time()
    translated_image = None

    # --- Input Validation ---
    # Image specific validation
    img_valid, img_msg = validate_image(image)
    if not img_valid:
        raise gr.Error(img_msg, print_exception=False)

    # Common validation (API Key, Model, Font, Rendering Params)
    yolo_model_path, font_dir = _validate_common_inputs(
        api_key, selected_model, font_pack, max_font_size, min_font_size, line_spacing, font_hinting
    )
    
    temp_image_path = None
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if isinstance(image, str):
            input_path = Path(image)
            original_ext = input_path.suffix.lower()
            image_path_for_processing = str(input_path)
        else:
            original_ext = '.png'
            temp_image_path = Path(tempfile.mktemp(suffix='.png'))
            image.save(temp_image_path)
            image_path_for_processing = str(temp_image_path)
            
        if output_format == "auto":
            output_ext = original_ext
        elif output_format == "png":
            output_ext = '.png'
        elif output_format == "jpeg":
            output_ext = '.jpg'
        else:
            output_ext = original_ext
            
        save_path = Path("./output") / f"MangaTranslator_{timestamp}{output_ext}"
        
        if output_format == "jpeg" or (output_format == "auto" and (output_ext == '.jpg' or output_ext == '.jpeg')):
            image_mode = "RGB"
        else:
            image_mode = "RGBA"
            
        output_dir = Path("./output")
        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir / f"MangaTranslator_{timestamp}{output_ext}"

        # Determine image mode based on output format
        if output_format == "jpeg" or (output_format == "auto" and (original_ext in ['.jpg', '.jpeg'])):
            image_mode = "RGB"
        else:
            image_mode = "RGBA" # Default to RGBA for PNG or other formats

        # --- Configuration Setup ---
        use_otsu_config_val = use_otsu_threshold

        config = MangaTranslatorConfig(
            yolo_model_path=str(yolo_model_path),
            verbose=verbose,
            device=device or target_device,
            detection=DetectionConfig(
                confidence=confidence
            ),
            cleaning=CleaningConfig(
                dilation_kernel_size=dilation_kernel_size,
                dilation_iterations=dilation_iterations,
                use_otsu_threshold=use_otsu_config_val,
                min_contour_area=min_contour_area,
                closing_kernel_size=closing_kernel_size,
                closing_iterations=closing_iterations,
                constraint_erosion_kernel_size=constraint_erosion_kernel_size,
                constraint_erosion_iterations=constraint_erosion_iterations
            ),
            translation=TranslationConfig(
                api_key=api_key,
                gemini_model=gemini_model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                input_language=input_language,
                output_language=output_language,
            ),
            rendering=RenderingConfig(
                font_dir=str(font_dir),
                max_font_size=max_font_size,
                min_font_size=min_font_size,
                line_spacing=line_spacing,
                use_subpixel_rendering=use_subpixel_rendering,
                font_hinting=font_hinting,
                use_ligatures=use_ligatures
            ),
            output=OutputConfig(
                jpeg_quality=jpeg_quality,
                png_compression=png_compression,
                image_mode=image_mode
            )
        )

        # --- Execute Translation ---
        translated_image = translate_and_render(
            image_path=image_path_for_processing,
            config=config,
            output_path=save_path
        )
        
        width, height = translated_image.size
        
        processing_time = time.time() - start_time
        
        translator_status_msg = (
            f"{SUCCESS_PREFIX}Translation completed!\n"
            f"• Image size: {width}x{height} pixels\n"
            f"• Source language: {config.translation.input_language}\n"
            f"• Target language: {config.translation.output_language}\n"
            f"• Reading Direction: {config.translation.reading_direction.upper()}\n"
            f"• Font pack: {font_pack}\n"
            f"• Model used: {selected_model}\n"
            f"• Cleaning Params: DKS:{dilation_kernel_size}, DI:{dilation_iterations}, Otsu:{use_otsu_threshold}, "
            f"MCA:{min_contour_area}, CKS:{closing_kernel_size}, CI:{closing_iterations}, "
            f"CEKS:{constraint_erosion_kernel_size}, CEI:{constraint_erosion_iterations}\n"
            f"• Temperature: {temperature}\n"
            f"• Top-p: {top_p}\n"
            f"• Top-k: {top_k}\n"
            f"• Output format: {output_format}\n"
            f"• JPEG quality: {jpeg_quality}\n"
            f"• PNG compression: {png_compression}\n"
            f"• Processing time: {processing_time:.2f} seconds\n"
            f"• Saved to: {save_path}"
        )
        
        result = translated_image.copy() if translated_image else None
        return result, translator_status_msg
    
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"{ERROR_PREFIX}{str(e)}", print_exception=False)
    finally:
        if temp_image_path and temp_image_path.exists():
            try:
                os.remove(temp_image_path)
            except Exception as e_clean:
                print(f"Warning: Could not remove temporary image file {temp_image_path}: {e_clean}")

def update_model_dropdown():
    """Update the model dropdown list"""
    try:
        models = get_available_models(MODELS_DIR)
        return gr.Dropdown(choices=models), f"{SUCCESS_PREFIX}Found {len(models)} models"
    except Exception as e:
        return gr.Dropdown(choices=[]), f"{ERROR_PREFIX}{str(e)}"

def update_font_dropdown():
    """Update the font pack dropdown list"""
    try:
        font_packs, default_font = get_available_font_packs()
        return gr.Dropdown(choices=font_packs, value=default_font), f"{SUCCESS_PREFIX}Found {len(font_packs)} font packs"
    except Exception as e:
        return gr.Dropdown(choices=[]), f"{ERROR_PREFIX}{str(e)}"

def refresh_models_and_fonts():
    """Update both model dropdown and font dropdown lists"""
    try:
        models = get_available_models(MODELS_DIR)
        saved_settings = get_saved_settings()
        current_model = saved_settings.get('yolo_model')
        selected_model_val = current_model if current_model in models else (models[0] if models else None)
        model_result = gr.Dropdown(choices=models, value=selected_model_val)
        
        font_packs, _ = get_available_font_packs()
        current_font = saved_settings.get('font_pack')
        selected_font_val = current_font if current_font in font_packs else (font_packs[0] if font_packs else None)
        font_result = gr.Dropdown(choices=font_packs, value=selected_font_val)
        
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
        else:
            gr.Warning(f"No font packs found. Found {model_text}")
            
        return model_result, font_result
    except Exception as e:
        gr.Error(f"Error refreshing resources: {str(e)}")
        models = get_available_models(MODELS_DIR)
        saved_settings = get_saved_settings()
        current_model = saved_settings.get('yolo_model')
        selected_model_val = current_model if current_model in models else None

        font_packs, _ = get_available_font_packs()
        current_font = saved_settings.get('font_pack')
        selected_font_val = current_font if current_font in font_packs else None

        return gr.Dropdown(choices=models, value=selected_model_val), gr.Dropdown(choices=font_packs, value=selected_font_val)

def process_batch(input_dir_or_files: Union[str, list], # Type hint added
                  api_key: str, input_language="Japanese", output_language="English",
                  gemini_model="gemini-2.0-flash", font_pack=None, confidence=0.35, dilation_kernel_size=7,
                  dilation_iterations=1, use_otsu_threshold=False, min_contour_area=50, closing_kernel_size=7, # Changed threshold_value
                  closing_iterations=1, constraint_erosion_kernel_size=5, constraint_erosion_iterations=1,
                  temperature=0.1, top_p=0.95, top_k=1,
                  max_font_size=14, min_font_size=8, line_spacing=1.0,
                  use_subpixel_rendering=False, font_hinting="none", use_ligatures=False,
                  verbose=False, selected_model="", jpeg_quality=95,
                  png_compression=6, progress=gr.Progress(), device=None):
    """
    Process a batch of manga images with translation.
    
    This function handles validation of inputs, batch processing of images through
    the translation pipeline, and generation of a status report.
    
    Args:
        input_dir (str or list): Directory path containing images to translate or a list of file paths
        api_key (str): Gemini API key for translation
        input_language (str, optional): Source language. Defaults to "Japanese".
        output_language (str, optional): Target language. Defaults to "English".
        gemini_model (str, optional): Gemini model to use. Defaults to "gemini-2.0-flash".
        font_pack (str, optional): Font pack to use. Defaults to None.
        confidence (float, optional): Bubble detection confidence threshold. Defaults to 0.35.
        dilation_kernel_size (int): Kernel size for dilating initial YOLO mask for ROI.
        dilation_iterations (int): Iterations for dilation.
        use_otsu_threshold (bool): If True, use Otsu's method for thresholding instead of the fixed value (210).
        min_contour_area (int): Minimum area for a contour to be considered part of the bubble interior.
        closing_kernel_size (int): Kernel size for morphological closing to smooth mask and include text.
        closing_iterations (int): Iterations for closing.
        constraint_erosion_kernel_size (int): Kernel size for eroding the original YOLO mask for edge constraint.
        constraint_erosion_iterations (int): Iterations for constraint erosion.
        temperature (float, optional): Temperature for Gemini model. Defaults to 0.1.
        top_p (float, optional): Top-p parameter for Gemini model. Defaults to 0.95.
        top_k (int, optional): Top-k parameter for Gemini model. Defaults to 1.
        max_font_size (int, optional): Max font size for rendering. Defaults to 14.
        min_font_size (int, optional): Min font size for rendering. Defaults to 8.
        line_spacing (float, optional): Line spacing multiplier. Defaults to 1.0.
        use_subpixel_rendering (bool, optional): Enable subpixel rendering. Defaults to False.
        font_hinting (str, optional): Font hinting level ('none', 'slight', 'normal', 'full'). Defaults to "none".
        use_ligatures (bool, optional): Enable standard ligatures. Defaults to False.
        verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
        selected_model (str, optional): YOLO model filename to use. Defaults to "".
        jpeg_quality (int, optional): JPEG quality setting (1-100). Defaults to 95.
        png_compression (int, optional): PNG compression level (0-9). Defaults to 6.
        progress (gr.Progress, optional): Gradio progress component. Defaults to gr.Progress().
        device (torch.device, optional): The target device to run models on.
    
    Returns:
        tuple: 
            - gallery_images (list): List of all processed images to display in the gallery
            - batch_status_msg (str): Detailed status message with processing statistics
    
    Raises:
        gr.Error: If inputs are invalid or processing fails
    """
    start_time = time.time()

    # --- Input Validation ---
    # Common validation (API Key, Model, Font, Rendering Params)
    yolo_model_path, font_dir = _validate_common_inputs(
        api_key, selected_model, font_pack, max_font_size, min_font_size, line_spacing, font_hinting
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path("./output") / timestamp
    
    def update_progress(value, desc="Processing..."):
        progress(value, desc=desc)
    
    temp_dir_path_obj = None
    try:
        # --- Configuration Setup ---
        use_otsu_config_val = use_otsu_threshold

        config = MangaTranslatorConfig(
            yolo_model_path=str(yolo_model_path),
            verbose=verbose,
            device=device or target_device,
            detection=DetectionConfig(
                confidence=confidence
            ),
            cleaning=CleaningConfig(
                dilation_kernel_size=dilation_kernel_size,
                dilation_iterations=dilation_iterations,
                use_otsu_threshold=use_otsu_config_val,
                min_contour_area=min_contour_area,
                closing_kernel_size=closing_kernel_size,
                closing_iterations=closing_iterations,
                constraint_erosion_kernel_size=constraint_erosion_kernel_size,
                constraint_erosion_iterations=constraint_erosion_iterations
            ),
            translation=TranslationConfig(
                api_key=api_key,
                gemini_model=gemini_model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                input_language=input_language,
                output_language=output_language,
            ),
            rendering=RenderingConfig(
                font_dir=str(font_dir),
                max_font_size=max_font_size,
                min_font_size=min_font_size,
                line_spacing=line_spacing,
                use_subpixel_rendering=use_subpixel_rendering,
                font_hinting=font_hinting,
                use_ligatures=use_ligatures
            ),
            output=OutputConfig(
                jpeg_quality=jpeg_quality,
                png_compression=png_compression,
                image_mode="RGBA"
            )
        )

        # --- Execute Batch Processing ---
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
                    image_files_to_copy.append(p)

            if not image_files_to_copy:
                 temp_dir_path_obj.cleanup()
                 raise gr.Error(f"{ERROR_PREFIX}No valid image files (.jpg, .jpeg, .png, .webp) found in the selection.", print_exception=False)

            progress(0.1, desc="Preparing files...")
            for img_file in image_files_to_copy:
                try:
                    # Use copy2 to preserve metadata if possible
                    shutil.copy2(img_file, temp_dir_path / img_file.name)
                except Exception as copy_err:
                    print(f"Warning: Could not copy file {img_file.name}: {copy_err}")

            process_dir = temp_dir_path_obj

        elif isinstance(input_dir_or_files, str):
             input_path = Path(input_dir_or_files)
             if not input_path.exists() or not input_path.is_dir():
                 raise gr.Error(f"{ERROR_PREFIX}Input directory '{input_dir_or_files}' does not exist or is not a directory.", print_exception=False)
             process_dir = input_path
        else:
             raise gr.Error(f"{ERROR_PREFIX}Invalid input type for batch processing.", print_exception=False)


        progress(0.2, desc="Starting batch processing...")

        results = batch_translate_images(
            input_dir=process_dir,
            config=config,
            output_dir=output_path,
            progress_callback=update_progress
        )
        
        # --- Result Handling ---
        progress(0.9, desc="Processing complete, preparing results...")
        
        success_count = results["success_count"]
        error_count = results["error_count"]
        errors = results["errors"]
        total_images = success_count + error_count
        
        gallery_images = []
        if output_path.exists():
            processed_files = list(output_path.glob("*.*"))
            if processed_files:
                processed_files.sort(key=lambda x: tuple(int(part) if part.isdigit() else part for part in re.split(r'(\d+)', x.stem)))
                gallery_images = [str(file_path) for file_path in processed_files]
        
        processing_time = time.time() - start_time
        seconds_per_image = processing_time / success_count if success_count > 0 else 0
        
        error_summary = ""
        if error_count > 0:
            error_summary = f"\n• Warning: {error_count} image(s) failed to process."
            if verbose and errors:
                for filename, error_msg in errors.items():
                    print(f"Error processing {filename}: {error_msg}")
        
        batch_status_msg = (
            f"{SUCCESS_PREFIX}Batch translation completed!\n"
            f"• Source language: {config.translation.input_language}\n"
            f"• Target language: {config.translation.output_language}\n"
            f"• Reading Direction: {config.translation.reading_direction.upper()}\n"
            f"• Font pack: {font_pack}\n"
            f"• Model used: {selected_model}\n"
            f"• Cleaning Params: DKS:{dilation_kernel_size}, DI:{dilation_iterations}, Otsu:{use_otsu_threshold}, "
            f"MCA:{min_contour_area}, CKS:{closing_kernel_size}, CI:{closing_iterations}, "
            f"CEKS:{constraint_erosion_kernel_size}, CEI:{constraint_erosion_iterations}\n"
            f"• Temperature: {temperature}\n"
            f"• Top-p: {top_p}\n"
            f"• Top-k: {top_k}\n"
            f"• Output format: {output_format}\n"
            f"• JPEG quality: {jpeg_quality}\n"
            f"• PNG compression: {png_compression}\n"
            f"• Successful translations: {success_count}/{total_images}{error_summary}\n"
            f"• Total processing time: {processing_time:.2f} seconds ({seconds_per_image:.2f} seconds/image)\n"
            f"• Output directory: {output_path}"
        )
        
        progress(1.0, desc="Batch processing complete")
        return gallery_images, batch_status_msg

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"{ERROR_PREFIX}An unexpected error occurred during batch processing: {str(e)}", print_exception=True)
    finally:
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
    translated_image_state = gr.State(value=None)
    
    gr.Markdown("# Manga Translator")
    
    model_choices = get_available_models(MODELS_DIR)
    font_choices, initial_default_font = get_available_font_packs()
    saved_settings = get_saved_settings()

    saved_model_name = saved_settings.get('yolo_model')
    default_model = saved_model_name if saved_model_name in model_choices else (model_choices[0] if model_choices else None)

    saved_font_pack = saved_settings.get('font_pack')
    default_font = saved_font_pack if saved_font_pack in font_choices else (initial_default_font if initial_default_font else None)
    batch_saved_font_pack = saved_settings.get('batch_font_pack')
    batch_default_font = batch_saved_font_pack if batch_saved_font_pack in font_choices else (initial_default_font if initial_default_font else None)

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
                    
                    with gr.Row():
                        gemini_model = gr.Dropdown(
                            ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-pro-exp-02-05", "gemini-2.5-pro-exp-03-25"], 
                            label="Model (Gemini)", 
                            value=saved_settings.get("gemini_model", "gemini-2.0-flash")
                        )
                    
                    with gr.Row():
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
                        clear_button = gr.ClearButton([input_image, output_image, status_message, translated_image_state], value="Clear")
        
        with gr.TabItem("Batch"):
            model_choices_batch = get_available_models(MODELS_DIR)
            font_choices_batch, initial_default_font_batch = get_available_font_packs()
            saved_settings_batch = get_saved_settings()
            
            saved_model_name_batch = saved_settings_batch.get('yolo_model')
            default_model_batch = saved_model_name_batch if saved_model_name_batch in model_choices_batch else (model_choices_batch[0] if model_choices_batch else None)

            saved_font_pack_batch = saved_settings_batch.get('batch_font_pack')
            default_font_batch = saved_font_pack_batch if saved_font_pack_batch in font_choices_batch else (initial_default_font_batch if initial_default_font_batch else None)
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_dir = gr.File(
                        label="Upload Directory",
                        file_count="directory",
                        type="filepath"
                    )
                    
                    with gr.Row():
                        batch_gemini_model = gr.Dropdown(
                            ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-thinking-exp-01-21"], 
                            label="Model (Gemini)", 
                            value=saved_settings.get("batch_gemini_model", "gemini-2.0-flash")
                        )
                    
                    with gr.Row():
                        batch_font_dropdown = gr.Dropdown(
                            choices=font_choices,
                            label="Text Font",
                            value=default_font_batch,
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
                        batch_clear_button = gr.ClearButton([input_dir, batch_output_gallery, batch_status_message], value="Clear")
        
        with gr.TabItem("Config", elem_id="settings-tab-container"):
            saved_settings = get_saved_settings()

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
                        model_dropdown = gr.Dropdown(
                            choices=model_choices, label="YOLO Model",
                            value=default_model,
                            filterable=False
                        )
                        confidence = gr.Slider(
                            0.1, 1.0, value=saved_settings.get("confidence", 0.35), step=0.05,
                            label="Bubble Detection Confidence"
                        )
                        config_reading_direction = gr.Radio(
                            choices=["rtl", "ltr"],
                            label="Reading Direction",
                            value=saved_settings.get("reading_direction", "rtl"),
                            info="Order for sorting bubbles (rtl=Manga, ltr=Comics).",
                            elem_id="config_reading_direction"
                        )
                    setting_groups.append(group_detection)

                    with gr.Group(visible=False, elem_classes="settings-group") as group_cleaning:
                        gr.Markdown("### Mask Cleaning & Refinement")
                        gr.Markdown("#### ROI Expansion")
                        dilation_kernel_size = gr.Slider(
                            1, 15, 
                            value=saved_settings.get("dilation_kernel_size", 7), 
                            step=2, 
                            label="Dilation Kernel Size", 
                            info="Controls how much the initial bubble detection grows outwards. Increase if the bubble's edge or text is missed."
                        )
                        dilation_iterations = gr.Slider(
                            1, 5, 
                            value=saved_settings.get("dilation_iterations", 1), 
                            step=1, 
                            label="Dilation Iterations", 
                            info="Controls how many times to apply the above bubble detection growth."
                        )
                        gr.Markdown("#### Bubble Interior Isolation")
                        use_otsu_threshold = gr.Checkbox(
                            value=saved_settings.get("use_otsu_threshold", False),
                            label="Use Automatic Thresholding (Otsu)",
                            info="Automatically determine the brightness threshold instead of using the fixed value (210). Recommended for varied lighting."
                        )
                        min_contour_area = gr.Slider(
                            0, 500, 
                            value=saved_settings.get("min_contour_area", 50), 
                            step=10, 
                            label="Min Bubble Area", 
                            info="Removes tiny detected regions. Increase to ignore small specks or dots inside the bubble area."
                        )
                        gr.Markdown("#### Mask Smoothing")
                        closing_kernel_size = gr.Slider(
                            1, 15, 
                            value=saved_settings.get("closing_kernel_size", 7), 
                            step=2, 
                            label="Closing Kernel Size", 
                            info="Controls the size of gaps between outline and text to fill. Increase to capture text very close to the bubble outline"
                        )
                        closing_iterations = gr.Slider(
                            1, 5, 
                            value=saved_settings.get("closing_iterations", 1), 
                            step=1, 
                            label="Closing Iterations", 
                            info="Controls how many times to apply the above gap filling."
                        )
                        gr.Markdown("#### Edge Constraint")
                        constraint_erosion_kernel_size = gr.Slider(
                            1, 12, 
                            value=saved_settings.get("constraint_erosion_kernel_size", 5), 
                            step=2, 
                            label="Constraint Erosion Kernel Size", 
                            info="Controls how much to shrink the cleaned mask inwards. Increase if the bubble's outline is being erased."
                        )
                        constraint_erosion_iterations = gr.Slider(
                            1, 3, 
                            value=saved_settings.get("constraint_erosion_iterations", 1), 
                            step=1, 
                            label="Constraint Erosion Iterations", 
                            info="Controls how many times to apply the above shrinkage."
                        )
                    setting_groups.append(group_cleaning)

                    with gr.Group(visible=False, elem_classes="settings-group") as group_translation:
                        gr.Markdown("### LLM Settings")
                        api_key = gr.Textbox(
                            label="Gemini API Key", placeholder="Enter your Gemini API key", type="password",
                            value=saved_settings.get('gemini_api_key', ''), show_copy_button=False,
                            info="API keys are stored locally"
                        )
                        
                        temperature = gr.Slider(
                            0, 2, value=saved_settings.get("temperature", 0.1), step=0.05,
                            label="Temperature"
                        )
                        top_p = gr.Slider(
                            0, 1, value=saved_settings.get("top_p", 0.95), step=0.05,
                            label="Top P"
                        )
                        top_k = gr.Slider(
                            0, 64, value=saved_settings.get("top_k", 1), step=1,
                            label="Top K"
                        )
                    setting_groups.append(group_translation)

                    with gr.Group(visible=False, elem_classes="settings-group") as group_rendering:
                        gr.Markdown("### Text Rendering")
                        max_font_size = gr.Slider(
                            5, 50, value=saved_settings.get("max_font_size", 14), step=1,
                            label="Max Font Size (px)",
                            info="The largest font size the renderer will attempt."
                        )
                        min_font_size = gr.Slider(
                             5, 50, value=saved_settings.get("min_font_size", 8), step=1,
                             label="Min Font Size (px)",
                             info="The smallest font size the renderer will attempt before giving up."
                        )
                        line_spacing = gr.Slider(
                            0.5, 2.0, value=saved_settings.get("line_spacing", 1.0), step=0.05,
                            label="Line Spacing Multiplier",
                            info="Adjusts the vertical space between lines of text (1.0 = standard)."
                        )
                        use_subpixel_rendering = gr.Checkbox(
                            value=saved_settings.get("use_subpixel_rendering", False),
                            label="Use Subpixel Rendering",
                            info="May improve text clarity on some LCD screens, but can cause color fringing."
                        )
                        font_hinting = gr.Radio(
                            choices=["none", "slight", "normal", "full"],
                            value=saved_settings.get("font_hinting", "none"),
                            label="Font Hinting",
                            info="Adjusts glyph outlines to fit pixel grid. 'None' is often best for high-res displays."
                        )
                        use_ligatures = gr.Checkbox(
                            value=saved_settings.get("use_ligatures", False),
                            label="Use Standard Ligatures (e.g., fi, fl)",
                            info="Enables common letter combinations to be rendered as single glyphs if supported by the font."
                        )
                    setting_groups.append(group_rendering)

                    with gr.Group(visible=False, elem_classes="settings-group") as group_output:
                        gr.Markdown("### Image Output")
                        output_format = gr.Radio(
                            choices=["auto", "png", "jpeg"], label="Image Output Format",
                            value=saved_settings.get("output_format", "auto"),
                            info="'auto' uses the same format as the input image"
                        )
                        jpeg_quality = gr.Slider(
                            1, 100, value=saved_settings.get("jpeg_quality", 95), step=1,
                            label="JPEG Quality (higher = better quality)",
                            interactive=saved_settings.get("output_format", "auto") != "png"
                        )
                        png_compression = gr.Slider(
                            0, 9, value=saved_settings.get("png_compression", 6), step=1,
                            label="PNG Compression Level (higher = smaller file)",
                            interactive=saved_settings.get("output_format", "auto") != "jpeg"
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
                            value=saved_settings.get("verbose", False),
                            label="Verbose Logging"
                        )
                    setting_groups.append(group_other)

            def switch_settings_view(selected_group_index):
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

            output_components_for_switch = setting_groups + nav_buttons
            
            nav_button_detection.click(fn=partial(switch_settings_view, 0), outputs=output_components_for_switch, queue=False)
            nav_button_cleaning.click(fn=partial(switch_settings_view, 1), outputs=output_components_for_switch, queue=False)
            nav_button_translation.click(fn=partial(switch_settings_view, 2), outputs=output_components_for_switch, queue=False)
            nav_button_rendering.click(fn=partial(switch_settings_view, 3), outputs=output_components_for_switch, queue=False)
            nav_button_output.click(fn=partial(switch_settings_view, 4), outputs=output_components_for_switch, queue=False)
            nav_button_other.click(fn=partial(switch_settings_view, 5), outputs=output_components_for_switch, queue=False)

            output_format.change(
                fn=lambda format: [gr.update(interactive=format != "png"), gr.update(interactive=format != "jpeg")],
                inputs=output_format, outputs=[jpeg_quality, png_compression]
            )
        
        with gr.TabItem("Setup"):
            gr.Markdown("## Initial Setup")
            gr.Markdown("""
            1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/)
            2. **Optional**: Place a custom YOLO model in the 'models' directory.
               - The default model is [yolov8m_seg-speech-bubble.pt](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
            3. Place fonts in their own folders inside the 'fonts' directory.
               - Each font should have its own dedicated folder containing all its variants (regular, bold, italic, etc.)
               - The default font "CC Wild Words" includes Roman, Italic, and Bold Italic variants
               <br><br>
            """)
            
            gr.Markdown("""
            ## Font Management
            
            You can add more fonts by placing folders containing font files (and their variants) into the 'fonts' directory.
            
            Each subdirectory in the 'fonts' directory will be treated as a separate font pack in the dropdown menu.
            
            Example directory structure:
            ```
            fonts/
            ├── CC Wild Words/
            │   ├── CC Wild Words Roman.ttf
            │   ├── CC Wild Words Bold.ttf
            │   └── CC Wild Words Bold Italic.ttf
            └── Another Font/
                ├── AnotherFont-Regular.ttf
                ├── AnotherFont-Bold.ttf
                └── AnotherFont-BoldItalic.ttf
            ```
            <br>
            """)
            
            gr.Markdown("""
            ## Using Your Own YOLO Model
            
            If you want to use a custom YOLO model, place the model file in the 'models' folder, refresh the models list (via the Other section) and select your model from the dropdown in the Config tab.
            
            **Note: The YOLO model must have segmentation and be trained *specifically* for speech bubble detection.**
            """)

    save_config_btn.click(
        fn=lambda api_key_val, mdl_dd_val, conf_val, config_rd_val,
                 dks_val, di_val, otsu_val, mca_val, cks_val, ci_val, ceks_val, cei_val,
                 max_fs_val, min_fs_val, ls_val, subpix_val, hint_val, liga_val,
                 temp_val, tp_val, tk_val,
                 out_fmt_val, jpeg_q_val, png_comp_val, verb_val,
                 input_lang_val, output_lang_val, gm_val, fp_val,
                 b_input_lang_val, b_output_lang_val, b_gm_val, b_fp_val:
            save_config({
                'gemini_api_key': api_key_val, 'yolo_model': mdl_dd_val,
                'confidence': conf_val,
                'dilation_kernel_size': dks_val, 'dilation_iterations': di_val,
                'use_otsu_threshold': otsu_val, 'min_contour_area': mca_val,
                'closing_kernel_size': cks_val, 'closing_iterations': ci_val,
                'constraint_erosion_kernel_size': ceks_val, 'constraint_erosion_iterations': cei_val,
                'max_font_size': max_fs_val, 'min_font_size': min_fs_val, 'line_spacing': ls_val, 'use_subpixel_rendering': subpix_val, 'font_hinting': hint_val, 'use_ligatures': liga_val,
                'temperature': temp_val, 'top_p': tp_val, 'top_k': tk_val,
                'output_format': out_fmt_val, 'jpeg_quality': jpeg_q_val, 'png_compression': png_comp_val,
                'verbose': verb_val,
                'input_language': input_lang_val, 'output_language': output_lang_val, 'gemini_model': gm_val, 'font_pack': fp_val,
                'reading_direction': config_rd_val,
                'batch_input_language': b_input_lang_val, 'batch_output_language': b_output_lang_val, 'batch_gemini_model': b_gm_val, 'batch_font_pack': b_fp_val,
                'batch_reading_direction': config_rd_val
            })[1],
        inputs=[
            api_key, model_dropdown, confidence, config_reading_direction,
            dilation_kernel_size, dilation_iterations, use_otsu_threshold, min_contour_area,
            closing_kernel_size, closing_iterations, constraint_erosion_kernel_size, constraint_erosion_iterations,
            max_font_size, min_font_size, line_spacing, use_subpixel_rendering, font_hinting, use_ligatures,
            temperature, top_p, top_k,
            output_format, jpeg_quality, png_compression, verbose,
            input_language, output_language, gemini_model, font_dropdown,
            batch_input_language, batch_output_language, batch_gemini_model, batch_font_dropdown
        ],
        outputs=[config_status],
        queue=False
    ).then(
        fn=lambda: None,
        inputs=None, outputs=None,
        js="""
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
        """,
        queue=False
    )

    def apply_defaults():
        defaults = reset_to_defaults()
        available_models = get_available_models(MODELS_DIR)
        reset_model = defaults.get('yolo_model') if defaults.get('yolo_model') in available_models else (available_models[0] if available_models else None)
        available_fonts, _ = get_available_font_packs()
        reset_font = defaults.get('font_pack') if defaults.get('font_pack') in available_fonts else (available_fonts[0] if available_fonts else None)
        batch_reset_font = defaults.get('batch_font_pack') if defaults.get('batch_font_pack') in available_fonts else (available_fonts[0] if available_fonts else None)
        return [
            reset_model,
            defaults['confidence'], defaults['reading_direction'],
            defaults['dilation_kernel_size'], defaults['dilation_iterations'],
            defaults['use_otsu_threshold'], defaults['min_contour_area'],
            defaults['closing_kernel_size'], defaults['closing_iterations'],
            defaults['constraint_erosion_kernel_size'], defaults['constraint_erosion_iterations'],
            defaults['max_font_size'], defaults['min_font_size'], defaults['line_spacing'], defaults['use_subpixel_rendering'], defaults['font_hinting'], defaults['use_ligatures'],
            defaults.get('gemini_api_key', ''), defaults['temperature'], defaults['top_p'], defaults['top_k'],
            defaults['output_format'], defaults['jpeg_quality'], defaults['png_compression'],
            defaults['verbose'],
            defaults['input_language'], defaults['output_language'], defaults['gemini_model'],
            reset_font,
            defaults['batch_input_language'], defaults['batch_output_language'],
            defaults['batch_gemini_model'], batch_reset_font,
            f"{SUCCESS_PREFIX}Settings reset to defaults"
        ]

    reset_defaults_btn.click(
        fn=apply_defaults, inputs=[],
        outputs=[
            model_dropdown,
            confidence, config_reading_direction,
            dilation_kernel_size, dilation_iterations, use_otsu_threshold, min_contour_area,
            closing_kernel_size, closing_iterations, constraint_erosion_kernel_size, constraint_erosion_iterations,
            max_font_size, min_font_size, line_spacing, use_subpixel_rendering, font_hinting, use_ligatures,
            api_key, temperature, top_p, top_k,
            output_format, jpeg_quality, png_compression,
            verbose,
            input_language, output_language, gemini_model, font_dropdown,
            batch_input_language, batch_output_language, batch_gemini_model, batch_font_dropdown,
            config_status
        ], queue=False
    )

    translate_button.click(
        fn=lambda: gr.update(interactive=False, value="Translating..."), outputs=translate_button, queue=False
    ).then(
        partial(translate_manga, device=target_device),
        inputs=[
            input_image,
            api_key,
            input_language,
            output_language,
            gemini_model,
            confidence,
            dilation_kernel_size, dilation_iterations, use_otsu_threshold, min_contour_area,
            closing_kernel_size, closing_iterations, constraint_erosion_kernel_size, constraint_erosion_iterations,
            temperature,
            top_p,
            top_k,
            model_dropdown,
            font_dropdown,
            max_font_size, min_font_size, line_spacing, use_subpixel_rendering, font_hinting, use_ligatures,
            verbose,
            jpeg_quality,
            png_compression,
            output_format
        ],
        outputs=[output_image, status_message]
    ).then(
        fn=lambda img, _: img, inputs=[output_image, status_message], outputs=[translated_image_state], queue=False
    ).then(
        fn=lambda: gr.update(interactive=True, value="Translate"), outputs=translate_button, queue=False
    )

    batch_process_button.click(
        fn=lambda: gr.update(interactive=False, value="Processing..."), outputs=batch_process_button, queue=False
    ).then(
        partial(process_batch, device=target_device),
        inputs=[
            input_dir,
            api_key,
            batch_input_language,
            batch_output_language,
            batch_gemini_model,
            batch_font_dropdown,
            confidence,
            dilation_kernel_size, dilation_iterations, use_otsu_threshold, min_contour_area,
            closing_kernel_size, closing_iterations, constraint_erosion_kernel_size, constraint_erosion_iterations,
            temperature,
            top_p,
            top_k,
            max_font_size, min_font_size, line_spacing, use_subpixel_rendering, font_hinting, use_ligatures,
            verbose,
            model_dropdown,
            jpeg_quality,
            png_compression
        ],
        outputs=[batch_output_gallery, batch_status_message]
    ).then(
        fn=lambda: gr.update(interactive=True, value="Start Batch Processing"), outputs=batch_process_button, queue=False
    )
    
    refresh_resources_button.click(
        fn=refresh_models_and_fonts,
        inputs=[],
        outputs=[model_dropdown, font_dropdown],
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
        lambda model_update, font_update: [model_update, font_update],
        inputs=[model_dropdown, font_dropdown],
        outputs=[model_dropdown, batch_font_dropdown]
    ).then(
        lambda model_update, font_update: [model_update, font_update],
        inputs=[model_dropdown, font_dropdown],
        outputs=[model_dropdown, font_dropdown]
    ).then(
        fn=None,
        inputs=None, outputs=None,
        js="""
            function() {
                setTimeout(() => {
                    const refreshButton = document.querySelector('.config-refresh-button button');
                     if (refreshButton) {
                        refreshButton.textContent = 'Refresh Models / Fonts';
                        refreshButton.disabled = false;
                    }
                }, 100); // Small delay
            }
        """,
        queue=False
    )

    app.queue()

if __name__ == "__main__":
    app.launch(inbrowser=args.open_browser, server_port=args.port, show_error=True)
