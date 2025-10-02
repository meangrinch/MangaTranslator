import argparse
import base64
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import cv2
import torch
from PIL import Image

from core.models import (
    DetectionConfig,
    CleaningConfig,
    TranslationConfig,
    RenderingConfig,
    OutputConfig,
    MangaTranslatorConfig,
)
from core.validation import autodetect_yolo_model_path
from utils.logging import log_message

from .cleaning import clean_speech_bubbles
from .detection import detect_speech_bubbles
from .image_utils import cv2_to_pil, pil_to_cv2, save_image_with_compression
from .rendering import render_text_skia
from .translation import call_translation_api_batch, sort_bubbles_by_reading_order


def translate_and_render(
    image_path: Union[str, Path], config: MangaTranslatorConfig, output_path: Optional[Union[str, Path]] = None
):
    """
    Main function to translate manga speech bubbles and render translations using a config object.

    Args:
        image_path (str or Path): Path to input image
        config (MangaTranslatorConfig): Configuration object containing all settings.
        output_path (str or Path, optional): Path to save the final image. If None, image is not saved.

    Returns:
        PIL.Image: Final translated image
    """
    start_time = time.time()
    image_path = Path(image_path)
    verbose = config.verbose
    device = config.device

    log_message(f"Using device: {device}", verbose=verbose)

    try:
        pil_original = Image.open(image_path)
    except FileNotFoundError:
        log_message(f"Error: Input image not found at {image_path}", always_print=True)
        raise
    except Exception as e:
        log_message(f"Error opening image {image_path}: {e}", always_print=True)
        raise

    desired_format = config.output.output_format
    output_ext_for_mode = Path(output_path).suffix.lower() if output_path else image_path.suffix.lower()

    if desired_format == "jpeg" or (desired_format == "auto" and output_ext_for_mode in [".jpg", ".jpeg"]):
        target_mode = "RGB"
    else:  # Default to RGBA for PNG, WEBP, or other formats in auto mode
        target_mode = "RGBA"
    log_message(f"Target mode: {target_mode}", verbose=verbose)

    pil_image_processed = pil_original
    if pil_image_processed.mode != target_mode:
        if target_mode == "RGB":
            if (
                pil_image_processed.mode == "RGBA"
                or pil_image_processed.mode == "LA"
                or (pil_image_processed.mode == "P" and "transparency" in pil_image_processed.info)
            ):
                log_message(f"Converting {pil_image_processed.mode} to RGB (flattening transparency)", verbose=verbose)
                background = Image.new("RGB", pil_image_processed.size, (255, 255, 255))
                try:
                    mask = None
                    if pil_image_processed.mode == "RGBA":
                        mask = pil_image_processed.split()[3]
                    elif pil_image_processed.mode == "LA":
                        mask = pil_image_processed.split()[1]
                    elif pil_image_processed.mode == "P" and "transparency" in pil_image_processed.info:
                        temp_rgba = pil_image_processed.convert("RGBA")
                        mask = temp_rgba.split()[3]

                    if mask:
                        background.paste(pil_image_processed, mask=mask)
                        pil_image_processed = background
                    else:
                        pil_image_processed = pil_image_processed.convert("RGB")
                except Exception as paste_err:
                    log_message(f"Warning: Paste failed, trying alpha_composite: {paste_err}", verbose=verbose)
                    try:
                        background_comp = Image.new("RGB", pil_image_processed.size, (255, 255, 255))
                        img_rgba_for_composite = (
                            pil_image_processed
                            if pil_image_processed.mode == "RGBA"
                            else pil_image_processed.convert("RGBA")
                        )
                        pil_image_processed = Image.alpha_composite(
                            background_comp.convert("RGBA"), img_rgba_for_composite
                        ).convert("RGB")
                        log_message("Alpha composite conversion successful", verbose=verbose)
                    except Exception as composite_err:
                        log_message(
                            f"Warning: Alpha composite failed, using simple convert: {composite_err}",
                            verbose=verbose
                        )
                        pil_image_processed = pil_image_processed.convert("RGB")  # Final fallback conversion
            else:  # Non-transparent conversion to RGB
                log_message(f"Converting {pil_image_processed.mode} to RGB", verbose=verbose)
                pil_image_processed = pil_image_processed.convert("RGB")
        elif target_mode == "RGBA":
            log_message(f"Converting {pil_image_processed.mode} to RGBA", verbose=verbose)
            pil_image_processed = pil_image_processed.convert("RGBA")

    original_cv_image = pil_to_cv2(pil_image_processed)

    # --- Encode Full Image for Context (optional) ---
    full_image_b64 = None
    if config.translation.send_full_page_context:
        try:
            is_success, buffer = cv2.imencode(".jpg", original_cv_image)
            if not is_success:
                raise ValueError("Full image encoding to JPEG failed")
            full_image_b64 = base64.b64encode(buffer).decode("utf-8")
            log_message("Encoded full image for context", verbose=verbose)
        except Exception as e:
            log_message(f"Warning: Failed to encode full image context: {e}", always_print=True)

    # --- Detection ---
    log_message("Detecting speech bubbles...", verbose=verbose)
    try:
        bubble_data = detect_speech_bubbles(
            image_path,
            config.yolo_model_path,
            config.detection.confidence,
            verbose=verbose,
            device=device,
            use_sam2=config.detection.use_sam2,
            sam2_model_id=config.detection.sam2_model_id,
        )
    except Exception as e:
        log_message(f"Error during detection: {e}", always_print=True)
        bubble_data = []

    final_image_to_save = pil_image_processed

    # --- Proceed only if bubbles were detected ---
    if not bubble_data:
        log_message("No speech bubbles detected", always_print=True)
    else:
        log_message(f"Detected {len(bubble_data)} bubbles", verbose=verbose)

        # --- Cleaning ---
        log_message("Cleaning speech bubbles...", verbose=verbose)
        try:
            use_otsu = config.cleaning.use_otsu_threshold

            cleaned_image_cv, processed_bubbles_info = clean_speech_bubbles(
                image_path,
                config.yolo_model_path,
                config.detection.confidence,
                pre_computed_detections=bubble_data,
                device=device,
                thresholding_value=config.cleaning.thresholding_value,
                use_otsu_threshold=use_otsu,
                roi_shrink_px=config.cleaning.roi_shrink_px,
                verbose=verbose,
            )
            log_message(f"Cleaned {len(processed_bubbles_info)} bubbles", verbose=verbose)
        except Exception as e:
            log_message(f"Error during cleaning: {e}", always_print=True)
            cleaned_image_cv = original_cv_image.copy()
            processed_bubbles_info = []

        pil_cleaned_image = cv2_to_pil(cleaned_image_cv)
        if pil_cleaned_image.mode != target_mode:
            log_message(f"Converting cleaned image to {target_mode}", verbose=verbose)
            pil_cleaned_image = pil_cleaned_image.convert(target_mode)
        final_image_to_save = pil_cleaned_image

        # --- Check for Cleaning Only Mode ---
        if config.cleaning_only:
            log_message("Cleaning only mode - skipping translation", always_print=True)
        else:
            # --- Prepare images for Translation ---
            log_message("Preparing bubble images...", verbose=verbose)
            for bubble in bubble_data:
                x1, y1, x2, y2 = bubble["bbox"]

                bubble_image_cv = original_cv_image[y1:y2, x1:x2].copy()

                try:
                    is_success, buffer = cv2.imencode(".jpg", bubble_image_cv)
                    if not is_success:
                        raise ValueError("cv2.imencode failed")
                    image_b64 = base64.b64encode(buffer).decode("utf-8")
                    bubble["image_b64"] = image_b64
                    log_message(f"Bubble {x1},{y1} ({x2 - x1}x{y2 - y1})", verbose=verbose)
                except Exception as e:
                    log_message(f"Error encoding bubble {bubble['bbox']}: {e}", verbose=verbose)
                    bubble["image_b64"] = None

            valid_bubble_data = [b for b in bubble_data if b.get("image_b64")]
            if not valid_bubble_data:
                log_message("No valid bubble images for translation", always_print=True)
            else:  # Only proceed if we have valid bubble data
                # --- Sort and Translate ---
                reading_direction = config.translation.reading_direction
                log_message(f"Sorting bubbles ({reading_direction.upper()})", verbose=verbose)

                sorted_bubble_data = sort_bubbles_by_reading_order(
                    valid_bubble_data, reading_direction
                )

                bubble_images_b64 = [bubble["image_b64"] for bubble in sorted_bubble_data]
                translated_texts = []
                if not bubble_images_b64:
                    log_message("No valid bubbles after sorting", always_print=True)
                else:
                    log_message(
                        f"Translating {len(bubble_images_b64)} bubbles: "
                        f"{config.translation.input_language} → {config.translation.output_language}",
                        always_print=True,
                    )
                    try:
                        translated_texts = call_translation_api_batch(
                            config=config.translation,
                            images_b64=bubble_images_b64,
                            full_image_b64=full_image_b64 or "",
                            debug=verbose,
                        )
                    except Exception as e:
                        log_message(f"Translation API error: {e}", always_print=True)
                        translated_texts = [
                            "[Translation Error: API call raised exception]" for _ in sorted_bubble_data
                        ]

                # --- Render Translations ---
                bubble_render_info_map = {
                    tuple(info["bbox"]): {
                        "color": info["color"],
                        "mask": info.get("mask"),
                    }
                    for info in processed_bubbles_info
                    if "bbox" in info and "color" in info and "mask" in info
                }
                log_message("Rendering translations...", verbose=verbose)
                if len(translated_texts) == len(sorted_bubble_data):
                    for i, bubble in enumerate(sorted_bubble_data):
                        bubble["translation"] = translated_texts[i]
                        bbox = bubble["bbox"]
                        text = bubble.get("translation", "")

                        if (
                            not text
                            or text.startswith("API Error")
                            or text.startswith("[Translation Error]")
                            or text.startswith("[Translation Error:")
                            or text.strip() in {
                                "[OCR FAILED]",
                                "[Empty response / no content]",
                                f"[{config.translation.provider}: API call failed/blocked]",
                                f"[{config.translation.provider}: OCR call failed/blocked]",
                                f"[{config.translation.provider}: Failed to parse response]",
                            }
                        ):
                            log_message(f"Skipping bubble {bbox} - invalid translation", verbose=verbose)
                            continue

                        log_message(f"Rendering bubble {bbox}: '{text[:30]}...'", verbose=verbose)

                        render_info = bubble_render_info_map.get(tuple(bbox))
                        bubble_color_bgr = (255, 255, 255)
                        cleaned_mask = None
                        if render_info:
                            bubble_color_bgr = render_info["color"]
                            cleaned_mask = render_info.get("mask")

                        rendered_image, success = render_text_skia(
                            pil_image=pil_cleaned_image,
                            text=text,
                            bbox=bbox,
                            font_dir=config.rendering.font_dir,
                            cleaned_mask=cleaned_mask,
                            bubble_color_bgr=bubble_color_bgr,
                            min_font_size=config.rendering.min_font_size,
                            max_font_size=config.rendering.max_font_size,
                            line_spacing_mult=config.rendering.line_spacing,
                            use_subpixel_rendering=config.rendering.use_subpixel_rendering,
                            font_hinting=config.rendering.font_hinting,
                            use_ligatures=config.rendering.use_ligatures,
                            hyphenate_before_scaling=config.rendering.hyphenate_before_scaling,
                            hyphen_penalty=config.rendering.hyphen_penalty,
                            hyphenation_min_word_length=config.rendering.hyphenation_min_word_length,
                            badness_exponent=config.rendering.badness_exponent,
                            padding_pixels=config.rendering.padding_pixels,
                            verbose=verbose,
                        )

                        if success:
                            pil_cleaned_image = rendered_image
                            final_image_to_save = pil_cleaned_image
                        else:
                            log_message(f"Failed to render bubble {bbox}", verbose=verbose)
                else:
                    log_message(
                        f"Warning: Bubble/translation count mismatch "
                        f"({len(sorted_bubble_data)}/{len(translated_texts)})",
                        always_print=True,
                    )

    # --- Save Output ---
    if output_path:
        if final_image_to_save.mode != target_mode:
            log_message(f"Converting final image to {target_mode}", verbose=verbose)
            final_image_to_save = final_image_to_save.convert(target_mode)

        save_image_with_compression(
            final_image_to_save,
            output_path,
            jpeg_quality=config.output.jpeg_quality,
            png_compression=config.output.png_compression,
            verbose=verbose,
        )

    end_time = time.time()
    processing_time = end_time - start_time
    log_message(f"Processing completed in {processing_time:.2f}s", always_print=True)

    return final_image_to_save


def batch_translate_images(
    input_dir: Union[str, Path],
    config: MangaTranslatorConfig,
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Process all images in a directory using a configuration object.

    Args:
        input_dir (str or Path): Directory containing images to process
        config (MangaTranslatorConfig): Configuration object containing all settings.
        output_dir (str or Path, optional): Directory to save translated images.
                                            If None, uses input_dir / "output_translated".
        progress_callback (callable, optional): Function to call with progress updates (0.0-1.0, message).

    Returns:
        dict: Processing results with keys:
            - "success_count": Number of successfully processed images
            - "error_count": Number of images that failed to process
            - "errors": Dictionary mapping filenames to error messages
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        log_message(f"Input path '{input_dir}' is not a directory", always_print=True)
        return {"success_count": 0, "error_count": 0, "errors": {}}

    if output_dir:
        output_dir = Path(output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./output") / timestamp

    os.makedirs(output_dir, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        log_message(f"No image files found in '{input_dir}'", always_print=True)
        return {"success_count": 0, "error_count": 0, "errors": {}}

    results = {"success_count": 0, "error_count": 0, "errors": {}}

    total_images = len(image_files)
    start_batch_time = time.time()

    log_message(f"Starting batch processing: {total_images} images", always_print=True)

    if progress_callback:
        progress_callback(0.0, f"Starting batch processing of {total_images} images...")

    for i, img_path in enumerate(image_files):
        try:
            if progress_callback:
                current_progress = i / total_images
                progress_callback(current_progress, f"Processing image {i + 1}/{total_images}: {img_path.name}")

            original_ext = img_path.suffix.lower()
            desired_format = config.output.output_format
            if desired_format == "jpeg":
                output_ext = ".jpg"
            elif desired_format == "png":
                output_ext = ".png"
            elif desired_format == "auto":
                output_ext = original_ext
            else:
                output_ext = original_ext
                log_message(
                    f"Warning: Invalid output_format '{desired_format}' in config. "
                    f"Using original extension '{original_ext}'.",
                    always_print=True,
                )

            output_path = output_dir / f"{img_path.stem}_translated{output_ext}"
            log_message(f"Processing {i + 1}/{total_images}: {img_path.name}", always_print=True)

            translate_and_render(img_path, config, output_path)

            results["success_count"] += 1

            if progress_callback:
                completed_progress = (i + 1) / total_images
                progress_callback(completed_progress, f"Completed {i + 1}/{total_images} images")

        except Exception as e:
            log_message(f"Error processing {img_path.name}: {str(e)}", always_print=True)
            results["error_count"] += 1
            results["errors"][img_path.name] = str(e)

            if progress_callback:
                completed_progress = (i + 1) / total_images
                progress_callback(completed_progress, f"Completed {i + 1}/{total_images} images (with errors)")

    if progress_callback:
        progress_callback(1.0, "Processing complete")

    end_batch_time = time.time()
    total_batch_time = end_batch_time - start_batch_time
    seconds_per_image = total_batch_time / total_images if total_images > 0 else 0

    log_message(
        f"Batch complete: {results['success_count']}/{total_images} images in "
        f"{total_batch_time:.2f}s ({seconds_per_image:.2f}s/image)",
        always_print=True,
    )
    if results["error_count"] > 0:
        log_message(f"Failed: {results['error_count']} images", always_print=True)
        for filename, error_msg in results["errors"].items():
            log_message(f"  - {filename}: {error_msg}", always_print=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Translate manga/comic speech bubbles using a configuration approach")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or directory (if using --batch)")
    parser.add_argument(
        "--output", type=str, required=False, help="Path to save the translated image or directory (if using --batch)"
    )
    parser.add_argument("--batch", action="store_true", help="Process all images in the input directory")
    # --- Provider and API Key Arguments ---
    parser.add_argument(
        "--provider",
        type=str,
        default="Google",
        choices=["Google", "OpenAI", "Anthropic", "xAI", "OpenRouter", "OpenAI-Compatible"],
        help="LLM provider to use for translation",
    )
    parser.add_argument(
        "--google-api-key",
        dest="google_api_key",
        type=str,
        default=None,
        help="Google API key (overrides GOOGLE_API_KEY env var if --provider is Google)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var if --provider is OpenAI)",
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var if --provider is Anthropic)",
    )
    parser.add_argument(
        "--xai-api-key",
        type=str,
        default=None,
        help="xAI API key (overrides XAI_API_KEY env var if --provider is xAI)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=None,
        help="OpenRouter API key (overrides OPENROUTER_API_KEY env var if --provider is OpenRouter)",
    )
    parser.add_argument(
        "--openai-compatible-url",
        type=str,
        default="http://localhost:11434/v1",
        help="Base URL for the OpenAI-Compatible endpoint (e.g., http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--openai-compatible-api-key",
        type=str,
        default=None,
        help="Optional API key for the OpenAI-Compatible endpoint (overrides OPENAI_COMPATIBLE_API_KEY env var)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for the selected provider (e.g., 'gemini-2.0-flash'). "
        "If not provided, a default will be attempted based on the provider.",
    )
    parser.add_argument("--font-dir", type=str, default="./fonts", help="Directory containing font files")
    parser.add_argument(
        "--input-language",
        type=str,
        default="Japanese",
        help="Source language (use 'Auto' to autodetect)",
    )
    parser.add_argument("--output-language", type=str, default="English", help="Target language")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for detection")
    parser.add_argument(
        "--no-sam2",
        dest="use_sam2",
        action="store_false",
        help="Disable SAM 2.1 guided segmentation",
    )
    parser.set_defaults(use_sam2=True)
    parser.add_argument(
        "--reading-direction",
        type=str,
        default="rtl",
        choices=["rtl", "ltr"],
        help="Reading direction for sorting bubbles (rtl or ltr)",
    )
    # Cleaning args
    parser.add_argument(
        "--use-otsu-threshold",
        action="store_true",
        help="Use Otsu's method for thresholding instead of the fixed value",
    )
    parser.add_argument(
        "--thresholding-value",
        type=int,
        default=190,
        help=(
            "Fixed threshold value for text detection (0-255). "
            "Lower values help clean edge-hugging text."
        ),
    )
    parser.add_argument(
        "--roi-shrink-px",
        type=int,
        default=4,
        help=(
            "Shrink the threshold ROI inward by N pixels (0-8) before fill. "
            "Lower helps clean edge-hugging text; higher preserves outlines."
        ),
    )
    # Translation args
    parser.add_argument("--temperature", type=float, default=0.1, help="Controls randomness in output (0.0-2.0)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling parameter (0.0-1.0)")
    parser.add_argument("--top-k", type=int, default=1, help="Limits to top k tokens")
    parser.add_argument(
        "--translation-mode",
        type=str,
        default="one-step",
        choices=["one-step", "two-step"],
        help="Translation process mode (one-step or two-step)",
    )
    parser.add_argument(
        "--openrouter-reasoning-override",
        action="store_true",
        help="Forces max output tokens to 8192 (for reasoning-capable LLM issues).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help=(
            "Internal reasoning effort for OpenAI reasoning models (o1/o3/o4-mini/gpt-5*). "
            "Note: 'minimal' is only supported by gpt-5 series."
        ),
    )
    # Rendering args
    parser.add_argument("--max-font-size", type=int, default=14, help="Max font size for rendering text.")
    parser.add_argument("--min-font-size", type=int, default=8, help="Min font size for rendering text.")
    parser.add_argument("--line-spacing", type=float, default=1.0, help="Line spacing multiplier for rendered text.")
    parser.add_argument(
        "--no-hyphenate-before-scaling",
        dest="hyphenate_before_scaling",
        action="store_false",
        help="Disable hyphenation of long words before reducing font size.",
    )
    parser.set_defaults(hyphenate_before_scaling=True)
    parser.add_argument(
        "--hyphen-penalty",
        type=float,
        default=1000.0,
        help="Penalty for hyphenated line breaks in text layout (100-2000). Increase to discourage hyphenation.",
    )
    parser.add_argument(
        "--hyphenation-min-word-length",
        type=int,
        default=8,
        help="Minimum word length for hyphenation (6-10).",
    )
    parser.add_argument(
        "--badness-exponent",
        type=float,
        default=3.0,
        help="Exponent for line badness calculation in text layout (2-4). Increase to avoid loose lines.",
    )
    parser.add_argument(
        "--padding-pixels",
        type=float,
        default=8.0,
        help="Padding between text and the edge of the speech bubble (2-12). "
             "Increase for more space between text and bubble boundaries.",
    )
    # Output args
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG compression quality (1-100)")
    parser.add_argument("--png-compression", type=int, default=6, help="PNG compression level (0-9)")
    parser.add_argument(
        "--image-mode", type=str, default="RGBA", choices=["RGB", "RGBA"], help="Processing image mode (RGB or RGBA)"
    )
    # General args
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--cleaning-only",
        action="store_true",
        help="Only perform detection and cleaning, skip translation and rendering.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable 'thinking' capabilities for Gemini 2.5 Flash models.",
    )
    # Full page context toggle
    parser.add_argument(
        "--no-full-page-context",
        dest="send_full_page_context",
        action="store_false",
        help="Disable sending the full page image as LLM context.",
    )
    parser.set_defaults(send_full_page_context=True)
    parser.set_defaults(verbose=False, cpu=False, cleaning_only=False, enable_thinking=False)

    args = parser.parse_args()

    # --- Create Config Object ---
    provider = args.provider
    api_key = None
    api_key_arg_name = ""
    api_key_env_var = ""
    compatible_url = None

    if provider == "Google":
        api_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")
        api_key_arg_name = "--google-api-key"
        api_key_env_var = "GOOGLE_API_KEY"
        default_model = "gemini-2.0-flash"
    elif provider == "OpenAI":
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        api_key_arg_name = "--openai-api-key"
        api_key_env_var = "OPENAI_API_KEY"
        default_model = "gpt-4o"
    elif provider == "Anthropic":
        api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        api_key_arg_name = "--anthropic-api-key"
        api_key_env_var = "ANTHROPIC_API_KEY"
        default_model = "claude-3.7-sonnet-latest"
    elif provider == "xAI":
        api_key = args.xai_api_key or os.environ.get("XAI_API_KEY")
        api_key_arg_name = "--xai-api-key"
        api_key_env_var = "XAI_API_KEY"
        default_model = "grok-4-fast-reasoning"
    elif provider == "OpenRouter":
        api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        api_key_arg_name = "--openrouter-api-key"
        api_key_env_var = "OPENROUTER_API_KEY"
        default_model = "openrouter/auto"
    elif provider == "OpenAI-Compatible":
        compatible_url = args.openai_compatible_url
        api_key = args.openai_compatible_api_key or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
        api_key_arg_name = "--openai-compatible-api-key"
        api_key_env_var = "OPENAI_COMPATIBLE_API_KEY"
        default_model = "default"

    if provider != "OpenAI-Compatible" and not api_key:
        print(
            f"Warning: {provider} API key not provided via {api_key_arg_name} or {api_key_env_var} "
            f"environment variable. Translation will likely fail."
        )

    model_name = args.model_name or default_model
    if not args.model_name:
        print(f"Using default model for {provider}: {model_name}")

    target_device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"Using {'CPU' if target_device.type == 'cpu' else 'CUDA'} device.")

    use_otsu_config_val = args.use_otsu_threshold

    config = MangaTranslatorConfig(
        yolo_model_path="",
        verbose=args.verbose,
        device=target_device,
        cleaning_only=args.cleaning_only,
        detection=DetectionConfig(confidence=args.conf, use_sam2=args.use_sam2),
        cleaning=CleaningConfig(
            thresholding_value=args.thresholding_value,
            use_otsu_threshold=use_otsu_config_val,
            roi_shrink_px=max(0, min(8, int(args.roi_shrink_px))),
        ),
        translation=TranslationConfig(
            provider=provider,
            google_api_key=api_key if provider == "Google" else os.environ.get("GOOGLE_API_KEY", ""),
            openai_api_key=api_key if provider == "OpenAI" else os.environ.get("OPENAI_API_KEY", ""),
            anthropic_api_key=api_key if provider == "Anthropic" else os.environ.get("ANTHROPIC_API_KEY", ""),
            xai_api_key=api_key if provider == "xAI" else os.environ.get("XAI_API_KEY", ""),
            openrouter_api_key=api_key if provider == "OpenRouter" else os.environ.get("OPENROUTER_API_KEY", ""),
            openai_compatible_url=compatible_url,
            openai_compatible_api_key=(
                api_key if provider == "OpenAI-Compatible" else os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
            ),
            model_name=model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            openrouter_reasoning_override=args.openrouter_reasoning_override,
            input_language=args.input_language,
            output_language=args.output_language,
            reading_direction=args.reading_direction,
            translation_mode=args.translation_mode,
            enable_thinking=args.enable_thinking,
            reasoning_effort=args.reasoning_effort,
            send_full_page_context=args.send_full_page_context,
        ),
        rendering=RenderingConfig(
            font_dir=args.font_dir,
            max_font_size=args.max_font_size,
            min_font_size=args.min_font_size,
            line_spacing=args.line_spacing,
            hyphenate_before_scaling=args.hyphenate_before_scaling,
            hyphen_penalty=args.hyphen_penalty,
            hyphenation_min_word_length=args.hyphenation_min_word_length,
            badness_exponent=args.badness_exponent,
            padding_pixels=args.padding_pixels,
        ),
        output=OutputConfig(
            jpeg_quality=args.jpeg_quality, png_compression=args.png_compression, image_mode=args.image_mode
        ),
    )

    # --- Execute ---
    if args.batch:
        input_path = Path(args.input)
        if not input_path.is_dir():
            print(f"Error: --batch requires --input '{args.input}' to be a directory.")
            exit(1)

        output_dir = Path(args.output) if args.output else None

        if args.output:
            output_dir = Path(args.output)
            if not output_dir.exists():
                print(f"Creating output directory: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
            elif not output_dir.is_dir():
                print(f"Error: Specified --output '{output_dir}' is not a directory.")
                exit(1)

        batch_translate_images(input_path, config, output_dir)
    else:
        input_path = Path(args.input)
        if not input_path.is_file():
            print(f"Error: Input '{args.input}' is not a valid file.")
            exit(1)

        output_path_arg = args.output
        if not output_path_arg:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            original_ext = input_path.suffix.lower()
            output_ext = original_ext
            output_dir = Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"MangaTranslator_{timestamp}{output_ext}"
            print(f"--output not specified, using default: {output_path}")
        else:
            output_path = Path(output_path_arg)
            if not output_path.parent.exists():
                print(f"Creating directory for output file: {output_path.parent}")
                output_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect YOLO model for CLI using ./models/
        try:
            models_base = Path("./models")
            detected_model_path = autodetect_yolo_model_path(models_base)
            config.yolo_model_path = str(detected_model_path)
        except Exception as autodetect_err:
            log_message(f"Error auto-detecting YOLO model: {autodetect_err}", always_print=True)
            log_message(
                "CLI requires a model at ./models/*.pt. Please add it and retry.",
                always_print=True,
            )
            exit(1)

        try:
            log_message(f"Processing {input_path}...", always_print=True)
            translate_and_render(input_path, config, output_path)
            log_message(f"Translation complete. Result saved to {output_path}", always_print=True)
        except Exception as e:
            log_message(f"Error processing {input_path}: {e}", always_print=True)


if __name__ == "__main__":
    main()
