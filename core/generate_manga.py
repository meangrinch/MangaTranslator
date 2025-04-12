import argparse
import base64
import os
import time
from pathlib import Path
from typing import Optional, Union, Callable, Dict, Any

import cv2
from PIL import Image
import torch

from utils.settings import (
    DetectionConfig,
    CleaningConfig,
    TranslationConfig,
    RenderingConfig,
    OutputConfig,
    MangaTranslatorConfig,
)
from .detection import detect_speech_bubbles
from .cleaning import clean_speech_bubbles
from .image_utils import pil_to_cv2, cv2_to_pil, save_image_with_compression
from .translation import call_translation_api_batch, sort_bubbles_by_reading_order
from utils.logging import log_message
from .rendering import render_text_skia


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

    original_mode = pil_original.mode
    log_message(f"Original image mode: {original_mode}", verbose=verbose)

    desired_format = config.output.output_format
    output_ext_for_mode = Path(output_path).suffix.lower() if output_path else image_path.suffix.lower()

    if desired_format == "jpeg" or (desired_format == "auto" and output_ext_for_mode in [".jpg", ".jpeg"]):
        target_mode = "RGB"
    else:  # Default to RGBA for PNG, WEBP, or other formats in auto mode
        target_mode = "RGBA"
    log_message(
        f"Target image mode determined as: {target_mode} "
        f"(based on format: {desired_format}, output_ext: {output_ext_for_mode})",
        verbose=verbose,
    )

    pil_image_processed = pil_original
    if pil_image_processed.mode != target_mode:
        if target_mode == "RGB":
            # Handle transparency when converting to RGB
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
                        # Convert P with transparency to RGBA to get mask
                        temp_rgba = pil_image_processed.convert("RGBA")
                        mask = temp_rgba.split()[3]

                    if mask:
                        background.paste(pil_image_processed, mask=mask)
                        pil_image_processed = background
                    else:
                        pil_image_processed = pil_image_processed.convert("RGB")
                except Exception as paste_err:
                    log_message(
                        f"Warning: Error during RGBA/LA/P -> RGB conversion paste: {paste_err}. "
                        f"Trying alpha_composite.",
                        verbose=verbose,
                    )
                    try:
                        # Attempt alpha compositing onto a white background as an alternative
                        background_comp = Image.new("RGB", pil_image_processed.size, (255, 255, 255))
                        img_rgba_for_composite = (
                            pil_image_processed
                            if pil_image_processed.mode == "RGBA"
                            else pil_image_processed.convert("RGBA")
                        )
                        pil_image_processed = Image.alpha_composite(
                            background_comp.convert("RGBA"), img_rgba_for_composite
                        ).convert("RGB")
                        log_message("Successfully used alpha_composite for RGB conversion.", verbose=verbose)
                    except Exception as composite_err:
                        log_message(
                            f"Warning: alpha_composite also failed: {composite_err}. Using simple convert.",
                            verbose=verbose,
                        )
                        pil_image_processed = pil_image_processed.convert("RGB")  # Final fallback conversion
            else:  # Non-transparent conversion to RGB
                log_message(f"Converting {pil_image_processed.mode} to RGB", verbose=verbose)
                pil_image_processed = pil_image_processed.convert("RGB")
        elif target_mode == "RGBA":
            log_message(f"Converting {pil_image_processed.mode} to RGBA", verbose=verbose)
            pil_image_processed = pil_image_processed.convert("RGBA")

    original_cv_image = pil_to_cv2(pil_image_processed)

    # --- Encode Full Image for Context ---
    full_image_b64 = None
    try:
        is_success, buffer = cv2.imencode(".jpg", original_cv_image)
        if not is_success:
            raise ValueError("Full image encoding to JPEG failed")
        full_image_b64 = base64.b64encode(buffer).decode("utf-8")
        log_message("Encoded full image for translation context.", verbose=verbose)
    except Exception as e:
        log_message(f"Error encoding full image to base64: {e}. Translation might lack context.", always_print=True)
        # Continue without full image context if encoding fails

    # --- Detection ---
    log_message("Detecting speech bubbles...", verbose=verbose)
    try:
        bubble_data = detect_speech_bubbles(
            image_path, config.yolo_model_path, config.detection.confidence, verbose=verbose, device=device
        )
    except Exception as e:
        log_message(f"Error during bubble detection: {e}. Proceeding without bubbles.", always_print=True)
        bubble_data = []

    final_image_to_save = pil_image_processed

    # --- Proceed only if bubbles were detected ---
    if not bubble_data:
        log_message(
            "No speech bubbles detected or detection failed. Skipping cleaning, translation, and rendering.",
            always_print=True,
        )
    else:
        log_message(f"Detected {len(bubble_data)} potential bubbles.", verbose=verbose)

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
                dilation_kernel_size=config.cleaning.dilation_kernel_size,
                dilation_iterations=config.cleaning.dilation_iterations,
                use_otsu_threshold=use_otsu,
                min_contour_area=config.cleaning.min_contour_area,
                closing_kernel_size=config.cleaning.closing_kernel_size,
                closing_iterations=config.cleaning.closing_iterations,
                closing_kernel_shape=config.cleaning.closing_kernel_shape,
                constraint_erosion_kernel_size=config.cleaning.constraint_erosion_kernel_size,
                constraint_erosion_iterations=config.cleaning.constraint_erosion_iterations,
                verbose=verbose,
            )
            log_message(f"Cleaning returned info for {len(processed_bubbles_info)} bubbles.", verbose=verbose)
        except Exception as e:
            log_message(f"Error during bubble cleaning: {e}. Proceeding with uncleaned image.", always_print=True)
            cleaned_image_cv = original_cv_image.copy()
            processed_bubbles_info = []

        pil_cleaned_image = cv2_to_pil(cleaned_image_cv)
        if pil_cleaned_image.mode != target_mode:
            log_message(
                f"Converting cleaned CV image mode from {pil_cleaned_image.mode} back to target {target_mode}",
                verbose=verbose,
            )
            pil_cleaned_image = pil_cleaned_image.convert(target_mode)
        final_image_to_save = pil_cleaned_image

        # --- Check for Cleaning Only Mode ---
        if config.cleaning_only:
            log_message(
                "Cleaning only mode enabled. Skipping translation and rendering.", verbose=verbose, always_print=True
            )
        else:
            # --- Prepare images for Translation ---
            log_message("Preparing bubble images for translation...", verbose=verbose)
            for bubble in bubble_data:
                x1, y1, x2, y2 = bubble["bbox"]

                # Crop from the *original* processed image (before cleaning)
                bubble_image_cv = original_cv_image[y1:y2, x1:x2].copy()

                try:
                    is_success, buffer = cv2.imencode(".jpg", bubble_image_cv)
                    if not is_success:
                        raise ValueError("cv2.imencode failed")
                    image_b64 = base64.b64encode(buffer).decode("utf-8")
                    bubble["image_b64"] = image_b64
                    log_message(f"Collected bubble at ({x1}, {y1}), size {x2-x1}x{y2-y1}", verbose=verbose)
                except Exception as e:
                    log_message(f"Error encoding bubble at {bubble['bbox']}: {e}", verbose=verbose, always_print=True)
                    bubble["image_b64"] = None

            valid_bubble_data = [b for b in bubble_data if b.get("image_b64")]
            if not valid_bubble_data:
                log_message(
                    "No valid bubble images could be prepared for translation. Skipping translation and rendering.",
                    always_print=True,
                )
            else:  # Only proceed if we have valid bubble data
                # --- Sort and Translate ---
                reading_direction = config.translation.reading_direction
                log_message(f"Sorting bubbles in {reading_direction.upper()} reading order...", verbose=verbose)
                image_width = original_cv_image.shape[1]
                image_height = original_cv_image.shape[0]

                sorted_bubble_data = sort_bubbles_by_reading_order(
                    valid_bubble_data, image_height, image_width, reading_direction
                )

                bubble_images_b64 = [bubble["image_b64"] for bubble in sorted_bubble_data]
                translated_texts = []
                if not bubble_images_b64:
                    log_message(
                        "No valid bubble images to translate after sorting/filtering. "
                        "Skipping translation and rendering.",
                        always_print=True,
                    )
                elif full_image_b64 is None:
                    log_message(
                        "Full image context is missing due to encoding error. Skipping translation and rendering.",
                        always_print=True,
                    )
                else:
                    # Only attempt translation if we have bubbles and context
                    log_message(
                        f"Translating {len(bubble_images_b64)} speech bubbles from {config.translation.input_language} "
                        f"to {config.translation.output_language} ({reading_direction.upper()})...",
                        always_print=True,
                    )
                    try:
                        translated_texts = call_translation_api_batch(
                            config=config.translation,
                            images_b64=bubble_images_b64,
                            full_image_b64=full_image_b64,
                            debug=verbose,
                        )
                    except Exception as e:
                        log_message(f"Error during API translation: {e}", always_print=True)
                        translated_texts = ["[Translation Error]" for _ in sorted_bubble_data]

                # --- Render Translations ---
                # Map original bbox tuple to color and LIR bbox
                bubble_render_info_map = {
                    tuple(info["bbox"]): {"color": info["color"], "lir_bbox": info.get("lir_bbox")}
                    for info in processed_bubbles_info
                    if "bbox" in info and "color" in info
                }
                log_message("Rendering translations onto image...", verbose=verbose)
                if len(translated_texts) == len(sorted_bubble_data):
                    for i, bubble in enumerate(sorted_bubble_data):
                        bubble["translation"] = translated_texts[i]
                        bbox = bubble["bbox"]
                        text = bubble.get("translation", "")

                        if (
                            not text
                            or text.startswith("API Error")
                            or text == "No text detected or translation failed"
                            or text.startswith("[No translation")
                            or text.startswith("[Translation Error]")
                        ):
                            log_message(
                                f"Skipping rendering for bubble {bbox} due to empty or error translation: '{text}'",
                                verbose=verbose,
                            )
                            continue

                        log_message(f"Rendering text for bubble {bbox}: '{text[:30]}...'", verbose=verbose)

                        render_info = bubble_render_info_map.get(tuple(bbox))
                        bubble_color_bgr = render_info["color"] if render_info else (255, 255, 255)
                        lir_bbox_coords = render_info.get("lir_bbox") if render_info else None

                        rendered_image, success = render_text_skia(
                            pil_image=pil_cleaned_image,
                            text=text,
                            bbox=bbox,
                            lir_bbox=lir_bbox_coords,
                            bubble_color_bgr=bubble_color_bgr,
                            font_dir=config.rendering.font_dir,
                            min_font_size=config.rendering.min_font_size,
                            max_font_size=config.rendering.max_font_size,
                            line_spacing_mult=config.rendering.line_spacing,
                            use_subpixel_rendering=config.rendering.use_subpixel_rendering,
                            font_hinting=config.rendering.font_hinting,
                            use_ligatures=config.rendering.use_ligatures,
                            verbose=verbose,
                        )

                        if success:
                            pil_cleaned_image = rendered_image
                            final_image_to_save = pil_cleaned_image
                        else:
                            log_message(
                                f"Failed to render text in bubble {bbox} using Skia.",
                                verbose=verbose,
                                always_print=True,
                            )
                else:
                    log_message(
                        f"Warning: Mismatch between number of bubbles ({len(sorted_bubble_data)}) "
                        f"and translations ({len(translated_texts)}). Skipping rendering.",
                        always_print=True,
                    )

    # --- Save Output ---
    if output_path:
        # Ensure the final image is in the correct mode before saving
        if final_image_to_save.mode != target_mode:
            log_message(
                f"Converting final image mode from {final_image_to_save.mode} to target {target_mode} before saving.",
                verbose=verbose,
            )
            final_image_to_save = final_image_to_save.convert(target_mode)

        log_message(f"Saving final image to {output_path} (Mode: {final_image_to_save.mode})", verbose=verbose)
        save_image_with_compression(
            final_image_to_save,
            output_path,
            jpeg_quality=config.output.jpeg_quality,
            png_compression=config.output.png_compression,
            verbose=verbose,
        )

    end_time = time.time()
    processing_time = end_time - start_time
    log_message(f"Processing finished in {processing_time:.2f} seconds.", always_print=True)

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

    log_message(f"Starting batch processing of {total_images} images...", always_print=True)

    if progress_callback:
        progress_callback(0.0, f"Starting batch processing of {total_images} images...")

    for i, img_path in enumerate(image_files):
        try:
            if progress_callback:
                current_progress = i / total_images
                progress_callback(current_progress, f"Processing image {i+1}/{total_images}: {img_path.name}")

            # Determine correct output extension based on config
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
            log_message(f"Image {i+1}/{total_images}: Processing {img_path.name}", always_print=True)

            translate_and_render(img_path, config, output_path)

            results["success_count"] += 1

            if progress_callback:
                completed_progress = (i + 1) / total_images
                progress_callback(completed_progress, f"Completed {i+1}/{total_images} images")

        except Exception as e:
            log_message(f"Error processing {img_path.name}: {str(e)}", always_print=True)
            results["error_count"] += 1
            results["errors"][img_path.name] = str(e)

            if progress_callback:
                completed_progress = (i + 1) / total_images
                progress_callback(completed_progress, f"Completed {i+1}/{total_images} images (with errors)")

    if progress_callback:
        progress_callback(1.0, "Processing complete")

    end_batch_time = time.time()
    total_batch_time = end_batch_time - start_batch_time
    seconds_per_image = total_batch_time / total_images if total_images > 0 else 0

    log_message(
        f"\nBatch processing complete: {results['success_count']} of {total_images} images processed "
        f"in {total_batch_time:.2f} seconds ({seconds_per_image:.2f} seconds/image).\n",
        always_print=True,
    )
    if results["error_count"] > 0:
        log_message(f"Failed to process {results['error_count']} images.", always_print=True)
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
    parser.add_argument("--yolo-model", type=str, required=True, help="Path to YOLO model")
    # --- Provider and API Key Arguments ---
    parser.add_argument(
        "--provider",
        type=str,
        default="Gemini",
        choices=["Gemini", "OpenAI", "Anthropic", "OpenRouter", "OpenAI-compatible"],
        help="LLM provider to use for translation",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (overrides GOOGLE_API_KEY env var if --provider is Gemini)",
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
        help="Model name for the selected provider (e.g., 'gemini-2.0 flash'). "
        "If not provided, a default will be attempted based on the provider.",
    )
    parser.add_argument("--font-dir", type=str, default="./fonts", help="Directory containing font files")
    parser.add_argument("--input-language", type=str, default="Japanese", help="Source language")
    parser.add_argument("--output-language", type=str, default="English", help="Target language")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for detection")
    parser.add_argument(
        "--reading-direction",
        type=str,
        default="rtl",
        choices=["rtl", "ltr"],
        help="Reading direction for sorting bubbles (rtl or ltr)",
    )
    # Cleaning args
    parser.add_argument("--dilation-kernel-size", type=int, default=7, help="ROI Dilation Kernel Size")
    parser.add_argument("--dilation-iterations", type=int, default=1, help="ROI Dilation Iterations")
    parser.add_argument(
        "--use-otsu-threshold",
        action="store_true",
        help="Use Otsu's method for thresholding instead of the fixed value (210)",
    )
    parser.add_argument("--min-contour-area", type=int, default=50, help="Min Bubble Contour Area")
    parser.add_argument("--closing-kernel-size", type=int, default=7, help="Mask Closing Kernel Size")
    parser.add_argument("--closing-iterations", type=int, default=1, help="Mask Closing Iterations")
    parser.add_argument(
        "--constraint-erosion-kernel-size", type=int, default=9, help="Edge Constraint Erosion Kernel Size"
    )
    parser.add_argument(
        "--constraint-erosion-iterations", type=int, default=1, help="Edge Constraint Erosion Iterations"
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
    # Rendering args
    parser.add_argument("--max-font-size", type=int, default=14, help="Max font size for rendering text.")
    parser.add_argument("--min-font-size", type=int, default=8, help="Min font size for rendering text.")
    parser.add_argument("--line-spacing", type=float, default=1.0, help="Line spacing multiplier for rendered text.")
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
    parser.set_defaults(verbose=False, cpu=False, cleaning_only=False)

    args = parser.parse_args()

    # --- Create Config Object ---
    # Determine API key based on provider
    provider = args.provider
    api_key = None
    api_key_arg_name = ""
    api_key_env_var = ""
    compatible_url = None

    if provider == "Gemini":
        api_key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
        api_key_arg_name = "--gemini-api-key"
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
        yolo_model_path=args.yolo_model,
        verbose=args.verbose,
        device=target_device,
        cleaning_only=args.cleaning_only,
        detection=DetectionConfig(confidence=args.conf),
        cleaning=CleaningConfig(
            dilation_kernel_size=args.dilation_kernel_size,
            dilation_iterations=args.dilation_iterations,
            use_otsu_threshold=use_otsu_config_val,
            min_contour_area=args.min_contour_area,
            closing_kernel_size=args.closing_kernel_size,
            closing_iterations=args.closing_iterations,
            constraint_erosion_kernel_size=args.constraint_erosion_kernel_size,
            constraint_erosion_iterations=args.constraint_erosion_iterations,
        ),
        translation=TranslationConfig(
            provider=provider,
            gemini_api_key=api_key if provider == "Gemini" else os.environ.get("GOOGLE_API_KEY", ""),
            openai_api_key=api_key if provider == "OpenAI" else os.environ.get("OPENAI_API_KEY", ""),
            anthropic_api_key=api_key if provider == "Anthropic" else os.environ.get("ANTHROPIC_API_KEY", ""),
            openrouter_api_key=api_key if provider == "OpenRouter" else os.environ.get("OPENROUTER_API_KEY", ""),
            openai_compatible_url=compatible_url,  # Pass the URL
            openai_compatible_api_key=(
                api_key if provider == "OpenAI-Compatible" else os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
            ),  # Pass the key
            model_name=model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            input_language=args.input_language,
            output_language=args.output_language,
            reading_direction=args.reading_direction,
            translation_mode=args.translation_mode,  # Pass the mode
        ),
        rendering=RenderingConfig(
            font_dir=args.font_dir,
            max_font_size=args.max_font_size,
            min_font_size=args.min_font_size,
            line_spacing=args.line_spacing,
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

        try:
            log_message(f"Processing {input_path}...", always_print=True)
            translate_and_render(input_path, config, output_path)
            log_message(f"Translation complete. Result saved to {output_path}", always_print=True)
        except Exception as e:
            log_message(f"Error processing {input_path}: {e}", always_print=True)


if __name__ == "__main__":
    main()
