import argparse
import base64
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import torch
from PIL import Image

from core.caching import get_cache
from core.config import (CleaningConfig, DetectionConfig,
                         MangaTranslatorConfig, OutputConfig,
                         OutsideTextConfig, TranslationConfig)
from core.validation import validate_mutually_exclusive_modes
from utils.exceptions import (CleaningError, FontError, ImageProcessingError,
                              RenderingError, TranslationError)
from utils.logging import log_message

from .image.cleaning import clean_speech_bubbles
from .image.detection import detect_speech_bubbles
from .image.image_utils import (convert_image_to_target_mode, cv2_to_pil,
                                pil_to_cv2, resize_to_max_side,
                                save_image_with_compression, upscale_image,
                                upscale_image_to_dimension)
from .ml.model_manager import get_model_manager
from .outside_text_processor import process_outside_text
from .services.translation import (call_translation_api_batch,
                                   prepare_bubble_images_for_translation,
                                   sort_bubbles_by_reading_order)
from .text.text_renderer import RenderingConfig, render_text_skia


def get_image_encoding_params(pil_image_format: Optional[str]) -> Tuple[str, str]:
    """Returns (mime_type, cv2_ext) for a given PIL image format."""
    if pil_image_format and pil_image_format.upper() == "PNG":
        return "image/png", ".png"
    return "image/jpeg", ".jpg"


def translate_and_render(
    image_path: Union[str, Path],
    config: MangaTranslatorConfig,
    output_path: Optional[Union[str, Path]] = None,
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
        image_format = pil_original.format
        mime_type, cv2_ext = get_image_encoding_params(image_format)
        log_message(
            f"Original image format: {image_format} -> MIME: {mime_type}",
            verbose=verbose,
        )
    except FileNotFoundError:
        log_message(f"Error: Input image not found at {image_path}", always_print=True)
        raise
    except Exception as e:
        log_message(f"Error opening image {image_path}: {e}", always_print=True)
        raise

    desired_format = config.output.output_format
    output_ext_for_mode = (
        Path(output_path).suffix.lower() if output_path else image_path.suffix.lower()
    )

    if desired_format == "jpeg" or (
        desired_format == "auto" and output_ext_for_mode in [".jpg", ".jpeg"]
    ):
        target_mode = "RGB"
    else:  # Default to RGBA for PNG, WEBP, or other formats in auto mode
        target_mode = "RGBA"
    log_message(f"Target mode: {target_mode}", verbose=verbose)

    pil_image_processed = convert_image_to_target_mode(
        pil_original, target_mode, verbose
    )

    get_cache().set_current_image(pil_image_processed, verbose)

    original_cv_image = pil_to_cv2(pil_image_processed)

    # Process outside text detection and inpainting
    pil_image_processed, outside_text_data = process_outside_text(
        pil_image_processed, config, image_path, image_format, verbose
    )
    original_cv_image = pil_to_cv2(pil_image_processed)

    full_image_b64 = None
    full_image_mime_type = None
    if config.translation.send_full_page_context:
        try:
            # Prepare full page context based on upscale method
            context_image_pil = cv2_to_pil(original_cv_image)

            if config.translation.upscale_method == "model":
                # Use upscaling model for full page context
                model_manager = get_model_manager()
                with model_manager.upscale_context() as upscale_model:
                    context_image_pil = upscale_image_to_dimension(
                        upscale_model,
                        context_image_pil,
                        config.translation.context_image_max_side_pixels,
                        config.device,
                        "max",
                        verbose,
                    )
                    # Resize to exact target dimension (downscale if needed)
                    context_image_pil = resize_to_max_side(
                        context_image_pil,
                        config.translation.context_image_max_side_pixels,
                        verbose=verbose,
                    )
                    log_message(
                        "Upscaled full image for context with model", verbose=verbose
                    )
            elif config.translation.upscale_method == "lanczos":
                # Use LANCZOS resampling (current behavior)
                context_image_pil = resize_to_max_side(
                    context_image_pil,
                    config.translation.context_image_max_side_pixels,
                    verbose=verbose,
                )
                log_message(
                    "Resized full image for context with LANCZOS", verbose=verbose
                )
            else:  # upscale_method == "none"
                # No resizing/upscaling
                log_message(
                    "Using full image for context without resizing", verbose=verbose
                )

            context_image_cv = pil_to_cv2(context_image_pil)
            is_success, buffer = cv2.imencode(cv2_ext, context_image_cv)
            if not is_success:
                raise ImageProcessingError(f"Full image encoding to {cv2_ext} failed")
            full_image_b64 = base64.b64encode(buffer).decode("utf-8")
            full_image_mime_type = mime_type
            log_message("Encoded full image for context", verbose=verbose)
        except Exception as e:
            log_message(
                f"Warning: Failed to encode full image context: {e}", always_print=True
            )

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
                pil_image_processed,
                config.yolo_model_path,
                config.detection.confidence,
                pre_computed_detections=bubble_data,
                device=device,
                thresholding_value=config.cleaning.thresholding_value,
                use_otsu_threshold=use_otsu,
                roi_shrink_px=config.cleaning.roi_shrink_px,
                verbose=verbose,
            )
            log_message(
                f"Cleaned {len(processed_bubbles_info)} bubbles", verbose=verbose
            )
        except CleaningError as e:
            log_message(f"Cleaning failed: {e}", always_print=True)
            cleaned_image_cv = original_cv_image.copy()
            processed_bubbles_info = []
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

            model_manager = get_model_manager()
            with model_manager.upscale_context() as upscale_model:
                bubble_data = prepare_bubble_images_for_translation(
                    bubble_data,
                    original_cv_image,
                    upscale_model,
                    config.device,
                    mime_type,
                    config.translation.bubble_min_side_pixels,
                    config.translation.upscale_method,
                    verbose,
                )

            log_message(
                f"Upscaled {len(bubble_data)} bubble images for translation",
                always_print=True,
            )
            valid_bubble_data = [b for b in bubble_data if b.get("image_b64")]
            if not valid_bubble_data:
                log_message("No valid bubble images for translation", always_print=True)
            else:  # Only proceed if we have valid bubble data
                # --- Sort and Translate ---
                reading_direction = config.translation.reading_direction
                # Merge outside text data with speech bubbles for reading order calculation
                if outside_text_data:
                    log_message(
                        f"Including {len(outside_text_data)} outside text regions in reading order calculation",
                        verbose=verbose,
                    )
                    # Combine speech bubbles and OSB text for unified reading order sorting
                    all_text_data = valid_bubble_data + outside_text_data
                else:
                    all_text_data = valid_bubble_data

                log_message(
                    f"Sorting all text elements ({reading_direction.upper()})",
                    verbose=verbose,
                )

                # Sort all text elements (speech bubbles + OSB text) by reading order
                sorted_bubble_data = sort_bubbles_by_reading_order(
                    all_text_data, reading_direction
                )

                bubble_images_b64 = [
                    bubble["image_b64"]
                    for bubble in sorted_bubble_data
                    if "image_b64" in bubble
                ]
                bubble_mime_types = [
                    bubble["mime_type"]
                    for bubble in sorted_bubble_data
                    if "image_b64" in bubble and "mime_type" in bubble
                ]
                translated_texts = []
                if not bubble_images_b64:
                    log_message("No valid bubbles after sorting", always_print=True)
                else:
                    if getattr(config, "test_mode", False):
                        placeholder_long = "Lorem **ipsum** *dolor* sit amet, consectetur adipiscing elit."
                        placeholder_short = "Lorem **ipsum** *dolor* sit amet..."
                        placeholder_osb = "Lorem"
                        log_message(
                            f"Test mode: generating placeholders for {len(sorted_bubble_data)} bubbles",
                            always_print=True,
                        )
                        # Map for rendering info used in probe
                        bubble_render_info_map_probe = {
                            tuple(info["bbox"]): {
                                "color": info["color"],
                                "mask": info.get("mask"),
                            }
                            for info in processed_bubbles_info
                            if "bbox" in info and "color" in info and "mask" in info
                        }
                        for i, bubble in enumerate(sorted_bubble_data):
                            bbox = bubble["bbox"]
                            is_outside_text = bubble.get("is_outside_text", False)

                            # Use simple "Lorem ipsum" for OSB text in test mode
                            if is_outside_text:
                                translated_texts.append(placeholder_osb)
                                continue

                            probe_info = bubble_render_info_map_probe.get(
                                tuple(bbox), {}
                            )
                            bubble_color_bgr = probe_info.get("color", (255, 255, 255))
                            cleaned_mask = probe_info.get("mask")
                            # Probe fit at max size without mutating the working image
                            _probe_canvas = pil_cleaned_image.copy()
                            probe_config = RenderingConfig(
                                min_font_size=config.rendering.max_font_size,
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
                            )
                            try:
                                _ = render_text_skia(
                                    pil_image=_probe_canvas,
                                    text=placeholder_long,
                                    bbox=bbox,
                                    font_dir=config.rendering.font_dir,
                                    cleaned_mask=cleaned_mask,
                                    bubble_color_bgr=bubble_color_bgr,
                                    config=probe_config,
                                    verbose=verbose,
                                    bubble_id=str(i + 1),
                                )
                                fits = True
                            except (RenderingError, FontError) as e:
                                log_message(
                                    f"Probe rendering failed: {e}", verbose=verbose
                                )
                                fits = False
                            except Exception as e:
                                log_message(
                                    f"Probe rendering unexpected error: {e}",
                                    always_print=True,
                                )
                                fits = False
                            translated_texts.append(
                                placeholder_long if fits else placeholder_short
                            )
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
                                mime_types=bubble_mime_types,
                                full_image_mime_type=full_image_mime_type
                                or "image/jpeg",
                                bubble_metadata=sorted_bubble_data,
                                debug=verbose,
                            )
                        except TranslationError as e:
                            log_message(f"Translation failed: {e}", always_print=True)
                            translated_texts = [f"[Translation Error: {e}]"] * len(
                                bubble_images_b64
                            )
                        except Exception as e:
                            log_message(
                                f"Translation API error: {e}", always_print=True
                            )
                            translated_texts = [
                                "[Translation Error: API call raised exception]"
                                for _ in sorted_bubble_data
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
                        is_outside_text = bubble.get("is_outside_text", False)

                        # Convert OSB text to uppercase
                        if is_outside_text and text:
                            text = text.upper()
                            bubble["translation"] = text

                        if (
                            not text
                            or text.startswith("API Error")
                            or text.startswith("[Translation Error]")
                            or text.startswith("[Translation Error:")
                            or text.strip()
                            in {
                                "[OCR FAILED]",
                                "[Empty response / no content]",
                                f"[{config.translation.provider}: API call failed/blocked]",
                                f"[{config.translation.provider}: OCR call failed/blocked]",
                                f"[{config.translation.provider}: Failed to parse response]",
                            }
                        ):
                            entry_type = "outside text" if is_outside_text else "bubble"
                            log_message(
                                f"Skipping {entry_type} {bbox} - invalid translation",
                                verbose=verbose,
                            )
                            continue

                        # Use OSB-specific settings for outside text, regular settings for speech bubbles
                        if is_outside_text:
                            log_message(
                                f"Rendering outside text {bbox}: '{text[:30]}...'",
                                verbose=verbose,
                            )
                            font_dir = (
                                config.outside_text.osb_font_name
                                if config.outside_text.osb_font_name
                                else config.rendering.font_dir
                            )
                            min_font = config.outside_text.osb_min_font_size
                            max_font = config.outside_text.osb_max_font_size
                            line_spacing = config.outside_text.osb_line_spacing
                            use_ligs = config.outside_text.osb_use_ligatures
                            # Outside text was inpainted, no mask needed
                            cleaned_mask = None
                            # Use the detected text color from outside_text_processor
                            is_dark_text = bubble.get("is_dark_text", True)
                            # Set bubble_color_bgr to mimic the original text color
                            # Dark text → dark background value → white rendering
                            # Light text → light background value → black rendering
                            bubble_color_bgr = (
                                (50, 50, 50) if is_dark_text else (255, 255, 255)
                            )
                            # Orientation & aspect decision for OSB
                            angle_signed = float(bubble.get("orientation_deg", 0.0))
                            x1, y1, x2, y2 = bbox
                            w = max(1.0, float(x2 - x1))
                            h = max(1.0, float(y2 - y1))
                            aspect_ratio = float(bubble.get("aspect_ratio", h / w))
                            # Determine verticality based on residual from perfect vertical (±90°)
                            residual_to_vertical = (
                                angle_signed - 90.0
                                if angle_signed >= 0.0
                                else angle_signed + 90.0
                            )
                            near_vertical = abs(residual_to_vertical) <= 5.0
                            portrait = aspect_ratio >= 1.5
                            vertical_stack = bool(near_vertical and portrait)
                            # Carry orientation:
                            # - If stacked: rotate only by the residual from perfect vertical
                            # - Else: rotate by full angle (negative due to Y-down)
                            rotation_deg = (
                                residual_to_vertical if vertical_stack else angle_signed
                            )
                        else:
                            log_message(
                                f"Rendering bubble {bbox}: '{text[:30]}...'",
                                verbose=verbose,
                            )
                            font_dir = config.rendering.font_dir
                            min_font = config.rendering.min_font_size
                            max_font = config.rendering.max_font_size
                            line_spacing = config.rendering.line_spacing
                            use_ligs = config.rendering.use_ligatures
                            render_info = bubble_render_info_map.get(tuple(bbox))
                            bubble_color_bgr = (255, 255, 255)
                            cleaned_mask = None
                            if render_info:
                                bubble_color_bgr = render_info["color"]
                                cleaned_mask = render_info.get("mask")
                            # No rotation/stacking for regular bubbles
                            vertical_stack = False
                            rotation_deg = 0.0

                        render_config = RenderingConfig(
                            min_font_size=min_font,
                            max_font_size=max_font,
                            line_spacing_mult=line_spacing,
                            use_subpixel_rendering=(
                                config.outside_text.osb_use_subpixel_rendering
                                if is_outside_text
                                else config.rendering.use_subpixel_rendering
                            ),
                            font_hinting=(
                                config.outside_text.osb_font_hinting
                                if is_outside_text
                                else config.rendering.font_hinting
                            ),
                            use_ligatures=use_ligs,
                            hyphenate_before_scaling=config.rendering.hyphenate_before_scaling,
                            hyphen_penalty=config.rendering.hyphen_penalty,
                            hyphenation_min_word_length=config.rendering.hyphenation_min_word_length,
                            badness_exponent=config.rendering.badness_exponent,
                            padding_pixels=config.rendering.padding_pixels,
                            outline_width=(
                                config.outside_text.osb_outline_width
                                if is_outside_text
                                else 0.0
                            ),
                        )
                        try:
                            rendered_image = render_text_skia(
                                pil_image=pil_cleaned_image,
                                text=text,
                                bbox=bbox,
                                font_dir=font_dir,
                                cleaned_mask=cleaned_mask,
                                bubble_color_bgr=bubble_color_bgr,
                                config=render_config,
                                verbose=verbose,
                                bubble_id=str(i + 1),
                                rotation_deg=rotation_deg,
                                vertical_stack=vertical_stack,
                            )
                            success = True
                        except (RenderingError, FontError) as e:
                            log_message(f"Text rendering failed: {e}", verbose=verbose)
                            rendered_image = pil_cleaned_image
                            success = False

                        if success:
                            pil_cleaned_image = rendered_image
                            final_image_to_save = pil_cleaned_image
                        else:
                            log_message(
                                f"Failed to render bubble {bbox}", verbose=verbose
                            )
                else:
                    log_message(
                        f"Warning: Bubble/translation count mismatch "
                        f"({len(sorted_bubble_data)}/{len(translated_texts)})",
                        always_print=True,
                    )

    # --- Final Image Upscaling (optional) ---
    if config.output.upscale_final_image:
        log_message("Upscaling final image...", verbose=verbose, always_print=True)
        final_image_to_save = upscale_image(
            final_image_to_save,
            config.output.upscale_final_image_factor,
            verbose=verbose,
        )

    # --- Save Output ---
    if output_path:
        if final_image_to_save.mode != target_mode:
            log_message(f"Converting final image to {target_mode}", verbose=verbose)
            final_image_to_save = final_image_to_save.convert(target_mode)

        try:
            save_image_with_compression(
                final_image_to_save,
                output_path,
                jpeg_quality=config.output.jpeg_quality,
                png_compression=config.output.png_compression,
                verbose=verbose,
            )
        except ImageProcessingError as e:
            log_message(f"Failed to save image: {e}", always_print=True)
            raise

    end_time = time.time()
    processing_time = end_time - start_time
    log_message(f"Processing completed in {processing_time:.2f}s", always_print=True)

    return final_image_to_save


def batch_translate_images(
    input_dir: Union[str, Path],
    config: MangaTranslatorConfig,
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    preserve_structure: bool = False,
) -> Dict[str, Any]:
    """
    Process all images in a directory using a configuration object.

    Args:
        input_dir (str or Path): Directory containing images to process
        config (MangaTranslatorConfig): Configuration object containing all settings.
        output_dir (str or Path, optional): Directory to save translated images.
                                            If None, uses input_dir / "output_translated".
        progress_callback (callable, optional): Function to call with progress updates (0.0-1.0, message).
        preserve_structure (bool): If True, recursively process subdirectories and preserve folder structure
                                   in the output. If False, only processes files in the root directory.

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

    if preserve_structure:
        # Recursively find all image files preserving directory structure
        image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
    else:
        image_files = [
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

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
            # Calculate relative path from input directory for structure preservation
            if preserve_structure:
                relative_path = img_path.relative_to(input_dir)
                # Create output subdirectory structure
                output_subdir = output_dir / relative_path.parent
                os.makedirs(output_subdir, exist_ok=True)
                # Use relative path for output filename
                output_filename = f"{relative_path.stem}_translated"
                display_path = str(relative_path)
                error_key = str(relative_path)
            else:
                output_subdir = output_dir
                output_filename = f"{img_path.stem}_translated"
                display_path = img_path.name
                error_key = img_path.name

            if progress_callback:
                current_progress = i / total_images
                progress_callback(
                    current_progress,
                    f"Processing image {i + 1}/{total_images}: {display_path}",
                )

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

            output_path = output_subdir / f"{output_filename}{output_ext}"
            log_message(
                f"Processing {i + 1}/{total_images}: {display_path}", always_print=True
            )

            translate_and_render(img_path, config, output_path)

            results["success_count"] += 1

            if progress_callback:
                completed_progress = (i + 1) / total_images
                progress_callback(
                    completed_progress, f"Completed {i + 1}/{total_images} images"
                )

        except Exception as e:
            log_message(f"Error processing {display_path}: {str(e)}", always_print=True)
            results["error_count"] += 1
            results["errors"][error_key] = str(e)

            if progress_callback:
                completed_progress = (i + 1) / total_images
                progress_callback(
                    completed_progress,
                    f"Completed {i + 1}/{total_images} images (with errors)",
                )

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
    parser = argparse.ArgumentParser(
        description="Translate manga/comic speech bubbles using a configuration approach"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image, directory, or ZIP archive (if using --batch)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save the translated image or directory (if using --batch)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all images in the input directory or ZIP archive (preserves folder structure for ZIP files)",
    )
    # --- Provider and API Key Arguments ---
    parser.add_argument(
        "--provider",
        type=str,
        default="Google",
        choices=[
            "Google",
            "OpenAI",
            "Anthropic",
            "xAI",
            "OpenRouter",
            "OpenAI-Compatible",
        ],
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
        help="Model name for the selected provider (e.g., 'gemini-2.5-flash'). "
        "If not provided, a default will be attempted based on the provider.",
    )
    parser.add_argument(
        "--font-dir",
        type=str,
        default="./fonts",
        help="Directory containing font files",
    )
    parser.add_argument(
        "--input-language",
        type=str,
        default="Japanese",
        help="Source language",
    )
    parser.add_argument(
        "--output-language", type=str, default="English", help="Target language"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35, help="Confidence threshold for detection"
    )
    parser.add_argument(
        "--no-sam2",
        dest="use_sam2",
        action="store_false",
        help="Disable SAM 2.1 segmentation",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Controls creativity. Lower is more deterministic, higher is more random (0.0-2.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Controls diversity. Lower is more focused, higher is more random (0.0-1.0)",
    )
    parser.add_argument(
        "--top-k", type=int, default=1, help="Limits sampling pool to top K tokens"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of tokens in the response (2048-32768). "
            "Default: 4096 for non-reasoning models, 16384 for reasoning models"
        ),
    )
    parser.add_argument(
        "--translation-mode",
        type=str,
        default="one-step",
        choices=["one-step", "two-step"],
        help=(
            "Method for translation ('one-step' combines OCR/Translate, 'two-step' separates them). "
            "'two-step' might improve translation quality for less-capable LLMs"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help=(
            "Controls internal reasoning effort for OpenAI/OpenRouter OpenAI reasoning models "
            "(o1/o3/o4-mini/gpt-5*). Note: 'minimal' is only supported by gpt-5 series."
        ),
    )
    # Rendering args
    parser.add_argument(
        "--max-font-size",
        type=int,
        default=14,
        help="Max font size for rendering text (px)",
    )
    parser.add_argument(
        "--min-font-size",
        type=int,
        default=8,
        help="Min font size for rendering text (px)",
    )
    parser.add_argument(
        "--line-spacing",
        type=float,
        default=1.0,
        help="Line spacing for rendering text (1.0 = standard)",
    )
    parser.add_argument(
        "--no-subpixel-rendering",
        dest="use_subpixel_rendering",
        action="store_false",
        help="Disable subpixel rendering for speech bubble text (disable for OLED displays)",
    )
    parser.set_defaults(use_subpixel_rendering=True)
    parser.add_argument(
        "--font-hinting",
        type=str,
        choices=["none", "slight", "normal", "full"],
        default="none",
        help="Font hinting mode for speech bubble text",
    )
    parser.add_argument(
        "--use-ligatures",
        action="store_true",
        help="Enable standard ligatures for speech bubble text (e.g., fi, fl)",
    )
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
        help="Minimum word length required for hyphenation (6-10)",
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
        default=5.0,
        help="Padding between text and the edge of the speech bubble (2-12). "
        "Increase for more space between text and bubble boundaries.",
    )
    # Output args
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG compression quality (1-100)",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=6,
        help="PNG compression level (0-9)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["auto", "png", "jpeg"],
        default="auto",
        help="Output image format (auto uses input format)",
    )
    parser.add_argument(
        "--upscale-final-image",
        action="store_true",
        help="Upscale final translated image using 2x-AnimeSharpV4_RCAN",
    )
    parser.add_argument(
        "--upscale-final-image-factor",
        type=float,
        default=2.0,
        help="Upscale factor for final image (1.0-8.0)",
    )
    # General args
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--cleaning-only",
        action="store_true",
        help="Skip translation and text rendering, output only the cleaned speech bubbles",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Skip translation and render placeholder text (lorem ipsum)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enables 'thinking' capabilities for Gemini 2.5 Flash (Lite), Claude reasoning models, "
            "and OpenRouter Gemini/Anthropic/Grok models"
        ),
    )
    parser.add_argument(
        "--enable-grounding",
        action="store_true",
        help=(
            "Enable Google Search grounding for Gemini models (Google and OpenRouter)"
        ),
    )
    parser.add_argument(
        "--special-instructions",
        type=str,
        default=None,
        help="Optional special instructions for the LLM (formatting, context, character names, etc.)",
    )
    # Full page context toggle
    parser.add_argument(
        "--no-full-page-context",
        dest="send_full_page_context",
        action="store_false",
        help=(
            "Disable including the full page image as context for translation. Enable if "
            "encountering refusals or using less-capable LLMs"
        ),
    )
    # Translation upscaling method
    parser.add_argument(
        "--upscale-method",
        type=str,
        choices=["model", "lanczos", "none"],
        default="model",
        help=(
            "Method for upscaling images before translation API. "
            "model: Use 2x-AnimeSharpV4 upscaling model (best quality, slower), "
            "lanczos: Use LANCZOS resampling (fast, good quality), "
            "none: No upscaling (may affect OCR quality for small text)"
        ),
    )

    # --- Outside Speech Bubble (OSB) Text Settings ---
    parser.add_argument(
        "--osb-enable",
        action="store_true",
        help="Enable outside speech bubble text detection and removal",
    )
    parser.add_argument(
        "--osb-huggingface-token",
        type=str,
        default=None,
        help="HuggingFace token for Flux Kontext model downloads (overrides HUGGINGFACE_TOKEN env var)",
    )
    parser.add_argument(
        "--osb-flux-steps",
        type=int,
        default=8,
        help=(
            "Number of denoising steps for Flux Kontext (1-30). "
            "15 is best for quality (diminishing returns beyond); "
            "below 6 shows noticeable degradation."
        ),
    )
    parser.add_argument(
        "--osb-flux-residual-threshold",
        type=float,
        default=0.12,
        help="Residual diff threshold for Flux inference (0.0-1.0)",
    )
    parser.add_argument(
        "--osb-seed",
        type=int,
        default=1,
        help="Seed for reproducible inpainting (-1 = random)",
    )
    parser.add_argument(
        "--osb-font-name",
        type=str,
        default=None,
        help="Font name for OSB text rendering (default: use main font)",
    )
    parser.add_argument(
        "--osb-max-font-size",
        type=int,
        default=64,
        help="Maximum font size for OSB text (5-96px)",
    )
    parser.add_argument(
        "--osb-min-font-size",
        type=int,
        default=12,
        help="Minimum font size for OSB text (5-50px)",
    )
    parser.add_argument(
        "--osb-use-ligatures",
        action="store_true",
        help="Enable standard ligatures for OSB text (e.g., fi, fl)",
    )
    parser.add_argument(
        "--osb-outline-width",
        type=float,
        default=3.0,
        help="Outline width for OSB text (0-10px)",
    )
    parser.add_argument(
        "--osb-line-spacing",
        type=float,
        default=1.0,
        help="Line spacing multiplier for OSB text (0.5-2.0)",
    )
    parser.add_argument(
        "--osb-use-subpixel",
        action="store_true",
        default=True,
        help="Enable subpixel rendering for OSB text (disable for OLED displays)",
    )
    parser.add_argument(
        "--osb-font-hinting",
        type=str,
        choices=["none", "slight", "normal", "full"],
        default="none",
        help="Font hinting mode for OSB text",
    )
    parser.add_argument(
        "--osb-bbox-expansion",
        type=float,
        default=0.1,
        help="Bounding box expansion percent for OSB detection",
    )
    parser.add_argument(
        "--osb-easyocr-min-size",
        type=int,
        default=200,
        help="Minimum text region size in pixels for OSB text detection",
    )
    parser.add_argument(
        "--bubble-min-side-pixels",
        type=int,
        default=128,
        help="Target minimum side length for speech bubble upscaling",
    )
    parser.add_argument(
        "--context-image-max-side-pixels",
        type=int,
        default=1536,
        help="Target maximum side length for full page image",
    )
    parser.add_argument(
        "--osb-min-side-pixels",
        type=int,
        default=128,
        help="Target minimum side length for outside speech bubble upscaling",
    )

    parser.set_defaults(send_full_page_context=True)
    parser.set_defaults(
        verbose=False,
        cpu=False,
        cleaning_only=False,
        enable_thinking=False,
        enable_grounding=False,
    )

    args = parser.parse_args()

    # --- Validate mutually exclusive flags ---
    try:
        validate_mutually_exclusive_modes(args.cleaning_only, args.test_mode)
    except Exception as e:
        parser.error(str(e))

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
        api_key = args.openai_compatible_api_key or os.environ.get(
            "OPENAI_COMPATIBLE_API_KEY"
        )
        api_key_arg_name = "--openai-compatible-api-key"
        api_key_env_var = "OPENAI_COMPATIBLE_API_KEY"
        default_model = "default"

    if provider != "OpenAI-Compatible" and not api_key:
        log_message(
            f"Warning: {provider} API key not provided via {api_key_arg_name} or {api_key_env_var} "
            f"environment variable. Translation will likely fail.",
            always_print=True,
        )

    model_name = args.model_name or default_model
    if not args.model_name:
        log_message(f"Using default model for {provider}: {model_name}", verbose=True)

    target_device = (
        torch.device("cpu")
        if args.cpu or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    log_message(
        f"Using {'CPU' if target_device.type == 'cpu' else 'CUDA'} device.",
        always_print=True,
    )

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
            google_api_key=(
                api_key
                if provider == "Google"
                else os.environ.get("GOOGLE_API_KEY", "")
            ),
            openai_api_key=(
                api_key
                if provider == "OpenAI"
                else os.environ.get("OPENAI_API_KEY", "")
            ),
            anthropic_api_key=(
                api_key
                if provider == "Anthropic"
                else os.environ.get("ANTHROPIC_API_KEY", "")
            ),
            xai_api_key=(
                api_key if provider == "xAI" else os.environ.get("XAI_API_KEY", "")
            ),
            openrouter_api_key=(
                api_key
                if provider == "OpenRouter"
                else os.environ.get("OPENROUTER_API_KEY", "")
            ),
            openai_compatible_url=compatible_url,
            openai_compatible_api_key=(
                api_key
                if provider == "OpenAI-Compatible"
                else os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
            ),
            model_name=model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            input_language=args.input_language,
            output_language=args.output_language,
            reading_direction=args.reading_direction,
            translation_mode=args.translation_mode,
            enable_thinking=args.enable_thinking,
            enable_grounding=args.enable_grounding,
            reasoning_effort=args.reasoning_effort,
            send_full_page_context=args.send_full_page_context,
            upscale_method=args.upscale_method,
            bubble_min_side_pixels=args.bubble_min_side_pixels,
            context_image_max_side_pixels=args.context_image_max_side_pixels,
            osb_min_side_pixels=args.osb_min_side_pixels,
            special_instructions=args.special_instructions,
        ),
        rendering=RenderingConfig(
            font_dir=args.font_dir,
            max_font_size=args.max_font_size,
            min_font_size=args.min_font_size,
            line_spacing=args.line_spacing,
            use_subpixel_rendering=args.use_subpixel_rendering,
            font_hinting=args.font_hinting,
            use_ligatures=args.use_ligatures,
            hyphenate_before_scaling=args.hyphenate_before_scaling,
            hyphen_penalty=args.hyphen_penalty,
            hyphenation_min_word_length=args.hyphenation_min_word_length,
            badness_exponent=args.badness_exponent,
            padding_pixels=args.padding_pixels,
        ),
        output=OutputConfig(
            output_format=args.output_format,
            jpeg_quality=args.jpeg_quality,
            png_compression=args.png_compression,
            upscale_final_image=args.upscale_final_image,
            upscale_final_image_factor=args.upscale_final_image_factor,
        ),
        outside_text=OutsideTextConfig(
            enabled=args.osb_enable,
            huggingface_token=args.osb_huggingface_token
            or os.environ.get("HUGGINGFACE_TOKEN", ""),
            flux_num_inference_steps=args.osb_flux_steps,
            flux_residual_diff_threshold=args.osb_flux_residual_threshold,
            seed=args.osb_seed,
            osb_font_name=args.osb_font_name,
            osb_max_font_size=args.osb_max_font_size,
            osb_min_font_size=args.osb_min_font_size,
            osb_use_ligatures=args.osb_use_ligatures,
            osb_outline_width=args.osb_outline_width,
            osb_line_spacing=args.osb_line_spacing,
            osb_use_subpixel_rendering=args.osb_use_subpixel,
            osb_font_hinting=args.osb_font_hinting,
            bbox_expansion_percent=args.osb_bbox_expansion,
            easyocr_min_size=args.osb_easyocr_min_size,
        ),
        test_mode=args.test_mode,
    )

    # --- Execute ---
    if args.batch:
        input_path = Path(args.input)
        zip_temp_dir_obj = None
        preserve_structure = False

        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            log_message(f"Detected ZIP archive: {input_path.name}", always_print=True)
            try:
                temp_dir_obj = tempfile.TemporaryDirectory()
                temp_dir_path = Path(temp_dir_obj.name)
                zip_temp_dir_obj = temp_dir_obj

                with zipfile.ZipFile(input_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir_path)

                log_message(
                    "Extracted ZIP archive to temporary directory",
                    always_print=True,
                )

                input_path = temp_dir_path
                preserve_structure = True
            except zipfile.BadZipFile:
                log_message(
                    f"Error: '{args.input}' is not a valid ZIP archive.",
                    always_print=True,
                )
                exit(1)
            except Exception as e:
                log_message(
                    f"Error extracting ZIP archive: {str(e)}",
                    always_print=True,
                )
                exit(1)
        elif not input_path.is_dir():
            log_message(
                f"Error: --batch requires --input '{args.input}' to be a directory or ZIP archive.",
                always_print=True,
            )
            exit(1)

        output_dir = Path(args.output) if args.output else None

        if args.output:
            output_dir = Path(args.output)
            if not output_dir.exists():
                log_message(
                    f"Creating output directory: {output_dir}", always_print=True
                )
                output_dir.mkdir(parents=True, exist_ok=True)
            elif not output_dir.is_dir():
                log_message(
                    f"Error: Specified --output '{output_dir}' is not a directory.",
                    always_print=True,
                )
                exit(1)

        try:
            batch_translate_images(
                input_path, config, output_dir, preserve_structure=preserve_structure
            )
        finally:
            if zip_temp_dir_obj:
                try:
                    zip_temp_dir_obj.cleanup()
                    log_message(
                        "Cleaned up ZIP extraction temporary directory",
                        always_print=True,
                    )
                except Exception as e_clean:
                    log_message(
                        f"Warning: Failed to clean up temporary directory: {e_clean}",
                        always_print=True,
                    )
    else:
        input_path = Path(args.input)
        if not input_path.is_file():
            log_message(
                f"Error: Input '{args.input}' is not a valid file.", always_print=True
            )
            exit(1)

        output_path_arg = args.output
        if not output_path_arg:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            original_ext = input_path.suffix.lower()
            output_ext = original_ext
            output_dir = Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"MangaTranslator_{timestamp}{output_ext}"
            log_message(
                f"--output not specified, using default: {output_path}",
                always_print=True,
            )
        else:
            output_path = Path(output_path_arg)
            if not output_path.parent.exists():
                log_message(
                    f"Creating directory for output file: {output_path.parent}",
                    always_print=True,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            log_message(f"Processing {input_path}...", always_print=True)
            translate_and_render(input_path, config, output_path)
            log_message(
                f"Translation complete. Result saved to {output_path}",
                always_print=True,
            )
        except Exception as e:
            log_message(f"Error processing {input_path}: {e}", always_print=True)


if __name__ == "__main__":
    main()
