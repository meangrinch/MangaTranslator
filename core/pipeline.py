import base64
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import cv2
from PIL import Image

from core.caching import get_cache
from core.config import MangaTranslatorConfig, PreprocessingConfig, RenderingConfig
from core.scaling import scale_font_size, scale_length, scale_scalar
from utils.exceptions import (
    CancellationError,
    CleaningError,
    FontError,
    ImageProcessingError,
    RenderingError,
    TranslationError,
)
from utils.logging import log_message

from .image.cleaning import clean_speech_bubbles, retry_cleaning_with_otsu
from .image.detection import detect_panels, detect_speech_bubbles
from .image.image_utils import (
    convert_image_to_target_mode,
    cv2_to_pil,
    pil_to_cv2,
    resize_to_max_side,
    save_image_with_compression,
    upscale_image,
    upscale_image_to_dimension,
)
from .image.sorting import sort_bubbles_by_reading_order
from .ml.model_manager import get_model_manager
from .outside_text_processor import process_outside_text
from .services.translation import (
    call_translation_api_batch,
    prepare_bubble_images_for_translation,
)
from .text.text_processing import is_latin_style_language
from .text.text_renderer import render_text_skia

if TYPE_CHECKING:
    from ui.cancellation import CancellationManager


def get_image_encoding_params(pil_image_format: Optional[str]) -> Tuple[str, str]:
    """Returns (mime_type, cv2_ext) for a given PIL image format."""
    if pil_image_format and pil_image_format.upper() == "PNG":
        return "image/png", ".png"
    return "image/jpeg", ".jpg"


def _resolve_pre_upscale_factor(
    pre_cfg: Optional[PreprocessingConfig],
    verbose: bool = False,
) -> float:
    if pre_cfg is None or not pre_cfg.enabled:
        return 1.0

    factor = max(1.0, min(float(pre_cfg.factor or 1.0), 8.0))
    if factor <= 1.01:
        return 1.0

    log_message(f"Initial upscaling enabled: {factor:.2f}x", verbose=verbose)
    return factor


def _apply_pre_upscale_if_needed(
    image: Image.Image,
    config: MangaTranslatorConfig,
    verbose: bool = False,
) -> Tuple[Image.Image, float]:
    factor = _resolve_pre_upscale_factor(
        getattr(config, "preprocessing", None), verbose
    )
    if factor == 1.0:
        return image, 1.0

    # Use the output upscale model setting for initial upscaling as well
    model_type = (
        getattr(config.output, "image_upscale_model", "model_lite")
        if hasattr(config, "output")
        else "model_lite"
    )
    upscaled = upscale_image(image, factor, model_type=model_type, verbose=verbose)
    return upscaled, factor


def translate_and_render(
    image_path: Union[str, Path],
    config: MangaTranslatorConfig,
    output_path: Optional[Union[str, Path]] = None,
    cancellation_manager: Optional["CancellationManager"] = None,
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

    if cancellation_manager and cancellation_manager.is_cancelled():
        raise TranslationError("Process cancelled by user.")

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
    pil_image_processed, _ = _apply_pre_upscale_if_needed(
        pil_image_processed, config, verbose
    )

    # Check for Upscaling Only Mode (skip detection, cleaning, and translation)
    if config.upscaling_only:
        log_message(
            "Upscaling only mode - skipping detection and translation",
            always_print=True,
        )
        final_image_to_save = pil_image_processed

        if config.output.upscale_final_image:
            log_message("Upscaling final image...", verbose=verbose, always_print=True)
            final_image_to_save = upscale_image(
                final_image_to_save,
                config.output.image_upscale_factor,
                model_type=config.output.image_upscale_model,
                verbose=verbose,
            )

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
        log_message(
            f"Processing completed in {processing_time:.2f}s", always_print=True
        )

        return final_image_to_save

    # Calculate dynamic processing scale based on image area relative to 1MP (if enabled)
    if config.preprocessing.auto_scale:
        width, height = pil_image_processed.size
        processing_scale = math.sqrt((width * height) / 1_000_000)
        log_message(
            f"Dynamic processing scale: {processing_scale:.2f}x", verbose=verbose
        )
    else:
        processing_scale = 1.0

    get_cache().set_current_image(pil_image_processed, verbose)

    original_cv_image = pil_to_cv2(pil_image_processed)

    # Detect speech bubbles first so OSB processing can respect bubble regions
    log_message("Detecting speech bubbles...", verbose=verbose)
    try:
        bubble_data, text_free_boxes = detect_speech_bubbles(
            image_path,
            config.yolo_model_path,
            config.detection.confidence,
            verbose=verbose,
            device=device,
            use_sam2=config.detection.use_sam2,
            conjoined_detection=config.detection.conjoined_detection,
            conjoined_confidence=config.detection.conjoined_confidence,
            image_override=pil_image_processed,
            osb_enabled=config.outside_text.enabled,
            osb_text_verification=config.detection.use_osb_text_verification,
            osb_text_hf_token=config.outside_text.huggingface_token,
        )
    except Exception as e:
        log_message(f"Error during detection: {e}", always_print=True)
        bubble_data = []
        text_free_boxes = []

    # Process outside text detection and inpainting (bubble-aware)
    pil_image_processed, outside_text_data = process_outside_text(
        pil_image_processed,
        config,
        image_path,
        image_format,
        verbose,
        bubble_data=bubble_data,
        text_free_boxes=text_free_boxes,
    )
    original_cv_image = pil_to_cv2(pil_image_processed)

    full_image_b64 = None
    full_image_mime_type = None
    if config.translation.send_full_page_context:
        try:
            # processing_scale is intentionally not used for context_image_max_side_pixels
            context_image_pil = cv2_to_pil(original_cv_image)
            effective_context_max_side = scale_length(
                config.translation.context_image_max_side_pixels,
                None,
                minimum=512,
                maximum=4096,
            )

            # Disable upscaling in test_mode
            context_upscale_method = (
                "none" if config.test_mode else config.translation.upscale_method
            )

            if context_upscale_method == "model":
                # Use upscaling model for full page context
                model_manager = get_model_manager()
                with model_manager.upscale_context() as upscale_model:
                    context_image_pil = upscale_image_to_dimension(
                        upscale_model,
                        context_image_pil,
                        effective_context_max_side,
                        config.device,
                        "max",
                        "model",
                        verbose,
                    )
                    # Resize to exact target dimension (downscale if needed)
                    context_image_pil = resize_to_max_side(
                        context_image_pil,
                        effective_context_max_side,
                        verbose=verbose,
                    )
                    log_message(
                        "Upscaled full image for context with model", verbose=verbose
                    )
            elif context_upscale_method == "model_lite":
                # Use upscaling lite model for full page context
                model_manager = get_model_manager()
                with model_manager.upscale_lite_context() as upscale_model:
                    context_image_pil = upscale_image_to_dimension(
                        upscale_model,
                        context_image_pil,
                        effective_context_max_side,
                        config.device,
                        "max",
                        "model_lite",
                        verbose,
                    )
                    # Resize to exact target dimension (downscale if needed)
                    context_image_pil = resize_to_max_side(
                        context_image_pil,
                        effective_context_max_side,
                        verbose=verbose,
                    )
                    log_message(
                        "Upscaled full image for context with lite model",
                        verbose=verbose,
                    )
            elif context_upscale_method == "lanczos":
                # Use LANCZOS resampling
                context_image_pil = resize_to_max_side(
                    context_image_pil,
                    effective_context_max_side,
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

    if cancellation_manager and cancellation_manager.is_cancelled():
        raise CancellationError("Process cancelled by user.")

    final_image_to_save = pil_image_processed

    if not bubble_data and not outside_text_data:
        log_message("No speech bubbles or outside text detected", always_print=True)
    else:
        if bubble_data:
            log_message(f"Detected {len(bubble_data)} bubbles", verbose=verbose)
        if outside_text_data:
            log_message(
                f"Detected {len(outside_text_data)} outside text regions",
                verbose=verbose,
            )

        if cancellation_manager and cancellation_manager.is_cancelled():
            raise CancellationError("Process cancelled by user.")

        if bubble_data:
            log_message("Cleaning speech bubbles...", verbose=verbose)
            try:
                use_otsu = config.cleaning.use_otsu_threshold
                if config.cleaning.inpaint_colored_bubbles:
                    log_message(
                        "Flux inpainting enabled for colored bubbles",
                        verbose=verbose,
                    )

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
                    processing_scale=processing_scale,
                    conjoined_confidence=config.detection.conjoined_confidence,
                    inpaint_colored_bubbles=config.cleaning.inpaint_colored_bubbles,
                    flux_hf_token=config.outside_text.huggingface_token,
                    flux_num_inference_steps=config.outside_text.flux_num_inference_steps,
                    flux_residual_diff_threshold=config.outside_text.flux_residual_diff_threshold,
                    flux_seed=config.outside_text.seed,
                    osb_text_verification=config.detection.use_osb_text_verification,
                    osb_text_hf_token=config.outside_text.huggingface_token,
                    force_cv2_inpainting=config.outside_text.force_cv2_inpainting,
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
                log_message(
                    f"Converting cleaned image to {target_mode}", verbose=verbose
                )
                pil_cleaned_image = pil_cleaned_image.convert(target_mode)
            final_image_to_save = pil_cleaned_image
        else:
            processed_bubbles_info = []
            pil_cleaned_image = pil_image_processed
            if pil_cleaned_image.mode != target_mode:
                log_message(f"Converting image to {target_mode}", verbose=verbose)
                pil_cleaned_image = pil_cleaned_image.convert(target_mode)
            final_image_to_save = pil_cleaned_image

        # Check for Cleaning Only Mode
        if config.cleaning_only:
            log_message("Cleaning only mode - skipping translation", always_print=True)
        else:
            main_min_font = scale_font_size(
                config.rendering.min_font_size, processing_scale, minimum=4, maximum=256
            )
            main_max_font = scale_font_size(
                config.rendering.max_font_size,
                processing_scale,
                minimum=main_min_font,
                maximum=384,
            )
            padding_pixels = scale_scalar(
                config.rendering.padding_pixels,
                processing_scale,
                minimum=1.0,
                maximum=80.0,
            )
            osb_min_font = scale_font_size(
                config.outside_text.osb_min_font_size,
                processing_scale,
                minimum=4,
                maximum=512,
            )
            osb_max_font = scale_font_size(
                config.outside_text.osb_max_font_size,
                processing_scale,
                minimum=osb_min_font,
                maximum=640,
            )
            osb_outline_width = scale_scalar(
                config.outside_text.osb_outline_width,
                processing_scale,
                minimum=0.0,
                maximum=24.0,
            )
            # Prepare images for Translation
            log_message("Preparing bubble images...", verbose=verbose)

            # Disable upscaling in test_mode
            bubble_upscale_method = (
                "none" if config.test_mode else config.translation.upscale_method
            )

            model_manager = get_model_manager()
            # Use appropriate context manager based on upscale_method
            if bubble_upscale_method == "model":
                context_manager = model_manager.upscale_context()
            elif bubble_upscale_method == "model_lite":
                context_manager = model_manager.upscale_lite_context()
            else:
                # For lanczos/none, create a dummy context manager that yields None
                from contextlib import nullcontext

                context_manager = nullcontext(None)

            with context_manager as upscale_model:
                bubble_data = prepare_bubble_images_for_translation(
                    bubble_data,
                    original_cv_image,
                    upscale_model,
                    config.device,
                    mime_type,
                    config.translation.bubble_min_side_pixels,
                    bubble_upscale_method,
                    verbose,
                )

            if bubble_upscale_method != "none":
                log_message(
                    f"Upscaled {len(bubble_data)} bubble images for translation",
                    always_print=True,
                )
            else:
                log_message(
                    f"Prepared {len(bubble_data)} bubble images for translation",
                    always_print=True,
                )
            valid_bubble_data = [b for b in bubble_data if b.get("image_b64")]
            if not valid_bubble_data and not outside_text_data:
                log_message(
                    "No valid bubble images or outside text for translation",
                    always_print=True,
                )
            else:  # Proceed if we have valid bubble data or outside text
                if cancellation_manager and cancellation_manager.is_cancelled():
                    raise CancellationError("Process cancelled by user.")

                # Sort and Translate
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

                # Detect panels if panel-aware sorting is enabled
                panels = None
                if config.detection.use_panel_sorting:
                    try:
                        log_message(
                            "Detecting panels for panel-aware sorting...",
                            verbose=verbose,
                        )
                        panels = detect_panels(
                            image_path,
                            confidence=config.detection.panel_confidence,
                            device=config.device,
                            verbose=verbose,
                        )
                        if panels:
                            log_message(
                                f"Detected {len(panels)} panels for sorting",
                                always_print=True,
                            )
                        else:
                            log_message(
                                "No panels detected, using global sorting",
                                verbose=verbose,
                            )
                    except Exception as e:
                        log_message(
                            f"Panel detection failed: {e}. Using global sorting.",
                            always_print=True,
                        )
                        panels = None

                # Sort all text elements (speech bubbles + OSB text) by reading order
                sorted_bubble_data = sort_bubbles_by_reading_order(
                    all_text_data, reading_direction, panels=panels
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
                                min_font_size=main_max_font,
                                max_font_size=main_max_font,
                                line_spacing_mult=config.rendering.line_spacing_mult,
                                use_subpixel_rendering=config.rendering.use_subpixel_rendering,
                                font_hinting=config.rendering.font_hinting,
                                use_ligatures=config.rendering.use_ligatures,
                                hyphenate_before_scaling=config.rendering.hyphenate_before_scaling,
                                hyphen_penalty=config.rendering.hyphen_penalty,
                                hyphenation_min_word_length=config.rendering.hyphenation_min_word_length,
                                badness_exponent=config.rendering.badness_exponent,
                                padding_pixels=padding_pixels,
                                supersampling_factor=1,  # No supersampling for probe
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
                            error_str = str(e).lower()
                            critical_tokens = (
                                "429",
                                "rate limit",
                                "rate-limit",
                                "auth",
                                "unauthorized",
                                "forbidden",
                                "payment",
                                "quota",
                                "empty response",
                                "api failed",
                            )
                            if any(token in error_str for token in critical_tokens):
                                raise

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

                        valid_translations = [
                            t
                            for t in translated_texts
                            if t
                            and not t.startswith("[Translation Error")
                            and not t.startswith("API Error")
                            and t.strip()
                            not in {
                                "[OCR FAILED]",
                                "[Empty response / no content]",
                                f"[{config.translation.provider}: API call failed/blocked]",
                                f"[{config.translation.provider}: OCR call failed/blocked]",
                                f"[{config.translation.provider}: Failed to parse response]",
                            }
                        ]

                        if bubble_images_b64 and not valid_translations:
                            raise TranslationError(
                                "Total translation failure: All bubbles failed."
                            )

                # Render Translations
                bubble_render_info_map = {
                    tuple(info["bbox"]): {
                        "color": info["color"],
                        "mask": info.get("mask"),
                        "base_mask": info.get("base_mask"),
                        "is_sam": info.get("is_sam", False),
                        "is_colored": info.get("is_colored", False),
                        "text_bbox": info.get("text_bbox"),
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
                            min_font = osb_min_font
                            max_font = osb_max_font
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
                            # OSB renders default to horizontal; vertical stacking is fallback-only
                            rotation_deg = 0.0
                            vertical_stack = False
                        else:
                            log_message(
                                f"Rendering bubble {bbox}: '{text[:30]}...'",
                                verbose=verbose,
                            )
                            font_dir = config.rendering.font_dir
                            min_font = main_min_font
                            max_font = main_max_font
                            line_spacing = config.rendering.line_spacing_mult
                            use_ligs = config.rendering.use_ligatures
                            render_info = bubble_render_info_map.get(tuple(bbox))
                            bubble_color_bgr = (255, 255, 255)
                            cleaned_mask = None
                            base_mask = None
                            is_sam_mask = False
                            if render_info:
                                bubble_color_bgr = render_info["color"]
                                cleaned_mask = render_info.get("mask")
                                base_mask = render_info.get("base_mask")
                                is_sam_mask = render_info.get("is_sam", False)
                            # No rotation/stacking for regular bubbles
                            vertical_stack = False
                            rotation_deg = 0.0

                        # Only apply hyphenation for Latin-style languages
                        should_hyphenate = config.rendering.hyphenate_before_scaling
                        if not is_latin_style_language(
                            config.translation.output_language
                        ):
                            should_hyphenate = False

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
                            hyphenate_before_scaling=should_hyphenate,
                            hyphen_penalty=config.rendering.hyphen_penalty,
                            hyphenation_min_word_length=config.rendering.hyphenation_min_word_length,
                            badness_exponent=config.rendering.badness_exponent,
                            padding_pixels=padding_pixels,
                            outline_width=(
                                osb_outline_width if is_outside_text else 0.0
                            ),
                            supersampling_factor=config.rendering.supersampling_factor,
                        )
                        success = False
                        if is_outside_text:
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
                                    raise_on_safe_error=False,
                                )
                                success = True
                            except Exception as e:
                                log_message(
                                    f"Text rendering failed: {e}", verbose=verbose
                                )
                                rendered_image = pil_cleaned_image
                                success = False

                                # Absolute last-chance fallback: force vertical stacking before giving up
                                if not vertical_stack:
                                    # Fallback uses neutral rotation since we no longer track orientation
                                    forced_stack_rotation = 0.0
                                    try:
                                        log_message(
                                            "OSB render failed, retrying with vertical-stack fallback",
                                            verbose=verbose,
                                            always_print=True,
                                        )
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
                                            rotation_deg=forced_stack_rotation,
                                            vertical_stack=True,
                                            raise_on_safe_error=False,
                                        )
                                        log_message(
                                            "Vertical-stack fallback succeeded",
                                            verbose=verbose,
                                        )
                                        success = True
                                    except Exception as e2:
                                        log_message(
                                            f"Vertical-stack fallback failed: {e2}",
                                            verbose=verbose,
                                        )
                                        # Restore original OSB patch if available
                                        if "original_crop_pil" in bubble:
                                            log_message(
                                                f"Restoring original OSB patch for {bbox}",
                                                verbose=verbose,
                                                always_print=True,
                                            )
                                            rendered_image = pil_cleaned_image.copy()
                                            original_patch = bubble["original_crop_pil"]
                                            rendered_image.paste(
                                                original_patch, (bbox[0], bbox[1])
                                            )
                                            success = True
                                        else:
                                            rendered_image = pil_cleaned_image
                                            success = False
                                else:
                                    if "original_crop_pil" in bubble:
                                        log_message(
                                            f"Restoring original OSB patch for {bbox}",
                                            verbose=verbose,
                                            always_print=True,
                                        )
                                        rendered_image = pil_cleaned_image.copy()
                                        original_patch = bubble["original_crop_pil"]
                                        rendered_image.paste(
                                            original_patch, (bbox[0], bbox[1])
                                        )
                                        success = True
                                    else:
                                        rendered_image = pil_cleaned_image
                                        success = False
                        else:
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
                                    raise_on_safe_error=True,
                                )
                                success = True
                            except ImageProcessingError as e:
                                safe_area_failed = (
                                    "Safe area calculation failed" in str(e)
                                )
                                retry_result = None
                                if safe_area_failed and base_mask is not None:
                                    log_message(
                                        f"Safe area failed for bubble {bbox}, retrying mask with Otsu",
                                        verbose=verbose,
                                        always_print=True,
                                    )
                                    retry_result = retry_cleaning_with_otsu(
                                        original_cv_image,
                                        {
                                            "base_mask": base_mask,
                                            "bbox": bbox,
                                            "is_sam": is_sam_mask,
                                            "is_colored": (
                                                render_info.get("is_colored", False)
                                                if render_info
                                                else False
                                            ),
                                            "text_bbox": (
                                                render_info.get("text_bbox")
                                                if render_info
                                                else None
                                            ),
                                        },
                                        config.cleaning.thresholding_value,
                                        config.cleaning.roi_shrink_px,
                                        processing_scale,
                                        verbose=verbose,
                                        classify_colored=(
                                            config.cleaning.inpaint_colored_bubbles
                                        ),
                                    )

                                if (
                                    retry_result
                                    and retry_result.get("mask") is not None
                                ):
                                    cleaned_mask = retry_result["mask"]
                                    bubble_color_bgr = retry_result.get(
                                        "color", bubble_color_bgr
                                    )
                                    base_mask = retry_result.get("base_mask", base_mask)
                                    if render_info is not None:
                                        render_info.update(
                                            {
                                                "mask": cleaned_mask,
                                                "color": bubble_color_bgr,
                                                "base_mask": base_mask,
                                                "is_colored": retry_result.get(
                                                    "is_colored",
                                                    render_info.get(
                                                        "is_colored", False
                                                    ),
                                                ),
                                                "text_bbox": retry_result.get(
                                                    "text_bbox",
                                                    render_info.get("text_bbox"),
                                                ),
                                            }
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
                                            raise_on_safe_error=False,
                                        )
                                        success = True
                                    except (
                                        RenderingError,
                                        FontError,
                                        ImageProcessingError,
                                    ) as e2:
                                        log_message(
                                            f"Text rendering failed after Otsu retry: {e2}",
                                            verbose=verbose,
                                        )
                                        rendered_image = pil_cleaned_image
                                        success = False
                                if not success:
                                    # Final fallback to padded bbox path
                                    fallback_msg = (
                                        f"Safe area calculation failed for {bbox}, using padded bbox fallback"
                                        if safe_area_failed
                                        else f"Rendering retry fallback for {bbox}, using padded bbox method"
                                    )
                                    log_message(
                                        fallback_msg,
                                        verbose=verbose,
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
                                            raise_on_safe_error=False,
                                        )
                                        success = True
                                    except (RenderingError, FontError) as e2:
                                        log_message(
                                            f"Text rendering failed: {e2}",
                                            verbose=verbose,
                                        )
                                        rendered_image = pil_cleaned_image
                                        success = False
                            except (RenderingError, FontError) as e:
                                log_message(
                                    f"Text rendering failed: {e}", verbose=verbose
                                )
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

    # Final Image Upscaling (optional)
    if config.output.upscale_final_image:
        log_message("Upscaling final image...", verbose=verbose, always_print=True)
        final_image_to_save = upscale_image(
            final_image_to_save,
            config.output.image_upscale_factor,
            model_type=config.output.image_upscale_model,
            verbose=verbose,
        )

    # Save Output
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
    cancellation_manager: Optional["CancellationManager"] = None,
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

            if cancellation_manager and cancellation_manager.is_cancelled():
                raise CancellationError("Batch process cancelled by user.")

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

            translate_and_render(
                img_path, config, output_path, cancellation_manager=cancellation_manager
            )

            results["success_count"] += 1

            if progress_callback:
                completed_progress = (i + 1) / total_images
                progress_callback(
                    completed_progress, f"Completed {i + 1}/{total_images} images"
                )

        except CancellationError:
            log_message(
                f"Batch cancelled during processing of {display_path}",
                verbose=config.verbose,
            )
            raise
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
