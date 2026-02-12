import base64
import gc
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from core.config import MangaTranslatorConfig
from core.image.image_utils import cv2_to_pil, pil_to_cv2, process_bubble_image_cached
from core.image.inpainting import FluxKleinInpainter, FluxKontextInpainter
from core.image.ocr_detection import OutsideTextDetector, extract_text_with_manga_ocr
from core.ml.model_manager import get_model_manager
from utils.logging import log_message


def process_outside_text(
    pil_image: Image.Image,
    config: MangaTranslatorConfig,
    image_path: Union[str, Path],
    image_format: Optional[str],
    verbose: bool = False,
    bubble_data: Optional[List[Dict[str, Any]]] = None,
    text_free_boxes: Optional[List[List[float]]] = None,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Process outside text detection, inpainting, and prepare data for translation.

    This function handles the complete outside text processing pipeline:
    1. Detects text outside speech bubbles using OCR
    2. Inpaints the detected text regions using FluxKontext
    3. Prepares the outside text data for translation API calls

    Args:
        pil_image: The PIL image to process
        config: MangaTranslatorConfig containing all settings
        image_path: Path to the original image file
        image_format: Original image format (PNG, JPEG, etc.)
        processing_scale: The scale factor for image processing
        verbose: Whether to print detailed logging

    Returns:
        Tuple containing:
        - processed_pil_image: The image after outside text inpainting
        - outside_text_data: List of dicts with outside text information for translation
    """
    if not config.outside_text.enabled:
        return pil_image, []

    log_message("Detecting text outside speech bubbles...", verbose=verbose)

    try:
        outside_detector = OutsideTextDetector(
            device=config.device, hf_token=config.outside_text.huggingface_token
        )
        outside_text_results = outside_detector.detect_outside_text(
            str(image_path),
            yolo_model_path=config.yolo_model_path,
            confidence=config.outside_text.osb_confidence,
            conjoined_confidence=config.detection.conjoined_confidence,
            verbose=verbose,
            image_override=pil_image,
            existing_bubbles=bubble_data,
            text_free_boxes=text_free_boxes,
        )

        if not outside_text_results:
            log_message("No outside text regions found", verbose=verbose)
            outside_detector.unload_models()
            return pil_image, []

        img_w, img_h = pil_image.size

        # Filter out probable page numbers
        # Only run OCR on "suspicious" detections (small & in margin)
        if config.outside_text.enable_page_number_filtering and outside_text_results:
            suspicious_crops = []
            suspicious_indices = []
            safe_results = []

            margin_threshold = max(
                0.0, min(0.3, config.outside_text.page_filter_margin_threshold)
            )
            min_area_threshold = max(
                0.0, min(0.2, config.outside_text.page_filter_min_area_ratio)
            )

            for i, res in enumerate(outside_text_results):
                bbox, _ = res
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cy = (y1 + y2) / 2

                is_in_margin = (cy < img_h * margin_threshold) or (
                    cy > img_h * (1 - margin_threshold)
                )

                area = (x2 - x1) * (y2 - y1)
                is_small = area < (img_w * img_h * min_area_threshold)

                if is_in_margin and is_small:
                    suspicious_crops.append(pil_image.crop((x1, y1, x2, y2)))
                    suspicious_indices.append(i)
                else:
                    safe_results.append(res)

            if suspicious_crops:
                log_message(
                    f"Verifying {len(suspicious_crops)} suspicious OSB regions with OCR...",
                    verbose=verbose,
                )
                suspicious_texts = extract_text_with_manga_ocr(
                    suspicious_crops, verbose=verbose
                )

                kept_suspicious_count = 0
                for i, text in enumerate(suspicious_texts):
                    # Regex for page numbers: digits, "Page 20", "p. 20", etc.
                    is_page_number = bool(
                        re.match(
                            r"^\s*(?:page\.?|p\.?)?\s*\d+\s*$", text, re.IGNORECASE
                        )
                    )

                    if not is_page_number:
                        safe_results.append(outside_text_results[suspicious_indices[i]])
                        kept_suspicious_count += 1
                    else:
                        log_message(
                            f"Filtered out page number: '{text}'", verbose=verbose
                        )

                outside_text_results = safe_results
                log_message(
                    f"Remaining OSB regions after filtering: {len(outside_text_results)}",
                    verbose=verbose,
                )

        # Build a mask of all detected speech bubbles to prevent OSB inpainting overlap
        total_bubble_mask = np.zeros((img_h, img_w), dtype=bool)
        if bubble_data:
            for bubble in bubble_data:
                try:
                    mask = bubble.get("sam_mask") if isinstance(bubble, dict) else None
                    if mask is not None:
                        mask_np = np.asarray(mask)
                        if mask_np.ndim == 3:
                            mask_np = mask_np[..., 0]
                        mask_bool = mask_np > 0
                        if mask_bool.shape[0] == img_h and mask_bool.shape[1] == img_w:
                            total_bubble_mask |= mask_bool
                            continue

                    bbox = bubble.get("bbox") if isinstance(bubble, dict) else None
                    if bbox and len(bbox) == 4:
                        x0, y0, x1, y1 = [int(c) for c in bbox]
                        x0 = max(0, min(img_w, x0))
                        x1 = max(0, min(img_w, x1))
                        y0 = max(0, min(img_h, y0))
                        y1 = max(0, min(img_h, y1))
                        if x1 > x0 and y1 > y0:
                            total_bubble_mask[y0:y1, x0:x1] = True
                except Exception as e:
                    log_message(
                        f"Warning: Failed to apply bubble mask for OSB exclusion: {e}",
                        verbose=verbose,
                    )

        mime_type = (
            "image/png"
            if image_format and image_format.upper() == "PNG"
            else "image/jpeg"
        )
        cv2_ext = ".png" if image_format and image_format.upper() == "PNG" else ".jpg"

        # Probe original text color for OSB rendering
        original_text_colors = {}
        for ocr_result in outside_text_results:
            bbox_coords, conf = ocr_result
            x1, y1, x2, y2 = [int(c) for c in bbox_coords]
            bbox_tuple = (x1, y1, x2, y2)

            bbox_area_img = pil_image.crop((x1, y1, x2, y2))
            bbox_array = np.array(bbox_area_img)

            if bbox_array.shape[-1] == 4:
                bbox_array = bbox_array[..., :3]

            pixels = bbox_array.reshape(-1, 3)

            # Use K-Means to find 2 dominant colors
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(pixels)

            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            unique, counts = np.unique(labels, return_counts=True)
            dominant_cluster_idx = unique[np.argmax(counts)]

            # Dominant cluster is usually the background (text pixels are sparse)
            bg_color_rgb = centers[dominant_cluster_idx]
            # Use proper luminance calculation (ITU-R BT.601)
            bg_brightness = (
                0.299 * bg_color_rgb[0]
                + 0.587 * bg_color_rgb[1]
                + 0.114 * bg_color_rgb[2]
            )
            is_dark_text = (
                bg_brightness < 128
            )  # passed downstream; renderer inverts for text color
            original_text_colors[bbox_tuple] = is_dark_text

            log_message(
                f"OSB bbox {bbox_tuple}: "
                f"{'Dark' if is_dark_text else 'Light'} background detected "
                f"(luminance={bg_brightness:.1f})",
                verbose=verbose,
            )

        log_message("Inpainting outside text regions...", verbose=verbose)

        # Create inpainter based on selected method
        inpainting_method = config.outside_text.inpainting_method
        inpainter = None

        if inpainting_method == "flux_klein_9b":
            try:
                inpainter = FluxKleinInpainter(
                    variant="9b",
                    device=config.device,
                    huggingface_token=config.outside_text.huggingface_token,
                    num_inference_steps=config.outside_text.flux_num_inference_steps,
                    low_vram=config.outside_text.flux_low_vram,
                    luminance_correction=config.outside_text.flux_luminance_correction,
                    verbose=verbose,
                )
                log_message("Using Flux.2 Klein 9B for inpainting", verbose=verbose)
            except Exception as e:
                log_message(
                    f"Flux Klein 9B unavailable ({e}), falling back to OpenCV",
                    verbose=verbose,
                )

        if inpainting_method == "flux_klein_4b":
            try:
                inpainter = FluxKleinInpainter(
                    variant="4b",
                    device=config.device,
                    huggingface_token=config.outside_text.huggingface_token,
                    num_inference_steps=config.outside_text.flux_num_inference_steps,
                    low_vram=config.outside_text.flux_low_vram,
                    luminance_correction=config.outside_text.flux_luminance_correction,
                    verbose=verbose,
                )
                log_message("Using Flux.2 Klein 4B for inpainting", verbose=verbose)
            except Exception as e:
                log_message(
                    f"Flux Klein 4B unavailable ({e}), falling back to OpenCV",
                    verbose=verbose,
                )

        if inpainting_method == "flux_kontext":
            try:
                # Determine backend from config
                backend = config.outside_text.kontext_backend
                low_vram = (
                    config.outside_text.flux_low_vram if backend == "sdnq" else False
                )
                inpainter = FluxKontextInpainter(
                    device=config.device,
                    huggingface_token=config.outside_text.huggingface_token,
                    num_inference_steps=config.outside_text.flux_num_inference_steps,
                    residual_diff_threshold=config.outside_text.flux_residual_diff_threshold,
                    backend=backend,
                    low_vram=low_vram,
                )
                backend_label = "SDNQ" if backend == "sdnq" else "Nunchaku"
                log_message(
                    f"Using Flux.1 Kontext ({backend_label}) for inpainting",
                    verbose=verbose,
                )
            except Exception as e:
                log_message(
                    f"Flux Kontext unavailable ({e}), falling back to OpenCV",
                    verbose=verbose,
                )

        if inpainting_method == "opencv" or inpainter is None:
            inpainter = None
            log_message("Using OpenCV simple fill for inpainting", verbose=verbose)

        mask_groups, _ = outside_detector.get_text_masks(
            str(image_path),
            bbox_expansion_percent=config.outside_text.bbox_expansion_percent,
            text_box_proximity_ratio=config.outside_text.text_box_proximity_ratio,
            verbose=verbose,
            image_override=pil_image,
            existing_results=outside_text_results,
        )

        current_image = pil_image
        temp_files = []
        try:
            if mask_groups:
                base_seed = (
                    random.randint(1, 999999)
                    if config.outside_text.seed == -1
                    else config.outside_text.seed
                )

                flux_inpaints = 0
                cv2_inpaints = 0
                for i, group in enumerate(mask_groups):
                    log_message(
                        f"Inpainting outside text region {i + 1}/{len(mask_groups)}",
                        verbose=verbose,
                    )
                    combined_mask = group["combined_mask"]
                    combined_mask = np.logical_and(
                        combined_mask, np.logical_not(total_bubble_mask)
                    )
                    if not np.any(combined_mask):
                        log_message(
                            "Skipping outside text region after bubble masking (no remaining area)",
                            verbose=verbose,
                        )
                        continue
                    region_seed = base_seed + i if base_seed > 0 else base_seed

                    original_bbox_dict = group.get("original_bbox")
                    composite_clip_bbox = None
                    fill_color = None
                    fallback_fill_color = None
                    ox0 = oy0 = ox1 = oy1 = None
                    if original_bbox_dict:
                        ox = int(original_bbox_dict.get("x", 0))
                        oy = int(original_bbox_dict.get("y", 0))
                        ow = int(original_bbox_dict.get("width", 0))
                        oh = int(original_bbox_dict.get("height", 0))
                        if ow > 0 and oh > 0:
                            ox0 = max(0, min(img_w, ox))
                            oy0 = max(0, min(img_h, oy))
                            ox1 = max(0, min(img_w, ox + ow))
                            oy1 = max(0, min(img_h, oy + oh))
                            composite_clip_bbox = (ox, oy, ox + ow, oy + oh)

                            # Determine detected text color for this region to ensure contrast
                            group_bg_is_dark = None
                            if original_text_colors:
                                votes_dark = 0
                                votes_light = 0
                                gx1, gy1, gx2, gy2 = ox, oy, ox + ow, oy + oh

                                for (
                                    bx1,
                                    by1,
                                    bx2,
                                    by2,
                                ), t_dark in original_text_colors.items():
                                    # Check if center of OCR box is inside group box
                                    bcx = (bx1 + bx2) / 2
                                    bcy = (by1 + by2) / 2
                                    if (
                                        bcx >= gx1
                                        and bcx <= gx2
                                        and bcy >= gy1
                                        and bcy <= gy2
                                    ):
                                        if t_dark:
                                            votes_dark += 1
                                        else:
                                            votes_light += 1
                                if votes_dark > 0 or votes_light > 0:
                                    group_bg_is_dark = votes_dark >= votes_light

                                    # Detected value represents background brightness
                                    fallback_fill_color = (
                                        (0, 0, 0)
                                        if group_bg_is_dark
                                        else (255, 255, 255)
                                    )

                                    t_type = "Dark" if group_bg_is_dark else "Light"
                                    f_col = (
                                        "White"
                                        if fallback_fill_color == (255, 255, 255)
                                        else "Black"
                                    )
                                    log_message(
                                        f"OSB Region {i + 1}: Detected {t_type} background. "
                                        f"Fallback fill: {f_col}.",
                                        verbose=verbose,
                                    )

                            # Expanded sampling around the original bbox to find background color
                            expansion_px = 2
                            sx1 = max(0, ox - expansion_px)
                            sy1 = max(0, oy - expansion_px)
                            sx2 = min(img_w, ox + ow + expansion_px)
                            sy2 = min(img_h, oy + oh + expansion_px)

                            if sx2 > sx1 and sy2 > sy1:
                                mask_h, mask_w = sy2 - sy1, sx2 - sx1
                                local_mask = np.ones((mask_h, mask_w), dtype=bool)

                                lx0 = max(0, ox0 - sx1)
                                ly0 = max(0, oy0 - sy1)
                                lx1 = min(mask_w, ox1 - sx1)
                                ly1 = min(mask_h, oy1 - sy1)

                                if lx1 > lx0 and ly1 > ly0:
                                    local_mask[ly0:ly1, lx0:lx1] = False

                                border_pixels = None
                                min_border_pixels = 20
                                if np.count_nonzero(local_mask) >= min_border_pixels:
                                    sampling_crop = current_image.crop(
                                        (sx1, sy1, sx2, sy2)
                                    )
                                    crop_np = np.array(sampling_crop.convert("RGB"))
                                    border_pixels = crop_np[local_mask]

                                if border_pixels is not None and border_pixels.size > 0:
                                    white_thresh = 250
                                    black_thresh = 5
                                    ratio_threshold = 0.95

                                    white_ratio = np.mean(
                                        np.all(border_pixels >= white_thresh, axis=1)
                                    )
                                    black_ratio = np.mean(
                                        np.all(border_pixels <= black_thresh, axis=1)
                                    )

                                    if fallback_fill_color is None:
                                        fallback_fill_color = (
                                            (255, 255, 255)
                                            if white_ratio >= black_ratio
                                            else (0, 0, 0)
                                        )

                                    force_fill = inpainting_method == "opencv"
                                    should_simple_fill = (
                                        white_ratio >= ratio_threshold
                                        or black_ratio >= ratio_threshold
                                        or force_fill
                                    )

                                    if should_simple_fill:
                                        fill_color = fallback_fill_color

                                        if force_fill and not (
                                            white_ratio >= ratio_threshold
                                            or black_ratio >= ratio_threshold
                                        ):
                                            log_message(
                                                "Forcing CV2 fill: defaulting to "
                                                f"{'white' if fill_color == (255, 255, 255) else 'black'} background",
                                                verbose=verbose,
                                            )
                                        else:
                                            log_message(
                                                "Skipping Flux for OSB region: detected pure "
                                                f"{'white' if fill_color == (255, 255, 255) else 'black'} background",
                                                verbose=verbose,
                                            )

                    def apply_simple_fill(color_to_use):
                        new_img = current_image.copy()

                        if (
                            original_bbox_dict
                            and ox1 is not None
                            and ox0 is not None
                            and oy1 is not None
                            and oy0 is not None
                            and ox1 > ox0
                            and oy1 > oy0
                        ):
                            # Restricted fill logic: Clip mask to bbox
                            region_mask = combined_mask[oy0:oy1, ox0:ox1]
                            if not np.any(region_mask):
                                return new_img

                            mask_pil = Image.fromarray(
                                (region_mask * 255).astype(np.uint8), mode="L"
                            )
                            patch = Image.new(
                                "RGB", (ox1 - ox0, oy1 - oy0), color_to_use
                            )
                            new_img.paste(patch, (ox0, oy0), mask=mask_pil)
                        else:
                            # Full mask fill
                            mask_pil = Image.fromarray(
                                (combined_mask * 255).astype(np.uint8), mode="L"
                            )
                            patch = Image.new("RGB", new_img.size, color_to_use)
                            new_img.paste(patch, (0, 0), mask=mask_pil)

                        return new_img

                    if fill_color is not None:
                        current_image = apply_simple_fill(fill_color)
                        cv2_inpaints += 1
                        continue

                    flux_failed = False
                    flux_fail_reason = None
                    inpainted_image = None

                    if inpainter is None:
                        flux_failed = True
                        flux_fail_reason = "Flux inpainter unavailable"
                    else:
                        try:
                            inpainted_image = inpainter.inpaint_mask(
                                current_image,
                                combined_mask,
                                seed=region_seed,
                                verbose=verbose,
                                strict_mask_clipping=True,
                                composite_clip_bbox=composite_clip_bbox,
                            )
                            if inpainted_image is current_image:
                                flux_failed = True
                                flux_fail_reason = (
                                    "Flux returned original image (no inpaint)"
                                )
                        except Exception as e:
                            flux_failed = True
                            flux_fail_reason = f"Flux inpainting error: {e}"

                    if flux_failed:
                        fallback_color_to_use = (
                            fallback_fill_color
                            if fallback_fill_color
                            else (255, 255, 255)
                        )
                        log_message(
                            f"Flux failed for OSB region {i + 1}"
                            + (f" ({flux_fail_reason})" if flux_fail_reason else "")
                            + f"; falling back to CV2 fill ({fallback_color_to_use})",
                            always_print=True,
                        )
                        current_image = apply_simple_fill(fallback_color_to_use)
                        cv2_inpaints += 1
                        continue

                    flux_inpaints += 1
                    # Save to disk if more regions remain to reduce memory usage
                    if i < len(mask_groups) - 1:
                        temp_file = None
                        try:
                            temp_fd, temp_file = tempfile.mkstemp(suffix=".png")
                            os.close(temp_fd)
                            inpainted_image.save(temp_file, format="PNG")
                            temp_files.append(temp_file)

                            with Image.open(temp_file) as img_tmp:
                                img_tmp.load()
                                current_image = img_tmp.copy()

                            del inpainted_image
                            gc.collect()
                            log_message(
                                "Saved intermediate inpainting result to disk",
                                verbose=verbose,
                            )
                        except Exception as e:
                            log_message(
                                "Warning: Failed to save intermediate image to disk: "
                                f"{e}. Continuing with in-memory processing.",
                                verbose=verbose,
                            )
                            # Fallback to in-memory if disk save fails
                            current_image = inpainted_image
                            if temp_file and temp_file in temp_files:
                                temp_files.remove(temp_file)
                    else:
                        current_image = inpainted_image

                log_message("Outside text inpainting completed", verbose=verbose)
                log_message(
                    f"Inpainted {len(mask_groups)} outside text regions (Flux: {flux_inpaints}, CV2: {cv2_inpaints})",
                    always_print=True,
                )
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass

        outside_text_data = []
        original_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        for ocr_result in outside_text_results:
            bbox_coords, conf = ocr_result
            x1, y1, x2, y2 = [int(c) for c in bbox_coords]
            bbox_tuple = (x1, y1, x2, y2)

            outside_text_image_cv = original_cv_image[y1:y2, x1:x2].copy()

            outside_text_image_pil = cv2_to_pil(outside_text_image_cv)

            original_crop_pil = outside_text_image_pil.copy()

            # Disable upscaling in test_mode
            osb_upscale_method = (
                "none" if config.test_mode else config.translation.upscale_method
            )

            if osb_upscale_method == "model":
                model_manager = get_model_manager()
                with model_manager.upscale_context() as upscale_model:
                    final_text_pil = process_bubble_image_cached(
                        outside_text_image_pil,
                        upscale_model,
                        config.device,
                        config.translation.osb_min_side_pixels,
                        "min",
                        "model",
                        verbose,
                    )
            elif osb_upscale_method == "model_lite":
                model_manager = get_model_manager()
                with model_manager.upscale_lite_context() as upscale_model:
                    final_text_pil = process_bubble_image_cached(
                        outside_text_image_pil,
                        upscale_model,
                        config.device,
                        config.translation.osb_min_side_pixels,
                        "min",
                        "model_lite",
                        verbose,
                    )
            elif osb_upscale_method == "lanczos":
                w, h = outside_text_image_pil.size
                min_side = min(w, h)
                if min_side < config.translation.osb_min_side_pixels:
                    scale_factor = config.translation.osb_min_side_pixels / min_side
                    new_w = int(w * scale_factor)
                    new_h = int(h * scale_factor)
                    resized_text = outside_text_image_pil.resize(
                        (new_w, new_h), Image.LANCZOS
                    )
                else:
                    resized_text = outside_text_image_pil
                final_text_pil = resized_text
            else:
                final_text_pil = outside_text_image_pil

            outside_text_image_cv = pil_to_cv2(final_text_pil)

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            aspect_ratio = float(h) / float(w)

            try:
                is_success, buffer = cv2.imencode(cv2_ext, outside_text_image_cv)
                if is_success:
                    image_b64 = base64.b64encode(buffer).decode("utf-8")

                    outside_text_data.append(
                        {
                            "bbox": bbox_tuple,
                            "confidence": conf,
                            "is_outside_text": True,
                            "image_b64": image_b64,
                            "mime_type": mime_type,
                            "is_dark_text": original_text_colors.get(bbox_tuple, True),
                            "aspect_ratio": aspect_ratio,
                            "original_crop_pil": original_crop_pil,
                        }
                    )
            except Exception as e:
                log_message(
                    f"Error encoding outside text bbox {(x1, y1, x2, y2)}: {e}",
                    verbose=verbose,
                )

        return current_image, outside_text_data

    except Exception as e:
        log_message(
            f"Error during outside text detection/inpainting: {e}",
            always_print=True,
        )
        return pil_image, []
