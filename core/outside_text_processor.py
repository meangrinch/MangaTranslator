import base64
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from core.config import MangaTranslatorConfig
from core.image.image_utils import (cv2_to_pil, pil_to_cv2,
                                    process_bubble_image_cached)
from core.image.inpainting import FluxKontextInpainter
from core.image.ocr_detection import OCR_LANGUAGE_MAP, OutsideTextDetector
from core.ml.model_manager import get_model_manager
from core.scaling import scale_length
from utils.logging import log_message


def process_outside_text(
    pil_image: Image.Image,
    config: MangaTranslatorConfig,
    image_path: Union[str, Path],
    image_format: Optional[str],
    processing_scale: float,
    verbose: bool = False,
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

    easyocr_min_size = scale_length(
        config.outside_text.easyocr_min_size,
        processing_scale,
        minimum=32,
        maximum=4096,
    )

    try:
        ocr_language = OCR_LANGUAGE_MAP.get(config.translation.input_language, "ja")

        outside_detector = OutsideTextDetector(
            device=config.device, languages=[ocr_language]
        )
        outside_text_results = outside_detector.detect_outside_text(
            str(image_path),
            yolo_model_path=config.yolo_model_path,
            verbose=verbose,
            min_size=easyocr_min_size,
            image_override=pil_image,
        )

        if not outside_text_results:
            log_message("No outside text regions found", verbose=verbose)
            outside_detector.unload_models()
            return pil_image, []

        log_message(
            f"Found {len(outside_text_results)} outside text regions",
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
            bbox_coords, text, conf = ocr_result
            x_coords = [point[0] for point in bbox_coords]
            y_coords = [point[1] for point in bbox_coords]
            x1, x2 = int(min(x_coords)), int(max(x_coords))
            y1, y2 = int(min(y_coords)), int(max(y_coords))
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

            # The dominant color is assumed to be the text color
            text_color_rgb = centers[dominant_cluster_idx]
            # Use proper luminance calculation (ITU-R BT.601)
            text_brightness = (
                0.299 * text_color_rgb[0]
                + 0.587 * text_color_rgb[1]
                + 0.114 * text_color_rgb[2]
            )
            is_dark_text = text_brightness < 128
            original_text_colors[bbox_tuple] = is_dark_text

            log_message(
                f"OSB bbox {bbox_tuple}: "
                f"{'Dark' if is_dark_text else 'Light'} text detected "
                f"(luminance={text_brightness:.1f})",
                verbose=verbose,
            )

        log_message("Inpainting outside text regions...", verbose=verbose)
        inpainter = FluxKontextInpainter(
            device=config.device,
            huggingface_token=config.outside_text.huggingface_token,
            num_inference_steps=config.outside_text.flux_num_inference_steps,
            residual_diff_threshold=config.outside_text.flux_residual_diff_threshold,
        )

        mask_groups, _ = outside_detector.get_text_masks(
            str(image_path),
            bbox_expansion_percent=config.outside_text.bbox_expansion_percent,
            verbose=verbose,
            min_size=easyocr_min_size,
            image_override=pil_image,
            existing_results=outside_text_results,
        )

        current_image = pil_image
        if mask_groups:
            base_seed = (
                random.randint(1, 999999)
                if config.outside_text.seed == -1
                else config.outside_text.seed
            )

            for i, group in enumerate(mask_groups):
                log_message(
                    f"Inpainting outside text region {i + 1}/{len(mask_groups)}",
                    verbose=verbose,
                )
                combined_mask = group["combined_mask"]
                region_seed = base_seed + i if base_seed > 0 else base_seed
                inpainted_image = inpainter.inpaint_mask(
                    current_image,
                    combined_mask,
                    seed=region_seed,
                    verbose=verbose,
                )
                current_image = inpainted_image

            log_message("Outside text inpainting completed", verbose=verbose)
            log_message(
                f"Inpainted {len(mask_groups)} outside text regions", always_print=True
            )

        outside_text_data = []
        original_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        for ocr_result in outside_text_results:
            bbox_coords, text, conf = ocr_result
            x_coords = [point[0] for point in bbox_coords]
            y_coords = [point[1] for point in bbox_coords]
            x1, x2 = int(min(x_coords)), int(max(x_coords))
            y1, y2 = int(min(y_coords)), int(max(y_coords))
            bbox_tuple = (x1, y1, x2, y2)

            outside_text_image_cv = original_cv_image[y1:y2, x1:x2].copy()

            outside_text_image_pil = cv2_to_pil(outside_text_image_cv)

            if config.translation.upscale_method == "model":
                model_manager = get_model_manager()
                with model_manager.upscale_context() as upscale_model:
                    final_text_pil = process_bubble_image_cached(
                        outside_text_image_pil,
                        upscale_model,
                        config.device,
                        config.translation.osb_min_side_pixels,
                        "min",
                        verbose,
                    )
            elif config.translation.upscale_method == "lanczos":
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

            # Estimate text orientation angle from the EasyOCR quadrilateral
            signed_angle_deg = 0.0
            try:
                pts = [(float(px), float(py)) for px, py in bbox_coords]
                edges = []
                for idx in range(4):
                    xA, yA = pts[idx]
                    xB, yB = pts[(idx + 1) % 4]
                    dx, dy = (xB - xA), (yB - yA)
                    length_sq = dx * dx + dy * dy
                    edges.append((length_sq, dx, dy))
                if edges:
                    edges.sort(reverse=True)
                    _, dx_max, dy_max = edges[0]
                    angle = np.degrees(np.arctan2(dy_max, dx_max))
                    if angle > 90.0:
                        angle -= 180.0
                    elif angle < -90.0:
                        angle += 180.0
                    signed_angle_deg = float(angle)
            except Exception:
                signed_angle_deg = 0.0

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
                            "original_text": text,
                            "confidence": conf,
                            "is_outside_text": True,
                            "image_b64": image_b64,
                            "mime_type": mime_type,
                            "is_dark_text": original_text_colors.get(bbox_tuple, True),
                            "orientation_deg": signed_angle_deg,
                            "aspect_ratio": aspect_ratio,
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
