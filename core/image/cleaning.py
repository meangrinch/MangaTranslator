import gc
import os
import random
import tempfile
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from core.scaling import scale_area, scale_kernel, scale_scalar
from utils.exceptions import CleaningError, ImageProcessingError, ValidationError
from utils.logging import log_message

from .detection import detect_speech_bubbles
from .image_utils import pil_to_cv2
from .inpainting import FluxKontextInpainter

# Cleaning parameters
GRAYSCALE_MIDPOINT = 128  # Threshold for determining black vs white bubbles
MIN_CONTOUR_AREA = 50  # Minimum area threshold for filtering small contours
DILATION_KERNEL_SIZE = (7, 7)  # Kernel size for morphological dilation
EROSION_KERNEL_SIZE = (5, 5)  # Kernel size for morphological erosion
DISTANCE_TRANSFORM_MASK_SIZE = 5  # Mask size for distance transform

# Classification thresholds for colored bubbles
BRIGHT_RATIO_THRESHOLD = 0.50
DARK_RATIO_THRESHOLD = 0.50
BRIGHT_DOM_RATIO_MIN = 0.30
DARK_DOM_RATIO_MIN = 0.30
BRIGHT_DARK_RATIO_MAX = 0.10
DARK_BRIGHT_RATIO_MAX = 0.10


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Ensure mask is uint8 binary (0/255).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def process_single_bubble(
    base_mask,
    img_gray,
    img_height,
    img_width,
    thresholding_value,
    use_otsu_threshold,
    roi_shrink_px,
    verbose,
    detection_bbox=None,
    is_sam=False,
    dilation_kernel=None,
    constraint_erosion_kernel=None,
    min_contour_area: float = MIN_CONTOUR_AREA,
    classify_colored: bool = False,
):
    """
    Process a single speech bubble mask to extract text regions and determine fill color.

    Args:
        base_mask (numpy.ndarray): The base mask (SAM or YOLO) for the bubble
        img_gray (numpy.ndarray): Grayscale image
        img_height (int): Image height
        img_width (int): Image width
        thresholding_value (int): Fixed threshold value for text detection
        use_otsu_threshold (bool): Whether to use Otsu's method for thresholding
        roi_shrink_px (int): Pixels to shrink ROI inwards
        verbose (bool): Whether to print verbose messages
        detection_bbox: Bounding box for logging (optional)
        is_sam (bool): Whether this is a SAM mask (for logging)

    Returns:
        tuple: (final_mask, fill_color_bgr, is_colored, sample_color_bgr, text_bbox)

    Raises:
        CleaningError: If processing fails
    """
    try:
        base_mask = _normalize_mask(base_mask)

        if dilation_kernel is None:
            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, DILATION_KERNEL_SIZE
            )
        if constraint_erosion_kernel is None:
            constraint_erosion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, EROSION_KERNEL_SIZE
            )
        masked_pixels = img_gray[base_mask == 255]
        if masked_pixels.size == 0:
            log_message(
                f"{'[SAM]' if is_sam else ''}Skipping detection {detection_bbox}: empty mask",
                verbose=verbose,
            )
            raise CleaningError(f"Empty mask for detection {detection_bbox}")

        mean_pixel_value = np.mean(masked_pixels)
        is_black_bubble = mean_pixel_value < GRAYSCALE_MIDPOINT
        fill_color_bgr = (0, 0, 0) if is_black_bubble else (255, 255, 255)
        is_colored_bubble = False
        sample_color_bgr: tuple[int, int, int] = fill_color_bgr

        log_message(
            f"{'[SAM]' if is_sam else ''}Detection {detection_bbox}: "
            f"{'Black' if is_black_bubble else 'White'} bubble (mean={mean_pixel_value:.1f})",
            verbose=verbose,
        )

        roi_mask = cv2.dilate(base_mask, dilation_kernel, iterations=1)
        roi_gray = np.zeros_like(img_gray)
        roi_indices = roi_mask == 255
        roi_gray[roi_indices] = img_gray[roi_indices]

        # Invert for black bubbles to detect text properly
        roi_for_thresholding = (
            cv2.bitwise_not(roi_gray) if is_black_bubble else roi_gray
        )
        thresholded_roi = np.zeros_like(img_gray)

        if use_otsu_threshold:
            roi_pixels_for_otsu = roi_for_thresholding[roi_indices]
            thresh_val, _ = cv2.threshold(
                roi_pixels_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            log_message(
                f"{'[SAM]' if is_sam else ''}  Otsu threshold: {thresh_val}",
                verbose=verbose,
            )
            _, thresholded_roi = cv2.threshold(
                roi_for_thresholding, thresh_val, 255, cv2.THRESH_BINARY
            )
        else:
            _, thresholded_roi = cv2.threshold(
                roi_for_thresholding, thresholding_value, 255, cv2.THRESH_BINARY
            )

        thresholded_roi = cv2.bitwise_and(thresholded_roi, roi_mask)

        # Shrink ROI to avoid border artifacts
        dist_map = cv2.distanceTransform(
            roi_mask, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK_SIZE
        )
        shrunk_roi_mask = np.where(dist_map >= float(roi_shrink_px), 255, 0).astype(
            np.uint8
        )
        thresholded_roi = cv2.bitwise_and(thresholded_roi, shrunk_roi_mask)

        # Use eroded mask to avoid erasing bubble outlines
        eroded_constraint_mask = cv2.erode(
            base_mask, constraint_erosion_kernel, iterations=1
        )

        contours, _ = cv2.findContours(
            thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= min_contour_area:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            if (
                0 <= cx < img_width
                and 0 <= cy < img_height
                and eroded_constraint_mask[cy, cx] == 255
            ):
                valid_contours.append(cnt)

        log_message(
            f"{'[SAM]' if is_sam else ''}Detection {detection_bbox}: {len(valid_contours)} text fragments found",
            verbose=verbose,
        )

        text_bbox = None
        if valid_contours:
            validated_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.drawContours(
                validated_mask, valid_contours, -1, 255, thickness=cv2.FILLED
            )

            # Re-contour to get clean boundary from validated mask
            boundary_contours, _ = cv2.findContours(
                validated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if boundary_contours:
                largest_contour = max(boundary_contours, key=cv2.contourArea)
                final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.drawContours(
                    final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED
                )
                x, y, w, h = cv2.boundingRect(largest_contour)
                text_bbox = (x, y, x + w, y + h)

                if classify_colored:
                    # Sample bubble interior excluding text box and outline to determine if colored
                    sampling_mask = cv2.erode(
                        base_mask, constraint_erosion_kernel, iterations=2
                    )
                    if text_bbox:
                        x1, y1, x2, y2 = text_bbox
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_width, x2)
                        y2 = min(img_height, y2)
                        sampling_mask[y1:y2, x1:x2] = 0
                    sample_pixels = img_gray[sampling_mask == 255]
                    if sample_pixels.size == 0:
                        sample_pixels = masked_pixels

                    sample_values = sample_pixels.astype(np.uint8).flatten()
                    hist = np.bincount(sample_values, minlength=256)
                    dominant_val = (
                        int(hist.argmax()) if hist.size > 0 else int(mean_pixel_value)
                    )
                    dominant_count = int(hist.max()) if hist.size > 0 else 0
                    total_count = max(int(sample_values.size), 1)
                    dominant_ratio = dominant_count / float(total_count)
                    bright_ratio = float(
                        np.count_nonzero(sample_values >= 245)
                    ) / float(total_count)
                    dark_ratio = float(np.count_nonzero(sample_values <= 15)) / float(
                        total_count
                    )

                    log_prefix = "[SAM] " if is_sam else ""
                    if bright_ratio >= BRIGHT_RATIO_THRESHOLD or (
                        dominant_val >= 245
                        and dominant_ratio >= BRIGHT_DOM_RATIO_MIN
                        and dark_ratio <= BRIGHT_DARK_RATIO_MAX
                    ):
                        is_colored_bubble = False
                        fill_color_bgr = (255, 255, 255)
                        sample_color_bgr = (255, 255, 255)
                        log_message(
                            f"{log_prefix}Detection {detection_bbox}: white "
                            f"(mode={dominant_val}, dom_ratio={dominant_ratio:.2f}, "
                            f"bright_ratio={bright_ratio:.2f}, dark_ratio={dark_ratio:.2f})",
                            verbose=verbose,
                        )
                    elif dark_ratio >= DARK_RATIO_THRESHOLD or (
                        dominant_val <= 15
                        and dominant_ratio >= DARK_DOM_RATIO_MIN
                        and bright_ratio <= DARK_BRIGHT_RATIO_MAX
                    ):
                        is_colored_bubble = False
                        fill_color_bgr = (0, 0, 0)
                        sample_color_bgr = (0, 0, 0)
                        log_message(
                            f"{log_prefix}Detection {detection_bbox}: black "
                            f"(mode={dominant_val}, dom_ratio={dominant_ratio:.2f}, "
                            f"bright_ratio={bright_ratio:.2f}, dark_ratio={dark_ratio:.2f})",
                            verbose=verbose,
                        )
                    else:
                        is_colored_bubble = True
                        sample_color_bgr = (dominant_val, dominant_val, dominant_val)
                        log_message(
                            f"{log_prefix}Detection {detection_bbox}: "
                            f"colored/gradient (mode={dominant_val}, "
                            f"dom_ratio={dominant_ratio:.2f}, "
                            f"bright_ratio={bright_ratio:.2f}, "
                            f"dark_ratio={dark_ratio:.2f})",
                            verbose=verbose,
                        )
                return (
                    final_mask,
                    fill_color_bgr,
                    is_colored_bubble,
                    sample_color_bgr,
                    text_bbox,
                )

        raise CleaningError("Failed to process bubble mask")

    except Exception as e:
        log_message(
            f"Failed to process {'SAM' if is_sam else 'YOLO'} mask for {detection_bbox}",
            always_print=True,
        )
        raise CleaningError("Failed to process bubble mask") from e


def clean_speech_bubbles(
    image_input: Union[str, Path, Image.Image],
    model_path,
    confidence=0.6,
    pre_computed_detections=None,
    device=None,
    thresholding_value: int = 190,
    use_otsu_threshold: bool = False,
    roi_shrink_px: int = 4,
    verbose: bool = False,
    processing_scale: float = 1.0,
    conjoined_confidence=0.35,
    inpaint_colored_bubbles: bool = False,
    flux_hf_token: str = "",
    flux_num_inference_steps: int = 10,
    flux_residual_diff_threshold: float = 0.15,
    flux_seed: int = 1,
    osb_text_verification: bool = False,
    osb_text_hf_token: str = "",
):
    """
    Clean speech bubbles using YOLO/SAM masks and optional Flux inpainting for colored bubbles.

    Args:
        image_input (str, Path, or PIL.Image.Image): Path to input image or a PIL Image object.
        model_path (str): Path to YOLO model.
        confidence (float): Confidence threshold for detections.
        pre_computed_detections (list, optional): Pre-computed detections from previous call.
        device (torch.device, optional): The device to run detection model on if needed.
        thresholding_value (int): Fixed threshold value for text detection (0-255). Lower values (e.g., 190)
                                 are useful for uncleaned text close to bubble's edges.
        use_otsu_threshold (bool): If True, use Otsu's method for thresholding instead of the fixed value.
        roi_shrink_px (int): Number of pixels to shrink the ROI inwards before identification/fill.
        inpaint_colored_bubbles (bool): If True, detect non-white/black bubbles and inpaint text with Flux.
        flux_hf_token (str): Hugging Face token for Flux downloads (shared with outside-text removal).
        flux_num_inference_steps (int): Flux denoising steps for colored bubble inpainting.
        flux_residual_diff_threshold (float): Flux residual diff threshold for caching.
        flux_seed (int): Seed for Flux; -1 enables random per run.
        osb_text_verification (bool): When True, expand bubble boxes to fully cover OSB text detections.
        osb_text_hf_token (str): Optional token for OSB text model downloads.

    Returns:
        numpy.ndarray: Cleaned image with text removed.
        list[dict]: A list of dictionaries per bubble containing:
                    - 'mask' (np.ndarray): validated text mask (0/255)
                    - 'base_mask' (np.ndarray): normalized detection mask used for processing
                    - 'color' (tuple BGR): sampled bubble color
                    - 'bbox' (tuple): detection bounding box
                    - 'is_colored' (bool): whether bubble interior was classified colored
                    - 'text_bbox' (tuple|None): bounding box of detected text mask
                    - 'is_sam' (bool): whether detection originated from SAM
    Raises:
        ValueError: If the image cannot be loaded or if an image object is passed without pre-computed detections.
        RuntimeError: If model loading or bubble detection fails.
    """
    try:
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input)
            image_path = image_input
        else:
            pil_image = image_input
            image_path = None  # In-memory image has no path

        image = pil_to_cv2(pil_image)
        img_height, img_width = image.shape[:2]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cleaned_image = image.copy()

        if pre_computed_detections is not None:
            detections = pre_computed_detections
        elif image_path is not None:
            detections = detect_speech_bubbles(
                image_path,
                model_path,
                confidence,
                device=device,
                conjoined_confidence=conjoined_confidence,
                osb_text_verification=osb_text_verification,
                osb_text_hf_token=osb_text_hf_token,
            )
        else:
            raise ValidationError(
                "Bubble detection requires an image path, but an image object "
                "was provided without pre-computed detections."
            )

        processed_bubbles = []

        effective_roi_shrink_px = float(
            scale_scalar(
                roi_shrink_px,
                processing_scale,
                minimum=0.0,
                maximum=64.0,
            )
        )
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, scale_kernel(DILATION_KERNEL_SIZE, processing_scale)
        )
        constraint_erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, scale_kernel(EROSION_KERNEL_SIZE, processing_scale)
        )
        min_contour_area = scale_area(
            MIN_CONTOUR_AREA,
            processing_scale,
            minimum=MIN_CONTOUR_AREA,
            maximum=5000,
        )
        for detection in detections:
            final_mask = None
            fill_color_bgr = None
            is_colored_bubble = False
            sample_color_bgr: Optional[tuple[int, int, int]] = None
            text_bbox: Optional[tuple[int, int, int, int]] = None
            base_mask = None
            is_sam_mask = False

            sam_mask = detection.get("sam_mask")
            if sam_mask is not None:
                base_mask = _normalize_mask(sam_mask)
                is_sam_mask = True
                try:
                    (
                        final_mask,
                        fill_color_bgr,
                        is_colored_bubble,
                        sample_color_bgr,
                        text_bbox,
                    ) = process_single_bubble(
                        base_mask,
                        img_gray,
                        img_height,
                        img_width,
                        thresholding_value,
                        use_otsu_threshold,
                        effective_roi_shrink_px,
                        verbose,
                        detection.get("bbox"),
                        is_sam=True,
                        dilation_kernel=dilation_kernel,
                        constraint_erosion_kernel=constraint_erosion_kernel,
                        min_contour_area=min_contour_area,
                        classify_colored=inpaint_colored_bubbles,
                    )
                except Exception as e:
                    retry_success = False
                    if not use_otsu_threshold and base_mask is not None:
                        log_message(
                            f"Standard cleaning failed for {detection.get('bbox')}, retrying with Otsu...",
                            verbose=verbose,
                        )
                        retry_res = retry_cleaning_with_otsu(
                            image,
                            {
                                "base_mask": base_mask,
                                "bbox": detection.get("bbox"),
                                "is_sam": True,
                            },
                            thresholding_value,
                            roi_shrink_px,
                            processing_scale,
                            verbose,
                            inpaint_colored_bubbles,
                        )
                        if retry_res:
                            final_mask = retry_res["mask"]
                            fill_color_bgr = retry_res["color"]
                            sample_color_bgr = retry_res["color"]
                            is_colored_bubble = retry_res["is_colored"]
                            text_bbox = retry_res["text_bbox"]
                            retry_success = True
                            log_message(
                                f"Otsu retry successful for {detection.get('bbox')}",
                                verbose=verbose,
                            )
                        else:
                            log_message(
                                f"Otsu retry failed for {detection.get('bbox')}",
                                verbose=verbose,
                            )

                    if not retry_success:
                        error_msg = f"Error processing SAM mask for detection {detection.get('bbox')}: {e}"
                        log_message(error_msg, always_print=True)
                        continue
            else:
                if "mask_points" not in detection or not detection["mask_points"]:
                    log_message(
                        f"Skipping detection {detection.get('bbox')}: no mask points",
                        verbose=verbose,
                    )
                    continue

                try:
                    points_list = detection["mask_points"]
                    points = np.array(points_list, dtype=np.float32)

                    if len(points.shape) == 3 and points.shape[1] == 1:
                        points_int = np.round(points).astype(int)
                    elif len(points.shape) == 2 and points.shape[1] == 2:
                        points_int = np.round(points).astype(int).reshape((-1, 1, 2))
                    else:
                        log_message(
                            f"Skipping detection {detection.get('bbox')}: invalid mask format",
                            verbose=verbose,
                        )
                        continue

                    yolo_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    cv2.fillPoly(yolo_mask, [points_int], 255)
                    base_mask = _normalize_mask(yolo_mask)

                    (
                        final_mask,
                        fill_color_bgr,
                        is_colored_bubble,
                        sample_color_bgr,
                        text_bbox,
                    ) = process_single_bubble(
                        base_mask,
                        img_gray,
                        img_height,
                        img_width,
                        thresholding_value,
                        use_otsu_threshold,
                        effective_roi_shrink_px,
                        verbose,
                        detection.get("bbox"),
                        is_sam=False,
                        dilation_kernel=dilation_kernel,
                        constraint_erosion_kernel=constraint_erosion_kernel,
                        min_contour_area=min_contour_area,
                        classify_colored=inpaint_colored_bubbles,
                    )

                except Exception as e:
                    retry_success = False
                    if not use_otsu_threshold and base_mask is not None:
                        log_message(
                            f"Standard cleaning failed for {detection.get('bbox')}, retrying with Otsu...",
                            verbose=verbose,
                        )
                        retry_res = retry_cleaning_with_otsu(
                            image,
                            {
                                "base_mask": base_mask,
                                "bbox": detection.get("bbox"),
                                "is_sam": False,
                            },
                            thresholding_value,
                            roi_shrink_px,
                            processing_scale,
                            verbose,
                            inpaint_colored_bubbles,
                        )
                        if retry_res:
                            final_mask = retry_res["mask"]
                            fill_color_bgr = retry_res["color"]
                            sample_color_bgr = retry_res["color"]
                            is_colored_bubble = retry_res["is_colored"]
                            text_bbox = retry_res["text_bbox"]
                            retry_success = True
                            log_message(
                                f"Otsu retry successful for {detection.get('bbox')}",
                                verbose=verbose,
                            )
                        else:
                            log_message(
                                f"Otsu retry failed for {detection.get('bbox')}",
                                verbose=verbose,
                            )

                    if not retry_success:
                        error_msg = f"Error processing YOLO mask for detection {detection.get('bbox')}: {e}"
                        log_message(error_msg, always_print=True)
                        continue

            if final_mask is not None and fill_color_bgr is not None:
                processed_bubbles.append(
                    {
                        "mask": final_mask,
                        "base_mask": base_mask,
                        "color": (
                            sample_color_bgr if sample_color_bgr else fill_color_bgr
                        ),
                        "bbox": detection.get("bbox"),
                        "is_colored": is_colored_bubble,
                        "text_bbox": text_bbox,
                        "is_sam": is_sam_mask,
                        "inpainted": False,
                    }
                )
                log_message(
                    f"Detection {detection.get('bbox')}: processed successfully",
                    verbose=verbose,
                )

        # Optional Flux inpainting for colored bubbles (text-only mask)
        if inpaint_colored_bubbles:
            colored_bubbles = [
                b for b in processed_bubbles if b.get("is_colored", False)
            ]
            if colored_bubbles and flux_hf_token:
                log_message(
                    f"Inpainting {len(colored_bubbles)} colored bubbles with Flux",
                    always_print=True,
                )
                pil_working = Image.fromarray(
                    cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
                )
                base_seed = (
                    random.randint(1, 999999)
                    if flux_seed == -1
                    else max(0, int(flux_seed))
                )
                temp_files = []
                try:
                    inpainter = FluxKontextInpainter(
                        device=device,
                        huggingface_token=flux_hf_token,
                        num_inference_steps=int(flux_num_inference_steps),
                        residual_diff_threshold=float(flux_residual_diff_threshold),
                    )
                    for idx, bubble_info in enumerate(colored_bubbles):
                        mask_np = bubble_info["mask"]
                        mask_bool = mask_np.astype(bool)
                        region_seed = base_seed + idx if base_seed > 0 else base_seed
                        bbox_tuple = bubble_info.get("bbox")
                        ocr_params = {"type": "colored_bubble", "bbox": bbox_tuple}
                        try:
                            pil_working = inpainter.inpaint_mask(
                                pil_working,
                                mask_bool,
                                seed=region_seed,
                                verbose=verbose,
                                ocr_params=ocr_params,
                            )
                            bubble_info["inpainted"] = True
                            # Re-sample background brightness after inpaint for accurate text contrast
                            cv_after = cv2.cvtColor(
                                np.array(pil_working.convert("RGB")), cv2.COLOR_RGB2BGR
                            )
                            masked_after = cv_after[mask_bool]
                            if masked_after.size > 0:
                                mean_val = int(np.clip(np.mean(masked_after), 0, 255))
                                bubble_info["color"] = (mean_val, mean_val, mean_val)
                        except Exception as e:
                            log_message(
                                f"Flux inpainting failed for bubble {bbox_tuple}: {e}; falling back to standard fill",
                                always_print=True,
                            )
                            continue

                        # Save intermediate result to disk to free memory when multiple regions
                        if idx < len(colored_bubbles) - 1:
                            temp_file = None
                            try:
                                temp_fd, temp_file = tempfile.mkstemp(suffix=".png")
                                os.close(temp_fd)
                                pil_working.save(temp_file, format="PNG")
                                log_message(
                                    "Saved intermediate inpainting result to disk",
                                    verbose=verbose,
                                )
                                temp_files.append(temp_file)
                                with Image.open(temp_file) as img_tmp:
                                    img_tmp.load()
                                    pil_working = img_tmp.copy()
                                gc.collect()
                            except Exception as e:
                                log_message(
                                    f"Warning: Failed to save intermediate inpainting result: {e}",
                                    verbose=verbose,
                                )
                                if temp_file and temp_file in temp_files:
                                    temp_files.remove(temp_file)
                                # fall through with in-memory image

                    cleaned_image = cv2.cvtColor(
                        np.array(pil_working.convert("RGB")), cv2.COLOR_RGB2BGR
                    )
                except Exception as e:
                    log_message(
                        f"Flux inpainting aborted; falling back to standard fill: {e}",
                        always_print=True,
                    )
                finally:
                    for temp_file in temp_files:
                        if temp_file and os.path.exists(temp_file):
                            try:
                                os.remove(temp_file)
                            except Exception:
                                pass
            elif colored_bubbles:
                log_message(
                    "Colored bubbles detected but Flux inpainting skipped (missing "
                    "Hugging Face token); falling back to standard fill",
                    always_print=True,
                )

        # Group masks by color for efficient batch processing (skip already inpainted regions)
        if processed_bubbles:
            color_groups = {}
            for bubble_info in processed_bubbles:
                if bubble_info.get("inpainted", False):
                    continue
                color_key = bubble_info["color"]
                if color_key not in color_groups:
                    color_groups[color_key] = []
                color_groups[color_key].append(bubble_info["mask"])

            for color_bgr, masks in color_groups.items():
                combined_mask = np.bitwise_or.reduce(masks)

                if cleaned_image.shape[2] == 4:
                    cleaned_image[combined_mask == 255, :3] = (
                        color_bgr  # Preserve alpha channel
                    )
                else:
                    cleaned_image[combined_mask == 255] = color_bgr

        log_message(
            f"Cleaned {len(processed_bubbles)} speech bubbles", always_print=True
        )
        return cleaned_image, processed_bubbles
    except IOError as e:
        raise ImageProcessingError(f"Error loading image {image_input}: {str(e)}")
    except Exception as e:
        raise CleaningError(f"Error cleaning speech bubbles: {str(e)}")


def retry_cleaning_with_otsu(
    image_bgr: np.ndarray,
    bubble_info: dict,
    thresholding_value: int,
    roi_shrink_px: int,
    processing_scale: float = 1.0,
    verbose: bool = False,
    classify_colored: bool = False,
) -> Optional[dict]:
    """
    Retry cleaning for a single bubble using Otsu thresholding.

    Returns a bubble-info dict compatible with clean_speech_bubbles output,
    or None if retry fails.
    """
    base_mask = bubble_info.get("base_mask")
    if base_mask is None:
        log_message(
            f"Otsu retry skipped for {bubble_info.get('bbox')}: missing base_mask",
            verbose=verbose,
        )
        return None

    try:
        if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 4:
            img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY)
        else:
            img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        log_message(
            f"Otsu retry failed to convert image to grayscale: {e}",
            always_print=True,
        )
        return None

    img_height, img_width = img_gray.shape[:2]

    effective_roi_shrink_px = float(
        scale_scalar(
            roi_shrink_px,
            processing_scale,
            minimum=0.0,
            maximum=64.0,
        )
    )
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, scale_kernel(DILATION_KERNEL_SIZE, processing_scale)
    )
    constraint_erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, scale_kernel(EROSION_KERNEL_SIZE, processing_scale)
    )
    min_contour_area = scale_area(
        MIN_CONTOUR_AREA,
        processing_scale,
        minimum=MIN_CONTOUR_AREA,
        maximum=5000,
    )

    try:
        result = process_single_bubble(
            base_mask,
            img_gray,
            img_height,
            img_width,
            thresholding_value,
            True,  # force Otsu
            effective_roi_shrink_px,
            verbose,
            bubble_info.get("bbox"),
            bubble_info.get("is_sam", False),
            dilation_kernel=dilation_kernel,
            constraint_erosion_kernel=constraint_erosion_kernel,
            min_contour_area=min_contour_area,
            classify_colored=classify_colored,
        )
    except CleaningError as e:
        log_message(
            f"Otsu retry cleaning failed for {bubble_info.get('bbox')}: {e}",
            always_print=True,
        )
        return None
    except Exception as e:
        log_message(
            f"Otsu retry cleaning unexpected error for {bubble_info.get('bbox')}: {e}",
            always_print=True,
        )
        return None

    if not result:
        return None

    (
        final_mask,
        fill_color_bgr,
        is_colored_bubble,
        sample_color_bgr,
        text_bbox,
    ) = result

    bubble_color = sample_color_bgr if sample_color_bgr else fill_color_bgr

    log_message(
        f"Otsu retry succeeded for {bubble_info.get('bbox')}",
        verbose=verbose,
    )

    return {
        "mask": final_mask,
        "base_mask": _normalize_mask(base_mask),
        "color": bubble_color,
        "bbox": bubble_info.get("bbox"),
        "is_colored": is_colored_bubble,
        "text_bbox": text_bbox,
        "is_sam": bubble_info.get("is_sam", False),
    }
