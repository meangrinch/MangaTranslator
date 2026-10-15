import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from core.batch_coordinator import (
    expanded_mask_bbox,
    partition_non_overlapping_waves,
    paste_image_region,
)
from core.scaling import scale_area, scale_kernel, scale_scalar
from utils.cancellation import raise_if_cancelled
from utils.exceptions import (
    CancellationError,
    CleaningError,
    ImageProcessingError,
    ValidationError,
)
from utils.logging import log_message

from .detection import detect_speech_bubbles
from .image_utils import pil_to_cv2
from .inpainting import make_flux_inpainter

# Cleaning parameters
GRAYSCALE_MIDPOINT = 128  # Threshold for determining black vs white bubbles
MIN_CONTOUR_AREA = 50  # Minimum area threshold for filtering small contours
DILATION_KERNEL_SIZE = (7, 7)  # Kernel size for morphological dilation
EROSION_KERNEL_SIZE = (5, 5)  # Kernel size for morphological erosion
DISTANCE_TRANSFORM_MASK_SIZE = 5  # Mask size for distance transform

# Classification threshold for solid background bubbles (interior sampling)
# Lower threshold (0.65) accounts for residual text mask contours and erosion artifacts
SOLID_RATIO_THRESHOLD = 0.65


# Adaptive shrink defaults for conjoined junction zones (at 1.0x scale)
JUNCTION_ADJACENCY_MARGIN = 10  # Px margin around bbox intersection for junction zone
JUNCTION_MIN_SHRINK = 1.0  # Minimal shrink applied inside junction zones


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Ensure mask is uint8 binary (0/255).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _resample_inpainted_bubble_color(
    pil_working: Image.Image,
    bubble_info: dict[str, Any],
    mask_bool: np.ndarray,
) -> None:
    cv_after = cv2.cvtColor(np.array(pil_working.convert("RGB")), cv2.COLOR_RGB2BGR)
    masked_after = cv_after[mask_bool]
    if masked_after.size > 0:
        mean_val = int(np.clip(np.mean(masked_after), 0, 255))
        bubble_info["color"] = (mean_val, mean_val, mean_val)


def _inpaint_colored_bubbles_with_coordinator(
    pil_working: Image.Image,
    colored_bubbles: list[dict[str, Any]],
    inpainter: Any,
    base_seed: int,
    verbose: bool,
    request_coordinator: Any,
) -> Image.Image:
    candidates = []
    for idx, bubble_info in enumerate(colored_bubbles):
        mask_np = bubble_info["mask"]
        mask_bool = mask_np.astype(bool)
        context_bbox = expanded_mask_bbox(mask_bool, pil_working.size)
        candidates.append(
            {
                "index": idx,
                "bubble_info": bubble_info,
                "mask": mask_bool,
                "seed": base_seed + idx if base_seed > 0 else base_seed,
                "bbox_tuple": bubble_info.get("bbox"),
                "context_bbox": context_bbox,
            }
        )

    waves = partition_non_overlapping_waves(
        candidates,
        lambda candidate: candidate["context_bbox"],
    )
    log_message(
        f"Scheduling colored-bubble Flux in {len(waves)} wave(s)",
        verbose=verbose,
    )

    for wave in waves:
        base_image = pil_working

        def make_job(candidate, base_img=base_image):
            def job():
                image_for_job = base_img.copy()
                try:
                    result_image = inpainter.inpaint_mask(
                        image_for_job,
                        candidate["mask"],
                        seed=candidate["seed"],
                        verbose=verbose,
                        ocr_params={
                            "type": "colored_bubble",
                            "bbox": candidate["bbox_tuple"],
                        },
                    )
                    return {
                        "candidate": candidate,
                        "image": result_image,
                        "error": None,
                    }
                except CancellationError:
                    raise
                except Exception as e:
                    return {
                        "candidate": candidate,
                        "image": None,
                        "error": e,
                    }

            return job

        results = request_coordinator.map_ordered([make_job(c) for c in wave])
        for result in results:
            candidate = result["candidate"]
            bubble_info = candidate["bubble_info"]
            if result["error"] is not None:
                log_message(
                    f"Flux inpainting failed for bubble {candidate['bbox_tuple']}: "
                    f"{result['error']}; falling back to standard fill",
                    always_print=True,
                )
                continue

            result_image = result["image"]
            context_bbox = candidate["context_bbox"]
            if context_bbox is None:
                pil_working = result_image
            else:
                pil_working = paste_image_region(
                    pil_working, result_image, context_bbox
                )
            bubble_info["inpainted"] = True
            _resample_inpainted_bubble_color(
                pil_working, bubble_info, candidate["mask"]
            )

    return pil_working


def _polygon_mask(
    detection: dict[str, Any], img_height: int, img_width: int, verbose: bool
) -> np.ndarray | None:
    """Rasterize a YOLO detection's polygon, or None (logged) when it has no usable one."""
    bbox = detection.get("bbox")
    points_list = detection.get("mask_points")
    if points_list is None or len(points_list) == 0:
        log_message(f"Skipping detection {bbox}: no mask points", verbose=verbose)
        return None
    try:
        points = np.array(points_list, dtype=np.float32)
        if points.ndim == 3 and points.shape[1] == 1:
            points_int = np.round(points).astype(int)
        elif points.ndim == 2 and points.shape[1] == 2:
            points_int = np.round(points).astype(int).reshape((-1, 1, 2))
        else:
            log_message(
                f"Skipping detection {bbox}: invalid mask format", verbose=verbose
            )
            return None
        yolo_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(yolo_mask, [points_int], 255)
        return _normalize_mask(yolo_mask)
    except Exception as e:
        log_message(
            f"Error processing YOLO mask for detection {bbox}: {e}", always_print=True
        )
        return None


def _shift_bbox(bbox, dx: int, dy: int) -> tuple:
    x1, y1, x2, y2 = bbox
    return (x1 - dx, y1 - dy, x2 - dx, y2 - dy)


def _build_adaptive_shrink_mask(
    roi_mask: np.ndarray,
    roi_shrink_px: float,
    detection_bbox: tuple,
    neighbor_bboxes: list,
    processing_scale: float = 1.0,
) -> np.ndarray:
    """
    Shrink ROI uniformly except near conjoined neighbor edges, where only
    JUNCTION_MIN_SHRINK is applied.  Without this, the distance-transform
    shrink pinches off narrow junction passages and leaves remnant text.
    """
    adjacency_margin = max(1, round(JUNCTION_ADJACENCY_MARGIN * processing_scale))

    dist_map = cv2.distanceTransform(
        roi_mask, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK_SIZE
    )
    shrunk = np.where(dist_map >= roi_shrink_px, 255, 0).astype(np.uint8)

    x1, y1, x2, y2 = detection_bbox
    h, w = roi_mask.shape[:2]

    for other_bbox in neighbor_bboxes:
        ox1, oy1, ox2, oy2 = other_bbox

        if (
            x1 - adjacency_margin > ox2
            or ox1 - adjacency_margin > x2
            or y1 - adjacency_margin > oy2
            or oy1 - adjacency_margin > y2
        ):
            continue

        zone_x1 = max(0, max(x1, ox1) - adjacency_margin)
        zone_y1 = max(0, max(y1, oy1) - adjacency_margin)
        zone_x2 = min(w, min(x2, ox2) + adjacency_margin)
        zone_y2 = min(h, min(y2, oy2) + adjacency_margin)

        if zone_x2 <= zone_x1 or zone_y2 <= zone_y1:
            continue

        junction_slice = dist_map[zone_y1:zone_y2, zone_x1:zone_x2]
        # Not scaled: ROI edge pixels sit at distance exactly 1.0, so a scaled
        # threshold drops that ring and strands text straddling the split line
        restored = np.where(junction_slice >= JUNCTION_MIN_SHRINK, 255, 0).astype(
            np.uint8
        )
        np.bitwise_or(
            shrunk[zone_y1:zone_y2, zone_x1:zone_x2],
            restored,
            out=shrunk[zone_y1:zone_y2, zone_x1:zone_x2],
        )

    return shrunk


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
    neighbor_bboxes: list | None = None,
    processing_scale: float = 1.0,
    image_bgr: np.ndarray | None = None,
    auto_threshold: bool = False,
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
        dilation_kernel: Kernel for dilation
        constraint_erosion_kernel: Kernel for erosion
        min_contour_area: Min area for contours
        classify_colored: Whether to classify colored bubbles
        neighbor_bboxes: Neighboring bboxes for shrink logic
        processing_scale: Processing scale multiplier
        image_bgr (numpy.ndarray): BGR image for text color sampling
        auto_threshold (bool): Scale thresholding_value by the bubble's background level

    Returns:
        tuple: (final_mask, fill_color_bgr, is_colored, sample_color_bgr, text_bbox, text_color_bgr)

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

        # Work on a crop around the bubble instead of the whole page. The margin covers
        # the dilation, the double erosion and a zero ring for the distance transform,
        # so the results match full-page processing exactly.
        bx, by, bw, bh = cv2.boundingRect(base_mask)
        margin = (
            max(dilation_kernel.shape) // 2
            + 2 * (max(constraint_erosion_kernel.shape) // 2)
            + 2
        )
        ox, oy = max(0, bx - margin), max(0, by - margin)
        crop = (
            slice(oy, min(img_height, by + bh + margin)),
            slice(ox, min(img_width, bx + bw + margin)),
        )
        base_mask = base_mask[crop]
        img_gray = img_gray[crop]
        if image_bgr is not None:
            image_bgr = image_bgr[crop]
        crop_height, crop_width = base_mask.shape

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
            threshold = thresholding_value
            if auto_threshold:
                # thresholding_value is calibrated for a pure-white bubble; rescale it
                # so off-white or gray paper is thresholded at the same relative level
                interior = cv2.erode(base_mask, constraint_erosion_kernel, iterations=2)
                bg_pixels = roi_for_thresholding[interior == 255]
                if bg_pixels.size == 0:
                    bg_pixels = roi_for_thresholding[base_mask == 255]
                bg_level = float(np.percentile(bg_pixels, 75))
                threshold = round(thresholding_value * bg_level / 255)
                log_message(
                    f"{'[SAM]' if is_sam else ''}  Auto threshold: {threshold} "
                    f"(background={bg_level:.0f})",
                    verbose=verbose,
                )
            _, thresholded_roi = cv2.threshold(
                roi_for_thresholding, threshold, 255, cv2.THRESH_BINARY
            )

        thresholded_roi = cv2.bitwise_and(thresholded_roi, roi_mask)

        if neighbor_bboxes and detection_bbox is not None:
            shrunk_roi_mask = _build_adaptive_shrink_mask(
                roi_mask,
                float(roi_shrink_px),
                _shift_bbox(detection_bbox, ox, oy),
                [_shift_bbox(b, ox, oy) for b in neighbor_bboxes],
                processing_scale=processing_scale,
            )
        else:
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
                0 <= cx < crop_width
                and 0 <= cy < crop_height
                and eroded_constraint_mask[cy, cx] == 255
            ):
                valid_contours.append(cnt)

        log_message(
            f"{'[SAM]' if is_sam else ''}Detection {detection_bbox}: {len(valid_contours)} text fragments found",
            verbose=verbose,
        )

        text_bbox = None
        if valid_contours:
            validated_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
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
                    final_mask,
                    [largest_contour],
                    -1,
                    255,
                    thickness=cv2.FILLED,
                    offset=(ox, oy),
                )
                x, y, w, h = cv2.boundingRect(largest_contour)
                text_bbox = (x + ox, y + oy, x + ox + w, y + oy + h)

                if classify_colored:
                    # Sample bubble interior excluding exact text pixels and outline to determine if colored
                    sampling_mask = cv2.erode(
                        base_mask, constraint_erosion_kernel, iterations=2
                    )
                    text_mask = cv2.bitwise_and(
                        cv2.bitwise_not(thresholded_roi), shrunk_roi_mask
                    )
                    text_mask_dilated = cv2.dilate(
                        text_mask, np.ones((3, 3), np.uint8), iterations=1
                    )
                    sampling_mask[text_mask_dilated == 255] = 0

                    if image_bgr is not None:
                        sample_pixels_bgr = image_bgr[sampling_mask == 255]
                        if sample_pixels_bgr.size == 0:
                            sample_pixels_bgr = image_bgr[base_mask == 255]

                        if sample_pixels_bgr.size > 0:
                            med_bgr = np.median(sample_pixels_bgr, axis=0).astype(int)
                            diffs = np.max(
                                np.abs(sample_pixels_bgr.astype(int) - med_bgr), axis=1
                            )
                            solid_ratio = float(np.count_nonzero(diffs <= 15)) / float(
                                len(sample_pixels_bgr)
                            )

                            if (
                                med_bgr[0] >= 245
                                and med_bgr[1] >= 245
                                and med_bgr[2] >= 245
                            ):
                                fill_color_bgr = (255, 255, 255)
                            elif (
                                med_bgr[0] <= 10
                                and med_bgr[1] <= 10
                                and med_bgr[2] <= 10
                            ):
                                fill_color_bgr = (0, 0, 0)
                            else:
                                fill_color_bgr = (
                                    int(med_bgr[0]),
                                    int(med_bgr[1]),
                                    int(med_bgr[2]),
                                )
                        else:
                            fill_color_bgr = (255, 255, 255)
                            solid_ratio = 0.0
                    else:
                        sample_pixels = img_gray[sampling_mask == 255]
                        if sample_pixels.size == 0:
                            sample_pixels = masked_pixels
                        sample_values = sample_pixels.astype(np.uint8).flatten()
                        med_val = (
                            int(np.median(sample_values))
                            if sample_values.size > 0
                            else int(mean_pixel_value)
                        )
                        diffs = np.abs(sample_values.astype(int) - med_val)
                        solid_ratio = float(np.count_nonzero(diffs <= 15)) / float(
                            max(len(sample_values), 1)
                        )
                        if med_val >= 245:
                            fill_color_bgr = (255, 255, 255)
                        elif med_val <= 10:
                            fill_color_bgr = (0, 0, 0)
                        else:
                            fill_color_bgr = (med_val, med_val, med_val)

                    log_prefix = "[SAM] " if is_sam else ""
                    if solid_ratio >= SOLID_RATIO_THRESHOLD:
                        is_colored_bubble = False
                        sample_color_bgr = fill_color_bgr
                        log_message(
                            f"{log_prefix}Detection {detection_bbox}: solid color {fill_color_bgr} "
                            f"(solid_ratio={solid_ratio:.2f})",
                            verbose=verbose,
                        )
                    else:
                        is_colored_bubble = True
                        sample_color_bgr = fill_color_bgr
                        log_message(
                            f"{log_prefix}Detection {detection_bbox}: non-solid/gradient "
                            f"(solid_ratio={solid_ratio:.2f})",
                            verbose=verbose,
                        )

                text_color_bgr = None
                if image_bgr is not None:
                    text_mask = cv2.bitwise_and(
                        cv2.bitwise_not(thresholded_roi), shrunk_roi_mask
                    )
                    sample_mask = cv2.erode(
                        text_mask, np.ones((3, 3), np.uint8), iterations=1
                    )
                    text_pixels_bgr = image_bgr[sample_mask == 255]
                    # Fallback if erosion obliterates thin text
                    if text_pixels_bgr.size == 0:
                        text_pixels_bgr = image_bgr[text_mask == 255]

                    if text_pixels_bgr.size > 0:
                        sampled_bgr = tuple(
                            np.median(text_pixels_bgr, axis=0).astype(int)
                        )
                        hsv = cv2.cvtColor(
                            np.uint8([[sampled_bgr]]), cv2.COLOR_BGR2HSV
                        )[0][0]
                        if hsv[1] < 25:
                            b, g, r = (
                                fill_color_bgr[0],
                                fill_color_bgr[1],
                                fill_color_bgr[2],
                            )
                            luminance = 0.114 * b + 0.587 * g + 0.299 * r
                            text_color_bgr = (
                                (0, 0, 0) if luminance >= 128 else (255, 255, 255)
                            )
                        else:
                            text_color_bgr = sampled_bgr

                return (
                    final_mask,
                    fill_color_bgr,
                    is_colored_bubble,
                    sample_color_bgr,
                    text_bbox,
                    text_color_bgr,
                )

        raise CleaningError("Failed to process bubble mask")

    except Exception as e:
        log_message(
            f"Failed to process {'SAM' if is_sam else 'YOLO'} mask for {detection_bbox}",
            always_print=True,
        )
        raise CleaningError("Failed to process bubble mask") from e


def clean_speech_bubbles(
    image_input: str | Path | Image.Image,
    model_path,
    confidence=0.6,
    pre_computed_detections=None,
    device=None,
    thresholding_value: int = 200,
    use_otsu_threshold: bool = False,
    roi_shrink_px: int = 5,
    verbose: bool = False,
    processing_scale: float = 1.0,
    conjoined_confidence=0.35,
    auto_threshold: bool = False,
    inpaint_colored_bubbles: bool = False,
    flux_hf_token: str = "",
    flux_num_inference_steps: int = 4,
    flux_residual_diff_threshold: float = 0.15,
    flux_seed: int = 1,
    osb_text_verification: bool = False,
    osb_text_hf_token: str = "",
    inpaint_method: str = "flux_kontext",
    flux_backend: str = "sdnq",
    flux_low_vram: bool = True,
    flux_sdcpp_cache_mode: str = "none",
    flux_sdcpp_diffusion_quant: str = "Q4_K_M",
    flux_sdcpp_text_encoder_quant: str = "",
    flux_luminance_correction: bool = True,
    flux_upscale_small_crops: bool = True,
    bubble_detector_model: str = "yolo_2",
    request_coordinator: Any | None = None,
):
    """
    Clean speech bubbles using YOLO/SAM masks and optional Flux inpainting for colored bubbles.

    Args:
        image_input (str, Path, or PIL.Image.Image): Path to input image or a PIL Image object.
        model_path (str): Path to YOLO model.
        confidence (float): Confidence threshold for detections.
        pre_computed_detections (list, optional): Pre-computed detections from previous call.
        device (torch.device, optional): The device to run detection model on if needed.
        thresholding_value (int): Fixed threshold value for text detection (0-255). Lower values (e.g., 180)
                                 are useful for uncleaned text close to bubble's edges.
        use_otsu_threshold (bool): If True, use Otsu's method for thresholding instead of the fixed value.
        auto_threshold (bool): If True, scale thresholding_value by each bubble's background level
                               (ignored when use_otsu_threshold is set).
        roi_shrink_px (int): Number of pixels to shrink the ROI inwards before identification/fill.
        inpaint_colored_bubbles (bool): If True, detect non-solid background bubbles and inpaint text with Flux.
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
            detection_result = detect_speech_bubbles(
                image_path,
                model_path,
                confidence,
                device=device,
                conjoined_confidence=conjoined_confidence,
                osb_text_verification=osb_text_verification,
                osb_text_hf_token=osb_text_hf_token,
                bubble_detector_model=bubble_detector_model,
            )
            detections = (
                detection_result[0]
                if isinstance(detection_result, tuple)
                else detection_result
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
            raise_if_cancelled()
            bbox = detection.get("bbox")
            sam_mask = detection.get("sam_mask")
            is_sam_mask = sam_mask is not None
            if is_sam_mask:
                base_mask = _normalize_mask(sam_mask)
            else:
                base_mask = _polygon_mask(detection, img_height, img_width, verbose)
                if base_mask is None:
                    continue

            try:
                (
                    final_mask,
                    fill_color_bgr,
                    is_colored_bubble,
                    sample_color_bgr,
                    text_bbox,
                    text_color_bgr,
                ) = process_single_bubble(
                    base_mask,
                    img_gray,
                    img_height,
                    img_width,
                    thresholding_value,
                    use_otsu_threshold,
                    effective_roi_shrink_px,
                    verbose,
                    bbox,
                    is_sam=is_sam_mask,
                    dilation_kernel=dilation_kernel,
                    constraint_erosion_kernel=constraint_erosion_kernel,
                    min_contour_area=min_contour_area,
                    classify_colored=inpaint_colored_bubbles,
                    neighbor_bboxes=detection.get("conjoined_neighbor_bboxes"),
                    processing_scale=processing_scale,
                    image_bgr=image,
                    auto_threshold=auto_threshold,
                )
            except Exception as e:
                retry_res = None
                if not use_otsu_threshold:
                    log_message(
                        f"Standard cleaning failed for {bbox}, retrying with Otsu...",
                        verbose=verbose,
                    )
                    retry_res = retry_cleaning_with_otsu(
                        image,
                        {
                            "base_mask": base_mask,
                            "bbox": bbox,
                            "is_sam": is_sam_mask,
                            "neighbor_bboxes": detection.get(
                                "conjoined_neighbor_bboxes"
                            ),
                        },
                        thresholding_value,
                        roi_shrink_px,
                        processing_scale,
                        verbose,
                        inpaint_colored_bubbles,
                    )
                    log_message(
                        f"Otsu retry {'successful' if retry_res else 'failed'} for {bbox}",
                        verbose=verbose,
                    )
                if not retry_res:
                    log_message(
                        f"Error processing {'SAM' if is_sam_mask else 'YOLO'} mask "
                        f"for detection {bbox}: {e}",
                        always_print=True,
                    )
                    continue
                final_mask = retry_res["mask"]
                fill_color_bgr = sample_color_bgr = retry_res["color"]
                is_colored_bubble = retry_res["is_colored"]
                text_bbox = retry_res["text_bbox"]
                text_color_bgr = retry_res.get("text_color_bgr")

            processed_bubbles.append(
                {
                    "mask": final_mask,
                    "base_mask": base_mask,
                    "color": sample_color_bgr if sample_color_bgr else fill_color_bgr,
                    "bbox": bbox,
                    "is_colored": is_colored_bubble,
                    "text_bbox": text_bbox,
                    "text_color_bgr": text_color_bgr,
                    "is_sam": is_sam_mask,
                    "inpainted": False,
                }
            )
            log_message(f"Detection {bbox}: processed successfully", verbose=verbose)

        # Optional Flux inpainting for colored bubbles (text-only mask)
        if inpaint_colored_bubbles and inpaint_method not in ("opencv", "none"):
            colored_bubbles = [
                b for b in processed_bubbles if b.get("is_colored", False)
            ]
            if colored_bubbles:
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
                try:
                    inpainter = make_flux_inpainter(
                        inpaint_method,
                        device=device,
                        huggingface_token=flux_hf_token,
                        num_inference_steps=flux_num_inference_steps,
                        backend=flux_backend,
                        low_vram=flux_low_vram,
                        residual_diff_threshold=flux_residual_diff_threshold,
                        luminance_correction=flux_luminance_correction,
                        upscale_small_crops=flux_upscale_small_crops,
                        sdcpp_cache_mode=flux_sdcpp_cache_mode,
                        sdcpp_diffusion_quant=flux_sdcpp_diffusion_quant,
                        sdcpp_text_encoder_quant=flux_sdcpp_text_encoder_quant,
                        verbose=verbose,
                    )
                    if request_coordinator is not None and len(colored_bubbles) > 1:
                        pil_working = _inpaint_colored_bubbles_with_coordinator(
                            pil_working,
                            colored_bubbles,
                            inpainter,
                            base_seed,
                            verbose,
                            request_coordinator,
                        )
                    else:
                        for idx, bubble_info in enumerate(colored_bubbles):
                            raise_if_cancelled()
                            mask_np = bubble_info["mask"]
                            mask_bool = mask_np.astype(bool)
                            region_seed = (
                                base_seed + idx if base_seed > 0 else base_seed
                            )
                            bbox_tuple = bubble_info.get("bbox")
                            ocr_params = {
                                "type": "colored_bubble",
                                "bbox": bbox_tuple,
                            }
                            try:
                                inpaint_call = inpainter.inpaint_mask
                                if request_coordinator is not None:
                                    pil_working = request_coordinator.run(
                                        inpaint_call,
                                        pil_working,
                                        mask_bool,
                                        seed=region_seed,
                                        verbose=verbose,
                                        ocr_params=ocr_params,
                                    )
                                else:
                                    pil_working = inpaint_call(
                                        pil_working,
                                        mask_bool,
                                        seed=region_seed,
                                        verbose=verbose,
                                        ocr_params=ocr_params,
                                    )
                                bubble_info["inpainted"] = True
                                _resample_inpainted_bubble_color(
                                    pil_working, bubble_info, mask_bool
                                )
                            except CancellationError:
                                raise
                            except Exception as e:
                                log_message(
                                    f"Flux inpainting failed for bubble {bbox_tuple}: {e}; falling back to standard fill",
                                    always_print=True,
                                )
                                continue

                    cleaned_image = cv2.cvtColor(
                        np.array(pil_working.convert("RGB")), cv2.COLOR_RGB2BGR
                    )
                except CancellationError:
                    raise
                except Exception as e:
                    log_message(
                        f"Flux inpainting aborted; falling back to standard fill: {e}",
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
    except OSError as e:
        raise ImageProcessingError(f"Error loading image {image_input}: {e!s}") from e
    except CancellationError:
        raise
    except Exception as e:
        raise CleaningError(f"Error cleaning speech bubbles: {e!s}") from e


def retry_cleaning_with_otsu(
    image_bgr: np.ndarray,
    bubble_info: dict,
    thresholding_value: int,
    roi_shrink_px: int,
    processing_scale: float = 1.0,
    verbose: bool = False,
    classify_colored: bool = False,
) -> dict | None:
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
            neighbor_bboxes=bubble_info.get("neighbor_bboxes"),
            processing_scale=processing_scale,
            image_bgr=image_bgr,
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
        text_color_bgr,
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
        "text_color_bgr": text_color_bgr,
        "is_sam": bubble_info.get("is_sam", False),
    }
