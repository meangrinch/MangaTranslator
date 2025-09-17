import cv2
import largestinteriorrectangle
import numpy as np
from PIL import Image

from utils.logging import log_message

from .detection import detect_speech_bubbles
from .image_utils import pil_to_cv2


def clean_speech_bubbles(
    image_path,
    model_path,
    confidence=0.35,
    pre_computed_detections=None,
    device=None,
    thresholding_value: int = 210,
    use_otsu_threshold: bool = False,
    verbose: bool = False,
):
    """
    Clean speech bubbles in the given image using YOLO detection and refined masking.

    Args:
        image_path (str): Path to input image.
        model_path (str): Path to YOLO model.
        confidence (float): Confidence threshold for detections.
        pre_computed_detections (list, optional): Pre-computed detections from previous call.
        device (torch.device, optional): The device to run detection model on if needed.
        thresholding_value (int): Fixed threshold value for text detection (0-255). Lower values (e.g., 190)
                                 are useful for uncleaned text close to bubble's edges.
        use_otsu_threshold (bool): If True, use Otsu's method for thresholding instead of the fixed value.

    Returns:
        numpy.ndarray: Cleaned image with white bubbles.
        list[dict]: A list of dictionaries, each containing the 'mask' (numpy.ndarray)
                    and 'color' (tuple BGR) for each processed bubble.
    Raises:
        ValueError: If the image cannot be loaded.
        RuntimeError: If model loading or bubble detection fails.
    """
    try:
        pil_image = Image.open(image_path)
        image = pil_to_cv2(pil_image)
        img_height, img_width = image.shape[:2]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cleaned_image = image.copy()

        detections = (
            pre_computed_detections
            if pre_computed_detections is not None
            else detect_speech_bubbles(image_path, model_path, confidence, device=device)
        )

        processed_bubbles = []
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        constraint_erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )

        for detection in detections:
            # Prefer SAM mask if available
            sam_mask = detection.get("sam_mask")
            if sam_mask is not None:
                try:
                    # Normalize SAM mask to uint8 0/255
                    base_mask = sam_mask
                    if base_mask.dtype != np.uint8:
                        base_mask = base_mask.astype(np.uint8)
                    base_mask = np.where(base_mask > 0, 255, 0).astype(np.uint8)

                    # ROI similar to YOLO path (expand for thresholding robustness)
                    roi_mask = cv2.dilate(base_mask, dilation_kernel, iterations=1)

                    # Determine bubble color from pixels under the SAM mask
                    masked_pixels = img_gray[base_mask == 255]
                    if masked_pixels.size == 0:
                        log_message(
                            f"Skipping detection {detection.get('bbox')} due to empty SAM mask after indexing.",
                            verbose=verbose,
                        )
                        continue
                    mean_pixel_value = np.mean(masked_pixels)
                    is_black_bubble = mean_pixel_value < 128
                    fill_color_bgr = (0, 0, 0) if is_black_bubble else (255, 255, 255)
                    log_message(
                        f"[SAM] Detection {detection.get('bbox')}: Mean={mean_pixel_value:.1f} -> "
                        f"{'Black' if is_black_bubble else 'White'} Bubble. Fill: {fill_color_bgr}",
                        verbose=verbose,
                    )

                    # Build ROI grayscale image
                    roi_gray = np.zeros_like(img_gray)
                    roi_indices = roi_mask == 255
                    if not np.any(roi_indices):
                        log_message(
                            f"[SAM] Skipping detection {detection.get('bbox')} due to empty ROI mask.",
                            verbose=verbose,
                        )
                        continue
                    roi_gray[roi_indices] = img_gray[roi_indices]

                    # Thresholding (invert for black bubbles)
                    roi_for_thresholding = cv2.bitwise_not(roi_gray) if is_black_bubble else roi_gray
                    thresholded_roi = np.zeros_like(img_gray)
                    if use_otsu_threshold:
                        if np.any(roi_indices):
                            roi_pixels_for_otsu = roi_for_thresholding[roi_indices]
                            if roi_pixels_for_otsu.size > 0:
                                thresh_val, _ = cv2.threshold(
                                    roi_pixels_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                                )
                                log_message(f"[SAM]   - Otsu threshold determined: {thresh_val}", verbose=verbose)
                                _, thresholded_roi = cv2.threshold(
                                    roi_for_thresholding, thresh_val, 255, cv2.THRESH_BINARY
                                )
                            else:
                                log_message(
                                    "[SAM]   - Skipping Otsu: No pixels in ROI for thresholding.",
                                    verbose=verbose,
                                )
                        else:
                            log_message("[SAM]   - Skipping Otsu: ROI mask is empty.", verbose=verbose)
                    else:
                        _, thresholded_roi = cv2.threshold(
                            roi_for_thresholding, thresholding_value, 255, cv2.THRESH_BINARY
                        )

                    # Restrict thresholded region to ROI
                    thresholded_roi = cv2.bitwise_and(thresholded_roi, roi_mask)

                    # Constrain interior to avoid erasing outlines (use eroded SAM mask)
                    eroded_constraint_mask = cv2.erode(
                        base_mask, constraint_erosion_kernel, iterations=1
                    )

                    final_mask = None
                    if is_black_bubble:
                        log_message(
                            f"[SAM] Detection {detection.get('bbox')}: Validated reconstruction (BLACK).",
                            verbose=verbose,
                        )
                        contours, _ = cv2.findContours(
                            thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        validated_mask = np.zeros_like(img_gray)
                        kept_count = 0
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area <= 50:
                                continue
                            m = cv2.moments(cnt)
                            if m["m00"] == 0:
                                continue
                            cx = int(m["m10"] / m["m00"])
                            cy = int(m["m01"] / m["m00"])
                            if 0 <= cx < img_width and 0 <= cy < img_height and eroded_constraint_mask[cy, cx] == 255:
                                cv2.drawContours(validated_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                                kept_count += 1
                        log_message(f"[SAM]   - Kept {kept_count} validated fragments.", verbose=verbose)

                        # Re-contour the stable validated mask to get definitive boundary
                        boundary_contours, _ = cv2.findContours(
                            validated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if boundary_contours:
                            largest_contour = max(boundary_contours, key=cv2.contourArea)
                            final_mask = np.zeros_like(img_gray)
                            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                        else:
                            final_mask = None
                    else:
                        log_message(
                            f"[SAM] Detection {detection.get('bbox')}: Validated reconstruction (WHITE).",
                            verbose=verbose,
                        )
                        contours, _ = cv2.findContours(
                            thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        validated_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        kept_count = 0
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if area <= 50:
                                continue
                            m = cv2.moments(cnt)
                            if m["m00"] == 0:
                                continue
                            cx = int(m["m10"] / m["m00"])
                            cy = int(m["m01"] / m["m00"])
                            if 0 <= cx < img_width and 0 <= cy < img_height and eroded_constraint_mask[cy, cx] == 255:
                                cv2.drawContours(validated_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                                kept_count += 1
                        log_message(f"[SAM]   - Kept {kept_count} validated fragments.", verbose=verbose)

                        boundary_contours, _ = cv2.findContours(
                            validated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if boundary_contours:
                            largest_contour = max(boundary_contours, key=cv2.contourArea)
                            final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                        else:
                            final_mask = None

                    if final_mask is not None:
                        lir_coords = None
                        if np.any(final_mask):
                            try:
                                lir_coords = largestinteriorrectangle.lir(final_mask.astype(bool))
                                log_message(
                                    f"[SAM]   - Calculated LIR: {lir_coords}",
                                    verbose=verbose,
                                )
                            except Exception as lir_e:
                                log_message(
                                    f"[SAM]   - LIR calculation failed: {lir_e}",
                                    verbose=verbose,
                                    is_error=True,
                                )
                                lir_coords = None

                        processed_bubbles.append(
                            {
                                "mask": final_mask,
                                "color": fill_color_bgr,
                                "bbox": detection.get("bbox"),
                                "lir_bbox": lir_coords,
                            }
                        )
                        log_message(
                            (
                                f"[SAM] Detection {detection.get('bbox')}: Stored final mask, "
                                f"color {fill_color_bgr}, and LIR {lir_coords}."
                            ),
                            verbose=verbose,
                        )
                        continue
                except Exception as e:
                    log_message(
                        f"Warning: Failed to refine SAM mask for {detection.get('bbox')}: {e}",
                        always_print=True,
                    )

            if "mask_points" not in detection or not detection["mask_points"]:
                log_message(f"Skipping detection without mask points: {detection.get('bbox')}", verbose=verbose)
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
                        f"Unexpected mask points format for detection {detection.get('bbox')}. Skipping.",
                        verbose=verbose,
                    )
                    continue

                original_yolo_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.fillPoly(original_yolo_mask, [points_int], 255)

                roi_mask = cv2.dilate(original_yolo_mask, dilation_kernel, iterations=1)

                masked_pixels = img_gray[original_yolo_mask == 255]
                if masked_pixels.size == 0:
                    log_message(
                        f"Skipping detection {detection.get('bbox')} due to empty mask after indexing.", verbose=verbose
                    )
                    continue

                mean_pixel_value = np.mean(masked_pixels)
                is_black_bubble = mean_pixel_value < 128
                fill_color_bgr = (0, 0, 0) if is_black_bubble else (255, 255, 255)
                log_message(
                    f"Detection {detection.get('bbox')}: Mean={mean_pixel_value:.1f} -> "
                    f"{'Black' if is_black_bubble else 'White'} Bubble. Fill: {fill_color_bgr}",
                    verbose=verbose,
                )

                roi_gray = np.zeros_like(img_gray)
                roi_indices = roi_mask == 255
                if not np.any(roi_indices):
                    log_message(f"Skipping detection {detection.get('bbox')} due to empty ROI mask.", verbose=verbose)
                    continue
                roi_gray[roi_indices] = img_gray[roi_indices]

                roi_for_thresholding = cv2.bitwise_not(roi_gray) if is_black_bubble else roi_gray

                thresholded_roi = np.zeros_like(img_gray)

                if use_otsu_threshold:
                    if np.any(roi_indices):
                        roi_pixels_for_otsu = roi_for_thresholding[roi_indices]
                        if roi_pixels_for_otsu.size > 0:
                            thresh_val, _ = cv2.threshold(
                                roi_pixels_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                            )
                            log_message(f"  - Otsu threshold determined: {thresh_val}", verbose=verbose)
                            _, thresholded_roi = cv2.threshold(roi_for_thresholding, thresh_val, 255, cv2.THRESH_BINARY)
                        else:
                            log_message("  - Skipping Otsu: No pixels in ROI for thresholding.", verbose=verbose)
                    else:
                        log_message("  - Skipping Otsu: ROI mask is empty.", verbose=verbose)
                else:
                    _, thresholded_roi = cv2.threshold(roi_for_thresholding, thresholding_value, 255, cv2.THRESH_BINARY)

                thresholded_roi = cv2.bitwise_and(thresholded_roi, roi_mask)

                final_mask = None
                eroded_constraint_mask = cv2.erode(
                    original_yolo_mask, constraint_erosion_kernel, iterations=1
                )

                if is_black_bubble:
                    log_message(
                        f"Detection {detection.get('bbox')}: Applying validated fragment reconstruction (BLACK).",
                        verbose=verbose,
                    )

                    contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    validated_mask = np.zeros_like(img_gray)
                    kept_count = 0
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area <= 50:
                            continue
                        m = cv2.moments(cnt)
                        if m["m00"] == 0:
                            continue
                        cx = int(m["m10"] / m["m00"])
                        cy = int(m["m01"] / m["m00"])
                        if 0 <= cx < img_width and 0 <= cy < img_height and eroded_constraint_mask[cy, cx] == 255:
                            cv2.drawContours(validated_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                            kept_count += 1
                    log_message(f"  - Kept {kept_count} validated fragments.", verbose=verbose)

                    boundary_contours, _ = cv2.findContours(
                        validated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if boundary_contours:
                        largest_contour = max(boundary_contours, key=cv2.contourArea)
                        final_mask = np.zeros_like(img_gray)
                        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                    else:
                        final_mask = None

                else:
                    log_message(
                        f"Detection {detection.get('bbox')}: Applying validated fragment reconstruction (WHITE).",
                        verbose=verbose,
                    )
                    contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    validated_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    kept_count = 0
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area <= 50:
                            continue
                        m = cv2.moments(cnt)
                        if m["m00"] == 0:
                            continue
                        cx = int(m["m10"] / m["m00"])
                        cy = int(m["m01"] / m["m00"])
                        if 0 <= cx < img_width and 0 <= cy < img_height and eroded_constraint_mask[cy, cx] == 255:
                            cv2.drawContours(validated_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                            kept_count += 1
                    log_message(f"  - Kept {kept_count} validated fragments.", verbose=verbose)

                    boundary_contours, _ = cv2.findContours(
                        validated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if boundary_contours:
                        largest_contour = max(boundary_contours, key=cv2.contourArea)
                        final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                    else:
                        final_mask = None

                if final_mask is not None:
                    lir_coords = None
                    if np.any(final_mask):
                        try:
                            # Ensure mask is uint8, 255 filled (already should be)
                            lir_coords = largestinteriorrectangle.lir(
                                final_mask.astype(bool)
                            )  # Convert to boolean for lir function
                            log_message(f"  - Calculated LIR: {lir_coords}", verbose=verbose)
                        except Exception as lir_e:
                            log_message(f"  - LIR calculation failed: {lir_e}", verbose=verbose, is_error=True)
                            lir_coords = None
                    else:
                        log_message("  - Skipping LIR calculation: Final mask is empty.", verbose=verbose)

                    processed_bubbles.append(
                        {
                            "mask": final_mask,
                            "color": fill_color_bgr,
                            "bbox": detection.get("bbox"),
                            "lir_bbox": lir_coords,
                        }
                    )
                    log_message(
                        f"Detection {detection.get('bbox')}: Stored final mask, color {fill_color_bgr}, "
                        f"and LIR {lir_coords}.",
                        verbose=verbose,
                    )

            except Exception as e:
                error_msg = f"Error processing mask for detection {detection.get('bbox')}: {e}"
                log_message(error_msg, always_print=True, is_error=True)

        for bubble_info in processed_bubbles:
            mask = bubble_info["mask"]
            color_bgr = bubble_info["color"]
            num_channels = cleaned_image.shape[2] if len(cleaned_image.shape) == 3 else 1

            if num_channels == 4 and len(color_bgr) == 3:
                fill_color = (*color_bgr, 255)
            else:
                fill_color = color_bgr

            fill_color_image = np.full_like(cleaned_image, fill_color)
            cleaned_image = np.where(np.expand_dims(mask, axis=2) == 255, fill_color_image, cleaned_image)

        return cleaned_image, processed_bubbles
    except IOError as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error cleaning speech bubbles: {str(e)}")
