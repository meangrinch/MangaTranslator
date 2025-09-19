import cv2
import numpy as np
from PIL import Image

from utils.logging import log_message

from .detection import detect_speech_bubbles
from .image_utils import pil_to_cv2


def _process_single_bubble(
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
    constraint_erosion_kernel=None
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
        tuple: (final_mask, fill_color_bgr) or (None, None) if processing failed
    """
    try:
        if base_mask.dtype != np.uint8:
            base_mask = base_mask.astype(np.uint8)
        base_mask = np.where(base_mask > 0, 255, 0).astype(np.uint8)

        if dilation_kernel is None:
            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        if constraint_erosion_kernel is None:
            constraint_erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        masked_pixels = img_gray[base_mask == 255]
        if masked_pixels.size == 0:
            log_message(
                f"{'[SAM]' if is_sam else ''}Skipping detection {detection_bbox}: empty mask",
                verbose=verbose,
            )
            return None, None

        mean_pixel_value = np.mean(masked_pixels)
        is_black_bubble = mean_pixel_value < 128
        fill_color_bgr = (0, 0, 0) if is_black_bubble else (255, 255, 255)

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
        roi_for_thresholding = cv2.bitwise_not(roi_gray) if is_black_bubble else roi_gray
        thresholded_roi = np.zeros_like(img_gray)

        if use_otsu_threshold:
            roi_pixels_for_otsu = roi_for_thresholding[roi_indices]
            thresh_val, _ = cv2.threshold(
                roi_pixels_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            log_message(
                f"{'[SAM]' if is_sam else ''}  Otsu threshold: {thresh_val}",
                verbose=verbose
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
        dist_map = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 3)
        shrunk_roi_mask = np.where(dist_map >= float(roi_shrink_px), 255, 0).astype(np.uint8)
        thresholded_roi = cv2.bitwise_and(thresholded_roi, shrunk_roi_mask)

        # Use eroded mask to avoid erasing bubble outlines
        eroded_constraint_mask = cv2.erode(base_mask, constraint_erosion_kernel, iterations=1)

        contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 50:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            if (0 <= cx < img_width and 0 <= cy < img_height
                    and eroded_constraint_mask[cy, cx] == 255):
                valid_contours.append(cnt)

        log_message(
            f"{'[SAM]' if is_sam else ''}Detection {detection_bbox}: {len(valid_contours)} text fragments found",
            verbose=verbose,
        )

        if valid_contours:
            validated_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.drawContours(validated_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

            # Re-contour to get clean boundary from validated mask
            boundary_contours, _ = cv2.findContours(
                validated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if boundary_contours:
                largest_contour = max(boundary_contours, key=cv2.contourArea)
                final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                return final_mask, fill_color_bgr

        return None, None

    except Exception as e:
        log_message(
            f"Failed to process {'SAM' if is_sam else 'YOLO'} mask for {detection_bbox}: {e}",
            always_print=True,
            is_error=True,
        )
        return None, None


def clean_speech_bubbles(
    image_path,
    model_path,
    confidence=0.35,
    pre_computed_detections=None,
    device=None,
    thresholding_value: int = 190,
    use_otsu_threshold: bool = False,
    roi_shrink_px: int = 4,
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
        roi_shrink_px (int): Number of pixels to shrink the ROI inwards before identification/fill.

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

        # Create kernels once for efficiency
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        constraint_erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for detection in detections:
            final_mask = None
            fill_color_bgr = None

            sam_mask = detection.get("sam_mask")
            if sam_mask is not None:
                final_mask, fill_color_bgr = _process_single_bubble(
                    sam_mask, img_gray, img_height, img_width,
                    thresholding_value, use_otsu_threshold, roi_shrink_px,
                    verbose, detection.get('bbox'), is_sam=True,
                    dilation_kernel=dilation_kernel,
                    constraint_erosion_kernel=constraint_erosion_kernel
                )
            else:
                if "mask_points" not in detection or not detection["mask_points"]:
                    log_message(f"Skipping detection {detection.get('bbox')}: no mask points", verbose=verbose)
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

                    final_mask, fill_color_bgr = _process_single_bubble(
                        yolo_mask, img_gray, img_height, img_width,
                        thresholding_value, use_otsu_threshold, roi_shrink_px,
                        verbose, detection.get('bbox'), is_sam=False,
                        dilation_kernel=dilation_kernel,
                        constraint_erosion_kernel=constraint_erosion_kernel
                    )

                except Exception as e:
                    error_msg = f"Error processing YOLO mask for detection {detection.get('bbox')}: {e}"
                    log_message(error_msg, always_print=True, is_error=True)
                    continue

            if final_mask is not None and fill_color_bgr is not None:
                processed_bubbles.append({
                    "mask": final_mask,
                    "color": fill_color_bgr,
                    "bbox": detection.get("bbox"),
                })
                log_message(
                    f"Detection {detection.get('bbox')}: processed successfully",
                    verbose=verbose,
                )

        # Group masks by color for efficient batch processing
        if processed_bubbles:
            color_groups = {}
            for bubble_info in processed_bubbles:
                color_key = bubble_info["color"]
                if color_key not in color_groups:
                    color_groups[color_key] = []
                color_groups[color_key].append(bubble_info["mask"])

            for color_bgr, masks in color_groups.items():
                combined_mask = np.bitwise_or.reduce(masks)
                num_channels = cleaned_image.shape[2] if len(cleaned_image.shape) == 3 else 1
                if num_channels == 4 and len(color_bgr) == 3:
                    fill_color = (*color_bgr, 255)
                else:
                    fill_color = color_bgr

                cleaned_image[combined_mask == 255] = fill_color

        return cleaned_image, processed_bubbles
    except IOError as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error cleaning speech bubbles: {str(e)}")
