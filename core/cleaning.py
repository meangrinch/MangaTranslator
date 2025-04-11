import cv2
import largestinteriorrectangle
import numpy as np
from PIL import Image

from .image_utils import pil_to_cv2
from .detection import detect_speech_bubbles
from .common_utils import log_message


def clean_speech_bubbles(
    image_path,
    model_path,
    confidence=0.35,
    pre_computed_detections=None,
    device=None,
    dilation_kernel_size=7,
    dilation_iterations=1,
    use_otsu_threshold: bool = False,
    min_contour_area=50,
    closing_kernel_size=7,
    closing_iterations=1,
    closing_kernel_shape=cv2.MORPH_ELLIPSE,
    constraint_erosion_kernel_size=5,
    constraint_erosion_iterations=1,
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
        dilation_kernel_size (int): Kernel size for dilating initial YOLO mask for ROI.
        dilation_iterations (int): Iterations for dilation.
        use_otsu_threshold (bool): If True, use Otsu's method for thresholding instead of the fixed value (210).
        min_contour_area (int): Minimum area for a contour to be considered part of the bubble interior.
        closing_kernel_size (int): Kernel size for morphological closing to smooth mask and include text.
        closing_iterations (int): Iterations for closing.
        closing_kernel_shape (int): Shape for the closing kernel (e.g., cv2.MORPH_RECT, cv2.MORPH_ELLIPSE).
        constraint_erosion_kernel_size (int): Kernel size for eroding the original YOLO mask for edge constraint.
        constraint_erosion_iterations (int): Iterations for constraint erosion.

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
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        closing_kernel = cv2.getStructuringElement(closing_kernel_shape, (closing_kernel_size, closing_kernel_size))
        constraint_erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (constraint_erosion_kernel_size, constraint_erosion_kernel_size)
        )

        for detection in detections:
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

                roi_mask = cv2.dilate(original_yolo_mask, dilation_kernel, iterations=dilation_iterations)

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
                    fixed_threshold = 210
                    _, thresholded_roi = cv2.threshold(roi_for_thresholding, fixed_threshold, 255, cv2.THRESH_BINARY)

                thresholded_roi = cv2.bitwise_and(thresholded_roi, roi_mask)

                final_mask = None
                eroded_constraint_mask = cv2.erode(
                    original_yolo_mask, constraint_erosion_kernel, iterations=constraint_erosion_iterations
                )

                if is_black_bubble:
                    log_message(
                        f"Detection {detection.get('bbox')}: Applying BLACK bubble refinement logic.", verbose=verbose
                    )

                    contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    refined_background_shape_mask = None
                    if contours:
                        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                        if valid_contours:
                            largest_contour = max(valid_contours, key=cv2.contourArea)
                            refined_background_shape_mask = np.zeros_like(img_gray)
                            cv2.drawContours(
                                refined_background_shape_mask, [largest_contour], -1, 255, thickness=cv2.FILLED
                            )
                            log_message("  - Found refined background shape.", verbose=verbose)
                        else:
                            log_message("  - No valid contours found for refined shape.", verbose=verbose)
                    else:
                        log_message("  - No contours found at all for refined shape.", verbose=verbose)

                    base_mask = original_yolo_mask.copy()
                    refined_mask = None
                    if refined_background_shape_mask is not None:
                        dilate_iter = 1
                        dilated_shape_mask = cv2.dilate(
                            refined_background_shape_mask, closing_kernel, iterations=dilate_iter
                        )
                        refined_mask = cv2.bitwise_and(base_mask, dilated_shape_mask)
                        log_message("  - Refined mask using dilated shape intersection.", verbose=verbose)
                    else:
                        refined_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
                        log_message("  - No refined shape, using closed base mask (1 iter).", verbose=verbose)

                    internal_erosion_iterations = max(
                        1, (constraint_erosion_iterations + 2)
                    )  # 2 additional iterations for black bubbles
                    eroded_refined_mask = cv2.erode(
                        refined_mask, constraint_erosion_kernel, iterations=internal_erosion_iterations
                    )
                    closed_eroded_refined_mask = cv2.morphologyEx(
                        eroded_refined_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations
                    )
                    final_mask = cv2.bitwise_and(closed_eroded_refined_mask, eroded_constraint_mask)

                else:
                    log_message(
                        f"Detection {detection.get('bbox')}: Applying WHITE bubble refinement logic (original).",
                        verbose=verbose,
                    )
                    contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                        if valid_contours:
                            largest_contour = max(valid_contours, key=cv2.contourArea)
                            interior_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                            cv2.drawContours(interior_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                            closed_mask = cv2.morphologyEx(
                                interior_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations
                            )
                            final_mask = cv2.bitwise_and(closed_mask, eroded_constraint_mask)
                            log_message(
                                "  - Generated mask from largest contour, closed, and constrained.", verbose=verbose
                            )
                        else:
                            log_message("  - No valid contours found.", verbose=verbose)
                    else:
                        log_message("  - No contours found at all.", verbose=verbose)

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
