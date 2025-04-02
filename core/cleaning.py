import cv2
import numpy as np
from PIL import Image

from .image_utils import pil_to_cv2
from .detection import detect_speech_bubbles

def clean_speech_bubbles(
    image_path,
    model_path,
    confidence=0.35,
    pre_computed_detections=None,
    device=None,
    dilation_kernel_size=7,
    dilation_iterations=1,
    threshold_value=210,
    min_contour_area=50,
    closing_kernel_size=7,
    closing_iterations=1,
    closing_kernel_shape=cv2.MORPH_ELLIPSE,
    constraint_erosion_kernel_size=5,
    constraint_erosion_iterations=1,
    verbose: bool = False # Added verbose parameter
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
        threshold_value (int or None): Fixed threshold for isolating bright bubble interior. None uses Otsu.
        min_contour_area (int): Minimum area for a contour to be considered part of the bubble interior.
        closing_kernel_size (int): Kernel size for morphological closing to smooth mask and include text.
        closing_iterations (int): Iterations for closing.
        closing_kernel_shape (int): Shape for the closing kernel (e.g., cv2.MORPH_RECT, cv2.MORPH_ELLIPSE).
        constraint_erosion_kernel_size (int): Kernel size for eroding the original YOLO mask for edge constraint.
        constraint_erosion_iterations (int): Iterations for constraint erosion.

    Returns:
        numpy.ndarray: Cleaned image with white bubbles.

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

        detections = pre_computed_detections if pre_computed_detections is not None else detect_speech_bubbles(
            image_path, model_path, confidence, device=device
        )

        bubble_masks = []

        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
        )
        closing_kernel = cv2.getStructuringElement(
            closing_kernel_shape, (closing_kernel_size, closing_kernel_size)
        )
        constraint_erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (constraint_erosion_kernel_size, constraint_erosion_kernel_size)
        )

        for detection in detections:
            # Add verbose logging here if needed in the future
            if "mask_points" not in detection or not detection["mask_points"]:
                if verbose: print(f"Verbose: Skipping detection without mask points: {detection.get('bbox')}")
                continue

            try:
                points_list = detection["mask_points"]
                points = np.array(points_list, dtype=np.float32)

                if len(points.shape) == 3 and points.shape[1] == 1:
                    points_int = np.round(points).astype(int)
                elif len(points.shape) == 2 and points.shape[1] == 2:
                    points_int = np.round(points).astype(int).reshape((-1, 1, 2))
                else:
                    if verbose: print(f"Verbose: Unexpected mask points format for detection {detection.get('bbox')}. Skipping.")
                    continue

                original_yolo_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.fillPoly(original_yolo_mask, [points_int], 255)

                roi_mask = cv2.dilate(original_yolo_mask, dilation_kernel, iterations=dilation_iterations)

                roi_gray = np.zeros_like(img_gray)
                roi_indices = roi_mask == 255
                if not np.any(roi_indices):
                    continue
                roi_gray[roi_indices] = img_gray[roi_indices]

                if threshold_value != -1: # Use fixed threshold
                    # Apply fixed threshold directly to the ROI (roi_gray already contains ROI pixels or zeros)
                    _, thresholded_roi = cv2.threshold(roi_gray, threshold_value, 255, cv2.THRESH_BINARY)
                else: # Use Otsu's method
                    thresholded_roi = np.zeros_like(img_gray) # Initialize mask
                    if np.any(roi_indices): # Check if there are any pixels in the ROI
                        roi_pixels = img_gray[roi_indices] # Get grayscale pixels from the original gray image within the ROI
                        if roi_pixels.size > 0: # Ensure there are pixels to process
                            # Calculate Otsu threshold based *only* on pixels within the ROI
                            thresh_val, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            # Apply threshold to the ROI pixels and place result in the full-size mask
                            thresholded_roi[roi_indices] = np.where(roi_pixels > thresh_val, 255, 0).astype(np.uint8)

                thresholded_roi_refined = cv2.bitwise_and(thresholded_roi, original_yolo_mask)

                contours, _ = cv2.findContours(thresholded_roi_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                final_mask = None
                if contours:
                    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                    if valid_contours:
                        largest_contour = max(valid_contours, key=cv2.contourArea)

                        interior_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        cv2.drawContours(interior_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

                        closed_mask = cv2.morphologyEx(interior_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations)

                        eroded_constraint_mask = cv2.erode(original_yolo_mask, constraint_erosion_kernel, iterations=constraint_erosion_iterations)

                        final_mask = cv2.bitwise_and(closed_mask, eroded_constraint_mask)

                if final_mask is not None:
                    bubble_masks.append(final_mask)

            except Exception as e:
                # Always print errors, but add more detail if verbose
                error_msg = f"Error processing mask for detection {detection.get('bbox')}: {e}"
                if verbose:
                    import traceback
                    error_msg += f"\n{traceback.format_exc()}"
                print(error_msg)

        if bubble_masks:
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for mask in bubble_masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            white_fill = np.ones_like(image) * 255
            cleaned_image = np.where(
                np.expand_dims(combined_mask, axis=2) == 255,
                white_fill,
                cleaned_image
            )

        return cleaned_image

    except IOError as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error cleaning speech bubbles: {str(e)}")