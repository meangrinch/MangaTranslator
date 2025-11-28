from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from core.caching import get_cache
from core.ml.model_manager import get_model_manager
from utils.exceptions import ImageProcessingError, ModelError
from utils.logging import log_message

# Detection Parameters
IOA_THRESHOLD = 0.80  # 80% IoA threshold for conjoined bubble detection
SAM_MASK_THRESHOLD = 0.5  # SAM2 mask binarization threshold


def _calculate_ioa(box_inner, box_outer):
    """Calculate Intersection over Area (IoA) for two bounding boxes.

    IoA = intersection_area / area_of_inner_box

    Args:
        box_inner: Tuple or list of (x0, y0, x1, y1) for the inner box
        box_outer: Tuple or list of (x0, y0, x1, y1) for the outer box

    Returns:
        float: IoA value between 0 and 1
    """
    x_inner_min, y_inner_min, x_inner_max, y_inner_max = box_inner
    x_outer_min, y_outer_min, x_outer_max, y_outer_max = box_outer

    inter_x_min = max(x_inner_min, x_outer_min)
    inter_y_min = max(y_inner_min, y_outer_min)
    inter_x_max = min(x_inner_max, x_outer_max)
    inter_y_max = min(y_inner_max, y_outer_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h

    area_inner = (x_inner_max - x_inner_min) * (y_inner_max - y_inner_min)
    return intersection / area_inner if area_inner > 0 else 0.0


def _categorize_detections(primary_boxes, secondary_boxes, ioa_threshold=IOA_THRESHOLD):
    """Categorize detections into simple and conjoined bubbles.

    Args:
        primary_boxes: Tensor of primary YOLO detection boxes (N, 4)
        secondary_boxes: Tensor of secondary YOLO detection boxes (M, 4)
        ioa_threshold: Threshold for determining if a secondary box is contained in a primary box

    Returns:
        tuple: (conjoined_indices, simple_indices)
            - conjoined_indices: List of tuples (primary_idx, [secondary_indices])
            - simple_indices: List of primary indices that are simple bubbles
    """
    conjoined_indices = []
    processed_secondary_indices = set()

    for i, p_box in enumerate(primary_boxes):
        contained_indices = []
        for j, s_box in enumerate(secondary_boxes):
            if j in processed_secondary_indices:
                continue
            ioa = _calculate_ioa(s_box.tolist(), p_box.tolist())
            if ioa > ioa_threshold:
                contained_indices.append(j)

        if len(contained_indices) >= 2:
            conjoined_indices.append((i, contained_indices))
            processed_secondary_indices.update(contained_indices)

    primary_simple_indices = [
        i
        for i in range(len(primary_boxes))
        if i not in [c[0] for c in conjoined_indices]
    ]

    return conjoined_indices, primary_simple_indices


def _process_simple_bubbles(
    image, primary_boxes, simple_indices, processor, sam_model, device
):
    """Process simple (non-conjoined) speech bubbles using SAM2.

    Args:
        image: PIL Image
        primary_boxes: Tensor of primary YOLO detection boxes
        simple_indices: List of indices for simple bubbles
        processor: SAM2 processor
        sam_model: SAM2 model
        device: PyTorch device

    Returns:
        list: List of numpy boolean masks for simple bubbles
    """
    if not simple_indices:
        return []

    simple_boxes_to_sam = primary_boxes[simple_indices].unsqueeze(0).cpu()
    inputs = processor(image, input_boxes=simple_boxes_to_sam, return_tensors="pt")

    # Cast floating point tensors to model's dtype before moving to device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point():
            inputs[key] = inputs[key].to(sam_model.dtype)

    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = sam_model(multimask_output=False, **inputs)

    masks_tensor = processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"]
    )[0][:, 0]
    simple_masks_np = (masks_tensor > SAM_MASK_THRESHOLD).cpu().numpy()
    return [mask for mask in simple_masks_np]


def _fallback_to_yolo_mask(primary_results, i, mask_type="points"):
    """Extract YOLO mask as fallback when SAM2 fails.

    Args:
        primary_results: YOLO detection results
        i: Detection index
        mask_type: Type of mask to extract ("points" or "binary")

    Returns:
        Mask data or None if extraction fails
    """
    if getattr(primary_results, "masks", None) is None:
        return None

    try:
        masks = primary_results.masks
        if len(masks) <= i:
            return None

        if mask_type == "points":
            mask_points = masks[i].xy[0]
            return (
                mask_points.tolist() if hasattr(mask_points, "tolist") else mask_points
            )
        elif mask_type == "binary":
            mask_tensor = masks.data[i]
            orig_h, orig_w = primary_results.orig_shape
            mask_resized = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            binary_mask = (mask_resized > SAM_MASK_THRESHOLD).cpu().numpy()
            return binary_mask.astype(np.uint8) * 255
        else:
            return None

    except (IndexError, AttributeError) as e:
        log_message(
            f"Could not extract YOLO mask for detection {i}: {e}",
            always_print=True,
        )
        return None


def detect_speech_bubbles(
    image_path: Path,
    model_path,
    confidence=0.35,
    verbose=False,
    device=None,
    use_sam2: bool = True,
    conjoined_detection: bool = True,
    image_override: Optional[Image.Image] = None,
):
    """Detect speech bubbles using dual YOLO models and SAM2.

    For conjoined bubbles detected by the secondary model, uses the inner bounding boxes
    directly and processes each as a separate simple bubble through SAM2.

    Args:
        image_path (Path): Path to the input image
        model_path (str): Path to the primary YOLO segmentation model
        confidence (float): Confidence threshold for detections
        verbose (bool): Whether to show detailed processing information
        device (torch.device, optional): The device to run the model on. Autodetects if None.
        use_sam2 (bool): Whether to use SAM2.1 for enhanced segmentation
        conjoined_detection (bool): Whether to enable conjoined bubble detection using secondary YOLO model

    Returns:
        list: List of dictionaries containing detection information (bbox, class, confidence, sam_mask)
    """
    detections = []

    _device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    log_message(f"Using device: {_device}", verbose=verbose)
    try:
        if image_override is not None:
            image_pil = (
                image_override
                if image_override.mode == "RGB"
                else image_override.convert("RGB")
            )
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        else:
            image_cv = cv2.imread(str(image_path))
            if image_cv is None:
                raise ImageProcessingError(f"Could not read image at {image_path}")
            image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        log_message(
            f"Processing image: {image_path.name} ({image_cv.shape[1]}x{image_cv.shape[0]})",
            verbose=verbose,
        )
    except Exception as e:
        raise ImageProcessingError(f"Error loading image: {e}")

    model_manager = get_model_manager()
    cache = get_cache()
    try:
        primary_model = model_manager.load_yolo_speech_bubble(model_path)
        log_message(f"Loaded primary YOLO model: {model_path}", verbose=verbose)
    except Exception as e:
        raise ModelError(f"Error loading primary model: {e}")

    yolo_cache_key = cache.get_yolo_cache_key(image_pil, model_path, confidence)
    cached_yolo = cache.get_yolo_detection(yolo_cache_key)

    if cached_yolo is not None:
        log_message("Using cached YOLO detections", verbose=verbose)
        primary_results, primary_boxes = cached_yolo
    else:
        primary_results = primary_model(
            image_cv, conf=confidence, device=_device, verbose=False
        )[0]
        primary_boxes = (
            primary_results.boxes.xyxy
            if primary_results.boxes is not None
            else torch.tensor([])
        )
        cache.set_yolo_detection(yolo_cache_key, (primary_results, primary_boxes))

    if len(primary_boxes) == 0:
        log_message("No detections found", verbose=verbose)
        return detections

    log_message(
        f"Detected {len(primary_boxes)} speech bubbles with YOLO", always_print=True
    )
    log_message(
        f"Primary YOLO found {len(primary_boxes)} speech bubbles", verbose=verbose
    )

    secondary_boxes = torch.tensor([])
    if use_sam2 and conjoined_detection:
        try:
            secondary_model = model_manager.load_yolo_conjoined_bubble()
            log_message(
                "Loaded secondary YOLO model for conjoined detection", verbose=verbose
            )

            secondary_results = secondary_model(
                image_cv, conf=confidence, device=_device, verbose=False
            )[0]
            secondary_boxes = (
                secondary_results.boxes.xyxy
                if secondary_results.boxes is not None
                else torch.tensor([])
            )
        except Exception as e:
            log_message(
                f"Warning: Could not load/run secondary YOLO model: {e}. Proceeding without conjoined detection.",
                verbose=verbose,
            )
            secondary_boxes = torch.tensor([])

    if not use_sam2:
        log_message("SAM2 disabled, using YOLO segmentation masks", verbose=verbose)
        for i, box in enumerate(primary_boxes):
            x0_f, y0_f, x1_f, y1_f = box.tolist()
            conf = float(primary_results.boxes.conf[i])
            cls_id = int(primary_results.boxes.cls[i])
            cls_name = primary_model.names[cls_id]

            detection = {
                "bbox": (
                    int(round(x0_f)),
                    int(round(y0_f)),
                    int(round(x1_f)),
                    int(round(y1_f)),
                ),
                "confidence": conf,
                "class": cls_name,
            }

            detection["sam_mask"] = _fallback_to_yolo_mask(primary_results, i, "binary")

            detections.append(detection)
        return detections

    try:
        log_message("Applying SAM2.1 segmentation refinement", verbose=verbose)
        sam_cache_key = cache.get_sam_cache_key(
            image_pil, primary_boxes, use_sam2, conjoined_detection
        )
        cached_sam = cache.get_sam_masks(sam_cache_key)

        if cached_sam is not None:
            log_message("Using cached SAM masks", verbose=verbose)
            detections = cached_sam
            return detections

        processor, sam_model = model_manager.load_sam2()
        if len(secondary_boxes) > 0:
            log_message(
                "Categorizing detections (simple vs conjoined)...", verbose=verbose
            )
            conjoined_indices, simple_indices = _categorize_detections(
                primary_boxes, secondary_boxes, ioa_threshold=IOA_THRESHOLD
            )
            log_message(
                f"Found {len(simple_indices)} simple bubbles and {len(conjoined_indices)} conjoined groups",
                verbose=verbose,
            )
            if len(conjoined_indices) > 0:
                log_message(
                    f"Detected {len(conjoined_indices)} conjoined speech bubbles with second YOLO",
                    always_print=True,
                )
        else:
            conjoined_indices = []
            simple_indices = list(range(len(primary_boxes)))
            log_message(
                f"No secondary detections, processing all {len(simple_indices)} as simple bubbles",
                verbose=verbose,
            )
        # Collect all boxes to process as simple bubbles
        boxes_to_process = []

        # Add simple bubbles
        for idx in simple_indices:
            boxes_to_process.append(primary_boxes[idx])

        # Add secondary boxes from conjoined groups
        for _, s_indices in conjoined_indices:
            for s_idx in s_indices:
                boxes_to_process.append(secondary_boxes[s_idx])

        # Process all boxes as simple bubbles
        if boxes_to_process:
            all_boxes_tensor = torch.stack(boxes_to_process)
            all_masks = _process_simple_bubbles(
                image_pil,
                all_boxes_tensor,
                list(range(len(boxes_to_process))),
                processor,
                sam_model,
                _device,
            )
            all_boxes = boxes_to_process

            # Log processing summary
            total_boxes = len(boxes_to_process)
            simple_count = len(simple_indices)
            conjoined_count = sum(len(s_indices) for _, s_indices in conjoined_indices)

            if conjoined_indices:
                log_message(
                    f"Processing {total_boxes} bubbles ({simple_count} simple + "
                    f"{conjoined_count} from conjoined groups)...",
                    verbose=verbose,
                )
            else:
                log_message(
                    f"Processing {total_boxes} simple bubbles...", verbose=verbose
                )
        else:
            all_masks = []
            all_boxes = []

        log_message(f"Refined {len(all_masks)} masks with SAM2", always_print=True)
        log_message(f"Total masks generated: {len(all_masks)}", verbose=verbose)
        img_h, img_w = image_cv.shape[:2]
        for i, (mask, box) in enumerate(zip(all_masks, all_boxes)):
            x0_f, y0_f, x1_f, y1_f = box.tolist()

            x0 = int(np.floor(max(0, min(x0_f, img_w))))
            y0 = int(np.floor(max(0, min(y0_f, img_h))))
            x1 = int(np.ceil(max(0, min(x1_f, img_w))))
            y1 = int(np.ceil(max(0, min(y1_f, img_h))))

            if x1 <= x0 or y1 <= y0:
                continue
            bbox_mask = np.zeros((img_h, img_w), dtype=bool)
            bbox_mask[y0:y1, x0:x1] = True
            clipped_mask = np.logical_and(mask, bbox_mask)

            detection = {
                "bbox": (x0, y0, x1, y1),
                "confidence": 1.0,  # Masks from SAM are high confidence
                "class": "speech bubble",
                "sam_mask": clipped_mask.astype(np.uint8) * 255,
            }
            detections.append(detection)

        log_message("SAM2.1 segmentation completed successfully", verbose=verbose)
        cache.set_sam_masks(sam_cache_key, detections)

    except Exception as e:
        log_message(
            f"SAM2.1 segmentation failed: {e}. Falling back to YOLO segmentation masks.",
            always_print=True,
        )
        detections = []
        for i, box in enumerate(primary_boxes):
            x0_f, y0_f, x1_f, y1_f = box.tolist()
            conf = float(primary_results.boxes.conf[i])
            cls_id = int(primary_results.boxes.cls[i])
            cls_name = primary_model.names[cls_id]

            detection = {
                "bbox": (
                    int(round(x0_f)),
                    int(round(y0_f)),
                    int(round(x1_f)),
                    int(round(y1_f)),
                ),
                "confidence": conf,
                "class": cls_name,
            }
            detection["sam_mask"] = _fallback_to_yolo_mask(primary_results, i, "binary")

            detections.append(detection)

    return detections
