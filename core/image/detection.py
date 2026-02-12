from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from core.caching import get_cache
from core.device import get_best_device
from core.ml.model_manager import ModelType, get_model_manager
from utils.exceptions import ImageProcessingError, ModelError
from utils.logging import log_message

# Detection Parameters
IOA_THRESHOLD = 0.50  # 50% IoA threshold for conjoined bubble detection
SAM_MASK_THRESHOLD = 0.5  # SAM2 mask binarization threshold
IOA_OVERLAP_THRESHOLD = 0.5  # IoA threshold for general overlap detection between boxes
IOU_DUPLICATE_THRESHOLD = 0.7  # IoU threshold for duplicate primary detection


def _box_contains(inner, outer) -> bool:
    """Return True if inner box is fully contained in outer box."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return ix0 >= ox0 and iy0 >= oy0 and ix1 <= ox1 and iy1 <= oy1


def _expand_boxes_with_osb_text(
    image_cv,
    image_pil,
    primary_boxes: torch.Tensor,
    cache,
    model_manager,
    device,
    confidence: float,
    hf_token: str,
    verbose: bool,
):
    """Expand speech-bubble boxes to fully contain detected OSB text boxes."""
    if primary_boxes is None or len(primary_boxes) == 0:
        return primary_boxes

    try:
        model_path = str(model_manager.model_paths[ModelType.YOLO_OSBTEXT])
        cache_key = cache.get_yolo_cache_key(image_pil, model_path, confidence)
        cached = cache.get_yolo_detection(cache_key)

        if cached is not None:
            _, osb_boxes, _ = cached
        else:
            osb_model = model_manager.load_yolo_osbtext(token=hf_token)
            osb_results = osb_model(
                image_cv, conf=confidence, device=device, verbose=False
            )[0]
            osb_boxes = (
                osb_results.boxes.xyxy
                if osb_results.boxes is not None
                else torch.tensor([])
            )
            osb_confs = (
                osb_results.boxes.conf
                if osb_results.boxes is not None
                else torch.tensor([])
            )
            cache.set_yolo_detection(cache_key, (osb_results, osb_boxes, osb_confs))

        if osb_boxes is None or len(osb_boxes) == 0:
            return primary_boxes

        pb_np = primary_boxes.detach().cpu().numpy()
        osb_np = osb_boxes.detach().cpu().numpy()

        for t_box in osb_np:
            tx0, ty0, tx1, ty1 = t_box
            best_idx = None
            best_intersection = 0.0

            for i, b_box in enumerate(pb_np):
                bx0, by0, bx1, by1 = b_box
                inter_x0 = max(bx0, tx0)
                inter_y0 = max(by0, ty0)
                inter_x1 = min(bx1, tx1)
                inter_y1 = min(by1, ty1)
                inter_w = max(0.0, inter_x1 - inter_x0)
                inter_h = max(0.0, inter_y1 - inter_y0)
                intersection = inter_w * inter_h
                if intersection > best_intersection:
                    best_intersection = intersection
                    best_idx = i

            if best_idx is None or best_intersection <= 0.0:
                continue

            if _box_contains(t_box, pb_np[best_idx]):
                continue

            bx0, by0, bx1, by1 = pb_np[best_idx]
            pb_np[best_idx] = [
                min(bx0, tx0),
                min(by0, ty0),
                max(bx1, tx1),
                max(by1, ty1),
            ]

        return torch.tensor(
            pb_np, device=primary_boxes.device, dtype=primary_boxes.dtype
        )
    except Exception as e:
        log_message(f"OSB text verification skipped: {e}", verbose=verbose)
        return primary_boxes


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


def _calculate_iou(box_a, box_b):
    """Calculate Intersection over Union (IoU) for two bounding boxes.

    IoU = intersection_area / union_area

    Args:
        box_a: Tuple of (x0, y0, x1, y1)
        box_b: Tuple of (x0, y0, x1, y1)

    Returns:
        float: IoU value between 0 and 1
    """
    inter_x_min = max(box_a[0], box_b[0])
    inter_y_min = max(box_a[1], box_b[1])
    inter_x_max = min(box_a[2], box_b[2])
    inter_y_max = min(box_a[3], box_b[3])

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _deduplicate_primary_boxes(
    boxes: torch.Tensor, confidences: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, List[int]]:
    """Remove duplicate primary detections using IoU-based NMS.

    When two boxes have IoU > threshold, keeps the one with higher confidence.

    Args:
        boxes: Tensor of bounding boxes (N, 4)
        confidences: Tensor of confidence scores (N,)
        threshold: IoU threshold above which boxes are considered duplicates

    Returns:
        Tuple of (deduplicated boxes tensor, indices of kept boxes)
    """
    if len(boxes) <= 1:
        return boxes, list(range(len(boxes)))

    boxes_list = boxes.tolist()
    confs_list = confidences.tolist()
    n = len(boxes_list)

    # Sort by confidence (descending)
    indices = sorted(range(n), key=lambda i: confs_list[i], reverse=True)
    keep = []

    for i in indices:
        is_duplicate = False
        for k in keep:
            if _calculate_iou(boxes_list[i], boxes_list[k]) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(i)

    return boxes[keep], keep


def _remove_contained_boxes(
    boxes: torch.Tensor, threshold: float = 0.9
) -> torch.Tensor:
    """Remove boxes that are fully or almost fully contained within other boxes.

    Args:
        boxes: Tensor of bounding boxes (N, 4)
        threshold: IoA threshold above which inner boxes are considered contained

    Returns:
        Tensor of filtered bounding boxes
    """
    if len(boxes) <= 1:
        return boxes

    boxes_list = boxes.tolist()
    n = len(boxes_list)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue

            # Check if box i is contained in box j
            if _calculate_ioa(boxes_list[i], boxes_list[j]) > threshold:
                keep[i] = False
                break

    return boxes[keep]


def _get_cached_osb_text_boxes(cache, model_manager, image_pil, confidence):
    """Retrieve cached OSB text boxes as a numpy array, or None if unavailable."""
    try:
        model_path = str(model_manager.model_paths[ModelType.YOLO_OSBTEXT])
        cache_key = cache.get_yolo_cache_key(image_pil, model_path, confidence)
        cached = cache.get_yolo_detection(cache_key)
        if cached is not None:
            _, osb_boxes, _ = cached
            if osb_boxes is not None and len(osb_boxes) > 0:
                return (
                    osb_boxes.detach().cpu().numpy()
                    if hasattr(osb_boxes, "detach")
                    else np.asarray(osb_boxes)
                )
    except Exception:
        pass
    return None


def _match_text_boxes_to_bubbles(osb_text_boxes, boxes):
    """Associate each OSB text box with the bubble box it overlaps most.

    Returns:
        dict mapping bubble index → list of text box arrays
    """
    text_boxes_for = {i: [] for i in range(len(boxes))}
    for t_box in osb_text_boxes:
        tx0, ty0, tx1, ty1 = t_box[:4]
        best_idx = None
        best_area = 0.0
        for i, b_box in enumerate(boxes):
            bx0, by0, bx1, by1 = (
                b_box if not hasattr(b_box, "tolist") else b_box.tolist()
            )
            inter_w = max(0.0, min(bx1, tx1) - max(bx0, tx0))
            inter_h = max(0.0, min(by1, ty1) - max(by0, ty0))
            area = inter_w * inter_h
            if area > best_area:
                best_area = area
                best_idx = i
        if best_idx is not None and best_area > 0.0:
            text_boxes_for[best_idx].append(t_box)
    return text_boxes_for


def _compute_bisector_nudge(
    center_a, center_b, mid_x, mid_y, text_boxes_a, text_boxes_b
):
    """Compute offset to shift the perpendicular bisector so it doesn't cut text.

    Projects text-bbox corners onto the A→B axis.  Text belonging to bubble A
    should lie on the A-side (negative projection); if any corner projects
    positive it intrudes past the bisector — the bisector must shift toward B
    to clear it (and vice-versa for bubble B).

    Returns:
        (nudged_mid_x, nudged_mid_y) after applying the offset.
    """
    vec_x = center_b[0] - center_a[0]
    vec_y = center_b[1] - center_a[1]
    length = np.sqrt(vec_x**2 + vec_y**2)
    if length < 1e-6:
        return mid_x, mid_y

    uvec_x = vec_x / length
    uvec_y = vec_y / length

    # Max intrusion of A's text past the bisector (positive = toward B)
    shift_for_a = 0.0
    for t_box in text_boxes_a:
        corners = [
            (t_box[0], t_box[1]),
            (t_box[2], t_box[1]),
            (t_box[0], t_box[3]),
            (t_box[2], t_box[3]),
        ]
        for cx, cy in corners:
            proj = (cx - mid_x) * uvec_x + (cy - mid_y) * uvec_y
            if proj > shift_for_a:
                shift_for_a = proj

    # Max intrusion of B's text past the bisector (negative = toward A)
    shift_for_b = 0.0
    for t_box in text_boxes_b:
        corners = [
            (t_box[0], t_box[1]),
            (t_box[2], t_box[1]),
            (t_box[0], t_box[3]),
            (t_box[2], t_box[3]),
        ]
        for cx, cy in corners:
            proj = (cx - mid_x) * uvec_x + (cy - mid_y) * uvec_y
            if proj < shift_for_b:
                shift_for_b = proj

    # shift_for_a > 0 means move bisector toward B; shift_for_b < 0 means toward A
    offset = shift_for_a + shift_for_b  # net signed offset along A→B

    return mid_x + offset * uvec_x, mid_y + offset * uvec_y


def _resolve_mask_overlaps(
    masks: list,
    boxes: list,
    verbose: bool = False,
    osb_text_boxes: Optional[np.ndarray] = None,
) -> list:
    """Resolve overlapping mask regions by carving exclusive boundaries.

    For conjoined bubble masks that overlap, this function:
    1. Finds overlapping pixel regions between mask pairs
    2. Determines the split axis based on box arrangement
    3. If OSB text boxes are provided, nudges the split line to avoid
       cutting through detected text
    4. Assigns overlap pixels to the appropriate mask

    Args:
        masks: List of boolean numpy arrays (SAM output masks)
        boxes: List of corresponding bounding boxes [x0, y0, x1, y1]
        verbose: Whether to log operations
        osb_text_boxes: Optional (N,4) numpy array of OSB text detection boxes
            in xyxy format.  When provided, the perpendicular bisector is
            shifted so it does not cut through any text bbox.

    Returns:
        List of resolved masks with non-overlapping regions
    """
    if len(masks) <= 1:
        return masks

    # Pre-compute text-box-to-bubble associations once
    text_boxes_for = None
    if osb_text_boxes is not None and len(osb_text_boxes) > 0:
        text_boxes_for = _match_text_boxes_to_bubbles(osb_text_boxes, boxes)

    resolved = [mask.copy() for mask in masks]
    n = len(resolved)
    overlap_count = 0
    nudge_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            overlap = np.logical_and(resolved[i], resolved[j])
            if not np.any(overlap):
                continue

            overlap_count += 1
            box_a = boxes[i] if not hasattr(boxes[i], "tolist") else boxes[i].tolist()
            box_b = boxes[j] if not hasattr(boxes[j], "tolist") else boxes[j].tolist()

            center_a = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
            center_b = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
            overlap_coords = np.where(overlap)

            # Perpendicular bisector split via cross product
            mid_x = (center_a[0] + center_b[0]) / 2
            mid_y = (center_a[1] + center_b[1]) / 2
            vec_x = center_b[0] - center_a[0]
            vec_y = center_b[1] - center_a[1]

            # Nudge bisector to avoid cutting through OSB text boxes
            if text_boxes_for is not None:
                old_mid = (mid_x, mid_y)
                mid_x, mid_y = _compute_bisector_nudge(
                    center_a,
                    center_b,
                    mid_x,
                    mid_y,
                    text_boxes_for.get(i, []),
                    text_boxes_for.get(j, []),
                )
                if (mid_x, mid_y) != old_mid:
                    nudge_count += 1

            pixel_y = overlap_coords[0]
            pixel_x = overlap_coords[1]
            cross = (pixel_x - mid_x) * vec_y - (pixel_y - mid_y) * vec_x

            mask_a_pixels = cross >= 0
            mask_b_pixels = cross < 0

            resolved[i][pixel_y[mask_b_pixels], pixel_x[mask_b_pixels]] = False
            resolved[j][pixel_y[mask_a_pixels], pixel_x[mask_a_pixels]] = False

    if overlap_count > 0:
        log_message(
            f"Resolved {overlap_count} overlapping mask region(s)", verbose=verbose
        )
    if nudge_count > 0:
        log_message(
            f"Nudged bisector on {nudge_count} overlap(s) to avoid cutting text",
            verbose=verbose,
        )

    return resolved


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
    # Handle cases where one bubble is detected on the page and is conjoined
    if primary_boxes.ndim == 1 and primary_boxes.numel() == 4:
        primary_boxes = primary_boxes.unsqueeze(0)
    if secondary_boxes.ndim == 1 and secondary_boxes.numel() == 4:
        secondary_boxes = secondary_boxes.unsqueeze(0)

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

    primary_simple_indices = []
    conjoined_primary_indices = {c[0] for c in conjoined_indices}

    for i in range(len(primary_boxes)):
        if i in conjoined_primary_indices:
            continue

        # Check for duplication against processed secondary bubbles
        is_duplicate = False
        p_box_list = primary_boxes[i].tolist()

        for s_idx in processed_secondary_indices:
            s_box_list = secondary_boxes[s_idx].tolist()
            if _calculate_ioa(s_box_list, p_box_list) > ioa_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            primary_simple_indices.append(i)

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
                mask_tensor.float().unsqueeze(0).unsqueeze(0),
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
    confidence=0.6,
    verbose=False,
    device=None,
    sam_model: str = "sam2",
    conjoined_detection: bool = True,
    conjoined_confidence=0.35,
    image_override: Optional[Image.Image] = None,
    osb_enabled: bool = False,
    osb_text_verification: bool = False,
    osb_text_hf_token: str = "",
):
    """Detect speech bubbles using dual YOLO models and SAM2/SAM3.

    For conjoined bubbles detected by the secondary model, uses the inner bounding boxes
    directly and processes each as a separate simple bubble through SAM.

    Args:
        image_path (Path): Path to the input image
        model_path (str): Path to the primary YOLO segmentation model
        confidence (float): Confidence threshold for primary YOLO model detections
        verbose (bool): Whether to show detailed processing information
        device (torch.device, optional): The device to run the model on. Autodetects if None.
        sam_model (str): SAM model for segmentation refinement ("off", "sam2", "sam3")
        conjoined_detection (bool): Whether to enable conjoined bubble detection using secondary YOLO model
        conjoined_confidence (float): Confidence threshold for secondary YOLO model (conjoined bubble detection)
        osb_text_verification (bool): When True, expand bubble boxes to fully cover OSB text detections
        osb_text_hf_token (str): Optional token for gated model downloads (SAM3, OSB text)

    Returns:
        tuple[list, list]: (speech bubble detections, text_free boxes from secondary model)
    """
    detections = []
    text_free_boxes: List[List[float]] = []

    _device = device if device is not None else get_best_device()
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

    # Remove duplicate primary detections using IoU-based NMS
    if len(primary_boxes) > 1:
        original_count = len(primary_boxes)
        primary_boxes, _ = _deduplicate_primary_boxes(
            primary_boxes, primary_results.boxes.conf, IOU_DUPLICATE_THRESHOLD
        )
        if len(primary_boxes) < original_count:
            log_message(
                f"Removed {original_count - len(primary_boxes)} duplicate detections",
                verbose=verbose,
            )

    # Remove nested duplicate primary detections
    if len(primary_boxes) > 1:
        original_count = len(primary_boxes)
        primary_boxes = _remove_contained_boxes(primary_boxes)
        if len(primary_boxes) < original_count:
            log_message(
                f"Removed {original_count - len(primary_boxes)} contained detections",
                verbose=verbose,
            )

    if len(primary_boxes) == 0:
        log_message("No detections found", verbose=verbose)
        return detections, text_free_boxes

    log_message(
        f"Detected {len(primary_boxes)} speech bubbles with YOLO", always_print=True
    )

    secondary_boxes = torch.tensor([])
    use_sam = sam_model in ("sam2", "sam3")
    if use_sam:
        try:
            secondary_model = model_manager.load_yolo_conjoined_bubble()
            log_message(
                "Loaded secondary YOLO model for conjoined/fallback detection",
                verbose=verbose,
            )

            secondary_results = secondary_model(
                image_cv, conf=conjoined_confidence, device=_device, verbose=False
            )[0]
            secondary_boxes = (
                secondary_results.boxes.xyxy
                if secondary_results.boxes is not None
                else torch.tensor([])
            )

            # Remove nested duplicate secondary detections
            if len(secondary_boxes) > 1:
                orig_sec_count = len(secondary_boxes)
                secondary_boxes = _remove_contained_boxes(secondary_boxes)
                if len(secondary_boxes) < orig_sec_count:
                    log_message(
                        f"Removed {orig_sec_count - len(secondary_boxes)} contained secondary detections",
                        verbose=verbose,
                    )

            # Fallback: Add bubbles detected by secondary model but missed by primary
            if len(secondary_boxes) > 0 and hasattr(secondary_model, "names"):
                text_bubble_id = None
                text_free_id = None
                for cid, cname in secondary_model.names.items():
                    if cname == "text_bubble":
                        text_bubble_id = cid
                    elif cname == "text_free":
                        text_free_id = cid

                secondary_cls = secondary_results.boxes.cls

                # Collect text_free boxes regardless of OSB setting
                if text_free_id is not None:
                    for i, s_box in enumerate(secondary_boxes):
                        if int(secondary_cls[i]) == text_free_id:
                            text_free_boxes.append(s_box.tolist())

                if text_bubble_id is not None:
                    new_boxes = []
                    primary_boxes_list = (
                        primary_boxes.tolist() if len(primary_boxes) > 0 else []
                    )

                    for i, s_box in enumerate(secondary_boxes):
                        if int(secondary_cls[i]) != text_bubble_id:
                            continue

                        s_box_list = s_box.tolist()

                        is_covered = False

                        for p_box_list in primary_boxes_list:
                            ioa_s_in_p = _calculate_ioa(s_box_list, p_box_list)
                            ioa_p_in_s = _calculate_ioa(p_box_list, s_box_list)

                            if (
                                ioa_s_in_p > IOA_OVERLAP_THRESHOLD
                                or ioa_p_in_s > IOA_OVERLAP_THRESHOLD
                            ):
                                is_covered = True
                                break

                        if not is_covered:
                            new_boxes.append(s_box)

                    if new_boxes:
                        log_message(
                            f"Found {len(new_boxes)} missed bubbles from secondary model",
                            always_print=True,
                        )
                        new_boxes_tensor = torch.stack(new_boxes)
                        if len(primary_boxes) > 0:
                            primary_boxes = torch.cat(
                                (primary_boxes, new_boxes_tensor), dim=0
                            )
                        else:
                            primary_boxes = new_boxes_tensor

            # Remove text_free detections (route to OSB if enabled, discard otherwise)
            if text_free_boxes and len(primary_boxes) > 0:
                indices_to_remove = []
                primary_boxes_list = primary_boxes.tolist()

                for i, p_box in enumerate(primary_boxes_list):
                    overlaps_text_free = False
                    for tf_box in text_free_boxes:
                        if (
                            _calculate_ioa(p_box, tf_box) > IOA_OVERLAP_THRESHOLD
                            or _calculate_ioa(tf_box, p_box) > IOA_OVERLAP_THRESHOLD
                        ):
                            overlaps_text_free = True
                            break

                    if overlaps_text_free:
                        indices_to_remove.append(i)

                if indices_to_remove:
                    action = (
                        "routing to OSB pipeline"
                        if osb_enabled
                        else "discarding (OSB disabled)"
                    )
                    log_message(
                        f"Removing {len(indices_to_remove)} bubbles marked text_free ({action})",
                        always_print=True,
                    )
                    keep_indices = [
                        i
                        for i in range(len(primary_boxes))
                        if i not in indices_to_remove
                    ]
                    if keep_indices:
                        primary_boxes = primary_boxes[keep_indices]
                    else:
                        primary_boxes = torch.tensor([])

        except Exception as e:
            log_message(
                f"Warning: Could not load/run secondary YOLO model: {e}. "
                "Proceeding without conjoined/fallback detection.",
                verbose=verbose,
            )
            secondary_boxes = torch.tensor([])

    osb_text_boxes_np = None
    if osb_text_verification and len(primary_boxes) > 0:
        primary_boxes = _expand_boxes_with_osb_text(
            image_cv,
            image_pil,
            primary_boxes,
            cache,
            model_manager,
            _device,
            confidence,
            osb_text_hf_token,
            verbose,
        )
        # Retrieve cached OSB text boxes for bisector nudge during overlap resolution
        osb_text_boxes_np = _get_cached_osb_text_boxes(
            cache, model_manager, image_pil, confidence
        )

    if not use_sam:
        log_message("SAM disabled, using YOLO segmentation masks", verbose=verbose)
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
        return detections, text_free_boxes

    conjoined_indices = []
    simple_indices = list(range(len(primary_boxes)))
    try:
        sam_model_name = "SAM 3" if sam_model == "sam3" else "SAM 2.1"
        log_message(
            f"Applying {sam_model_name} segmentation refinement", verbose=verbose
        )
        sam_cache_key = cache.get_sam_cache_key(
            image_pil,
            primary_boxes,
            sam_model,
            conjoined_detection,
            conjoined_confidence,
        )
        cached_sam = cache.get_sam_masks(sam_cache_key)

        if cached_sam is not None:
            log_message("Using cached SAM masks", verbose=verbose)
            detections = cached_sam
            return detections, text_free_boxes

        # Load appropriate SAM model
        if sam_model == "sam3":
            processor, sam_model_instance = model_manager.load_sam3(
                token=osb_text_hf_token, verbose=verbose
            )
        else:
            processor, sam_model_instance = model_manager.load_sam2(verbose=verbose)
        if len(secondary_boxes) > 0 and conjoined_detection:
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
        boxes_to_process = []
        conjoined_mask_ranges = []

        for idx in simple_indices:
            boxes_to_process.append(primary_boxes[idx])

        for p_idx, s_indices in conjoined_indices:
            start_idx = len(boxes_to_process)
            for s_idx in s_indices:
                boxes_to_process.append(secondary_boxes[s_idx])
            end_idx = len(boxes_to_process)
            conjoined_mask_ranges.append((start_idx, end_idx, p_idx))

        if boxes_to_process:
            all_boxes_tensor = torch.stack(boxes_to_process)
            all_masks = _process_simple_bubbles(
                image_pil,
                all_boxes_tensor,
                list(range(len(boxes_to_process))),
                processor,
                sam_model_instance,
                _device,
            )

            for start_idx, end_idx, p_idx in conjoined_mask_ranges:
                group_masks = all_masks[start_idx:end_idx]
                group_boxes = boxes_to_process[start_idx:end_idx]

                # Scope OSB text boxes to those intersecting this group's primary bubble
                group_osb = None
                if osb_text_boxes_np is not None:
                    p_box = primary_boxes[p_idx].tolist()
                    px0, py0, px1, py1 = p_box
                    hits = []
                    for tb in osb_text_boxes_np:
                        if tb[0] < px1 and tb[2] > px0 and tb[1] < py1 and tb[3] > py0:
                            hits.append(tb)
                    if hits:
                        group_osb = np.array(hits)

                resolved_masks = _resolve_mask_overlaps(
                    group_masks,
                    group_boxes,
                    verbose=verbose,
                    osb_text_boxes=group_osb,
                )
                all_masks[start_idx:end_idx] = resolved_masks

            all_boxes = boxes_to_process

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

        log_message(
            f"Refined {len(all_masks)} masks with {sam_model_name}", always_print=True
        )
        log_message(f"Total masks generated: {len(all_masks)}", verbose=verbose)

        img_h, img_w = image_cv.shape[:2]

        conjoined_siblings: dict[int, list[tuple]] = {}
        for start_idx, end_idx, _ in conjoined_mask_ranges:
            group_bboxes = []
            for j in range(start_idx, end_idx):
                bx = boxes_to_process[j].tolist()
                group_bboxes.append(
                    (
                        int(np.floor(max(0, min(bx[0], img_w)))),
                        int(np.floor(max(0, min(bx[1], img_h)))),
                        int(np.ceil(max(0, min(bx[2], img_w)))),
                        int(np.ceil(max(0, min(bx[3], img_h)))),
                    )
                )
            for j in range(start_idx, end_idx):
                conjoined_siblings[j] = [
                    b for k, b in enumerate(group_bboxes) if k != j - start_idx
                ]

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
            if i in conjoined_siblings:
                detection["conjoined_neighbor_bboxes"] = conjoined_siblings[i]
            detections.append(detection)

        log_message(
            f"{sam_model_name} segmentation completed successfully", verbose=verbose
        )
        cache.set_sam_masks(sam_cache_key, detections)

    except Exception as e:
        log_message(
            f"{sam_model_name} segmentation failed: {e}. Falling back to YOLO segmentation masks.",
            always_print=True,
        )
        detections = []

        # Process primary boxes first in fallback to avoid duplicating secondary splits
        fallback_boxes = []
        if conjoined_detection and len(secondary_boxes) > 0 and conjoined_indices:
            for idx in simple_indices:
                fallback_boxes.append(("primary", idx, primary_boxes[idx]))
            for _, s_indices in conjoined_indices:
                for s_idx in s_indices:
                    fallback_boxes.append(("secondary", s_idx, secondary_boxes[s_idx]))
        elif len(primary_boxes) > 0:
            for idx in range(len(primary_boxes)):
                fallback_boxes.append(("primary", idx, primary_boxes[idx]))

        fallback_conjoined_siblings: dict[int, list[tuple]] = {}
        if conjoined_detection and len(secondary_boxes) > 0 and conjoined_indices:
            fb_idx = len(simple_indices)
            for _, s_indices in conjoined_indices:
                group_bboxes = []
                group_fb_indices = []
                for s_idx in s_indices:
                    bx = secondary_boxes[s_idx].tolist()
                    group_bboxes.append(
                        (
                            int(round(bx[0])),
                            int(round(bx[1])),
                            int(round(bx[2])),
                            int(round(bx[3])),
                        )
                    )
                    group_fb_indices.append(fb_idx)
                    fb_idx += 1
                for k, gi in enumerate(group_fb_indices):
                    fallback_conjoined_siblings[gi] = [
                        b for j, b in enumerate(group_bboxes) if j != k
                    ]

        img_h, img_w = image_cv.shape[:2]
        primary_fallback_count = 0
        secondary_fallback_count = 0

        for fb_i, (source, orig_idx, box) in enumerate(fallback_boxes):
            x0_f, y0_f, x1_f, y1_f = box.tolist()

            if source == "primary" and len(primary_results.boxes) > 0:
                safe_idx = min(orig_idx, len(primary_results.boxes.conf) - 1)
                conf = float(primary_results.boxes.conf[safe_idx])
                cls_id = int(primary_results.boxes.cls[safe_idx])
                cls_name = primary_model.names[cls_id]
                sam_mask = _fallback_to_yolo_mask(primary_results, safe_idx, "binary")
                primary_fallback_count += 1
            elif source == "secondary" and "secondary_results" in locals():
                try:
                    safe_idx = min(orig_idx, len(secondary_results.boxes.conf) - 1)
                    conf = float(secondary_results.boxes.conf[safe_idx])
                except Exception:
                    conf = conjoined_confidence
                cls_name = "speech_bubble"
                x0 = int(max(0, min(x0_f, img_w)))
                y0 = int(max(0, min(y0_f, img_h)))
                x1 = int(max(0, min(x1_f, img_w)))
                y1 = int(max(0, min(y1_f, img_h)))
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                mask[y0:y1, x0:x1] = 255
                sam_mask = mask
                secondary_fallback_count += 1
            else:
                conf = conjoined_confidence
                cls_name = "speech_bubble"
                x0 = int(max(0, min(x0_f, img_w)))
                y0 = int(max(0, min(y0_f, img_h)))
                x1 = int(max(0, min(x1_f, img_w)))
                y1 = int(max(0, min(y1_f, img_h)))
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                mask[y0:y1, x0:x1] = 255
                sam_mask = mask

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
            detection["sam_mask"] = sam_mask
            if fb_i in fallback_conjoined_siblings:
                detection["conjoined_neighbor_bboxes"] = fallback_conjoined_siblings[
                    fb_i
                ]

            detections.append(detection)

        log_message(
            f"Fallback segmentation used {len(detections)} boxes "
            f"(primary: {primary_fallback_count}, secondary splits: {secondary_fallback_count})",
            verbose=verbose,
        )

        return detections, text_free_boxes
    return detections, text_free_boxes


def detect_panels(
    image_path: Path,
    confidence: float = 0.25,
    device=None,
    verbose=False,
    image_override: Optional[Image.Image] = None,
) -> List[Tuple[int, int, int, int]]:
    """Detect manga/comic panels using YOLO model.

    Args:
        image_path (Path): Path to the input image
        confidence (float): Confidence threshold for panel YOLO detections
        device (torch.device, optional): The device to run the model on. Autodetects if None.
        verbose (bool): Whether to show detailed processing information
        image_override (Image.Image, optional): PIL Image to use instead of loading from path

    Returns:
        list: List of tuples (x1, y1, x2, y2) representing panel bounding boxes.
              Only includes detections with class "frame".
    """
    _device = device if device is not None else get_best_device()

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
            f"Processing image for panel detection: {image_path.name if image_path else 'override'} "
            f"({image_cv.shape[1]}x{image_cv.shape[0]})",
            verbose=verbose,
        )
    except Exception as e:
        raise ImageProcessingError(f"Error loading image: {e}")

    model_manager = get_model_manager()
    try:
        panel_model = model_manager.load_yolo_panel(verbose=verbose)
    except Exception as e:
        raise ModelError(f"Error loading panel model: {e}")

    try:
        results = panel_model(image_cv, conf=confidence, device=_device, verbose=False)[
            0
        ]
        boxes = results.boxes.xyxy if results.boxes is not None else torch.tensor([])
        classes = results.boxes.cls if results.boxes is not None else torch.tensor([])

        if len(boxes) == 0:
            log_message("No panels detected", verbose=verbose)
            return []

        # Filter for "frame" class (panel class)
        frame_class_id = None
        if hasattr(panel_model, "names"):
            for class_id, class_name in panel_model.names.items():
                if class_name.lower() == "frame":
                    frame_class_id = class_id
                    break

        panel_boxes = []
        for i, box in enumerate(boxes):
            # If we found a frame class ID, only include detections of that class
            # Otherwise, include all detections (fallback)
            if frame_class_id is not None:
                if int(classes[i]) != frame_class_id:
                    continue

            x0_f, y0_f, x1_f, y1_f = box.tolist()
            panel_boxes.append(
                (
                    int(round(x0_f)),
                    int(round(y0_f)),
                    int(round(x1_f)),
                    int(round(y1_f)),
                )
            )

        return panel_boxes

    except Exception as e:
        log_message(
            f"Panel detection failed: {e}. Proceeding without panel information.",
            always_print=True,
        )
        return []
