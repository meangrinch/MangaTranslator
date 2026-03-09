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
OSB_TEXT_MATCH_IOA_THRESHOLD = (
    0.2  # Minimum text-box overlap ratio for bubble assignment
)
AMBIGUOUS_TEXT_MATCH_RATIO = (
    0.85  # Skip nudge boxes that match sibling bubbles nearly equally
)
TEXT_NUDGE_BOX_INSET_RATIO = 0.08  # Use text-core box, not full detector padding
OVERLAP_NUDGE_INSET_RATIO = 0.08  # Keep nudged diagonal away from overlap-zone edges
MIN_OVERLAP_SPLIT_SHARE = 0.08  # Minimum share of overlap zone kept by each child
SYNTHETIC_CONJOINED_IOA_THRESHOLD = (
    0.15  # Primary bbox overlap signaling a split conjoined bubble
)
AXIS_DOMINANCE_RATIO = (
    3.0  # Center-offset ratio to classify a box pair as axis-aligned (~18° cone)
)
TEXT_SAFE_SPLIT_MARGIN_RATIO = (
    0.25  # Allow split to clip up to this fraction from a text box edge
)


def _box_contains(inner, outer) -> bool:
    """Return True if inner box is fully contained in outer box."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return ix0 >= ox0 and iy0 >= oy0 and ix1 <= ox1 and iy1 <= oy1


def _box_intersection_area(box_a, box_b) -> float:
    """Return the intersection area between two xyxy boxes."""
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _box_area(box) -> float:
    """Return the area of an xyxy box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _point_in_box(point_x: float, point_y: float, box) -> bool:
    """Return True if a point lies inside an xyxy box."""
    return box[0] <= point_x <= box[2] and box[1] <= point_y <= box[3]


def _mask_to_bbox(mask, fallback_box=None) -> tuple[int, int, int, int]:
    """Convert a binary mask to an xyxy bbox, falling back when empty."""
    mask_array = np.asarray(mask) > 0
    coords = np.where(mask_array)
    if coords[0].size == 0 or coords[1].size == 0:
        if fallback_box is None:
            return (0, 0, 0, 0)
        x0_f, y0_f, x1_f, y1_f = (
            fallback_box.tolist() if hasattr(fallback_box, "tolist") else fallback_box
        )
        return (
            int(round(x0_f)),
            int(round(y0_f)),
            int(round(x1_f)),
            int(round(y1_f)),
        )

    y_coords, x_coords = coords
    return (
        int(x_coords.min()),
        int(y_coords.min()),
        int(x_coords.max()) + 1,
        int(y_coords.max()) + 1,
    )


def _text_box_meaningfully_matches_box(t_box, b_box) -> bool:
    """Return True when a text box meaningfully belongs to a bubble box."""
    intersection = _box_intersection_area(t_box, b_box)
    if intersection <= 0.0:
        return False

    text_area = _box_area(t_box)
    if text_area <= 0.0:
        return False

    text_center_x = (t_box[0] + t_box[2]) / 2.0
    text_center_y = (t_box[1] + t_box[3]) / 2.0
    text_ioa = intersection / text_area
    return text_ioa >= OSB_TEXT_MATCH_IOA_THRESHOLD or _point_in_box(
        text_center_x, text_center_y, b_box
    )


def _get_nudge_box_corners(t_box) -> list[tuple[float, float]]:
    """Return a slightly inset box so detector padding does not overconstrain nudges."""
    x0, y0, x1, y1 = [float(v) for v in t_box[:4]]
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    inset_x = min(
        max(1.0, width * TEXT_NUDGE_BOX_INSET_RATIO),
        max(0.0, width / 2.0 - 0.5),
    )
    inset_y = min(
        max(1.0, height * TEXT_NUDGE_BOX_INSET_RATIO),
        max(0.0, height / 2.0 - 0.5),
    )
    return [
        (x0 + inset_x, y0 + inset_y),
        (x1 - inset_x, y0 + inset_y),
        (x0 + inset_x, y1 - inset_y),
        (x1 - inset_x, y1 - inset_y),
    ]


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
                image_cv,
                conf=confidence,
                device=device,
                verbose=False,
                imgsz=640,
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
            best_idx = None
            best_intersection = 0.0

            for i, b_box in enumerate(pb_np):
                intersection = _box_intersection_area(t_box, b_box)
                if intersection > best_intersection:
                    best_intersection = intersection
                    best_idx = i

            if best_idx is None or best_intersection <= 0.0:
                continue

            if not _text_box_meaningfully_matches_box(t_box, pb_np[best_idx]):
                continue

            if _box_contains(t_box, pb_np[best_idx]):
                continue

            b = pb_np[best_idx]
            pb_np[best_idx] = [
                min(b[0], t_box[0]),
                min(b[1], t_box[1]),
                max(b[2], t_box[2]),
                max(b[3], t_box[3]),
            ]

        return torch.tensor(
            pb_np, device=primary_boxes.device, dtype=primary_boxes.dtype
        )
    except Exception as e:
        log_message(f"OSB text verification skipped: {e}", verbose=verbose)
        return primary_boxes


def _calculate_ioa(box_inner, box_outer):
    """IoA = intersection_area / area_of_inner_box."""
    area_inner = _box_area(box_inner)
    if area_inner <= 0:
        return 0.0
    return _box_intersection_area(box_inner, box_outer) / area_inner


def _calculate_iou(box_a, box_b):
    """IoU = intersection_area / union_area."""
    intersection = _box_intersection_area(box_a, box_b)
    union = _box_area(box_a) + _box_area(box_b) - intersection
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
    boxes: torch.Tensor,
    indices: Optional[List[Tuple[str, int]]] = None,
    threshold: float = 0.9,
) -> Tuple[torch.Tensor, List[Tuple[str, int]]]:
    """Remove boxes that are fully or almost fully contained within other boxes.

    Args:
        boxes: Tensor of bounding boxes (N, 4)
        indices: Source/index mapping for each box position
        threshold: IoA threshold above which inner boxes are considered contained

    Returns:
        Tuple of filtered bounding boxes and retained source/index mappings
    """
    if indices is None:
        indices = [("primary", i) for i in range(len(boxes))]

    if len(boxes) <= 1:
        return boxes, indices

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

    kept_indices = [indices[i] for i, keep_box in enumerate(keep) if keep_box]
    return boxes[keep], kept_indices


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
        meaningful_matches = []
        for i, b_box in enumerate(boxes):
            box_list = b_box if not hasattr(b_box, "tolist") else b_box.tolist()
            area = _box_intersection_area(t_box[:4], box_list)
            if area <= 0.0:
                continue
            if _text_box_meaningfully_matches_box(t_box[:4], box_list):
                meaningful_matches.append((i, area))

        meaningful_matches.sort(key=lambda item: item[1], reverse=True)
        ambiguous = (
            len(meaningful_matches) > 1
            and meaningful_matches[1][1] / meaningful_matches[0][1]
            >= AMBIGUOUS_TEXT_MATCH_RATIO
        )
        if meaningful_matches and not ambiguous:
            text_boxes_for[meaningful_matches[0][0]].append(t_box)
    return text_boxes_for


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


def _detect_overlapping_primaries(
    primary_boxes,
    simple_indices,
    ioa_threshold=SYNTHETIC_CONJOINED_IOA_THRESHOLD,
    verbose=False,
):
    """Detect primary boxes that significantly overlap each other, indicating
    the primary YOLO model split a conjoined bubble into per-section bboxes.

    Uses union-find for transitive grouping (A↔B + B↔C → {A,B,C}).

    Returns:
        synthetic_groups: list of sorted member-index lists (each len ≥ 2)
        updated_simple_indices: simple_indices with grouped members removed
    """
    if len(simple_indices) < 2:
        return [], simple_indices

    parent_map: dict[int, int] = {}

    def find(x):
        while parent_map.get(x, x) != x:
            parent_map[x] = parent_map.get(parent_map[x], parent_map[x])
            x = parent_map[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent_map[ry] = rx

    has_overlap = False
    for a_pos in range(len(simple_indices)):
        for b_pos in range(a_pos + 1, len(simple_indices)):
            idx_a = simple_indices[a_pos]
            idx_b = simple_indices[b_pos]
            box_a = primary_boxes[idx_a].tolist()
            box_b = primary_boxes[idx_b].tolist()

            if (
                _calculate_ioa(box_a, box_b) > ioa_threshold
                or _calculate_ioa(box_b, box_a) > ioa_threshold
            ):
                union(idx_a, idx_b)
                has_overlap = True

    if not has_overlap:
        return [], simple_indices

    group_map: dict[int, list[int]] = {}
    for idx in simple_indices:
        root = find(idx)
        group_map.setdefault(root, []).append(idx)

    synthetic_groups = [
        sorted(members) for members in group_map.values() if len(members) >= 2
    ]
    if not synthetic_groups:
        return [], simple_indices

    grouped_set = {idx for grp in synthetic_groups for idx in grp}
    updated_simple = [idx for idx in simple_indices if idx not in grouped_set]

    total_members = sum(len(g) for g in synthetic_groups)
    log_message(
        f"Detected {len(synthetic_groups)} synthetic conjoined group(s) "
        f"from {total_members} overlapping primary detections",
        always_print=True,
    )
    return synthetic_groups, updated_simple


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

            # Only interpolate if the mask doesn't already match the original shape (e.g. if retina_masks=False)
            if mask_tensor.shape[-2:] != (orig_h, orig_w):
                mask_tensor = (
                    torch.nn.functional.interpolate(
                        mask_tensor.float().unsqueeze(0).unsqueeze(0),
                        size=(orig_h, orig_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

            binary_mask = (mask_tensor > SAM_MASK_THRESHOLD).cpu().numpy()
            return binary_mask.astype(np.uint8) * 255
        else:
            return None

    except (IndexError, AttributeError) as e:
        log_message(
            f"Could not extract YOLO mask for detection {i}: {e}",
            always_print=True,
        )
        return None


def _build_rect_mask_from_box(box, img_h: int, img_w: int) -> np.ndarray:
    """Create a full-image rectangular mask from a bounding box."""
    x0_f, y0_f, x1_f, y1_f = box.tolist() if hasattr(box, "tolist") else box
    x0 = int(np.floor(max(0, min(x0_f, img_w))))
    y0 = int(np.floor(max(0, min(y0_f, img_h))))
    x1 = int(np.ceil(max(0, min(x1_f, img_w))))
    y1 = int(np.ceil(max(0, min(y1_f, img_h))))

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def _get_group_osb_text_boxes(
    osb_text_boxes: Optional[np.ndarray], primary_box
) -> Optional[np.ndarray]:
    """Scope OSB text boxes to those intersecting a primary bubble box."""
    if osb_text_boxes is None or len(osb_text_boxes) == 0:
        return None

    px0, py0, px1, py1 = (
        primary_box.tolist() if hasattr(primary_box, "tolist") else primary_box
    )
    hits = []
    for tb in osb_text_boxes:
        if tb[0] < px1 and tb[2] > px0 and tb[1] < py1 and tb[3] > py0:
            hits.append(tb)
    return np.array(hits) if hits else None


def _seed_mask_from_box(parent_mask: np.ndarray, box) -> np.ndarray:
    """Create a seed mask by clipping the parent mask to a child bounding box."""
    img_h, img_w = parent_mask.shape
    box_mask = _build_rect_mask_from_box(box, img_h, img_w) > 0
    seed_mask = np.logical_and(parent_mask, box_mask)
    if np.any(seed_mask):
        return seed_mask

    if not np.any(parent_mask):
        return seed_mask

    x0_f, y0_f, x1_f, y1_f = box.tolist() if hasattr(box, "tolist") else box
    center_x = (x0_f + x1_f) / 2.0
    center_y = (y0_f + y1_f) / 2.0

    parent_coords = np.column_stack(np.where(parent_mask))
    if parent_coords.size == 0:
        return seed_mask

    distances = (parent_coords[:, 1] - center_x) ** 2 + (
        parent_coords[:, 0] - center_y
    ) ** 2
    nearest_y, nearest_x = parent_coords[int(np.argmin(distances))]
    seed_mask[nearest_y, nearest_x] = True
    return seed_mask


def _split_overlap_zone_with_line(
    overlap_mask: np.ndarray,
    center_a: tuple[float, float],
    center_b: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    text_boxes_a: Optional[list] = None,
    text_boxes_b: Optional[list] = None,
    require_text_safe_split: bool = False,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Split an overlap zone along a line, optionally requiring a text-safe offset."""
    line_vec_x = line_end[0] - line_start[0]
    line_vec_y = line_end[1] - line_start[1]
    line_length = np.hypot(line_vec_x, line_vec_y)
    if line_length < 1e-6:
        return None

    normal_x = line_vec_y / line_length
    normal_y = -line_vec_x / line_length

    def _signed_distance(point_x, point_y):
        return (point_x - line_start[0]) * normal_x + (
            point_y - line_start[1]
        ) * normal_y

    pixel_y, pixel_x = np.where(overlap_mask)
    if len(pixel_x) == 0:
        return None

    pixel_dist = _signed_distance(pixel_x, pixel_y)
    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]
    text_boxes_a = text_boxes_a or []
    text_boxes_b = text_boxes_b or []
    require_text_safe_split = (
        require_text_safe_split and bool(text_boxes_a) and bool(text_boxes_b)
    )

    offset = 0.0
    if require_text_safe_split:
        raw_lower_bound = float(np.min(pixel_dist))
        raw_upper_bound = float(np.max(pixel_dist))
        nudge_inset = max(
            1.0, (raw_upper_bound - raw_lower_bound) * OVERLAP_NUDGE_INSET_RATIO
        )
        lower_bound = raw_lower_bound + nudge_inset
        upper_bound = raw_upper_bound - nudge_inset

        if lower_bound > upper_bound:
            lower_bound = raw_lower_bound
            upper_bound = raw_upper_bound

        def _tighten_bounds(text_boxes, center_dist, lower, upper):
            if abs(center_dist) < 1e-6:
                return lower, upper

            corner_distances = []
            for t_box in text_boxes:
                for cx, cy in _get_nudge_box_corners(t_box):
                    corner_distances.append(_signed_distance(cx, cy))

            if not corner_distances:
                return lower, upper

            # Only constrain with corners on the text's own side of the
            # midline (d=0).  Corners that have crossed into sibling
            # territory represent tiny intrusions that cannot be protected
            # and would otherwise make the bounds infeasible.
            if center_dist > 0:
                own_side = [d for d in corner_distances if d > 0]
                if own_side:
                    upper = min(upper, min(own_side))
            else:
                own_side = [d for d in corner_distances if d < 0]
                if own_side:
                    lower = max(lower, max(own_side))
            return lower, upper

        center_a_dist = _signed_distance(center_a[0], center_a[1])
        center_b_dist = _signed_distance(center_b[0], center_b[1])
        lower_bound, upper_bound = _tighten_bounds(
            text_boxes_a, center_a_dist, lower_bound, upper_bound
        )
        lower_bound, upper_bound = _tighten_bounds(
            text_boxes_b, center_b_dist, lower_bound, upper_bound
        )

        if lower_bound > upper_bound:
            return None

        if lower_bound <= 0.0 <= upper_bound:
            midpoint_offset = (lower_bound + upper_bound) / 2.0
            offset = 0.0 if abs(midpoint_offset) < 1e-6 else midpoint_offset
        elif upper_bound < 0.0:
            offset = upper_bound
        else:
            offset = lower_bound

        # Post-validation: reject if the offset cuts through the interior
        # of any text box.  Allow edge-clipping (corners that barely
        # straddle the midline) by insetting the rejection zone.
        for t_box in list(text_boxes_a) + list(text_boxes_b):
            dists = [
                _signed_distance(cx, cy) for cx, cy in _get_nudge_box_corners(t_box)
            ]
            d_min, d_max = min(dists), max(dists)
            span = d_max - d_min
            if span < 1e-6:
                continue
            margin = span * TEXT_SAFE_SPLIT_MARGIN_RATIO
            if (d_min + margin) <= offset <= (d_max - margin):
                return None

    def _classify_pixels(split_offset: float):
        side_a = _signed_distance(center_a[0], center_a[1]) - split_offset
        side_b = _signed_distance(center_b[0], center_b[1]) - split_offset
        pixel_side = pixel_dist - split_offset

        if side_a == side_b:
            signed_projection = (pixel_x - (center_a[0] + center_b[0]) / 2.0) * dx + (
                pixel_y - (center_a[1] + center_b[1]) / 2.0
            ) * dy
            mask_a_pixels = signed_projection <= 0
            mask_b_pixels = signed_projection > 0
        else:
            if side_a < side_b:
                mask_a_pixels = pixel_side <= 0
                mask_b_pixels = pixel_side > 0
            else:
                mask_a_pixels = pixel_side >= 0
                mask_b_pixels = pixel_side < 0

        return mask_a_pixels, mask_b_pixels

    mask_a_pixels, mask_b_pixels = _classify_pixels(offset)
    if require_text_safe_split and offset != 0.0:
        total_overlap_pixels = len(pixel_x)
        min_pixels = max(
            1, int(np.ceil(total_overlap_pixels * MIN_OVERLAP_SPLIT_SHARE))
        )
        if (
            np.count_nonzero(mask_a_pixels) < min_pixels
            or np.count_nonzero(mask_b_pixels) < min_pixels
        ):
            return None

    mask_a = np.zeros_like(overlap_mask, dtype=bool)
    mask_b = np.zeros_like(overlap_mask, dtype=bool)
    mask_a[pixel_y[mask_a_pixels], pixel_x[mask_a_pixels]] = True
    mask_b[pixel_y[mask_b_pixels], pixel_x[mask_b_pixels]] = True
    return mask_a, mask_b


def _detect_group_arrangement(group_boxes: list) -> Optional[str]:
    """Determine if all boxes in a group are arranged along a single axis.

    Returns "horizontal" when every pair of box centres has a dominant
    horizontal offset (|dx| > AXIS_DOMINANCE_RATIO * |dy|), "vertical" when
    the dominant offset is vertical, or None for mixed/diagonal layouts.
    """
    if len(group_boxes) < 2:
        return None

    def _center(box):
        b = box.tolist() if hasattr(box, "tolist") else box
        return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0

    arrangement: Optional[str] = None
    for i in range(len(group_boxes)):
        ci = _center(group_boxes[i])
        for j in range(i + 1, len(group_boxes)):
            cj = _center(group_boxes[j])
            dx = abs(cj[0] - ci[0])
            dy = abs(cj[1] - ci[1])

            if dx > AXIS_DOMINANCE_RATIO * max(dy, 1e-6):
                pair_arr = "horizontal"
            elif dy > AXIS_DOMINANCE_RATIO * max(dx, 1e-6):
                pair_arr = "vertical"
            else:
                return None

            if arrangement is None:
                arrangement = pair_arr
            elif arrangement != pair_arr:
                return None

    return arrangement


def _split_overlap_zone_with_box_diagonal(
    overlap_mask: np.ndarray,
    box_a,
    box_b,
    text_boxes_a: Optional[list] = None,
    text_boxes_b: Optional[list] = None,
    group_arrangement: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an overlap zone, choosing axis-aligned or diagonal lines.

    When *group_arrangement* is ``"horizontal"`` or ``"vertical"`` (all group
    members share that axis), the corresponding axis-aligned split is tried
    first.  Otherwise the diagonal is preferred, matching the previous default.
    """
    box_a = box_a.tolist() if hasattr(box_a, "tolist") else box_a
    box_b = box_b.tolist() if hasattr(box_b, "tolist") else box_b

    ox0 = max(box_a[0], box_b[0])
    oy0 = max(box_a[1], box_b[1])
    ox1 = min(box_a[2], box_b[2])
    oy1 = min(box_a[3], box_b[3])

    if ox1 <= ox0 or oy1 <= oy0 or not np.any(overlap_mask):
        return np.zeros_like(overlap_mask, dtype=bool), np.zeros_like(
            overlap_mask, dtype=bool
        )

    center_a = ((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0)
    center_b = ((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0)
    dx = center_b[0] - center_a[0]
    dy = center_b[1] - center_a[1]

    # Same-sign diagonal offsets mean the boxes are NW/SE or SE/NW relative to
    # each other, so the desired split is the overlap anti-diagonal. Otherwise,
    # use the main diagonal.
    if dx * dy >= 0:
        line_start = (ox1, oy0)
        line_end = (ox0, oy1)
    else:
        line_start = (ox0, oy0)
        line_end = (ox1, oy1)

    overlap_mid_x = (ox0 + ox1) / 2.0
    overlap_mid_y = (oy0 + oy1) / 2.0
    diag_line = (line_start, line_end)
    h_line = ((ox0, overlap_mid_y), (ox1, overlap_mid_y))
    v_line = ((overlap_mid_x, oy0), (overlap_mid_x, oy1))

    # Horizontal arrangement → boxes side-by-side → vertical split preferred.
    # Vertical arrangement   → boxes stacked      → horizontal split preferred.
    if group_arrangement == "horizontal":
        split_candidates = [v_line, diag_line, h_line]
    elif group_arrangement == "vertical":
        split_candidates = [h_line, diag_line, v_line]
    else:
        split_candidates = [diag_line, h_line, v_line]

    text_boxes_a = text_boxes_a or []
    text_boxes_b = text_boxes_b or []
    if text_boxes_a and text_boxes_b:
        for candidate_start, candidate_end in split_candidates:
            split_masks = _split_overlap_zone_with_line(
                overlap_mask,
                center_a,
                center_b,
                candidate_start,
                candidate_end,
                text_boxes_a=text_boxes_a,
                text_boxes_b=text_boxes_b,
                require_text_safe_split=True,
            )
            if split_masks is not None:
                return split_masks

    # Non-text-safe fallback: try preferred candidate first, then diagonal.
    preferred_start, preferred_end = split_candidates[0]
    fallback_lines = [(preferred_start, preferred_end)]
    if (preferred_start, preferred_end) != (line_start, line_end):
        fallback_lines.append((line_start, line_end))
    for fallback_start, fallback_end in fallback_lines:
        split_masks = _split_overlap_zone_with_line(
            overlap_mask,
            center_a,
            center_b,
            fallback_start,
            fallback_end,
            text_boxes_a=text_boxes_a,
            text_boxes_b=text_boxes_b,
            require_text_safe_split=False,
        )
        if split_masks is not None:
            return split_masks

    return np.zeros_like(overlap_mask, dtype=bool), np.zeros_like(
        overlap_mask, dtype=bool
    )


def _expand_resolved_masks_within_parent(
    parent_mask: np.ndarray,
    resolved_masks: list[np.ndarray],
) -> list[np.ndarray]:
    """Grow resolved seed masks to cover the remaining parent-mask pixels."""
    if len(resolved_masks) == 0:
        return []

    assigned = np.zeros_like(parent_mask, dtype=bool)
    seed_masks = []
    distance_maps = []

    for mask in resolved_masks:
        seed = np.asarray(mask, dtype=bool)
        seed_masks.append(seed)
        assigned |= seed

        if np.any(seed):
            inv_seed = np.where(seed, 0, 1).astype(np.uint8)
            distance_maps.append(
                cv2.distanceTransform(inv_seed, cv2.DIST_L2, 5).astype(np.float32)
            )
        else:
            distance_maps.append(np.full(parent_mask.shape, np.inf, dtype=np.float32))

    remaining = np.logical_and(parent_mask, ~assigned)
    if not np.any(remaining):
        return [seed.astype(np.uint8) * 255 for seed in seed_masks]

    nearest_seed = np.argmin(np.stack(distance_maps, axis=0), axis=0)
    expanded_masks = []
    for idx, seed in enumerate(seed_masks):
        expanded = seed.copy()
        expanded[np.logical_and(remaining, nearest_seed == idx)] = True
        expanded_masks.append(expanded.astype(np.uint8) * 255)

    return expanded_masks


def _split_conjoined_mask(
    primary_mask,
    group_boxes: list,
    verbose: bool = False,
    osb_text_boxes: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Partition one mask into per-secondary masks for a conjoined group."""
    if primary_mask is None or len(group_boxes) == 0:
        return []

    base_mask = np.asarray(primary_mask) > 0
    if not np.any(base_mask):
        return [np.zeros_like(primary_mask, dtype=np.uint8) for _ in group_boxes]

    if len(group_boxes) == 1:
        return [base_mask.astype(np.uint8) * 255]

    img_h, img_w = base_mask.shape
    box_masks = [
        _build_rect_mask_from_box(box, img_h, img_w) > 0 for box in group_boxes
    ]
    resolved_masks = [np.logical_and(base_mask, box_mask) for box_mask in box_masks]
    text_boxes_for = None
    if osb_text_boxes is not None and len(osb_text_boxes) > 0:
        text_boxes_for = _match_text_boxes_to_bubbles(osb_text_boxes, group_boxes)

    for idx, mask in enumerate(resolved_masks):
        if not np.any(mask):
            resolved_masks[idx] = _seed_mask_from_box(base_mask, group_boxes[idx])

    group_arrangement = _detect_group_arrangement(group_boxes)

    overlap_count = 0
    for i in range(len(group_boxes)):
        for j in range(i + 1, len(group_boxes)):
            overlap_zone = np.logical_and(
                base_mask, np.logical_and(box_masks[i], box_masks[j])
            )
            if not np.any(overlap_zone):
                continue

            overlap_count += 1
            split_i, split_j = _split_overlap_zone_with_box_diagonal(
                overlap_zone,
                group_boxes[i],
                group_boxes[j],
                text_boxes_a=text_boxes_for.get(i, []) if text_boxes_for else None,
                text_boxes_b=text_boxes_for.get(j, []) if text_boxes_for else None,
                group_arrangement=group_arrangement,
            )

            resolved_masks[i][overlap_zone] = False
            resolved_masks[j][overlap_zone] = False
            resolved_masks[i] = np.logical_or(resolved_masks[i], split_i)
            resolved_masks[j] = np.logical_or(resolved_masks[j], split_j)

    if overlap_count > 0:
        log_message(
            f"Resolved {overlap_count} YOLO conjoined overlap zone(s)",
            verbose=verbose,
        )

    return _expand_resolved_masks_within_parent(base_mask, resolved_masks)


def _get_detection_metadata(
    source: str,
    orig_idx: int,
    primary_results,
    primary_model,
    secondary_results,
    conjoined_confidence: float,
):
    """Return confidence/class metadata for a detection source."""
    if (
        source == "primary"
        and primary_results is not None
        and len(primary_results.boxes) > 0
    ):
        safe_idx = min(orig_idx, len(primary_results.boxes.conf) - 1)
        conf = float(primary_results.boxes.conf[safe_idx])
        cls_id = int(primary_results.boxes.cls[safe_idx])
        return conf, primary_model.names[cls_id]

    if (
        source == "secondary"
        and secondary_results is not None
        and len(secondary_results.boxes) > 0
    ):
        safe_idx = min(orig_idx, len(secondary_results.boxes.conf) - 1)
        conf = float(secondary_results.boxes.conf[safe_idx])
        cls_id = int(secondary_results.boxes.cls[safe_idx])
        cls_name = (
            secondary_results.names.get(cls_id, "speech_bubble")
            if hasattr(secondary_results, "names")
            else "speech_bubble"
        )
        return conf, cls_name

    return conjoined_confidence, "speech_bubble"


def _build_segmentation_detections(
    primary_boxes,
    grouping_primary_boxes,
    primary_sources,
    primary_results,
    primary_model,
    secondary_boxes,
    secondary_sources,
    secondary_results,
    simple_indices,
    conjoined_indices,
    img_h: int,
    img_w: int,
    conjoined_confidence: float,
    osb_text_boxes_np: Optional[np.ndarray] = None,
    verbose: bool = False,
    sam_masks: Optional[list] = None,
    synthetic_conjoined_groups: Optional[list] = None,
):
    """Build final detections from masks, including conjoined-mask partitioning."""
    detections = []

    for idx in simple_indices:
        source, orig_idx = primary_sources[idx]
        box = primary_boxes[idx]
        conf, cls_name = _get_detection_metadata(
            source,
            orig_idx,
            primary_results,
            primary_model,
            secondary_results,
            conjoined_confidence,
        )

        sam_mask = None
        if sam_masks is not None and sam_masks[idx] is not None:
            sam_mask = sam_masks[idx]
        elif source == "primary":
            sam_mask = _fallback_to_yolo_mask(primary_results, orig_idx, "binary")

        if sam_mask is None:
            sam_mask = _build_rect_mask_from_box(box, img_h, img_w)

        x0_f, y0_f, x1_f, y1_f = box.tolist()
        detections.append(
            {
                "bbox": (
                    int(round(x0_f)),
                    int(round(y0_f)),
                    int(round(x1_f)),
                    int(round(y1_f)),
                ),
                "confidence": conf,
                "class": cls_name,
                "sam_mask": sam_mask,
            }
        )

    for p_idx, s_indices in conjoined_indices:
        parent_source, parent_orig_idx = primary_sources[p_idx]
        parent_box = primary_boxes[p_idx]
        grouping_parent_box = grouping_primary_boxes[p_idx]
        parent_mask = None

        if sam_masks is not None and sam_masks[p_idx] is not None:
            parent_mask = sam_masks[p_idx]
        elif parent_source == "primary":
            parent_mask = _fallback_to_yolo_mask(
                primary_results, parent_orig_idx, "binary"
            )

        if parent_mask is None:
            parent_mask = _build_rect_mask_from_box(parent_box, img_h, img_w)

        group_boxes = [secondary_boxes[s_idx] for s_idx in s_indices]
        group_osb = _get_group_osb_text_boxes(osb_text_boxes_np, grouping_parent_box)
        split_masks = _split_conjoined_mask(
            parent_mask,
            group_boxes,
            verbose=verbose,
            osb_text_boxes=group_osb,
        )

        group_bboxes = [
            _mask_to_bbox(split_masks[idx], fallback_box=group_boxes[idx])
            for idx in range(len(group_boxes))
        ]

        for local_idx, s_idx in enumerate(s_indices):
            source, orig_idx = secondary_sources[s_idx]
            box = secondary_boxes[s_idx]
            conf, cls_name = _get_detection_metadata(
                source,
                orig_idx,
                primary_results,
                primary_model,
                secondary_results,
                conjoined_confidence,
            )
            detection = {
                "bbox": group_bboxes[local_idx],
                "confidence": conf,
                "class": cls_name,
                "sam_mask": split_masks[local_idx],
            }
            detection["conjoined_neighbor_bboxes"] = [
                bbox for idx, bbox in enumerate(group_bboxes) if idx != local_idx
            ]
            detections.append(detection)

    # Synthetic conjoined groups (primary-only, no parent YOLO detection)
    for sg in synthetic_conjoined_groups or []:
        parent_mask = sg.get("parent_mask")
        parent_box = sg["parent_box"]
        member_indices = sg["member_indices"]

        if parent_mask is None:
            parent_mask = _build_rect_mask_from_box(parent_box, img_h, img_w)

        group_boxes = [grouping_primary_boxes[idx] for idx in member_indices]
        group_osb = _get_group_osb_text_boxes(osb_text_boxes_np, parent_box)
        split_masks = _split_conjoined_mask(
            parent_mask,
            group_boxes,
            verbose=verbose,
            osb_text_boxes=group_osb,
        )

        group_bboxes = [
            _mask_to_bbox(split_masks[local_idx], fallback_box=group_boxes[local_idx])
            for local_idx in range(len(group_boxes))
        ]

        for local_idx, p_idx in enumerate(member_indices):
            source, orig_idx = primary_sources[p_idx]
            conf, cls_name = _get_detection_metadata(
                source,
                orig_idx,
                primary_results,
                primary_model,
                secondary_results,
                conjoined_confidence,
            )
            detection = {
                "bbox": group_bboxes[local_idx],
                "confidence": conf,
                "class": cls_name,
                "sam_mask": split_masks[local_idx],
                "conjoined_neighbor_bboxes": [
                    bbox for idx, bbox in enumerate(group_bboxes) if idx != local_idx
                ],
            }
            detections.append(detection)

    return detections


def detect_speech_bubbles(
    image_path: Path,
    model_path,
    confidence=0.6,
    verbose=False,
    device=None,
    seg_model: str = "yolo",
    conjoined_detection: bool = True,
    conjoined_confidence=0.35,
    image_override: Optional[Image.Image] = None,
    osb_enabled: bool = False,
    osb_text_verification: bool = False,
    osb_text_hf_token: str = "",
):
    """Detect speech bubbles using dual YOLO models and optional SAM2/SAM3 refinement.

    For conjoined bubbles detected by the secondary model, uses the inner bounding boxes
    directly and processes each as a separate simple bubble through SAM.

    Args:
        image_path (Path): Path to the input image
        model_path (str): Path to the primary YOLO segmentation model
        confidence (float): Confidence threshold for primary YOLO model detections
        verbose (bool): Whether to show detailed processing information
        device (torch.device, optional): The device to run the model on. Autodetects if None.
        seg_model (str): Segmentation model ("yolo", "sam2", "sam3")
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
            image_cv,
            conf=confidence,
            device=_device,
            verbose=False,
            imgsz=640,
            retina_masks=True,
        )[0]
        primary_boxes = (
            primary_results.boxes.xyxy
            if primary_results.boxes is not None
            else torch.tensor([])
        )
        cache.set_yolo_detection(yolo_cache_key, (primary_results, primary_boxes))

    primary_sources = [("primary", idx) for idx in range(len(primary_boxes))]

    # Remove duplicate primary detections using IoU-based NMS
    if len(primary_boxes) > 1:
        original_count = len(primary_boxes)
        primary_boxes, keep_indices = _deduplicate_primary_boxes(
            primary_boxes, primary_results.boxes.conf, IOU_DUPLICATE_THRESHOLD
        )
        primary_sources = [primary_sources[idx] for idx in keep_indices]
        if len(primary_boxes) < original_count:
            log_message(
                f"Removed {original_count - len(primary_boxes)} duplicate detections",
                verbose=verbose,
            )

    # Remove nested duplicate primary detections
    if len(primary_boxes) > 1:
        original_count = len(primary_boxes)
        primary_boxes, primary_sources = _remove_contained_boxes(
            primary_boxes, primary_sources
        )
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
    secondary_sources = []
    secondary_results = None
    use_sam = seg_model in ("sam2", "sam3")
    if conjoined_detection:
        try:
            secondary_model = model_manager.load_yolo_conjoined_bubble()
            log_message(
                "Loaded secondary YOLO model for conjoined/fallback detection",
                verbose=verbose,
            )

            secondary_results = secondary_model(
                image_cv,
                conf=conjoined_confidence,
                device=_device,
                verbose=False,
                imgsz=1024,
            )[0]
            secondary_boxes = (
                secondary_results.boxes.xyxy
                if secondary_results.boxes is not None
                else torch.tensor([])
            )
            secondary_sources = [
                ("secondary", idx) for idx in range(len(secondary_boxes))
            ]

            # Remove nested duplicate secondary detections
            if len(secondary_boxes) > 1:
                orig_sec_count = len(secondary_boxes)
                secondary_boxes, secondary_sources = _remove_contained_boxes(
                    secondary_boxes, secondary_sources
                )
                if len(secondary_boxes) < orig_sec_count:
                    log_message(
                        f"Removed {orig_sec_count - len(secondary_boxes)} contained secondary detections",
                        verbose=verbose,
                    )

            # Filter secondary detections to text_bubble for conjoined processing,
            # while still collecting text_free boxes for OSB routing.
            if len(secondary_boxes) > 0 and hasattr(secondary_model, "names"):
                text_bubble_id = None
                text_free_id = None
                for cid, cname in secondary_model.names.items():
                    if cname == "text_bubble":
                        text_bubble_id = cid
                    elif cname == "text_free":
                        text_free_id = cid

                secondary_cls = secondary_results.boxes.cls

                filtered_boxes = []
                filtered_sources = []
                for i, s_box in enumerate(secondary_boxes):
                    _, secondary_idx = secondary_sources[i]
                    cls_id = int(secondary_cls[secondary_idx])
                    if text_free_id is not None and cls_id == text_free_id:
                        text_free_boxes.append(s_box.tolist())
                        continue
                    if text_bubble_id is None or cls_id == text_bubble_id:
                        filtered_boxes.append(s_box)
                        filtered_sources.append(secondary_sources[i])

                secondary_boxes = (
                    torch.stack(filtered_boxes)
                    if filtered_boxes
                    else secondary_boxes[:0]
                )
                secondary_sources = filtered_sources

                if len(secondary_boxes) > 0:
                    new_boxes = []
                    new_box_sources = []
                    primary_boxes_list = (
                        primary_boxes.tolist() if len(primary_boxes) > 0 else []
                    )

                    for i, s_box in enumerate(secondary_boxes):
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
                            new_box_sources.append(secondary_sources[i])

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
                        primary_sources.extend(new_box_sources)

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
                        primary_sources = [primary_sources[i] for i in keep_indices]
                    else:
                        primary_boxes = torch.tensor([])
                        primary_sources = []

        except Exception as e:
            log_message(
                f"Warning: Could not load/run secondary YOLO model: {e}. "
                "Proceeding without conjoined/fallback detection.",
                verbose=verbose,
            )
            secondary_boxes = torch.tensor([])
            secondary_sources = []

    if len(primary_boxes) == 0:
        return detections, text_free_boxes

    grouping_primary_boxes = primary_boxes.clone()

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

    conjoined_indices = []
    simple_indices = list(range(len(primary_boxes)))
    if len(secondary_boxes) > 0 and conjoined_detection:
        log_message("Categorizing detections (simple vs conjoined)...", verbose=verbose)
        conjoined_indices, simple_indices = _categorize_detections(
            grouping_primary_boxes, secondary_boxes, ioa_threshold=IOA_THRESHOLD
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
        log_message(
            f"No secondary detections, processing all {len(simple_indices)} as simple bubbles",
            verbose=verbose,
        )

    # Detect primary boxes that overlap each other — likely per-section bboxes
    # for a conjoined bubble that the primary YOLO failed to capture as one box.
    synthetic_conjoined_groups: list[dict] = []
    if len(simple_indices) > 1:
        synth_groups, simple_indices = _detect_overlapping_primaries(
            grouping_primary_boxes, simple_indices, verbose=verbose
        )
        for member_indices in synth_groups:
            member_stack = grouping_primary_boxes[member_indices]
            parent_box = torch.cat(
                [
                    member_stack[:, :2].min(dim=0).values,
                    member_stack[:, 2:].max(dim=0).values,
                ]
            )
            synthetic_conjoined_groups.append(
                {
                    "member_indices": member_indices,
                    "parent_box": parent_box,
                    "parent_mask": None,
                }
            )

    if not use_sam:
        log_message("SAM disabled, using YOLO segmentation masks", verbose=verbose)
        img_h, img_w = image_cv.shape[:2]
        detections = _build_segmentation_detections(
            primary_boxes,
            grouping_primary_boxes,
            primary_sources,
            primary_results,
            primary_model,
            secondary_boxes,
            secondary_sources,
            secondary_results,
            simple_indices,
            conjoined_indices,
            img_h,
            img_w,
            conjoined_confidence,
            osb_text_boxes_np=osb_text_boxes_np,
            verbose=verbose,
            synthetic_conjoined_groups=synthetic_conjoined_groups,
        )
        return detections, text_free_boxes

    try:
        sam_model_name = "SAM 3" if seg_model == "sam3" else "SAM 2.1"
        log_message(
            f"Applying {sam_model_name} segmentation refinement", verbose=verbose
        )
        sam_cache_key = cache.get_sam_cache_key(
            image_pil,
            primary_boxes,
            seg_model,
            conjoined_detection,
            conjoined_confidence,
        )
        cached_sam = cache.get_sam_masks(sam_cache_key)

        if cached_sam is not None:
            log_message("Using cached SAM masks", verbose=verbose)
            detections = cached_sam
            return detections, text_free_boxes

        # Load appropriate SAM model
        if seg_model == "sam3":
            processor, sam_model_instance = model_manager.load_sam3(
                token=osb_text_hf_token, verbose=verbose
            )
        else:
            processor, sam_model_instance = model_manager.load_sam2(verbose=verbose)
        boxes_to_process = []
        process_indices = []

        for idx in simple_indices:
            boxes_to_process.append(primary_boxes[idx])
            process_indices.append(idx)

        for p_idx, s_indices in conjoined_indices:
            boxes_to_process.append(primary_boxes[p_idx])
            process_indices.append(p_idx)

        # Append synthetic conjoined parent boxes to the same SAM batch
        synth_start_idx = len(boxes_to_process)
        for sg in synthetic_conjoined_groups:
            boxes_to_process.append(sg["parent_box"])

        img_h, img_w = image_cv.shape[:2]
        sam_masks_for_yolo = [None] * len(primary_boxes)
        if boxes_to_process:
            all_boxes_tensor = torch.stack(boxes_to_process)

            total_boxes = len(boxes_to_process)
            simple_count = len(simple_indices)
            conjoined_parent_count = len(conjoined_indices)
            synth_parent_count = len(synthetic_conjoined_groups)

            parts = []
            if simple_count:
                parts.append(f"{simple_count} simple")
            if conjoined_parent_count:
                parts.append(f"{conjoined_parent_count} conjoined parents")
            if synth_parent_count:
                parts.append(f"{synth_parent_count} synthetic conjoined parents")
            if len(parts) > 1:
                log_message(
                    f"Processing {total_boxes} bubbles " f"({' + '.join(parts)})...",
                    verbose=verbose,
                )
            else:
                log_message(
                    f"Processing {total_boxes} simple bubbles...",
                    verbose=verbose,
                )

            all_masks_bool = _process_simple_bubbles(
                image_pil,
                all_boxes_tensor,
                list(range(len(boxes_to_process))),
                processor,
                sam_model_instance,
                _device,
            )

            for i, (mask_bool, box) in enumerate(zip(all_masks_bool, boxes_to_process)):
                x0_f, y0_f, x1_f, y1_f = box.tolist()
                x0 = int(np.floor(max(0, min(x0_f, img_w))))
                y0 = int(np.floor(max(0, min(y0_f, img_h))))
                x1 = int(np.ceil(max(0, min(x1_f, img_w))))
                y1 = int(np.ceil(max(0, min(y1_f, img_h))))

                if x1 > x0 and y1 > y0:
                    bbox_mask = np.zeros((img_h, img_w), dtype=bool)
                    bbox_mask[y0:y1, x0:x1] = True
                    mask_bool = np.logical_and(mask_bool, bbox_mask)

                clipped_mask = mask_bool.astype(np.uint8) * 255

                if i < synth_start_idx:
                    sam_masks_for_yolo[process_indices[i]] = clipped_mask
                else:
                    sg_idx = i - synth_start_idx
                    synthetic_conjoined_groups[sg_idx]["parent_mask"] = clipped_mask

            log_message(
                f"Generated {len(all_masks_bool)} primary masks with {sam_model_name}",
                always_print=True,
            )

        img_h, img_w = image_cv.shape[:2]
        detections = _build_segmentation_detections(
            primary_boxes,
            grouping_primary_boxes,
            primary_sources,
            primary_results,
            primary_model,
            secondary_boxes,
            secondary_sources,
            secondary_results,
            simple_indices,
            conjoined_indices,
            img_h,
            img_w,
            conjoined_confidence,
            osb_text_boxes_np=osb_text_boxes_np,
            verbose=verbose,
            sam_masks=sam_masks_for_yolo,
            synthetic_conjoined_groups=synthetic_conjoined_groups,
        )

        log_message(
            f"{sam_model_name} segmentation completed successfully", verbose=verbose
        )
        cache.set_sam_masks(sam_cache_key, detections)

    except Exception as e:
        log_message(
            f"{sam_model_name} segmentation failed: {e}. Falling back to YOLO segmentation masks.",
            always_print=True,
        )
        img_h, img_w = image_cv.shape[:2]
        detections = _build_segmentation_detections(
            primary_boxes,
            grouping_primary_boxes,
            primary_sources,
            primary_results,
            primary_model,
            secondary_boxes,
            secondary_sources,
            secondary_results,
            simple_indices,
            conjoined_indices,
            img_h,
            img_w,
            conjoined_confidence,
            osb_text_boxes_np=osb_text_boxes_np,
            verbose=verbose,
            synthetic_conjoined_groups=synthetic_conjoined_groups,
        )

        log_message(
            f"Fallback segmentation used {len(detections)} boxes",
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
        results = panel_model(
            image_cv,
            conf=confidence,
            device=_device,
            verbose=False,
            imgsz=640,
        )[0]
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
