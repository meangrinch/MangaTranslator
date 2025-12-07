from typing import Any, Dict, List


def sort_bubbles_by_reading_order(detections, reading_direction="rtl", panels=None):
    """
    Sort text elements (speech bubbles and OSB text) into a robust reading order.

    Strategy:
    - If panels are provided:
      1. Assign bubbles to panels.
      2. Sort bubbles within each panel.
      3. Treat panels (with their bubbles) and unassigned bubbles as "spatial entities".
      4. Sort all spatial entities (panels + unassigned bubbles) together using the robust
         bands/columns algorithm. This ensures unassigned bubbles are interleaved correctly
         between panels based on their position.
    - Otherwise, use global sorting on all bubbles.

    Args:
        detections (list): List of detection dicts with a "bbox" key: (x1, y1, x2, y2).
        reading_direction (str): "rtl" or "ltr".
        panels (list, optional): List of panel bounding boxes as tuples (x1, y1, x2, y2).

    Returns:
        list: New list with the same detection dicts, sorted in reading order.
              Elements will have a "panel_id" key added if panels are provided.
    """

    if not detections:
        return []

    rtl = (reading_direction or "rtl").lower() == "rtl"

    def _get_features(bbox):
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return x1, y1, x2, y2, w, h, cx, cy

    def _spatial_sort(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort items spatially using bands and columns.
        Items must be dicts with at least a "bbox" key.
        Returns the sorted list of items.
        """
        if not items:
            return []

        y_overlap_ratio_threshold = 0.3
        y_center_band_factor = 0.35
        x_overlap_ratio_threshold = 0.3
        x_center_band_factor = 0.35

        enriched = []
        for item in items:
            x1, y1, x2, y2, w, h, cx, cy = _get_features(item["bbox"])
            enriched.append(
                {
                    "item": item,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "w": w,
                    "h": h,
                    "cx": cx,
                    "cy": cy,
                }
            )

        enriched.sort(key=lambda e: e["cy"])

        bands = []

        for e in enriched:
            y1 = e["y1"]
            y2 = e["y2"]
            h = e["h"]

            best_band_idx = -1
            best_score = -1.0
            for i, band in enumerate(bands):
                band_h = max(1.0, float(band["y_max"] - band["y_min"]))
                overlap_v = max(0.0, min(y2, band["y_max"]) - max(y1, band["y_min"]))
                overlap_ratio = overlap_v / min(h, band_h)
                center_delta_y = abs(e["cy"] - (band["y_min"] + band["y_max"]) / 2.0)

                same_row = (overlap_ratio >= y_overlap_ratio_threshold) or (
                    center_delta_y <= y_center_band_factor * min(h, band_h)
                )
                if same_row:
                    score = overlap_ratio - (center_delta_y / (h + band_h)) * 0.1
                    if score > best_score:
                        best_score = score
                        best_band_idx = i

            if best_band_idx == -1:
                bands.append({"y_min": y1, "y_max": y2, "items": [e]})
            else:
                band = bands[best_band_idx]
                band["items"].append(e)
                band["y_min"] = min(band["y_min"], y1)
                band["y_max"] = max(band["y_max"], y2)

        bands.sort(key=lambda b: b["y_min"])

        ordered_items = []
        for band in bands:
            items_in_band = band["items"]

            columns = []
            for e in items_in_band:
                x1 = e["x1"]
                x2 = e["x2"]
                w = e["w"]

                best_col_idx = -1
                best_score = -1.0
                for i, col in enumerate(columns):
                    col_w = max(1.0, float(col["x_max"] - col["x_min"]))
                    overlap_h = max(0.0, min(x2, col["x_max"]) - max(x1, col["x_min"]))
                    overlap_ratio = overlap_h / min(w, col_w)
                    col_center_x = (col["x_min"] + col["x_max"]) / 2.0
                    center_delta_x = abs(e["cx"] - col_center_x)

                    same_col = (overlap_ratio >= x_overlap_ratio_threshold) or (
                        center_delta_x <= x_center_band_factor * min(w, col_w)
                    )
                    if same_col:
                        score = overlap_ratio - (center_delta_x / (w + col_w)) * 0.1
                        if score > best_score:
                            best_score = score
                            best_col_idx = i

                if best_col_idx == -1:
                    columns.append({"x_min": x1, "x_max": x2, "items": [e]})
                else:
                    col = columns[best_col_idx]
                    col["items"].append(e)
                    col["x_min"] = min(col["x_min"], x1)
                    col["x_max"] = max(col["x_max"], x2)

            if rtl:
                columns.sort(key=lambda c: -((c["x_min"] + c["x_max"]) / 2.0))
            else:
                columns.sort(key=lambda c: ((c["x_min"] + c["x_max"]) / 2.0))

            for col in columns:
                col["items"].sort(key=lambda e: e["cy"])
                ordered_items.extend([e["item"] for e in col["items"]])

        return ordered_items

    def _point_in_box(x, y, box):
        px1, py1, px2, py2 = box
        return px1 <= x <= px2 and py1 <= y <= py2

    def _assign_to_panel(detection, panels_list):
        """Assign a detection to a panel based on center point or intersection."""
        x1, y1, x2, y2 = detection["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        for i, panel_box in enumerate(panels_list):
            if _point_in_box(cx, cy, panel_box):
                return i

        # Check if center is within a small distance (snapping)
        # This helps with bubbles that are just barely outside a panel
        det_w = x2 - x1
        det_h = y2 - y1
        det_diagonal = (det_w * det_w + det_h * det_h) ** 0.5
        snap_threshold = det_diagonal * 0.05  # 5% of detection's diagonal

        min_dist = float("inf")
        closest_panel_idx = -1

        for i, panel_box in enumerate(panels_list):
            px1, py1, px2, py2 = panel_box
            dx = max(px1 - cx, 0, cx - px2)
            dy = max(py1 - cy, 0, cy - py2)
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_panel_idx = i

        if min_dist <= snap_threshold:
            return closest_panel_idx

        best_panel = None
        best_intersection = 0.0
        det_area = (x2 - x1) * (y2 - y1)

        for i, panel_box in enumerate(panels_list):
            px1, py1, px2, py2 = panel_box
            inter_x1 = max(x1, px1)
            inter_y1 = max(y1, py1)
            inter_x2 = min(x2, px2)
            inter_y2 = min(y2, py2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                intersection_ratio = inter_area / det_area if det_area > 0 else 0.0
                if intersection_ratio > best_intersection:
                    best_intersection = intersection_ratio
                    best_panel = i

        # Only assign if intersection is significant (>10%)
        if best_intersection > 0.1:
            return best_panel

        return None

    if not panels:
        return _spatial_sort(detections)

    # We do NOT sort panels beforehand. sorting panels by simple Y is brittle.
    # Instead, we will sort "panel entities" together with "unassigned bubbles".
    panel_assignments = {i: [] for i in range(len(panels))}
    unassigned = []

    for det in detections:
        panel_id = _assign_to_panel(det, panels)
        if panel_id is not None:
            det["panel_id"] = panel_id
            panel_assignments[panel_id].append(det)
        else:
            det["panel_id"] = None
            unassigned.append(det)

    meta_entities = []

    for pid, p_bbox in enumerate(panels):
        bubbles_in_panel = panel_assignments[pid]
        if bubbles_in_panel:
            sorted_bubbles = _spatial_sort(bubbles_in_panel)
            meta_entities.append(
                {"bbox": p_bbox, "content": sorted_bubbles, "type": "panel", "id": pid}
            )

    for det in unassigned:
        meta_entities.append({"bbox": det["bbox"], "content": [det], "type": "bubble"})

    sorted_entities = _spatial_sort(meta_entities)

    final_result = []
    for entity in sorted_entities:
        final_result.extend(entity["content"])

    return final_result
