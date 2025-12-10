from typing import Any, Dict, List


def sort_bubbles_by_reading_order(detections, reading_direction="rtl", panels=None):
    """
    Hybrid Algorithm (veto system):
    - Macro: graph sort with ceiling + right-neighbor veto to enforce Z flow.
    - Micro: tuned spatial banding with looser thresholds for offset bubbles.
    """

    if not detections:
        return []

    rtl = (reading_direction or "rtl").lower() == "rtl"

    # Micro layout: keep slightly offset bubbles grouped into lines/columns.
    def _get_features(bbox):
        x1, y1, x2, y2 = bbox
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return x1, y1, x2, y2, w, h, cx, cy

    def _spatial_sort(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Robust spatial sort for bubbles (vertical columns + horizontal rows)."""
        if not items:
            return []

        # Tuned thresholds to keep slightly offset bubbles in the same line.
        y_overlap_ratio_threshold = 0.25
        y_center_band_factor = 0.5
        x_overlap_ratio_threshold = 0.2
        x_center_band_factor = 0.5

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
            y1, y2, h = e["y1"], e["y2"], e["h"]
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
                x1, x2, w = e["x1"], e["x2"], e["w"]
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

    # Macro layout: panel graph with root detection and dual veto for Z-flow.
    def _iou_x(boxA, boxB):
        xa1, _, xa2, _ = boxA
        xb1, _, xb2, _ = boxB
        inter = max(0, min(xa2, xb2) - max(xa1, xb1))
        union = (xa2 - xa1) + (xb2 - xb1) - inter
        return inter / union if union > 0 else 0

    def _iou_y_overlap(boxA, boxB):
        _, ya1, _, ya2 = boxA
        _, yb1, _, yb2 = boxB
        inter = max(0, min(ya2, yb2) - max(ya1, yb1))
        min_h = min(ya2 - ya1, yb2 - yb1)
        return inter / min_h if min_h > 0 else 0

    def sort_panels_strict(panels_list, rtl=True):
        if not panels_list:
            return []

        nodes = []
        for i, bbox in enumerate(panels_list):
            nodes.append(
                {
                    "id": i,
                    "bbox": bbox,
                    "center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    "visited": False,
                }
            )

        sorted_indices = []

        # Roots: panels with no panel above in the same column.
        root_nodes = []
        for n in nodes:
            is_root = True
            for parent in nodes:
                if n["id"] == parent["id"]:
                    continue
                is_above = parent["bbox"][3] <= (n["bbox"][1] + 50)
                x_overlap = _iou_x(parent["bbox"], n["bbox"])
                if is_above and x_overlap > 0.2:
                    is_root = False
                    break
            if is_root:
                root_nodes.append(n)

        if root_nodes:
            start_node = (
                max(root_nodes, key=lambda n: n["bbox"][2])
                if rtl
                else min(root_nodes, key=lambda n: n["bbox"][0])
            )
        else:
            start_node = min(nodes, key=lambda n: n["bbox"][1])

        current = start_node
        current["visited"] = True
        sorted_indices.append(current["id"])

        while len(sorted_indices) < len(nodes):
            c_box = current["bbox"]
            candidates = [n for n in nodes if not n["visited"]]
            if not candidates:
                break

            col_cand = None
            col_candidates = []
            for cand in candidates:
                cand_box = cand["bbox"]
                overlap = _iou_x(c_box, cand_box)
                is_below = cand_box[1] >= (c_box[1] + (c_box[3] - c_box[1]) * 0.5)
                if overlap > 0.2 and is_below:
                    dist_y = max(0, cand_box[1] - c_box[3])
                    col_candidates.append((dist_y, cand))

            if col_candidates:
                col_candidates.sort(
                    key=lambda x: (
                        int(x[0] / 50),
                        -x[1]["center"][0] if rtl else x[1]["center"][0],
                    )
                )
                col_cand = col_candidates[0][1]

            row_cand = None
            row_candidates = []
            for cand in candidates:
                cand_box = cand["bbox"]
                if rtl:
                    is_row_neighbor = cand_box[2] <= (c_box[0] + 50)
                    dist_x = c_box[0] - cand_box[2]
                else:
                    is_row_neighbor = cand_box[0] >= (c_box[2] - 50)
                    dist_x = cand_box[0] - c_box[2]

                if is_row_neighbor:
                    y_inter = max(
                        0, min(c_box[3], cand_box[3]) - max(c_box[1], cand_box[1])
                    )
                    if y_inter > 0:
                        row_candidates.append((dist_x, cand))

            if row_candidates:
                row_candidates.sort(key=lambda x: x[0])
                row_cand = row_candidates[0][1]

            # Dual veto: ceiling (topological) + right-neighbor (row start).
            if col_cand:
                is_blocked = False
                for other in candidates:
                    if other["id"] == col_cand["id"]:
                        continue
                    is_above = other["bbox"][3] <= (col_cand["bbox"][1] + 50)
                    x_overlap = _iou_x(other["bbox"], col_cand["bbox"])
                    if is_above and x_overlap > 0.2:
                        is_blocked = True
                        break
                    if rtl:
                        has_block_neighbor = other["bbox"][0] > (
                            col_cand["bbox"][0] + 20
                        )
                    else:
                        has_block_neighbor = other["bbox"][2] < (
                            col_cand["bbox"][2] - 20
                        )
                    y_overlap_ratio = _iou_y_overlap(col_cand["bbox"], other["bbox"])
                    if has_block_neighbor and y_overlap_ratio > 0.3:
                        is_blocked = True
                        break
                if is_blocked:
                    col_cand = None

            next_node = None
            if row_cand and not col_cand:
                next_node = row_cand
            elif col_cand and not row_cand:
                next_node = col_cand
            elif row_cand and col_cand:
                curr_h = c_box[3] - c_box[1]
                bottom_diff = abs(c_box[3] - row_cand["bbox"][3])
                is_row_aligned = bottom_diff < (curr_h * 0.25)
                next_node = row_cand if is_row_aligned else col_cand

            if not next_node:
                # Recompute roots among remaining nodes to find a new entry.
                sub_roots = []
                for n in candidates:
                    is_root = True
                    for parent in candidates:
                        if n["id"] == parent["id"]:
                            continue
                        is_above = parent["bbox"][3] <= (n["bbox"][1] + 50)
                        x_overlap = _iou_x(parent["bbox"], n["bbox"])
                        if is_above and x_overlap > 0.2:
                            is_root = False
                            break
                    if is_root:
                        sub_roots.append(n)

                if sub_roots:
                    next_node = (
                        max(sub_roots, key=lambda n: n["bbox"][2])
                        if rtl
                        else min(sub_roots, key=lambda n: n["bbox"][0])
                    )
                else:
                    next_node = min(candidates, key=lambda n: n["bbox"][1])

            current = next_node
            current["visited"] = True
            sorted_indices.append(current["id"])

        return sorted_indices

    if not panels:
        return _spatial_sort(detections)

    sorted_panel_indices = sort_panels_strict(panels, rtl)
    if not sorted_panel_indices:
        sorted_panel_indices = list(range(len(panels)))

    panel_bins = {pid: [] for pid in sorted_panel_indices}
    unassigned = []

    for detection in detections:
        bx1, by1, bx2, by2 = detection["bbox"]
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        assigned = False

        for i, pbbox in enumerate(panels):
            px1, py1, px2, py2 = pbbox
            if px1 <= bcx <= px2 and py1 <= bcy <= py2:
                panel_bins.setdefault(i, []).append(detection)
                detection["panel_id"] = i
                assigned = True
                break

        if not assigned:
            best_dist = float("inf")
            best_pid = -1
            for i, pbbox in enumerate(panels):
                px1, py1, px2, py2 = pbbox
                dx = max(px1 - bcx, 0, bcx - px2)
                dy = max(py1 - bcy, 0, bcy - py2)
                dist = (dx**2 + dy**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_pid = i

            if best_dist < 300:
                panel_bins.setdefault(best_pid, []).append(detection)
                detection["panel_id"] = best_pid
                assigned = True

        if not assigned:
            detection["panel_id"] = None
            unassigned.append(detection)

    final_order = []
    for pid in sorted_panel_indices:
        final_order.extend(_spatial_sort(panel_bins.get(pid, [])))

    if unassigned:
        final_order.extend(_spatial_sort(unassigned))

    return final_order
