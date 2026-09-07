"""Outside Speech Bubble (OSB) Text Tilt Detection.

Implements computer vision angle estimation and oriented bounding dimensions
extraction for outside-bubble text and sound effects in manga/comic images.
Follows the technical specification in tests/osb_tilt_design_spec.md.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

DEFAULT_ESTIMATION_MAX_SIDE: int = 256
MIN_INK_ELONGATION_RATIO: float = 1.25


def detect_crop_tilt_angle(
    crop_cv: np.ndarray,
    is_dark_text: bool = True,
    max_side: int = DEFAULT_ESTIMATION_MAX_SIDE,
    deadband_deg: float = 3.0,
    max_tilt_deg: float = 45.0,
    min_ink_elongation: float = MIN_INK_ELONGATION_RATIO,
) -> tuple[float, float, tuple[float, float], str]:
    """Estimate tilt angle and oriented dimensions from an original raw crop.

    Args:
        crop_cv: Raw BGR or grayscale image crop from original_bbox.
        is_dark_text: Prior indication of text polarity (True if ink is dark).
        max_side: Longest edge downscale cap for performance budget.
        deadband_deg: Threshold below which tilt snaps to 0.0 degrees.
        max_tilt_deg: Hard clamp limit on estimated angle.
        min_ink_elongation: Minimum ink elongation (r_pca / r_obb) required to detect tilt.

    Returns:
        (angle_deg, confidence, (obb_length, obb_thickness), tilt_axis)
        - angle_deg: Skia rotation angle in degrees.
          Horizontal: +angle slopes down-and-right, -angle slopes up-and-right.
          Vertical: +angle slopes down-and-left, -angle slopes down-and-right.
        - confidence: Float in [0.0, 1.0].
        - (obb_length, obb_thickness): Oriented dimensions in unscaled crop pixels.
        - tilt_axis: 'horizontal', 'vertical', or 'none'.
    """
    if crop_cv is None or crop_cv.size == 0:
        return 0.0, 0.0, (0.0, 0.0), "none"

    if len(crop_cv.shape) == 2:
        crop_cv = cv2.cvtColor(crop_cv, cv2.COLOR_GRAY2BGR)

    orig_h, orig_w = crop_cv.shape[:2]
    default_dims = (float(max(orig_w, orig_h)), float(min(orig_w, orig_h)))

    # Step 1.1: Size gate
    if (
        orig_h < 8
        or orig_w < 8
        or (orig_h < 16 and orig_w < 8)
        or (orig_w < 16 and orig_h < 8)
    ):
        return 0.0, 0.0, default_dims, "none"

    # Step 1.1: Downscale for perf budget
    max_dim = max(orig_h, orig_w)
    if max_dim > max_side:
        scale = float(max_side) / float(max_dim)
        scaled_w = max(1, round(orig_w * scale))
        scaled_h = max(1, round(orig_h * scale))
        crop = cv2.resize(crop_cv, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        crop = crop_cv.copy()

    h, w = crop.shape[:2]
    scale_inv = 1.0 / scale

    # Step 1.2: Background color & luminance sampling around border
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    border_pixels = crop[border_mask]
    bg_bgr = np.median(border_pixels, axis=0).astype(np.uint8)
    bg_lum = (
        0.299 * float(bg_bgr[2]) + 0.587 * float(bg_bgr[1]) + 0.114 * float(bg_bgr[0])
    )

    # Step 1.3: Screentone attenuation
    blurred = cv2.GaussianBlur(crop, (3, 3), 1.0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Step 1.4: Dual-polarity candidate mask competition
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    bg_lab = cv2.cvtColor(np.uint8([[bg_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(
        np.float32
    )
    delta_e = np.sqrt(np.sum((lab.astype(np.float32) - bg_lab) ** 2, axis=2))
    p95_de = float(np.percentile(delta_e, 95))

    if p95_de < 15.0:
        p22 = float(np.percentile(gray, 22))
        p78 = float(np.percentile(gray, 78))
        dark_raw = (gray <= min(135.0, p22)).astype(np.uint8) * 255
        bright_raw = (gray >= max(130.0, p78)).astype(np.uint8) * 255
    else:
        tau = max(25.0, 0.40 * p95_de)
        fg = delta_e >= tau
        dark_raw = (fg & (gray < bg_lum)).astype(np.uint8) * 255
        bright_raw = (fg & (gray >= bg_lum)).astype(np.uint8) * 255

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def clean_mask(m: np.ndarray) -> np.ndarray:
        opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k3)

    dark_mask = clean_mask(dark_raw)
    bright_mask = clean_mask(bright_raw)

    def score_mask(m: np.ndarray, is_prior: bool) -> tuple[float, np.ndarray]:
        ys, xs = np.where(m > 0)
        num_fg = len(xs)
        if num_fg < 15:
            return -1.0, m
        span_x = (np.max(xs) - np.min(xs) + 1) / float(w)
        span_y = (np.max(ys) - np.min(ys) + 1) / float(h)
        span_ratio = span_x * span_y
        box_area = (np.max(xs) - np.min(xs) + 1) * (np.max(ys) - np.min(ys) + 1)
        pixel_density = num_fg / float(box_area)
        row_consistency = len(np.unique(ys)) / float(h)
        edge_touch_count = np.count_nonzero(m[border_mask] > 0)
        edge_touch_penalty = edge_touch_count / float(num_fg)

        score = (span_ratio * pixel_density * row_consistency) - (
            1.5 * edge_touch_penalty
        )
        if is_prior:
            score *= 1.2
        return score, m

    score_dark, _ = score_mask(dark_mask, is_dark_text)
    score_bright, _ = score_mask(bright_mask, not is_dark_text)

    if score_dark <= 0.0 and score_bright <= 0.0:
        winning_mask = dark_mask if is_dark_text else bright_mask
    else:
        winning_mask = dark_mask if score_dark >= score_bright else bright_mask

    # Step 2: Component filtering & denoising
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        winning_mask, connectivity=8
    )
    crop_area = float(w * h)
    filtered_mask = np.zeros_like(winning_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]

        # Area gates
        if area < max(10, 0.00008 * crop_area) or area > (0.35 * crop_area):
            continue

        # Border touch gate
        touches = 0
        if cx == 0:
            touches += 1
        if cy == 0:
            touches += 1
        if cx + cw >= w:
            touches += 1
        if cy + ch >= h:
            touches += 1
        if touches >= 2:
            continue
        # Discard border-attached background framing (ceiling beams, wall pillars, panel frames)
        if touches >= 1:
            if (cx == 0 or cx + cw >= w) and (
                ch / float(h) > 0.30 or area > 0.12 * crop_area
            ):
                continue
            if (cy == 0 or cy + ch >= h) and (
                cw / float(w) > 0.30 or area > 0.12 * crop_area
            ):
                continue

        # Speed line filter: high aspect streak with small thickness
        aspect = max(cw, ch) / max(1.0, float(min(cw, ch)))
        thickness = min(cw, ch)
        if aspect > 20.0 and thickness < 2.5:
            continue

        filtered_mask[labels == i] = 255

    valid_pixels = np.count_nonzero(filtered_mask)
    if valid_pixels < 20 or valid_pixels < (0.005 * crop_area):
        return 0.0, 0.0, default_dims, "none"

    # Step 3: Aspect-Ratio Routing & Multi-Column CJK Guard
    ys, xs = np.where(filtered_mask > 0)
    pts = np.column_stack((xs, ys)).astype(np.float32)

    # PCA elongation check
    _, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
    ev1 = float(eigenvalues[0, 0])
    ev2 = float(eigenvalues[1, 0])
    r_pca = math.sqrt(ev1 / max(ev2, 1e-4))

    rect = cv2.minAreaRect(pts)
    _, (rect_w, rect_h), _ = rect
    r_obb = max(rect_w, rect_h) / max(1.0, min(rect_w, rect_h))
    obb_length = max(rect_w, rect_h) * scale_inv
    obb_thickness = min(rect_w, rect_h) * scale_inv
    oriented_dims = (float(obb_length), float(obb_thickness))

    if r_pca < min_ink_elongation and r_obb < min_ink_elongation:
        return 0.0, 0.0, oriented_dims, "none"

    aspect = float(h) / float(w)
    is_vertical = aspect > 1.3

    # Step 3.1: Vertical Path & Multi-Column Gutter Guard
    if is_vertical:
        tilt_axis = "vertical"
        col_counts = np.sum(filtered_mask > 0, axis=0).astype(np.float32)
        active_mask = col_counts > 0
        if not np.any(active_mask):
            return 0.0, 0.0, oriented_dims, "none"

        mean_active = float(np.mean(col_counts[active_mask]))
        threshold = max(3.0, 0.40 * mean_active)

        # Detect columns with morphological merge for small intra-column gaps
        col_binary = (col_counts > threshold).astype(np.uint8)
        gap_px = max(2, round(0.045 * w))
        k_col = np.ones((1, gap_px), dtype=np.uint8)
        col_merged = cv2.morphologyEx(
            col_binary.reshape(1, -1), cv2.MORPH_CLOSE, k_col
        )[0]

        # Extract contiguous column regions
        cols = []
        in_col = False
        start_x = 0
        for x_idx, val in enumerate(col_merged):
            if val and not in_col:
                in_col = True
                start_x = x_idx
            elif not val and in_col:
                in_col = False
                if (x_idx - start_x) >= max(3, int(0.05 * w)):
                    cols.append((start_x, x_idx))
        if in_col and (w - start_x) >= max(3, int(0.05 * w)):
            cols.append((start_x, w))

        # Check multi-column gutter guard on true parallel columns (each spanning >= 40% height)
        tall_cols = []
        for c_x1, c_x2 in cols:
            c_mask = (xs >= c_x1) & (xs < c_x2)
            c_xs = xs[c_mask]
            c_ys = ys[c_mask]
            if len(c_xs) >= 20 and (np.max(c_ys) - np.min(c_ys)) >= (0.40 * h):
                c_pts = np.column_stack((c_xs, c_ys)).astype(np.float32)
                _, c_evecs, _ = cv2.PCACompute2(c_pts, mean=None)
                cv1 = c_evecs[0]
                if cv1[1] < 0:
                    cv1 = -cv1
                col_angle = -math.degrees(math.atan2(cv1[0], cv1[1]))
                tall_cols.append((col_angle, len(c_xs)))

        if len(tall_cols) >= 2:
            spread = max(a for a, _ in tall_cols) - min(a for a, _ in tall_cols)
            avg_angle = float(
                np.average(
                    [a for a, _ in tall_cols], weights=[wt for _, wt in tall_cols]
                )
            )
            # If upright vertical columns agree within <= 7 deg and average is small:
            if spread <= 7.0 and abs(avg_angle) < 4.0:
                return 0.0, 1.0, oriented_dims, "vertical"
            if spread <= 7.0:
                # Consistent tilted columns
                clamped = max(-max_tilt_deg, min(max_tilt_deg, avg_angle))
                final_deg = 0.0 if abs(clamped) < deadband_deg else clamped
                return float(final_deg), 0.85, oriented_dims, "vertical"
            # Parallel columns disagree -> fail-safe 0.0
            return 0.0, 0.0, oriented_dims, "vertical"

        # Projection-profile optimization for vertical text deskewing
        # In Skia: negative angle rotates counter-clockwise (bottom to the right).
        # In OpenCV: positive angle rotates counter-clockwise, so passing theta deskews text at Skia angle theta.
        scores = []
        base_conc = 0.0
        c_0 = np.sum(filtered_mask > 0, axis=0).astype(np.float64)
        if np.sum(c_0) > 0:
            base_conc = np.sum(c_0**2) / (np.sum(c_0) ** 2)

        for s_deg in range(-int(max_tilt_deg), int(max_tilt_deg) + 1):
            m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(s_deg), 1.0)
            rot = cv2.warpAffine(filtered_mask, m, (w, h), flags=cv2.INTER_NEAREST)
            col_counts = np.sum(rot > 0, axis=0).astype(np.float64)
            total = np.sum(col_counts)
            if total <= 0:
                continue
            conc = np.sum(col_counts**2) / (total**2)
            scores.append((float(s_deg), conc))

        if scores and base_conc > 0:
            best_s, best_conc = max(scores, key=lambda sc: sc[1])
            gain = best_conc / max(base_conc, 1e-6)
            if gain >= 1.03 and abs(best_s) >= deadband_deg:
                conf = max(0.60, (gain - 1.0) / 0.12)
                return float(best_s), conf, oriented_dims, "vertical"

        # Single-column vertical path fallback using PCA / linear regression
        unique_y = np.unique(ys)
        if len(unique_y) >= 8:
            med_x = np.array([np.median(xs[ys == y_val]) for y_val in unique_y])
            slope = np.polyfit(unique_y, med_x, 1)[0]
            theta_row = -math.degrees(math.atan(slope))

            # PCA fallback
            v1 = eigenvectors[0]
            if v1[1] < 0:
                v1 = -v1
            theta_pca = (
                -math.degrees(math.atan2(v1[0], v1[1]))
                if abs(v1[1]) > 1e-4
                else theta_row
            )

            if abs(theta_row - theta_pca) <= 8.0:
                cand_angle = (theta_row + theta_pca) / 2.0
                clamped = max(-max_tilt_deg, min(max_tilt_deg, cand_angle))
                final_deg = 0.0 if abs(clamped) < deadband_deg else clamped
                return float(final_deg), 0.75, oriented_dims, "vertical"

        return 0.0, 0.0, oriented_dims, "vertical"

    # Step 3.2: Horizontal Path & Projection-Gain Verification
    tilt_axis = "horizontal"

    # Method A: Centroid Regression
    valid_comp = [
        i
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= max(8, int(crop_area * 0.0005))
    ]
    theta_a = None
    if len(valid_comp) >= 3:
        cxs = centroids[valid_comp, 0]
        cys = centroids[valid_comp, 1]
        weights = np.sqrt(stats[valid_comp, cv2.CC_STAT_AREA].astype(np.float32))
        try:
            slope_a = np.polyfit(cxs, cys, 1, w=weights)[0]
            theta_a = math.degrees(math.atan(slope_a))
        except Exception:
            theta_a = None

    # Method B: Column Medians
    unique_xs = np.unique(xs)
    theta_b = None
    if len(unique_xs) >= 8:
        med_ys = np.array([np.median(ys[xs == x_val]) for x_val in unique_xs])
        try:
            slope_b = np.polyfit(unique_xs, med_ys, 1)[0]
            theta_b = math.degrees(math.atan(slope_b))
        except Exception:
            theta_b = None

    # Method C: PCA major axis
    v1 = eigenvectors[0]
    if v1[0] < 0:
        v1 = -v1
    theta_c = math.degrees(math.atan2(v1[1], v1[0]))

    cands = [t for t in (theta_a, theta_b, theta_c) if t is not None]
    if not cands:
        return 0.0, 0.0, oriented_dims, "none"

    spread = max(cands) - min(cands)
    has_sign_conflict = (max(cands) > 4.0) and (min(cands) < -4.0)

    if spread <= 8.0:
        cand_angle = float(np.median(cands))
    elif theta_a is not None and (
        abs(theta_a - theta_c) <= 6.0
        or (theta_b is not None and abs(theta_a - theta_b) <= 6.0)
    ):
        cand_angle = float(theta_a)
    elif spread > 15.0 or has_sign_conflict:
        return 0.0, 0.0, oriented_dims, tilt_axis
    else:
        cand_angle = float(np.median(cands))

    if abs(cand_angle) < deadband_deg:
        return 0.0, 0.9, oriented_dims, tilt_axis

    # Projection-gain verification (geometric sign & angle proof)
    def calc_concentration(img_bin: np.ndarray) -> tuple[float, float]:
        row_counts = np.sum(img_bin > 0, axis=1).astype(np.float64)
        total = np.sum(row_counts)
        if total <= 0:
            return 0.0, 0.0
        c = float(np.sum(row_counts**2) / (total**2))
        peak_ratio = float(np.max(row_counts) / (np.mean(row_counts) + 1e-4))
        return c, peak_ratio

    c_orig, peak_orig = calc_concentration(filtered_mask)
    abs_cand = abs(cand_angle)

    # In OpenCV getRotationMatrix2D, positive angle rotates counter-clockwise.
    # To deskew a clockwise (+theta) tilt, rotate counter-clockwise (+abs_cand).
    # To deskew a counter-clockwise (-theta) tilt, rotate clockwise (-abs_cand).
    m_cw_deskew = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), abs_cand, 1.0)
    m_ccw_deskew = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -abs_cand, 1.0)

    rot_cw_deskew = cv2.warpAffine(
        filtered_mask, m_cw_deskew, (w, h), flags=cv2.INTER_NEAREST
    )
    rot_ccw_deskew = cv2.warpAffine(
        filtered_mask, m_ccw_deskew, (w, h), flags=cv2.INTER_NEAREST
    )

    c_cw, peak_cw = calc_concentration(rot_cw_deskew)
    c_ccw, peak_ccw = calc_concentration(rot_ccw_deskew)

    # For clockwise text (cand_angle > 0), the deskew rotation is counter-clockwise (m_cw_deskew, +abs_cand in cv2)
    # For counter-clockwise text (cand_angle < 0), the deskew rotation is clockwise (m_ccw_deskew, -abs_cand in cv2)
    if cand_angle >= 0:
        c_deskew, peak_deskew = c_cw, peak_cw
        c_opp = c_ccw
    else:
        c_deskew, peak_deskew = c_ccw, peak_ccw
        c_opp = c_cw

    # On strong sign conflict (> 1.3x higher concentration in opposite direction), fail safe to 0.0 per spec
    if c_opp > (c_deskew * 1.30) and abs(cand_angle) > 5.0:
        return 0.0, 0.0, oriented_dims, tilt_axis

    c_gain = c_deskew / max(c_orig, 1e-6)
    peak_gain = peak_deskew / max(peak_orig, 1e-6)

    accepted = (c_gain >= 1.05) or (peak_gain >= 1.03)
    if not accepted:
        return 0.0, 0.35, oriented_dims, tilt_axis

    final_angle = cand_angle

    if abs(final_angle) < deadband_deg:
        return 0.0, 0.9, oriented_dims, tilt_axis

    final_angle = max(-max_tilt_deg, min(max_tilt_deg, final_angle))
    gain_conf = min(1.0, max(0.5, (c_gain - 1.0) / 0.15))
    confidence = float(gain_conf)

    return float(final_angle), confidence, oriented_dims, tilt_axis


def compute_effective_tilt_angle(
    tilt_deg: float,
    tilt_conf: float,
    follow_tilt: bool = True,
    confidence_threshold: float = 0.5,
    min_tilt_deg: float = 3.0,
    max_tilt_deg: float = 20.0,
) -> float:
    """Compute clamped and deadband-filtered tilt angle for rendering/filling.

    Returns:
        float: Effective rotation angle in degrees (0.0 if disabled, unconfident, or below deadband).
    """
    if not follow_tilt:
        return 0.0
    if float(tilt_conf) < float(confidence_threshold):
        return 0.0
    clamped = max(-float(max_tilt_deg), min(float(max_tilt_deg), float(tilt_deg)))
    if abs(clamped) < float(min_tilt_deg):
        return 0.0
    return float(clamped)
