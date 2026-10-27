from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

# SAM-TS-L was tuned on Manga109, whose single pages are ~830 px wide; pages are
# shrunk toward that scale but never enlarged. At 2x it gave crisper strokes but
# masked sweat marks and screentone as text, and LaMa results were no better
REFERENCE_SHORT_SIDE = 830
WINDOW = 1024
STRIDE = 768
# At the reference scale, 7 px covers white letter outlines and anti-aliasing
REFERENCE_DILATION = 7
MIN_DILATION = 2
THRESHOLD = 0.5
# Composites also take fainter glyph parts connected to confident ones: a missed
# outline tail would survive the paste, while extra coverage only costs a little art
COMPOSITE_LOW_THRESHOLD = 0.15
# Tone matching of edit-model redraws samples flat pixels (level range within the
# window) where the redraw is within the tolerance of the original, and smooths the
# offset over a scale in dilation radii
TONE_FLAT_RANGE = 12
TONE_FLAT_WINDOW = 7
TONE_MATCH_TOLERANCE = 24
TONE_MATCH_MIN_SAMPLES = 16
TONE_SIGMA_RADII = 3


def glyph_scale(size: tuple[int, int]) -> float:
    """Scale that brings a (width, height) page to the model's text size."""
    return min(1.0, REFERENCE_SHORT_SIDE / min(size))


def glyph_dilation_radius(size: tuple[int, int]) -> int:
    """Glyph mask dilation in page pixels for a (width, height) page."""
    return max(
        MIN_DILATION, round(REFERENCE_DILATION * min(size) / REFERENCE_SHORT_SIDE)
    )


def _window_starts(length: int) -> list[int]:
    if length <= WINDOW:
        return [0]
    starts = list(range(0, length - WINDOW + 1, STRIDE))
    if starts[-1] + WINDOW < length:
        starts.append(length - WINDOW)
    return starts


def page_glyph_probability(
    image: Image.Image, predict: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Per-pixel text probability for the whole page, at the page's resolution."""
    width, height = image.size
    rgb = np.asarray(image.convert("RGB"))
    scale = glyph_scale(image.size)
    if scale < 1.0:
        scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        rgb = cv2.resize(rgb, scaled_size, interpolation=cv2.INTER_AREA)

    rows, cols = rgb.shape[:2]
    total = np.zeros((rows, cols), np.float32)
    counts = np.zeros((rows, cols), np.float32)
    for y in _window_starts(rows):
        for x in _window_starts(cols):
            window = rgb[y : y + WINDOW, x : x + WINDOW]
            h, w = window.shape[:2]
            total[y : y + h, x : x + w] += predict(window)
            counts[y : y + h, x : x + w] += 1
    prob = total / counts
    if scale < 1.0:
        prob = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
    return prob


def _glyph_pixels(prob: np.ndarray, low_threshold: float | None) -> np.ndarray:
    """Pixels above THRESHOLD, plus weaker ones connected to them if `low_threshold`."""
    strong = prob > THRESHOLD
    if low_threshold is None or not strong.any():
        return strong
    _, labels = cv2.connectedComponents(
        (prob > low_threshold).astype(np.uint8), connectivity=8
    )
    return np.isin(labels, np.unique(labels[strong]))


def refine_mask_to_glyphs(
    box_mask: np.ndarray,
    glyph_prob: np.ndarray,
    radius: int,
    low_threshold: float | None = None,
) -> np.ndarray:
    """Shrink each region of `box_mask` to its dilated glyphs.

    With `low_threshold`, faint glyph parts connected to confident ones count too.
    A region where the model finds no text keeps its whole box, so text the model
    misses is still erased.
    """
    box_mask = np.asarray(box_mask, dtype=bool)
    img_h, img_w = box_mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        box_mask.astype(np.uint8), connectivity=8
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    refined = np.zeros_like(box_mask)
    for label in range(1, count):
        x, y, w, h = stats[label, :4]
        # Glyphs cut by the box edge still dilate into it, so look past the edge
        x0, y0 = max(0, x - radius), max(0, y - radius)
        x1, y1 = min(img_w, x + w + radius), min(img_h, y + h + radius)
        region = labels[y0:y1, x0:x1] == label
        glyphs = _glyph_pixels(glyph_prob[y0:y1, x0:x1], low_threshold)
        glyphs = glyphs.astype(np.uint8)
        glyphs = cv2.dilate(glyphs, kernel).astype(bool) & region
        refined[y0:y1, x0:x1] |= glyphs if glyphs.any() else region
    return refined


def _is_flat(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((TONE_FLAT_WINDOW, TONE_FLAT_WINDOW), np.uint8)
    spread = (cv2.dilate(image, kernel) - cv2.erode(image, kernel)).reshape(
        image.shape[0], image.shape[1], -1
    )
    return spread.max(axis=2) < TONE_FLAT_RANGE


def _match_tone(
    before: np.ndarray, after: np.ndarray, samples: np.ndarray, sigma: float
) -> np.ndarray:
    """`after` shifted by the smoothed before-after offset measured at `samples`.

    Edit models redraw the paper a few levels off, which glyph-shaped composites would
    show as ghosts on flat areas. Only flat pixels where the redraw agrees with the
    original are sampled, since redrawn lines and texture sit slightly off and would
    bias the offset.
    """
    flat = samples & _is_flat(before) & _is_flat(after)
    after = after.astype(np.float32)
    diff = before.astype(np.float32) - after
    weight = (flat & (np.abs(diff).max(axis=2) < TONE_MATCH_TOLERANCE)).astype(
        np.float32
    )
    if weight.sum() < TONE_MATCH_MIN_SAMPLES:
        return after
    global_offset = (diff * weight[..., None]).sum(axis=(0, 1)) / weight.sum()
    density = cv2.GaussianBlur(weight, (0, 0), sigma)
    local = cv2.GaussianBlur(diff * weight[..., None], (0, 0), sigma)
    local = local / np.maximum(density, 1e-6)[..., None]
    # Far from any sample the local estimate is noise, so ease to the global offset
    trust = np.clip(density / 0.05, 0.0, 1.0)[..., None]
    return after + trust * local + (1 - trust) * global_offset


def composite_glyphs(
    before: np.ndarray,
    after: np.ndarray,
    box_mask: np.ndarray,
    glyph_prob: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Paste the inpainted `after` over `before` only on each box region's glyphs.

    Inside a region, art away from the dilated glyphs keeps its original pixels. The
    glyph edge fades back to the original over half the dilation, and the redraw is
    tone-matched to the original art it replaces. A region where the model finds no
    text takes `after` across its whole box, as before glyph masks.
    """
    box_mask = np.asarray(box_mask, dtype=bool)
    refined = refine_mask_to_glyphs(
        box_mask, glyph_prob, radius, COMPOSITE_LOW_THRESHOLD
    )
    out = before.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        box_mask.astype(np.uint8), connectivity=8
    )
    feather = max(1, radius // 2)
    for label in range(1, count):
        x, y, w, h = stats[label, :4]
        crop = (slice(y, y + h), slice(x, x + w))
        region = labels[crop] == label
        glyphs = refined[crop] & region
        if (glyphs == region).all():
            alpha = region.astype(np.float32)
            redraw = after[crop].astype(np.float32)
        else:
            # Distance to the kept art inside the region, so glyphs cut by the box
            # edge stay fully redrawn there
            inside = distance_transform_edt(glyphs | ~region)
            alpha = np.clip(inside / feather, 0.0, 1.0) * glyphs
            redraw = _match_tone(
                before[crop], after[crop], region & ~glyphs, TONE_SIGMA_RADII * radius
            )
        a = alpha[..., None]
        blended = redraw * a + before[crop].astype(np.float32) * (1 - a)
        out[crop] = np.where(
            region[..., None], np.clip(blended, 0, 255).round(), out[crop]
        ).astype(out.dtype)
    return out
