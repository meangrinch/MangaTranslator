"""Glyph-level text masks from a text segmentation model (Koharu SAM-TS-L).

The model is injected as `predict(window_rgb) -> probability`, so this module only
handles page scaling, tiling, and turning probabilities into inpainting masks.
"""

from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image

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


def refine_mask_to_glyphs(
    box_mask: np.ndarray, glyph_prob: np.ndarray, radius: int
) -> np.ndarray:
    """Shrink each region of `box_mask` to its dilated glyphs.

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
        glyphs = (glyph_prob[y0:y1, x0:x1] > THRESHOLD).astype(np.uint8)
        glyphs = cv2.dilate(glyphs, kernel).astype(bool) & region
        refined[y0:y1, x0:x1] |= glyphs if glyphs.any() else region
    return refined
