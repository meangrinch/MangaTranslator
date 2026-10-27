from collections.abc import Callable
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

Box = tuple[int, int, int, int]

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
# LaMa masks also take the glyphs the model finds inside each text box with the page at
# twice its text scale. Hollow and outlined SFX that the reference scale half-misses
# come out whole there, but outside text boxes that scale marks hair and hatching
UPSCALED_FACTOR = 2
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
# Text the OSB model missed: letters merge into one region when the gap between them
# is within this share of the larger one's letter size. Wider gaps reach art beside
# the text, and verifying merged regions as one lets that art through
TEXT_GROUP_GAP = 0.4
# Specks and regions smaller than these areas (pixels at the reference scale) are noise
MIN_LETTER_AREA = 20
MIN_TEXT_AREA = 150
# A letter belongs to a verified detection when this share of its box lies inside it
VERIFIED_LETTER_OVERLAP = 0.5


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


class UpscaledGlyphMap:
    """Text probability of a page at UPSCALED_FACTOR times the model's text scale.

    It is computed only in the windows that cover requested boxes, each at most once, so
    a box's values match those of the whole page at that scale.
    """

    def __init__(self, image: Image.Image):
        self.size = image.size
        width, height = image.size
        scale = UPSCALED_FACTOR * glyph_scale(image.size)
        scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        self._rgb = cv2.resize(
            np.asarray(image.convert("RGB")), scaled_size, interpolation=interpolation
        )
        rows, cols = self._rgb.shape[:2]
        self._total = np.zeros((rows, cols), np.float32)
        self._counts = np.zeros((rows, cols), np.float32)
        self._done: set[tuple[int, int]] = set()

    def box_probability(
        self, box: Box, predict: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Probability inside `box` (x0, y0, x1, y1), at the page's resolution."""
        x0, y0, x1, y1 = box
        width, height = self.size
        rows, cols = self._total.shape
        sx, sy = cols / width, rows / height
        # Scaled pixels that linear interpolation back to the box reads
        lo_x = max(0, int(np.floor((x0 + 0.5) * sx - 0.5)) - 1)
        hi_x = min(cols, int(np.ceil((x1 - 0.5) * sx - 0.5)) + 2)
        lo_y = max(0, int(np.floor((y0 + 0.5) * sy - 0.5)) - 1)
        hi_y = min(rows, int(np.ceil((y1 - 0.5) * sy - 0.5)) + 2)
        for y in _window_starts(rows):
            if y >= hi_y or y + WINDOW <= lo_y:
                continue
            for x in _window_starts(cols):
                if x >= hi_x or x + WINDOW <= lo_x or (y, x) in self._done:
                    continue
                window = self._rgb[y : y + WINDOW, x : x + WINDOW]
                h, w = window.shape[:2]
                self._total[y : y + h, x : x + w] += predict(window)
                self._counts[y : y + h, x : x + w] += 1
                self._done.add((y, x))
        prob = self._total[lo_y:hi_y, lo_x:hi_x] / self._counts[lo_y:hi_y, lo_x:hi_x]
        # Sampled like resizing the whole scaled map back to the page
        to_scaled = np.float32(
            [
                [sx, 0, (x0 + 0.5) * sx - 0.5 - lo_x],
                [0, sy, (y0 + 0.5) * sy - 0.5 - lo_y],
            ]
        )
        return cv2.warpAffine(
            prob,
            to_scaled,
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )


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


def refine_box_mask_to_glyphs(
    box_mask: np.ndarray,
    glyph_prob: np.ndarray,
    box: Box,
    box_prob: np.ndarray,
    radius: int,
) -> np.ndarray:
    """refine_mask_to_glyphs for a mask inside `box`, where `box_prob` (another pass's
    probability over the box) raises the page map. `glyph_prob` is not changed."""
    x0, y0, x1, y1 = box
    img_h, img_w = box_mask.shape
    # refine_mask_to_glyphs looks `radius` past each region, so this crop is exact
    cx0, cy0 = max(0, x0 - radius), max(0, y0 - radius)
    cx1, cy1 = min(img_w, x1 + radius), min(img_h, y1 + radius)
    prob = glyph_prob[cy0:cy1, cx0:cx1].copy()
    inside = prob[y0 - cy0 : y1 - cy0, x0 - cx0 : x1 - cx0]
    np.maximum(inside, box_prob, out=inside)
    refined = np.zeros_like(box_mask, dtype=bool)
    refined[cy0:cy1, cx0:cx1] = refine_mask_to_glyphs(
        box_mask[cy0:cy1, cx0:cx1], prob, radius
    )
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


class GlyphGroup(NamedTuple):
    """Bounds (x0, y0, x1, y1) of a text region and of each of its letters."""

    box: Box
    components: list[Box]


def _letter_size(box: Box) -> int:
    # Long strokes are measured by their width, so a line doesn't reach far letters
    w, h = box[2] - box[0], box[3] - box[1]
    return min(max(w, h), 2 * min(w, h) + 2)


def _gap(a: Box, b: Box) -> int:
    return max(max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]))


def group_glyphs(glyph: np.ndarray) -> list[GlyphGroup]:
    """Connected glyph pixels grouped into text regions by the gaps between them."""
    area_scale = (min(glyph.shape) / REFERENCE_SHORT_SIDE) ** 2
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(glyph, np.uint8), connectivity=8
    )
    letters = [
        (int(x), int(y), int(x + w), int(y + h), int(area))
        for x, y, w, h, area in stats[1:count]
        if area >= MIN_LETTER_AREA * area_scale
    ]
    letters.sort()
    sizes = [_letter_size(letter[:4]) for letter in letters]
    reach = TEXT_GROUP_GAP * max(sizes, default=0)
    parent = list(range(len(letters)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(letters):
        for j in range(i + 1, len(letters)):
            b = letters[j]
            # Sorted by left edge, so no later letter can be within reach
            if b[0] - a[2] > reach:
                break
            if _gap(a[:4], b[:4]) <= TEXT_GROUP_GAP * max(sizes[i], sizes[j]):
                parent[root(i)] = root(j)

    members: dict[int, list[int]] = {}
    for i in range(len(letters)):
        members.setdefault(root(i), []).append(i)
    groups = []
    for indices in members.values():
        if sum(letters[i][4] for i in indices) < MIN_TEXT_AREA * area_scale:
            continue
        boxes = [letters[i][:4] for i in indices]
        groups.append(GlyphGroup(_bounds(boxes), boxes))
    return groups


def _bounds(boxes: list[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _overlap_share(inner: Box, outer) -> float:
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return max(0, w) * max(0, h) / max(1, area)


def verified_glyph_box(group: GlyphGroup, detections: list) -> Box | None:
    """Bounds of the group's letters that lie inside a detection box, or None.

    Art merged into the group, such as a censor bar beside the letters, stays out.
    """
    kept = [
        box
        for box in group.components
        if any(_overlap_share(box, d) > VERIFIED_LETTER_OVERLAP for d in detections)
    ]
    return _bounds(kept) if kept else None
