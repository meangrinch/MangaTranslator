import base64
import math
import random
import re
import unicodedata
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from core.batch_coordinator import (
    expanded_mask_bbox,
    partition_non_overlapping_waves,
    paste_image_region,
)
from core.config import MangaTranslatorConfig
from core.image.glyph_mask import THRESHOLD as GLYPH_THRESHOLD
from core.image.glyph_mask import (
    composite_glyphs,
    glyph_dilation_radius,
    page_glyph_probability,
    refine_mask_to_glyphs,
)
from core.image.image_utils import (
    cv2_to_pil,
    pil_to_cv2,
    process_bubble_image_cached,
    resize_lanczos_antiring,
)
from core.image.inpainting import make_flux_inpainter
from core.image.ocr_detection import OutsideTextDetector, extract_text_with_manga_ocr
from core.image.tilt_detection import (
    compute_effective_tilt_angle,
    detect_crop_tilt_angle,
)
from core.ml.hi_sam import predict_text_probability
from core.ml.model_manager import get_model_manager
from utils.cancellation import raise_if_cancelled
from utils.exceptions import CancellationError
from utils.logging import log_message
from utils.model_metadata import flux_method_backends

# OSB Expansion and Classification Constants
OSB_EXPANSION_PIXEL_BUFFER = 5  # for bubbles, nearby OSB regions, panels
OSB_SOLID_RATIO_THRESHOLD = (
    0.95  # Stricter than interior sampling because border ring has no text mask noise
)
OSB_COLOR_TOLERANCE = 15
OSB_EXPANSION_PX = 2
OSB_WHITE_SNAP_THRESH = 245
OSB_BLACK_SNAP_THRESH = 10
# Share of strongly colored pixels that marks a color page; B/W scans with stray tints
# measured at most 1%, color pages 8% or more
OSB_COLOR_PAGE_MIN_FRACTION = 0.05


@dataclass
class _PageGlyphMap:
    """Text probability of the original page, computed on first use.

    Tilt detection, LaMa's glyph masks, and the edit models' glyph composites share it,
    so the text segmentation model runs at most once per page, and not at all on pages
    that need none of them.
    """

    image: Image.Image
    # LaMa reuses the text model on every page; the diffusion inpainters need its VRAM
    keep_model: bool
    verbose: bool = False
    prob: np.ndarray | None = None
    failed: bool = False

    @classmethod
    def for_page(
        cls, image: Image.Image, config: MangaTranslatorConfig, verbose: bool
    ) -> "_PageGlyphMap":
        return cls(
            image,
            keep_model=config.outside_text.inpainting_method == "lama_manga",
            verbose=verbose,
        )

    def get(self) -> np.ndarray | None:
        """The probability map, or None when the text model can't be loaded or run."""
        if self.prob is None and not self.failed:
            manager = get_model_manager()
            with manager.flux_inference_lock:
                try:
                    self._compute(manager)
                finally:
                    if not self.keep_model:
                        manager.unload_text_sam(verbose=self.verbose)
        return self.prob

    def _compute(self, manager) -> None:
        try:
            # No local reference to the model, so unloading it frees its memory
            self.prob = page_glyph_probability(
                self.image,
                partial(
                    predict_text_probability,
                    manager.load_text_sam(verbose=self.verbose),
                ),
            )
        except CancellationError:
            raise
        except Exception as e:
            self.failed = True
            log_message(
                f"Text segmentation unavailable ({e}); using heuristic tilt and "
                "whole text boxes for inpainting",
                always_print=True,
            )


def _composites_glyphs(config: MangaTranslatorConfig) -> bool:
    """Whether edit-model inpaints are pasted only over the letters."""
    return config.outside_text.inpainting_method not in (
        "lama_manga",
        "opencv",
        "none",
    ) and bool(config.outside_text.flux_preserve_art)


def _inpainting_uses_glyph_map(config: MangaTranslatorConfig) -> bool:
    return config.outside_text.inpainting_method == "lama_manga" or _composites_glyphs(
        config
    )


def _glyph_map_for_tilt(
    glyph_map: _PageGlyphMap, config: MangaTranslatorConfig
) -> _PageGlyphMap | None:
    """The page map if tilt detection should use it, else None for the ink heuristic.

    On the CPU the text model takes ~20 s per page, too slow to spend on tilt alone,
    so there tilt only shares the map that inpainting builds anyway.
    """
    if _inpainting_uses_glyph_map(config):
        return glyph_map
    device = config.device
    return None if device is None or device.type == "cpu" else glyph_map


@dataclass
class OutsideTextWork:
    """Prepared outside-text state: translation data ready, inpaint still pending."""

    pil_image: Image.Image
    config: MangaTranslatorConfig
    image_path: str | Path
    image_format: str | None
    verbose: bool
    outside_text_results: list
    raw_outside_text_results: list
    original_text_colors: dict
    total_bubble_mask: Any
    outside_detector: Any
    mask_groups: list
    img_w: int
    img_h: int
    mime_type: str
    cv2_ext: str
    outside_text_data: list[dict[str, Any]] = field(default_factory=list)
    is_color_page: bool = False
    glyph_map: _PageGlyphMap | None = None

    def __post_init__(self):
        if self.glyph_map is None:
            self.glyph_map = _PageGlyphMap.for_page(
                self.pil_image, self.config, self.verbose
            )


def _build_outside_text_data(
    pil_image: Image.Image,
    outside_text_results,
    raw_outside_text_results,
    original_text_colors,
    extracted_text_colors,
    none_skipped_clip_bboxes,
    mime_type: str,
    cv2_ext: str,
    config: MangaTranslatorConfig,
    verbose: bool,
    glyph_map: _PageGlyphMap | None,
) -> list[dict[str, Any]]:
    outside_text_data = []
    original_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    for expanded_res, raw_res in zip(
        outside_text_results, raw_outside_text_results, strict=False
    ):
        bbox_coords, conf = expanded_res
        raw_coords, _ = raw_res

        x1, y1, x2, y2 = [int(c) for c in bbox_coords]
        bbox_tuple = (x1, y1, x2, y2)
        raw_bbox_tuple = tuple(int(c) for c in raw_coords)
        rx1, ry1, rx2, ry2 = raw_bbox_tuple

        rx1 = max(0, min(original_cv_image.shape[1], rx1))
        ry1 = max(0, min(original_cv_image.shape[0], ry1))
        rx2 = max(0, min(original_cv_image.shape[1], rx2))
        ry2 = max(0, min(original_cv_image.shape[0], ry2))
        if rx2 > rx1 and ry2 > ry1:
            outside_text_image_cv = original_cv_image[ry1:ry2, rx1:rx2].copy()
        else:
            outside_text_image_cv = original_cv_image[y1:y2, x1:x2].copy()
        outside_text_image_pil = cv2_to_pil(outside_text_image_cv)
        original_crop_pil = cv2_to_pil(original_cv_image[y1:y2, x1:x2].copy())

        osb_upscale_method = (
            "none" if config.test_mode else config.translation.upscale_method
        )

        if osb_upscale_method == "model":
            model_manager = get_model_manager()
            upscale_model = model_manager.load_upscale(verbose=verbose)
            final_text_pil = process_bubble_image_cached(
                outside_text_image_pil,
                upscale_model,
                config.device,
                config.translation.osb_min_side_pixels,
                "min",
                "model",
                verbose,
            )
            model_manager.clear_cache()
        elif osb_upscale_method == "model_lite":
            model_manager = get_model_manager()
            upscale_model = model_manager.load_upscale_lite(verbose=verbose)
            final_text_pil = process_bubble_image_cached(
                outside_text_image_pil,
                upscale_model,
                config.device,
                config.translation.osb_min_side_pixels,
                "min",
                "model_lite",
                verbose,
            )
            model_manager.clear_cache()
        elif osb_upscale_method == "lanczos":
            w, h = outside_text_image_pil.size
            min_side = min(w, h)
            if min_side < config.translation.osb_min_side_pixels:
                scale_factor = config.translation.osb_min_side_pixels / min_side
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                resized_text = resize_lanczos_antiring(
                    outside_text_image_pil, (new_w, new_h)
                )
            else:
                resized_text = outside_text_image_pil
            final_text_pil = resized_text
        else:
            final_text_pil = outside_text_image_pil

        outside_text_image_cv = pil_to_cv2(final_text_pil)

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect_ratio = float(h) / float(w)

        needs_text_bg = False
        if none_skipped_clip_bboxes:
            bcx = (x1 + x2) / 2
            bcy = (y1 + y2) / 2
            for clip_bbox in none_skipped_clip_bboxes:
                cx1, cy1, cx2, cy2 = clip_bbox
                if cx1 <= bcx <= cx2 and cy1 <= bcy <= cy2:
                    needs_text_bg = True
                    break

        angle_deg, tilt_conf, obb_dims, tilt_axis = (
            0.0,
            0.0,
            (float(w), float(h)),
            "none",
        )
        osb_tilt_estimation_side_pixels = 256
        if config.outside_text.osb_follow_tilt and rx2 > rx1 and ry2 > ry1:
            glyph_prob = None if glyph_map is None else glyph_map.get()
            angle_deg, tilt_conf, obb_dims, tilt_axis = detect_crop_tilt_angle(
                crop_cv=original_cv_image[ry1:ry2, rx1:rx2],
                # original_text_colors flags a dark background, i.e. light ink
                is_dark_text=not original_text_colors.get(raw_bbox_tuple, False),
                max_side=osb_tilt_estimation_side_pixels,
                deadband_deg=config.outside_text.osb_min_tilt_deg,
                max_tilt_deg=config.outside_text.osb_max_tilt_deg,
                ink_mask=(
                    None
                    if glyph_prob is None
                    else glyph_prob[ry1:ry2, rx1:rx2] > GLYPH_THRESHOLD
                ),
            )

        try:
            is_success, buffer = cv2.imencode(cv2_ext, outside_text_image_cv)
            if is_success:
                image_b64 = base64.b64encode(buffer).decode("utf-8")
                outside_text_data.append(
                    {
                        "bbox": bbox_tuple,
                        "original_bbox": raw_bbox_tuple,
                        "confidence": conf,
                        "is_outside_text": True,
                        "image_b64": image_b64,
                        "mime_type": mime_type,
                        "is_dark_text": original_text_colors.get(raw_bbox_tuple, True),
                        "text_color_rgb": extracted_text_colors.get(raw_bbox_tuple),
                        "aspect_ratio": aspect_ratio,
                        "needs_text_background": needs_text_bg,
                        "original_crop_pil": original_crop_pil,
                        "tilt_deg": angle_deg,
                        "tilt_conf": tilt_conf,
                        "tilt_axis": tilt_axis,
                        "oriented_dimensions": obb_dims,
                    }
                )
        except Exception as e:
            log_message(
                f"Error encoding outside text bbox {(x1, y1, x2, y2)}: {e}",
                verbose=verbose,
            )

    return outside_text_data


def _apply_inpaint_render_metadata(
    outside_text_data: list[dict[str, Any]],
    extracted_text_colors: dict,
    none_skipped_clip_bboxes: set,
) -> None:
    """Update existing OSB entries in-place with render-only fields from inpainting."""
    for item in outside_text_data:
        raw_bbox_tuple = item.get("original_bbox")
        bbox = item.get("bbox")
        if raw_bbox_tuple is not None and raw_bbox_tuple in extracted_text_colors:
            item["text_color_rgb"] = extracted_text_colors[raw_bbox_tuple]
        elif item.get("text_color_rgb") is None and extracted_text_colors and bbox:
            x1, y1, x2, y2 = bbox
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            for key, color in extracted_text_colors.items():
                if key in (raw_bbox_tuple, bbox):
                    item["text_color_rgb"] = color
                    break
                if (
                    isinstance(key, tuple)
                    and len(key) == 4
                    and key[0] <= bcx <= key[2]
                    and key[1] <= bcy <= key[3]
                ):
                    item["text_color_rgb"] = color
                    break

        needs_text_bg = False
        if none_skipped_clip_bboxes and bbox is not None:
            x1, y1, x2, y2 = bbox
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            for clip_bbox in none_skipped_clip_bboxes:
                cx1, cy1, cx2, cy2 = clip_bbox
                if cx1 <= bcx <= cx2 and cy1 <= bcy <= cy2:
                    needs_text_bg = True
                    break
        item["needs_text_background"] = needs_text_bg


def _fill_mask(image: Image.Image, mask: np.ndarray, color) -> Image.Image:
    """Copy of `image` with `color` painted wherever `mask` is set."""
    filled = image.copy()
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return filled
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    region = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), mode="L")
    filled.paste(Image.new("RGB", (x1 - x0, y1 - y0), color), (x0, y0), mask=region)
    return filled


def _results_bounds(results: list, indices: list[int]) -> tuple[int, int, int, int]:
    """Union (x0, y0, x1, y1) of the boxes of `results[i]` for each index."""
    boxes = [results[i][0] for i in indices]
    return (
        int(min(box[0] for box in boxes)),
        int(min(box[1] for box in boxes)),
        int(max(box[2] for box in boxes)),
        int(max(box[3] for box in boxes)),
    )


def _border_color(pixels: np.ndarray) -> tuple[tuple[int, int, int], float]:
    """Median RGB of border pixels (snapped to pure white/black) and the share near it."""
    median = np.median(pixels, axis=0).astype(int)
    diffs = np.max(np.abs(pixels.astype(int) - median), axis=1)
    solid_ratio = float(np.mean(diffs <= OSB_COLOR_TOLERANCE))
    if np.all(median >= OSB_WHITE_SNAP_THRESH):
        color = (255, 255, 255)
    elif np.all(median <= OSB_BLACK_SNAP_THRESH):
        color = (0, 0, 0)
    else:
        color = (int(median[0]), int(median[1]), int(median[2]))
    return color, solid_ratio


class _Glyph(NamedTuple):
    """A solid component whose border mostly contrasts."""

    area: int
    ratio: float  # share of the border that contrasts
    on_frame: bool  # touches the crop edge, so it may be background
    x: int
    y: int
    mask: np.ndarray  # component pixels within its bounding box at (x, y)


def _outlined_glyph_components(
    crop_gray: np.ndarray,
    glyph_mask: np.ndarray,
    is_contrasting,
    min_core_ratio: float = 0.0,
) -> list[_Glyph]:
    """Each solid glyph in `glyph_mask` whose border mostly contrasts."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(glyph_mask)
    height, width = crop_gray.shape
    found = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 10:
            continue
        # Work inside the component's own box, padded for the 5x5 dilation; scanning
        # the whole region per component takes seconds on screentone
        x, y, w, h = (int(v) for v in stats[label, :4])
        y0, y1 = max(0, y - 3), min(height, y + h + 3)
        x0, x1 = max(0, x - 3), min(width, x + w + 3)
        comp = (labels[y0:y1, x0:x1] == label).astype(np.uint8) * 255
        core = np.count_nonzero(cv2.erode(comp, np.ones((2, 2), np.uint8)))
        if core == 0 or core / area < min_core_ratio:
            continue
        dilated = cv2.dilate(comp, np.ones((5, 5), np.uint8))
        boundary = cv2.bitwise_and(dilated, cv2.bitwise_not(comp))
        border = crop_gray[y0:y1, x0:x1][boundary == 255]
        if len(border) == 0:
            continue
        ratio = np.count_nonzero(is_contrasting(border)) / len(border)
        if ratio >= 0.40:
            on_frame = x == 0 or y == 0 or x + w >= width or y + h >= height
            mask = labels[y : y + h, x : x + w] == label
            found.append(_Glyph(area, ratio, on_frame, x, y, mask))
    return found


def _is_page_number(text: str) -> bool:
    """Match digits, "Page 20", "p. 20", "- 20 -", etc."""
    # manga-ocr returns full-width characters; NFKC folds them to ASCII
    text = unicodedata.normalize("NFKC", text)
    return bool(
        re.match(
            r"^\s*[-–—―ー]?\s*(?:page\.?|p\.?)?\s*\d+\s*[-–—―ー]?\s*$",
            text,
            re.IGNORECASE,
        )
    )


def _drop_small_osb_regions(
    outside_text_results: list, min_ignore: float, img_w: int, img_h: int, verbose: bool
) -> list:
    """Drop OSB detections smaller than `min_ignore` of the image area."""
    image_area = float(img_w * img_h)
    kept_results = []
    removed_count = 0
    for bbox, conf in outside_text_results:
        x1, y1, x2, y2 = bbox
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, image_area)
        if area_ratio < min_ignore:
            removed_count += 1
            log_message(
                f"Skipping small OSB region (area {area_ratio:.4%} < {min_ignore:.4%})",
                verbose=verbose,
            )
            continue
        kept_results.append((bbox, conf))
    outside_text_results = kept_results
    if removed_count:
        log_message(
            f"Filtered {removed_count} OSB region(s) below min area threshold "
            f"({min_ignore:.2%} of image); {len(outside_text_results)} remaining",
            verbose=verbose,
        )
    return outside_text_results


def _drop_page_numbers(
    pil_image: Image.Image,
    outside_text_results: list,
    config: MangaTranslatorConfig,
    img_w: int,
    img_h: int,
    verbose: bool,
) -> list:
    """Drop OSB detections that OCR as page numbers.

    Only small detections in the top/bottom margin are OCR'd."""
    suspicious_crops = []
    suspicious_indices = []
    safe_results = []

    margin_threshold = max(
        0.0, min(0.3, config.outside_text.page_filter_margin_threshold)
    )
    min_area_threshold = max(
        0.0, min(0.2, config.outside_text.page_filter_min_area_ratio)
    )

    for i, res in enumerate(outside_text_results):
        bbox, _ = res
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cy = (y1 + y2) / 2

        is_in_margin = (cy < img_h * margin_threshold) or (
            cy > img_h * (1 - margin_threshold)
        )

        area = (x2 - x1) * (y2 - y1)
        is_small = area < (img_w * img_h * min_area_threshold)

        if is_in_margin and is_small:
            suspicious_crops.append(pil_image.crop((x1, y1, x2, y2)))
            suspicious_indices.append(i)
        else:
            safe_results.append(res)

    if suspicious_crops:
        log_message(
            f"Verifying {len(suspicious_crops)} suspicious OSB regions with OCR...",
            verbose=verbose,
        )
        suspicious_texts = extract_text_with_manga_ocr(
            suspicious_crops, verbose=verbose
        )

        for i, text in enumerate(suspicious_texts):
            if not _is_page_number(text):
                safe_results.append(outside_text_results[suspicious_indices[i]])
            else:
                log_message(f"Filtered out page number: '{text}'", verbose=verbose)

        outside_text_results = safe_results
        log_message(
            f"Remaining OSB regions after filtering: {len(outside_text_results)}",
            verbose=verbose,
        )
    return outside_text_results


def _expand_osb_bboxes(
    outside_text_results: list,
    config: MangaTranslatorConfig,
    img_w: int,
    img_h: int,
    bubble_data: list[dict[str, Any]] | None,
    panels: list[tuple[int, int, int, int]] | None,
    verbose: bool,
) -> list:
    """Grow narrow/tall and tiny OSB boxes (which render too small) within their
    panel, without overlapping bubbles or other OSB boxes."""
    narrow_expansion_mult = getattr(
        config.outside_text, "osb_render_expansion_narrow_multiplier", 1.0
    )
    tiny_expansion_mult = getattr(
        config.outside_text, "osb_render_expansion_tiny_multiplier", 1.0
    )
    aspect_ratio_threshold = getattr(
        config.outside_text, "osb_render_expansion_aspect_ratio_threshold", 0.4
    )
    area_ratio_threshold = getattr(
        config.outside_text, "osb_render_expansion_area_ratio_threshold", 0.005
    )
    if max(narrow_expansion_mult, tiny_expansion_mult) <= 1.0:
        return outside_text_results
    log_message(
        "Expanding OSB bboxes "
        f"(narrow/tall: {narrow_expansion_mult}x, tiny: {tiny_expansion_mult}x)...",
        verbose=verbose,
    )
    expanded_results = []
    for i, res in enumerate(outside_text_results):
        bbox, conf = res
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        aspect_ratio = float(w) / float(max(1, h))
        area_ratio = (w * h) / float(max(1, img_w * img_h))
        is_narrow_tall = aspect_ratio <= aspect_ratio_threshold
        is_tiny = area_ratio < area_ratio_threshold

        expansion_mult = 1.0
        if is_narrow_tall:
            expansion_mult = max(expansion_mult, narrow_expansion_mult)
        if is_tiny:
            expansion_mult = max(expansion_mult, tiny_expansion_mult)

        if expansion_mult <= 1.0:
            expanded_results.append(([int(x1), int(y1), int(x2), int(y2)], conf))
            continue

        cx = x1 + w / 2
        cy = y1 + h / 2

        new_w = w * expansion_mult
        new_h = h * expansion_mult

        nx1 = int(cx - new_w / 2)
        ny1 = int(cy - new_h / 2)
        nx2 = int(cx + new_w / 2)
        ny2 = int(cy + new_h / 2)

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(img_w, nx2)
        ny2 = min(img_h, ny2)

        if panels:
            associated_panel = None
            for p in panels:
                px1, py1, px2, py2 = p
                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    associated_panel = p
                    break
            if associated_panel:
                px1, py1, px2, py2 = associated_panel
                panel_buffer = OSB_EXPANSION_PIXEL_BUFFER
                c_px1 = min(int(px1) + panel_buffer, int(px2))
                c_py1 = min(int(py1) + panel_buffer, int(py2))
                c_px2 = max(int(px2) - panel_buffer, int(px1))
                c_py2 = max(int(py2) - panel_buffer, int(py1))

                nx1 = max(c_px1, nx1)
                ny1 = max(c_py1, ny1)
                nx2 = min(c_px2, nx2)
                ny2 = min(c_py2, ny2)

        buffer = OSB_EXPANSION_PIXEL_BUFFER
        obstacles = []
        if bubble_data:
            for b in bubble_data:
                bb = b.get("bbox")
                if bb and len(bb) == 4:
                    bx1, by1, bx2, by2 = [int(c) for c in bb]
                    obstacles.append(
                        (
                            max(0, bx1 - buffer),
                            max(0, by1 - buffer),
                            min(img_w, bx2 + buffer),
                            min(img_h, by2 + buffer),
                        )
                    )

        for j, other_res in enumerate(outside_text_results):
            if i == j:
                continue

            if j < i:
                ob, _ = expanded_results[j]
            else:
                ob, _ = other_res
            obx1, oby1, obx2, oby2 = [int(c) for c in ob]

            obstacles.append(
                (
                    max(0, obx1 - buffer),
                    max(0, oby1 - buffer),
                    min(img_w, obx2 + buffer),
                    min(img_h, oby2 + buffer),
                )
            )

        for ox1, oy1, ox2, oy2 in obstacles:
            if not (nx2 <= ox1 or nx1 >= ox2 or ny2 <= oy1 or ny1 >= oy2):
                can_retract_nx2 = (nx2 - ox1) if (ox1 >= x2) else float("inf")
                can_retract_nx1 = (ox2 - nx1) if (ox2 <= x1) else float("inf")
                can_retract_ny2 = (ny2 - oy1) if (oy1 >= y2) else float("inf")
                can_retract_ny1 = (oy2 - ny1) if (oy2 <= y1) else float("inf")

                min_retract = min(
                    can_retract_nx2,
                    can_retract_nx1,
                    can_retract_ny2,
                    can_retract_ny1,
                )
                if min_retract != float("inf"):
                    if min_retract == can_retract_nx2:
                        nx2 = ox1
                    elif min_retract == can_retract_nx1:
                        nx1 = ox2
                    elif min_retract == can_retract_ny2:
                        ny2 = oy1
                    elif min_retract == can_retract_ny1:
                        ny1 = oy2

        nx1 = min(nx1, int(x1))
        ny1 = min(ny1, int(y1))
        nx2 = max(nx2, int(x2))
        ny2 = max(ny2, int(y2))

        expanded_results.append(([nx1, ny1, nx2, ny2], conf))

    outside_text_results = expanded_results
    return outside_text_results


def _build_total_bubble_mask(
    bubble_data: list[dict[str, Any]] | None, img_w: int, img_h: int, verbose: bool
) -> np.ndarray:
    """Mask of all detected speech bubbles (dilated), so OSB inpainting avoids them."""
    total_bubble_mask = np.zeros((img_h, img_w), dtype=bool)
    if bubble_data:
        for bubble in bubble_data:
            try:
                mask = bubble.get("sam_mask") if isinstance(bubble, dict) else None
                if mask is not None:
                    mask_np = np.asarray(mask)
                    if mask_np.ndim == 3:
                        mask_np = mask_np[..., 0]
                    mask_bool = mask_np > 0
                    if mask_bool.shape[0] == img_h and mask_bool.shape[1] == img_w:
                        total_bubble_mask |= mask_bool
                        continue

                bbox = bubble.get("bbox") if isinstance(bubble, dict) else None
                if bbox and len(bbox) == 4:
                    x0, y0, x1, y1 = [int(c) for c in bbox]
                    x0 = max(0, min(img_w, x0))
                    x1 = max(0, min(img_w, x1))
                    y0 = max(0, min(img_h, y0))
                    y1 = max(0, min(img_h, y1))
                    if x1 > x0 and y1 > y0:
                        total_bubble_mask[y0:y1, x0:x1] = True
            except Exception as e:
                log_message(
                    f"Warning: Failed to apply bubble mask for OSB exclusion: {e}",
                    verbose=verbose,
                )

        if np.any(total_bubble_mask):
            # Dilate the bubble mask to provide a safe buffer for OSB fill
            kernel = np.ones((11, 11), np.uint8)
            total_bubble_mask = cv2.dilate(
                total_bubble_mask.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
    return total_bubble_mask


def _probe_original_text_colors(
    pil_image: Image.Image, raw_outside_text_results: list, verbose: bool
) -> dict:
    """Map each OSB box to whether its background is dark (via 2-color K-Means)."""
    original_text_colors = {}
    for ocr_result in raw_outside_text_results:
        bbox_coords, _ = ocr_result
        x1, y1, x2, y2 = [int(c) for c in bbox_coords]
        bbox_tuple = (x1, y1, x2, y2)

        bbox_area_img = pil_image.crop((x1, y1, x2, y2))
        bbox_array = np.array(bbox_area_img)

        if bbox_array.shape[-1] == 4:
            bbox_array = bbox_array[..., :3]

        pixels = bbox_array.reshape(-1, 3)
        # Every 4th pixel is plenty to find two colors, at a tenth of the cost
        if len(pixels) >= 8:
            pixels = pixels[::4]

        # Use K-Means to find 2 dominant colors
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=1)
        kmeans.fit(pixels)

        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        unique, counts = np.unique(labels, return_counts=True)
        dominant_cluster_idx = unique[np.argmax(counts)]

        # Dominant cluster is usually the background (text pixels are sparse)
        bg_color_rgb = centers[dominant_cluster_idx]
        # Use proper luminance calculation (ITU-R BT.601)
        bg_brightness = (
            0.299 * bg_color_rgb[0] + 0.587 * bg_color_rgb[1] + 0.114 * bg_color_rgb[2]
        )
        is_dark_text = (
            bg_brightness < 128
        )  # passed downstream; renderer inverts for text color
        original_text_colors[bbox_tuple] = is_dark_text

        log_message(
            f"OSB bbox {bbox_tuple}: "
            f"{'Dark' if is_dark_text else 'Light'} background detected "
            f"(luminance={bg_brightness:.1f})",
            verbose=verbose,
        )
    return original_text_colors


def prepare_outside_text_work(
    pil_image: Image.Image,
    config: MangaTranslatorConfig,
    image_path: str | Path,
    image_format: str | None,
    verbose: bool = False,
    bubble_data: list[dict[str, Any]] | None = None,
    text_free_boxes: list[list[float]] | None = None,
    panels: list[tuple[int, int, int, int]] | None = None,
) -> OutsideTextWork | None:
    """Detect outside text and build translation crops without inpainting."""
    if not config.outside_text.enabled:
        return None

    log_message("Detecting text outside speech bubbles...", verbose=verbose)

    try:
        outside_detector = OutsideTextDetector(
            device=config.device, hf_token=config.outside_text.huggingface_token
        )
        outside_text_results = outside_detector.detect_outside_text(
            str(image_path),
            yolo_model_path=config.yolo_model_path,
            confidence=config.outside_text.osb_confidence,
            conjoined_confidence=config.detection.conjoined_confidence,
            verbose=verbose,
            image_override=pil_image,
            existing_bubbles=bubble_data,
            text_free_boxes=text_free_boxes,
            bubble_detector_model=config.detection.bubble_detector_model,
            min_area_ignore_ratio=config.outside_text.min_area_ignore_ratio,
            text_free_only=config.outside_text.osb_text_free_only,
        )

        if not outside_text_results:
            log_message("No outside text regions found", verbose=verbose)
            return None

        img_w, img_h = pil_image.size

        min_ignore = max(0.0, min(0.05, config.outside_text.min_area_ignore_ratio))
        if min_ignore > 0.0:
            outside_text_results = _drop_small_osb_regions(
                outside_text_results, min_ignore, img_w, img_h, verbose
            )
            if not outside_text_results:
                log_message(
                    "No outside text regions remaining after min area filter",
                    verbose=verbose,
                )
                return None

        # Filter out probable page numbers
        if config.outside_text.enable_page_number_filtering and outside_text_results:
            outside_text_results = _drop_page_numbers(
                pil_image, outside_text_results, config, img_w, img_h, verbose
            )

        raw_outside_text_results = outside_text_results.copy()

        outside_text_results = _expand_osb_bboxes(
            outside_text_results, config, img_w, img_h, bubble_data, panels, verbose
        )
        total_bubble_mask = _build_total_bubble_mask(bubble_data, img_w, img_h, verbose)

        mime_type = (
            "image/png"
            if image_format and image_format.upper() == "PNG"
            else "image/jpeg"
        )
        cv2_ext = ".png" if image_format and image_format.upper() == "PNG" else ".jpg"

        original_text_colors = _probe_original_text_colors(
            pil_image, raw_outside_text_results, verbose
        )

        mask_groups, _ = outside_detector.get_text_masks(
            str(image_path),
            bbox_expansion_percent_width=config.outside_text.bbox_expansion_percent_width,
            bbox_expansion_percent_height=config.outside_text.bbox_expansion_percent_height,
            text_box_proximity_ratio=config.outside_text.text_box_proximity_ratio,
            verbose=verbose,
            image_override=pil_image,
            existing_results=raw_outside_text_results,
        )

        glyph_map = _PageGlyphMap.for_page(pil_image, config, verbose)
        outside_text_data = _build_outside_text_data(
            pil_image=pil_image,
            outside_text_results=outside_text_results,
            raw_outside_text_results=raw_outside_text_results,
            original_text_colors=original_text_colors,
            extracted_text_colors={},
            none_skipped_clip_bboxes=set(),
            mime_type=mime_type,
            cv2_ext=cv2_ext,
            config=config,
            verbose=verbose,
            glyph_map=_glyph_map_for_tilt(glyph_map, config),
        )

        return OutsideTextWork(
            pil_image=pil_image,
            config=config,
            image_path=image_path,
            image_format=image_format,
            verbose=verbose,
            outside_text_results=outside_text_results,
            raw_outside_text_results=raw_outside_text_results,
            original_text_colors=original_text_colors,
            total_bubble_mask=total_bubble_mask,
            outside_detector=outside_detector,
            mask_groups=mask_groups,
            img_w=img_w,
            img_h=img_h,
            mime_type=mime_type,
            cv2_ext=cv2_ext,
            outside_text_data=outside_text_data,
            is_color_page=_is_color_page(pil_image),
            glyph_map=glyph_map,
        )

    except Exception as e:
        log_message(
            f"Error during outside text detection: {e}",
            always_print=True,
        )
        return None


def _build_osb_simple_fill_mask(
    img_w: int,
    img_h: int,
    mask_indices: list[int],
    outside_text_results: list,
    outside_text_data: list[dict[str, Any]],
    config: MangaTranslatorConfig,
    fallback_bounds: tuple[int | None, int | None, int | None, int | None]
    | None = None,
    total_bubble_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build a 2D boolean mask for solid-fill inpainting, respecting detected OSB text tilt."""
    rect_mask = np.zeros((img_h, img_w), dtype=bool)
    has_drawn = False

    if mask_indices and outside_text_results:
        for idx in mask_indices:
            if idx >= len(outside_text_results):
                continue
            coords = outside_text_results[idx][0]
            if (
                isinstance(coords, (list, tuple))
                and len(coords) == 1
                and isinstance(coords[0], (list, tuple))
            ):
                coords = coords[0]
            bx0 = max(0, int(coords[0]))
            by0 = max(0, int(coords[1]))
            bx1 = min(img_w, int(coords[2]))
            by1 = min(img_h, int(coords[3]))
            if bx1 <= bx0 or by1 <= by0:
                continue

            rot_deg = 0.0
            if idx < len(outside_text_data):
                item = outside_text_data[idx]
                rot_deg = compute_effective_tilt_angle(
                    tilt_deg=item.get("tilt_deg", 0.0),
                    tilt_conf=item.get("tilt_conf", 0.0),
                    follow_tilt=config.outside_text.osb_follow_tilt,
                    confidence_threshold=config.outside_text.osb_tilt_confidence_threshold,
                    min_tilt_deg=config.outside_text.osb_min_tilt_deg,
                    max_tilt_deg=config.outside_text.osb_max_tilt_deg,
                )

            if abs(rot_deg) > 0.01:
                bcx = (bx0 + bx1) / 2.0
                bcy = (by0 + by1) / 2.0
                bw = float(bx1 - bx0)
                bh = float(by1 - by0)
                shrink = max(0.70, math.cos(math.radians(rot_deg)))
                hw = (bw * shrink) / 2.0
                hh = (bh * shrink) / 2.0
                rad = math.radians(rot_deg)
                cos_t = math.cos(rad)
                sin_t = math.sin(rad)

                corners = np.array(
                    [
                        [bcx - hw * cos_t + hh * sin_t, bcy - hw * sin_t - hh * cos_t],
                        [bcx + hw * cos_t + hh * sin_t, bcy + hw * sin_t - hh * cos_t],
                        [bcx + hw * cos_t - hh * sin_t, bcy + hw * sin_t + hh * cos_t],
                        [bcx - hw * cos_t - hh * sin_t, bcy - hw * sin_t + hh * cos_t],
                    ],
                    dtype=np.int32,
                )

                mask_uint8 = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(mask_uint8, [corners], 255)
                rect_mask |= mask_uint8 > 0
                has_drawn = True
            else:
                rect_mask[by0:by1, bx0:bx1] = True
                has_drawn = True

    elif fallback_bounds:
        b_ox0, b_oy0, b_ox1, b_oy1 = fallback_bounds
        if (
            b_ox0 is not None
            and b_oy0 is not None
            and b_ox1 is not None
            and b_oy1 is not None
            and b_ox1 > b_ox0
            and b_oy1 > b_oy0
        ):
            rect_mask[b_oy0:b_oy1, b_ox0:b_ox1] = True
            has_drawn = True

    if not has_drawn:
        return None

    if total_bubble_mask is not None:
        rect_mask = np.logical_and(rect_mask, np.logical_not(total_bubble_mask))

    return rect_mask if np.any(rect_mask) else None


def _create_osb_inpainter(config: MangaTranslatorConfig, verbose: bool):
    """Create the inpainter for the configured OSB method, or None for OpenCV/none."""
    inpainting_method = config.outside_text.inpainting_method
    inpainter = None
    model_labels = {
        "flux_klein_9b": "Flux.2 Klein 9B",
        "flux_klein_4b": "Flux.2 Klein 4B",
        "flux_kontext": "Flux.1 Kontext",
        "flux3_klein": "Flux.3 Image Klein",
        "fibo2_edit_turbo": "Fibo 2.0 Edit Turbo",
        "qwen_image_2_1_turbo": "Qwen-Image 2.1 Turbo",
        "lama_manga": "LaMa Manga",
    }
    if inpainting_method in model_labels:
        osb = config.outside_text
        label = model_labels[inpainting_method]
        try:
            inpainter = make_flux_inpainter(
                inpainting_method,
                device=config.device,
                huggingface_token=osb.huggingface_token,
                num_inference_steps=osb.flux_num_inference_steps,
                backend=osb.flux_backend,
                low_vram=osb.flux_low_vram,
                residual_diff_threshold=osb.flux_residual_diff_threshold,
                luminance_correction=osb.flux_luminance_correction,
                upscale_small_crops=osb.flux_upscale_small_crops,
                sdcpp_cache_mode=osb.flux_sdcpp_cache_mode,
                sdcpp_diffusion_quant=osb.flux_sdcpp_diffusion_quant,
                sdcpp_text_encoder_quant=osb.flux_sdcpp_text_encoder_quant,
                verbose=verbose,
            )
            if flux_method_backends(inpainting_method):
                backend_label = {
                    "sdnq": "SDNQ",
                    "sdcpp": "sd.cpp",
                    "nunchaku": "Nunchaku",
                }.get(osb.flux_backend, osb.flux_backend)
                label = f"{label} ({backend_label})"
            log_message(f"Using {label} for inpainting", verbose=verbose)
        except Exception as e:
            log_message(
                f"{label} unavailable ({e}), falling back to OpenCV",
                verbose=verbose,
            )
    if inpainting_method == "none":
        inpainter = None
        log_message(
            "Using text background mode (no inpainting for non-solid regions)",
            verbose=verbose,
        )
    elif inpainting_method == "opencv" or inpainter is None:
        inpainter = None
        log_message("Using OpenCV simple fill for inpainting", verbose=verbose)
    return inpainter


def _border_pixels(
    image: Image.Image, bounds: tuple[int, int, int, int], img_w: int, img_h: int
) -> np.ndarray | None:
    """RGB pixels in an OSB_EXPANSION_PX ring around `bounds`, or None if off-image."""
    x0, y0, x1, y1 = bounds
    sx1 = max(0, x0 - OSB_EXPANSION_PX)
    sy1 = max(0, y0 - OSB_EXPANSION_PX)
    sx2 = min(img_w, x1 + OSB_EXPANSION_PX)
    sy2 = min(img_h, y1 + OSB_EXPANSION_PX)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    crop_np = np.array(image.crop((sx1, sy1, sx2, sy2)).convert("RGB"))
    ring = np.ones(crop_np.shape[:2], dtype=bool)
    ix1 = max(0, x0 - sx1)
    iy1 = max(0, y0 - sy1)
    ix2 = min(crop_np.shape[1], x1 - sx1)
    iy2 = min(crop_np.shape[0], y1 - sy1)
    if ix2 > ix1 and iy2 > iy1:
        ring[iy1:iy2, ix1:ix2] = False
    return crop_np[ring]


def _is_color_page(image: Image.Image) -> bool:
    """Whether enough of the page is strongly colored to use the color-page text detector."""
    hsv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2HSV)
    colorful = (hsv[..., 1] > 60) & (hsv[..., 2] > 40)
    return bool(colorful.mean() >= OSB_COLOR_PAGE_MIN_FRACTION)


def _has_glyph_evidence(glyphs: list[_Glyph]) -> bool:
    total = sum(g.area for g in glyphs)
    return len(glyphs) >= 2 or (len(glyphs) == 1 and total >= 30) or total >= 50


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m8)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def _overlap(a: np.ndarray, ax: int, ay: int, b: np.ndarray, bx: int, by: int) -> int:
    """Pixels set in both masks, each placed with its top-left corner at (x, y)."""
    x0, y0 = max(ax, bx), max(ay, by)
    x1 = min(ax + a.shape[1], bx + b.shape[1])
    y1 = min(ay + a.shape[0], by + b.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(
        np.count_nonzero(
            a[y0 - ay : y1 - ay, x0 - ax : x1 - ax]
            & b[y0 - by : y1 - by, x0 - bx : x1 - bx]
        )
    )


def _drop_containers(own: list[_Glyph], other: list[_Glyph]) -> list[_Glyph]:
    """Drop glyphs whose interior holds several glyphs and most of the other polarity's area.

    A caption frame or bubble interior is not a glyph. Glyphs that enclose the candidate
    (the frame around an interior) are its border, not its content, so their area is left
    out of the other polarity's total."""
    if not own or not other:
        return own
    total = sum(o.area for o in other)
    kept = []
    for glyph in own:
        h, w = glyph.mask.shape
        nearby = [
            o
            for o in other
            if o.x < glyph.x + w
            and glyph.x < o.x + o.mask.shape[1]
            and o.y < glyph.y + h
            and glyph.y < o.y + o.mask.shape[0]
        ]
        if len(nearby) < 3:
            kept.append(glyph)
            continue
        solid = _fill_holes(glyph.mask)
        inside = enclosing = n_inside = 0
        for o in nearby:
            if _overlap(o.mask, o.x, o.y, solid, glyph.x, glyph.y) >= 0.5 * o.area:
                inside += o.area
                n_inside += 1
            elif (
                _overlap(glyph.mask, glyph.x, glyph.y, _fill_holes(o.mask), o.x, o.y)
                >= 0.5 * glyph.area
            ):
                enclosing += o.area
        if n_inside < 3 or inside < 0.5 * (total - enclosing):
            kept.append(glyph)
    return kept


def _hollow_fill_area(gray: np.ndarray) -> int:
    """Area of white regions shaped like the fills of hollow (outlined white) letters.

    A fill is a solid, stroke-shaped white region ringed by clearly darker pixels. Compact
    regions (counters), rings around dark letters (white outlines) and interiors holding most
    of the ink (caption frames) are not fills. Unlike the glyph border test, the 1px ring
    still catches outlines too thin or gray to pass it."""
    height, width = gray.shape
    ink = gray < 128
    ink_labels = None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (gray > 200).astype(np.uint8), connectivity=4
    )
    total = 0
    for label in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < 20 or x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        y0, y1 = max(0, y - 2), min(height, y + h + 2)
        x0, x1 = max(0, x - 2), min(width, x + w + 2)
        m = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        sub = gray[y0:y1, x0:x1]
        ring = cv2.dilate(m, np.ones((3, 3), np.uint8)) & (1 - m)
        fill_median = float(np.median(sub[m == 1]))
        if np.mean(sub[ring == 1] <= fill_median - 80) < 0.6:
            continue
        radius = float(cv2.distanceTransform(m, cv2.DIST_L2, 3).max())
        if area / (4 * radius * radius) < 2.5:
            continue
        solid = _fill_holes(m)
        if solid.sum() - area >= 0.5 * area:
            continue
        _, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = int((hierarchy[0][:, 3] >= 0).sum()) if hierarchy is not None else 0
        if holes > 2:
            if ink_labels is None:
                _, ink_labels = cv2.connectedComponents(ink.astype(np.uint8))
            sub_ink = ink[y0:y1, x0:x1]
            enclosed = int((sub_ink & solid & (m == 0)).sum())
            # Ink just outside the region (its frame, or its own outline) is not its content
            outside = cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(bool) & ~solid
            border_ids = np.unique(ink_labels[y0:y1, x0:x1][outside & sub_ink])
            border_ids = border_ids[border_ids > 0]
            border = (
                int(np.isin(ink_labels, border_ids).sum()) if border_ids.size else 0
            )
            if enclosed >= 0.5 * max(1, int(ink.sum()) - border):
                continue
        total += area
    return total


def _hole_test_text_color(rgb: np.ndarray) -> tuple[int, int, int]:
    """Text color from an Otsu split: ink enclosing large holes is an outline around the text."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, light = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frame = np.concatenate([light[0], light[-1], light[:, 0], light[:, -1]])
    bg_cls = int(frame.mean() >= 0.5)
    min_area = max(8, gray.size // 2000)

    # Background-class regions not connected to the frame: counters or outlined fills
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (light == bg_cls).astype(np.uint8), connectivity=4
    )
    is_hole = stats[:, cv2.CC_STAT_AREA] >= min_area
    is_hole[0] = False
    is_hole[
        np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    ] = False
    holes = is_hole[labels]
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (light != bg_cls).astype(np.uint8), connectivity=8
    )
    is_ink = stats[:, cv2.CC_STAT_AREA] >= min_area
    is_ink[0] = False
    ink = is_ink[labels]

    use_holes = holes.sum() / max(1, ink.sum()) >= 0.4
    text_mask = holes if use_holes else ink
    text_cls = bg_cls if use_holes else 1 - bg_cls
    core = cv2.erode(text_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pixels = rgb[core] if core.sum() >= 10 else rgb[text_mask]
    if len(pixels):
        median = np.median(pixels, axis=0).astype(int)
        if cv2.cvtColor(np.uint8([[median]]), cv2.COLOR_RGB2HSV)[0][0][1] >= 25:
            return tuple(int(c) for c in median)
    # Snap by Otsu class, not luminance: small JPEG text is often mid-gray
    return (255, 255, 255) if text_cls == 1 else (0, 0, 0)


def _bw_page_text_color(
    pil_image: Image.Image, text_bounds: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    """Text color of an OSB region on a B/W page."""
    x0, y0, x1, y1 = text_bounds
    img_w, img_h = pil_image.size
    pad = 4
    rgb = np.array(
        pil_image.crop(
            (
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(img_w, x1 + pad),
                min(img_h, y1 + pad),
            )
        ).convert("RGB")
    )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Glyphs touching the padded crop edge are background, not text
    dark = [
        g
        for g in _outlined_glyph_components(
            gray, (gray < 50).astype(np.uint8) * 255, lambda border: border > 150
        )
        if not g.on_frame
    ]
    white = [
        g
        for g in _outlined_glyph_components(
            gray,
            (gray > 200).astype(np.uint8) * 255,
            lambda border: border < 90,
            min_core_ratio=0.05,
        )
        if not g.on_frame
    ]
    dark, white = _drop_containers(dark, white), _drop_containers(white, dark)
    # Strict borders keep enclosed patches of art from passing as glyphs
    dark = [g for g in dark if g.ratio >= 0.6]
    white = [g for g in white if g.ratio >= 0.6]
    dark_area = sum(g.area for g in dark)
    white_area = sum(g.area for g in white)
    min_area = 0.02 * gray.size
    has_dark = len(dark) >= 2 or dark_area >= min_area
    has_white = len(white) >= 2 or white_area >= min_area

    if has_dark and not has_white:
        return (0, 0, 0)
    if has_white and not has_dark:
        return (255, 255, 255)
    fill_area = _hollow_fill_area(gray)
    if has_dark and has_white:
        return (0, 0, 0) if dark_area >= max(white_area, fill_area) else (255, 255, 255)
    if fill_area >= 0.01 * gray.size:
        return (255, 255, 255)
    return _hole_test_text_color(rgb)


def _color_page_text_color(
    pil_image: Image.Image,
    text_bounds: tuple[int, int, int, int],
    ref_bg_rgb: np.ndarray,
    fallback_fill_color: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    """Text color of an OSB region on a color page."""
    rgb = np.array(pil_image.crop(text_bounds).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Light saturated colors (a yellow banner) clear the gray cutoff but aren't white text
    neutral = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1] <= 40
    dark = _outlined_glyph_components(
        gray, (gray < 50).astype(np.uint8) * 255, lambda border: border > 150
    )
    white = _outlined_glyph_components(
        gray,
        ((gray > 200) & neutral).astype(np.uint8) * 255,
        lambda border: border < 90,
        min_core_ratio=0.05,
    )
    has_dark, has_white = _has_glyph_evidence(dark), _has_glyph_evidence(white)
    if has_dark and not has_white:
        return (0, 0, 0)
    if has_white and not has_dark:
        return (255, 255, 255)
    if has_dark and has_white:
        # By area, not count: dark specks of art outnumber letters; glyphs on the crop
        # edge are mostly background
        dark_area = sum(g.area for g in dark if not g.on_frame)
        white_area = sum(g.area for g in white if not g.on_frame)
        if dark_area != white_area:
            return (0, 0, 0) if dark_area > white_area else (255, 255, 255)
    return _general_text_color(pil_image, text_bounds, ref_bg_rgb, fallback_fill_color)


def _detect_osb_text_color(
    pil_image: Image.Image,
    text_bounds: tuple[int, int, int, int],
    ref_bg_rgb: np.ndarray,
    fallback_fill_color: tuple[int, int, int] | None,
    is_color_page: bool,
) -> tuple[int, int, int]:
    """Text color of an OSB region, using the detector suited to the page type."""
    if is_color_page:
        return _color_page_text_color(
            pil_image, text_bounds, ref_bg_rgb, fallback_fill_color
        )
    return _bw_page_text_color(pil_image, text_bounds)


def _general_text_color(
    pil_image: Image.Image,
    text_bounds: tuple[int, int, int, int],
    ref_bg_rgb: np.ndarray,
    fallback_fill_color: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    """Text color of an OSB region: outlined monochrome, then chromatic, then contrast."""
    bg_lab = cv2.cvtColor(np.uint8([[ref_bg_rgb]]), cv2.COLOR_RGB2LAB)[0][0]
    crop_rgb = np.array(pil_image.crop(text_bounds).convert("RGB"))
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Detect outlined monochrome text (black text + white outline or white text + dark outline)
    detected_text_color = None

    # Glyph cores whose border contrasts: dark text with a light outline, and the reverse
    dark_glyph_comps = _outlined_glyph_components(
        crop_gray,
        (crop_gray < 50).astype(np.uint8) * 255,
        lambda border: border > 150,
    )
    white_glyph_comps = _outlined_glyph_components(
        crop_gray,
        (crop_gray > 200).astype(np.uint8) * 255,
        lambda border: border < 90,
        min_core_ratio=0.05,
    )
    has_dark_outlines = _has_glyph_evidence(dark_glyph_comps)
    has_white_outlines = _has_glyph_evidence(white_glyph_comps)

    if has_dark_outlines and not has_white_outlines:
        detected_text_color = (0, 0, 0)
    elif has_white_outlines and not has_dark_outlines:
        detected_text_color = (255, 255, 255)
    elif has_dark_outlines and has_white_outlines:
        avg_dark_ratio = float(np.mean([g.ratio for g in dark_glyph_comps]))
        avg_white_ratio = float(np.mean([g.ratio for g in white_glyph_comps]))
        if (
            len(dark_glyph_comps) > len(white_glyph_comps)
            and avg_dark_ratio >= avg_white_ratio * 0.8
        ):
            detected_text_color = (0, 0, 0)
        elif (
            len(white_glyph_comps) > len(dark_glyph_comps)
            and avg_white_ratio >= avg_dark_ratio * 0.8
        ):
            detected_text_color = (255, 255, 255)
        elif avg_dark_ratio > avg_white_ratio:
            detected_text_color = (0, 0, 0)
        else:
            detected_text_color = (255, 255, 255)

    # 3. Detect chromatic text via LAB contrast against background
    if detected_text_color is None:
        crop_lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        dist_map = np.linalg.norm(crop_lab - bg_lab.astype(np.float32), axis=2)

        robust_max_dist = np.percentile(dist_map, 95)
        CONTRAST_THRESHOLD = max(30, robust_max_dist * 0.6)
        contrast_mask = (dist_map > CONTRAST_THRESHOLD).astype(np.uint8) * 255

        kernel_3 = np.ones((3, 3), np.uint8)
        contrast_mask = cv2.morphologyEx(contrast_mask, cv2.MORPH_CLOSE, kernel_3)
        contrast_mask = cv2.erode(
            contrast_mask,
            np.ones((2, 2), np.uint8),
            iterations=1,
        )

        contours, _ = cv2.findContours(
            contrast_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        clean_mask = np.zeros_like(contrast_mask)
        MIN_COMPONENT_AREA = 4
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_COMPONENT_AREA:
                cv2.drawContours(
                    clean_mask,
                    [cnt],
                    -1,
                    255,
                    cv2.FILLED,
                )

        text_pixels_rgb = crop_rgb[clean_mask == 255]
        if len(text_pixels_rgb) >= 10:
            sampled_rgb = tuple(np.median(text_pixels_rgb, axis=0).astype(int))
            hsv = cv2.cvtColor(
                np.uint8([[sampled_rgb]]),
                cv2.COLOR_RGB2HSV,
            )[0][0]
            if hsv[1] >= 25:
                detected_text_color = sampled_rgb

    # 4. Fallback for non-outlined monochrome text: contrast against fill
    if detected_text_color is None:
        fill_for_contrast = (
            fallback_fill_color if fallback_fill_color is not None else (0, 0, 0)
        )
        fill_lum = (
            0.299 * fill_for_contrast[0]
            + 0.587 * fill_for_contrast[1]
            + 0.114 * fill_for_contrast[2]
        )
        detected_text_color = (0, 0, 0) if fill_lum >= 128 else (255, 255, 255)
    return detected_text_color


class _RegionFill(NamedTuple):
    """How to fill one OSB region, from `_decide_osb_region_fill`."""

    composite_clip_bbox: tuple[int, int, int, int] | None
    original_bounds: tuple[int | None, int | None, int | None, int | None]
    fill_color: tuple[int, int, int] | None  # None: the region needs inpainting
    fallback_fill_color: tuple[int, int, int] | None
    text_color: tuple[int, int, int] | None


def _decide_osb_region_fill(
    work: OutsideTextWork,
    current_image: Image.Image,
    group: dict[str, Any],
    region_number: int,
) -> _RegionFill:
    """Sample an OSB region's background and text colors to choose how to fill it."""
    img_w, img_h = work.img_w, work.img_h
    verbose = work.verbose
    composite_clip_bbox = None
    fill_color = None
    fallback_fill_color = None
    text_color = None
    bounds = (None, None, None, None)
    original_bbox_dict = group.get("original_bbox")
    if not original_bbox_dict:
        return _RegionFill(
            composite_clip_bbox, bounds, fill_color, fallback_fill_color, text_color
        )
    ox = int(original_bbox_dict.get("x", 0))
    oy = int(original_bbox_dict.get("y", 0))
    ow = int(original_bbox_dict.get("width", 0))
    oh = int(original_bbox_dict.get("height", 0))
    if ow <= 0 or oh <= 0:
        return _RegionFill(
            composite_clip_bbox, bounds, fill_color, fallback_fill_color, text_color
        )
    bounds = (
        max(0, min(img_w, ox)),
        max(0, min(img_h, oy)),
        max(0, min(img_w, ox + ow)),
        max(0, min(img_h, oy + oh)),
    )
    composite_clip_bbox = (ox, oy, ox + ow, oy + oh)

    # Determine detected text color for this region to ensure contrast
    if work.original_text_colors:
        votes_dark = 0
        votes_light = 0
        gx1, gy1, gx2, gy2 = composite_clip_bbox
        for (bx1, by1, bx2, by2), t_dark in work.original_text_colors.items():
            # Check if center of OCR box is inside group box
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2
            if gx1 <= bcx <= gx2 and gy1 <= bcy <= gy2:
                if t_dark:
                    votes_dark += 1
                else:
                    votes_light += 1
        if votes_dark > 0 or votes_light > 0:
            group_bg_is_dark = votes_dark >= votes_light

            # Detected value represents background brightness
            fallback_fill_color = (0, 0, 0) if group_bg_is_dark else (255, 255, 255)

            t_type = "Dark" if group_bg_is_dark else "Light"
            f_col = "White" if fallback_fill_color == (255, 255, 255) else "Black"
            log_message(
                f"OSB Region {region_number}: Detected {t_type} background. "
                f"Fallback fill: {f_col}.",
                verbose=verbose,
            )

    # Expanded sampling around the original bbox to find background color
    mask_indices = group.get("mask_indices", [])
    if mask_indices and work.raw_outside_text_results:
        text_bounds = _results_bounds(work.raw_outside_text_results, mask_indices)
    else:
        text_bounds = composite_clip_bbox
    border_pixels = _border_pixels(current_image, text_bounds, img_w, img_h)
    if border_pixels is None or len(border_pixels) < 20:
        return _RegionFill(
            composite_clip_bbox, bounds, fill_color, fallback_fill_color, text_color
        )

    detected_color, solid_ratio = _border_color(border_pixels)
    is_border_solid = solid_ratio >= OSB_SOLID_RATIO_THRESHOLD
    if is_border_solid or fallback_fill_color is None:
        fallback_fill_color = detected_color

    ref_bg_rgb = np.array(
        detected_color
        if is_border_solid
        else (fallback_fill_color if fallback_fill_color is not None else (0, 0, 0)),
        dtype=np.uint8,
    )
    text_color = _detect_osb_text_color(
        work.pil_image,
        text_bounds,
        ref_bg_rgb,
        fallback_fill_color,
        work.is_color_page,
    )

    force_fill = work.config.outside_text.inpainting_method == "opencv"

    # Get expanded bounds for this group to check solid color and for cv2 fill
    p_x0, p_y0, p_x1, p_y1 = composite_clip_bbox
    if mask_indices and work.outside_text_results:
        p_x0, p_y0, p_x1, p_y1 = _results_bounds(
            work.outside_text_results, mask_indices
        )
        p_x0, p_y0 = max(0, p_x0), max(0, p_y0)
        p_x1, p_y1 = min(img_w, p_x1), min(img_h, p_y1)

    # Simply check if the expanded boundary is solid color
    expanded_is_solid = is_border_solid
    if not force_fill:
        eborder_pixels = _border_pixels(
            current_image, (p_x0, p_y0, p_x1, p_y1), img_w, img_h
        )
        if eborder_pixels is not None and eborder_pixels.size > 0:
            edetected_color, esolid_ratio = _border_color(eborder_pixels)
            if esolid_ratio >= OSB_SOLID_RATIO_THRESHOLD:
                expanded_is_solid = True
                fallback_fill_color = edetected_color

    if expanded_is_solid or force_fill:
        fill_color = (
            fallback_fill_color if fallback_fill_color is not None else detected_color
        )
        if force_fill:
            log_message(
                f"Using OpenCV simple fill for OSB region: {fill_color} background",
                verbose=verbose,
            )
        else:
            log_message(
                f"Skipping Flux for OSB region: detected solid {fill_color} background",
                verbose=verbose,
            )

    return _RegionFill(
        composite_clip_bbox, bounds, fill_color, fallback_fill_color, text_color
    )


def _clip_mask_to_bbox(
    mask: np.ndarray, bbox: tuple[int, int, int, int] | None
) -> np.ndarray:
    """Copy of `mask` with everything outside `bbox` cleared; a plain copy without one."""
    if bbox is None:
        return mask.copy()
    img_h, img_w = mask.shape
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, min(img_w, x1)), max(0, min(img_w, x2))
    y1, y2 = max(0, min(img_h, y1)), max(0, min(img_h, y2))
    clipped = np.zeros_like(mask)
    if x2 > x1 and y2 > y1:
        clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def _glyph_inpaint_mask(
    work: OutsideTextWork,
    combined_mask: np.ndarray,
    composite_clip_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """The region's mask clipped to its text box, then shrunk to the dilated glyphs."""
    box_mask = _clip_mask_to_bbox(combined_mask, composite_clip_bbox)
    glyph_prob = work.glyph_map.get()
    if glyph_prob is None:
        return box_mask
    radius = glyph_dilation_radius(work.pil_image.size)
    return refine_mask_to_glyphs(box_mask, glyph_prob, radius)


def _composite_to_glyphs(
    work: OutsideTextWork,
    before: Image.Image,
    after: Image.Image,
    box_mask: np.ndarray,
) -> Image.Image:
    """Keep the original art between and around the letters of an edit-model inpaint.

    The edit models redraw their whole crop, so pasting the box would replace art
    the text never covered. LaMa already inpaints only the glyphs.
    """
    if not _composites_glyphs(work.config):
        return after
    glyph_prob = work.glyph_map.get()
    if glyph_prob is None:
        return after
    out = composite_glyphs(
        np.asarray(before),
        np.asarray(after),
        box_mask,
        glyph_prob,
        glyph_dilation_radius(work.pil_image.size),
    )
    return Image.fromarray(out)


def _run_grouped_flux(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    inpainter,
    seed: int,
    request_coordinator,
    img_w: int,
    img_h: int,
    verbose: bool,
) -> tuple[Image.Image, int, int]:
    """Inpaint all queued OSB regions in one Flux pass, falling back to CV2 fills.

    Returns the image and the number of regions done by Flux and by CV2.
    """
    grouped_mask = np.zeros((img_h, img_w), dtype=bool)
    for candidate in candidates:
        grouped_mask = np.logical_or(grouped_mask, candidate["mask"])

    if np.any(grouped_mask):
        log_message(
            f"Running grouped Flux inpainting for {len(candidates)} OSB regions",
            verbose=verbose,
        )
        try:
            inpaint_kwargs = {
                "seed": seed,
                "verbose": verbose,
                "strict_mask_clipping": True,
                "ocr_params": {
                    "type": "outside_text_group",
                    "regions": len(candidates),
                },
            }
            if request_coordinator is not None:
                inpainted_image = request_coordinator.run(
                    inpainter.inpaint_mask,
                    image,
                    grouped_mask,
                    **inpaint_kwargs,
                )
            else:
                inpainted_image = inpainter.inpaint_mask(
                    image,
                    grouped_mask,
                    **inpaint_kwargs,
                )
            if inpainted_image is image:
                raise RuntimeError("Flux returned original image (no inpaint)")
            return inpainted_image, len(candidates), 0
        except CancellationError:
            raise
        except Exception as e:
            log_message(
                f"Grouped Flux inpainting failed ({e}); falling back to CV2 fill",
                always_print=True,
            )
            for candidate in candidates:
                image = _fill_mask(
                    image,
                    candidate["fill_mask"],
                    candidate["fallback_color"],
                )
        return image, 0, len(candidates)
    return image, 0, 0


def _try_flux_inpaint(
    inpainter,
    image: Image.Image,
    mask: np.ndarray,
    seed: int,
    composite_clip_bbox,
    verbose: bool,
) -> tuple[Image.Image | None, str | None]:
    """Inpaint one OSB region with Flux; returns (image, None) or (None, failure reason)."""
    if inpainter is None:
        return None, "Flux inpainter unavailable"
    try:
        inpainted_image = inpainter.inpaint_mask(
            image,
            mask,
            seed=seed,
            verbose=verbose,
            strict_mask_clipping=True,
            composite_clip_bbox=composite_clip_bbox,
        )
    except CancellationError:
        raise
    except Exception as e:
        return None, f"Flux inpainting error: {e}"
    if inpainted_image is image:
        return None, "Flux returned original image (no inpaint)"
    return inpainted_image, None


def _run_flux_waves(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    inpainter,
    request_coordinator,
    fill_candidate,
    composite_candidate,
    verbose: bool,
) -> tuple[Image.Image, int, int]:
    """Run queued per-region Flux jobs in non-overlapping waves via the coordinator.

    Each result goes through `composite_candidate(before, after, candidate)`; regions
    whose job fails get `fill_candidate(image, candidate, color)` instead.
    Returns the image and the number of regions done by Flux and by CV2.
    """
    flux_count = 0
    cv2_count = 0
    waves = partition_non_overlapping_waves(
        candidates,
        lambda candidate: candidate["context_bbox"],
    )
    log_message(
        f"Scheduling OSB Flux in {len(waves)} wave(s)",
        verbose=verbose,
    )

    for wave in waves:
        base_image = image

        def make_job(candidate, base_img=base_image):
            def job():
                image_for_job = base_img.copy()
                try:
                    result_image = inpainter.inpaint_mask(
                        image_for_job,
                        candidate["mask"],
                        seed=candidate["seed"],
                        verbose=verbose,
                        strict_mask_clipping=True,
                        composite_clip_bbox=candidate["composite_clip_bbox"],
                    )
                    if result_image is image_for_job:
                        raise RuntimeError("Flux returned original image (no inpaint)")
                    return {
                        "candidate": candidate,
                        "image": result_image,
                        "error": None,
                    }
                except CancellationError:
                    raise
                except Exception as e:
                    return {
                        "candidate": candidate,
                        "image": None,
                        "error": e,
                    }

            return job

        results = request_coordinator.map_ordered(
            [make_job(candidate) for candidate in wave]
        )
        for result in results:
            candidate = result["candidate"]
            if result["error"] is not None:
                fallback_color_to_use = candidate["fallback_color"]
                log_message(
                    f"Flux failed for OSB region {candidate['index']}"
                    f" (Flux inpainting error: {result['error']}); "
                    f"falling back to CV2 fill ({fallback_color_to_use})",
                    always_print=True,
                )
                image = fill_candidate(image, candidate, fallback_color_to_use)
                cv2_count += 1
                continue

            result_image = composite_candidate(base_image, result["image"], candidate)
            context_bbox = candidate["context_bbox"]
            if context_bbox is None:
                image = result_image
            else:
                image = paste_image_region(
                    image,
                    result_image,
                    context_bbox,
                )
            flux_count += 1
    return image, flux_count, cv2_count


def finish_outside_text_work(
    work: OutsideTextWork,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Run OSB inpainting for prepared work and attach render metadata."""
    pil_image = work.pil_image
    config = work.config
    verbose = work.verbose
    outside_text_results = work.outside_text_results
    total_bubble_mask = work.total_bubble_mask
    mask_groups = work.mask_groups
    img_w = work.img_w
    img_h = work.img_h
    outside_text_data = work.outside_text_data

    extracted_text_colors = {}
    none_skipped_clip_bboxes = set()
    current_image = pil_image

    try:
        log_message("Inpainting outside text regions...", verbose=verbose)

        inpainting_method = config.outside_text.inpainting_method
        inpainter = _create_osb_inpainter(config, verbose)
        current_image = pil_image
        none_skipped_clip_bboxes = set()
        if mask_groups:
            base_seed = (
                random.randint(1, 999999)
                if config.outside_text.seed == -1
                else config.outside_text.seed
            )

            extracted_text_colors = {}
            flux_inpaints = 0
            cv2_inpaints = 0
            none_skips = 0
            group_flux_regions = bool(config.outside_text.flux_group_regions)
            grouped_flux_candidates = []
            request_coordinator = getattr(config, "request_coordinator", None)
            pending_flux_candidates = []

            def simple_fill(
                image, color, group, combined_mask, original_bbox_dict, bounds
            ):
                """Fill the group's (tilt-aware) text boxes, else its whole mask."""
                rect_mask = _build_osb_simple_fill_mask(
                    img_w=img_w,
                    img_h=img_h,
                    mask_indices=group.get("mask_indices", []),
                    outside_text_results=outside_text_results,
                    outside_text_data=outside_text_data,
                    config=config,
                    fallback_bounds=bounds if original_bbox_dict else None,
                    total_bubble_mask=total_bubble_mask,
                )
                return _fill_mask(
                    image, combined_mask if rect_mask is None else rect_mask, color
                )

            def apply_candidate_simple_fill(image, candidate, color_to_use):
                return simple_fill(
                    image,
                    color_to_use,
                    candidate["group"],
                    candidate["fill_mask"],
                    candidate["original_bbox_dict"],
                    candidate["original_bounds"],
                )

            def flush_pending_flux_candidates():
                nonlocal current_image, flux_inpaints, cv2_inpaints
                if not pending_flux_candidates:
                    return

                candidates = list(pending_flux_candidates)
                pending_flux_candidates.clear()
                current_image, flux_done, cv2_done = _run_flux_waves(
                    current_image,
                    candidates,
                    inpainter,
                    request_coordinator,
                    apply_candidate_simple_fill,
                    lambda before, after, candidate: _composite_to_glyphs(
                        work, before, after, candidate["box_mask"]
                    ),
                    verbose,
                )
                flux_inpaints += flux_done
                cv2_inpaints += cv2_done

            for i, group in enumerate(mask_groups):
                raise_if_cancelled()
                log_message(
                    f"Inpainting outside text region {i + 1}/{len(mask_groups)}",
                    verbose=verbose,
                )
                combined_mask = group["combined_mask"]
                combined_mask = np.logical_and(
                    combined_mask, np.logical_not(total_bubble_mask)
                )
                if not np.any(combined_mask):
                    log_message(
                        "Skipping outside text region after bubble masking (no remaining area)",
                        verbose=verbose,
                    )
                    continue
                region_seed = base_seed + i if base_seed > 0 else base_seed

                original_bbox_dict = group.get("original_bbox")
                region = _decide_osb_region_fill(
                    work, current_image, group, region_number=i + 1
                )
                composite_clip_bbox = region.composite_clip_bbox
                fill_color = region.fill_color
                fallback_fill_color = region.fallback_fill_color
                if region.text_color is not None:
                    extracted_text_colors[composite_clip_bbox] = region.text_color

                region_fill_args = (
                    group,
                    combined_mask,
                    original_bbox_dict,
                    region.original_bounds,
                )

                if fill_color is not None:
                    flush_pending_flux_candidates()
                    current_image = simple_fill(
                        current_image, fill_color, *region_fill_args
                    )
                    cv2_inpaints += 1
                    continue

                if inpainting_method == "none":
                    if composite_clip_bbox:
                        none_skipped_clip_bboxes.add(composite_clip_bbox)
                    none_skips += 1
                    log_message(
                        f"Skipping inpaint for non-solid OSB region {i + 1} (none mode)",
                        verbose=verbose,
                    )
                    continue

                # LaMa erases every masked pixel, so give it only the letters. The edit
                # models still see the box; when they are composited through the
                # glyphs, the map is built here, before their weights load.
                inpaint_mask = combined_mask
                box_mask = _clip_mask_to_bbox(combined_mask, composite_clip_bbox)
                if inpainting_method == "lama_manga" and inpainter is not None:
                    inpaint_mask = _glyph_inpaint_mask(
                        work, combined_mask, composite_clip_bbox
                    )
                elif inpainter is not None and _composites_glyphs(config):
                    work.glyph_map.get()

                if group_flux_regions and inpainter is not None:
                    flush_pending_flux_candidates()
                    grouped_mask = _clip_mask_to_bbox(inpaint_mask, composite_clip_bbox)

                    if not np.any(grouped_mask):
                        log_message(
                            f"Skipping OSB region {i + 1} after grouped clip (no remaining area)",
                            verbose=verbose,
                        )
                        continue

                    grouped_flux_candidates.append(
                        {
                            "index": i + 1,
                            "mask": grouped_mask,
                            "fill_mask": box_mask,
                            "fallback_color": (
                                fallback_fill_color
                                if fallback_fill_color
                                else (255, 255, 255)
                            ),
                        }
                    )
                    log_message(
                        f"Queued OSB region {i + 1} for grouped Flux inpainting",
                        verbose=verbose,
                    )
                    continue

                if request_coordinator is not None and inpainter is not None:
                    fallback_color_to_use = (
                        fallback_fill_color if fallback_fill_color else (255, 255, 255)
                    )
                    pending_flux_candidates.append(
                        {
                            "index": i + 1,
                            "mask": inpaint_mask.copy(),
                            "fill_mask": combined_mask.copy(),
                            "box_mask": box_mask,
                            "seed": region_seed,
                            "composite_clip_bbox": composite_clip_bbox,
                            "fallback_color": fallback_color_to_use,
                            "group": group,
                            "original_bbox_dict": original_bbox_dict,
                            "original_bounds": region.original_bounds,
                            "context_bbox": expanded_mask_bbox(
                                inpaint_mask, current_image.size
                            ),
                        }
                    )
                    log_message(
                        f"Queued OSB region {i + 1} for intra-page Flux scheduling",
                        verbose=verbose,
                    )
                    continue

                inpainted_image, flux_fail_reason = _try_flux_inpaint(
                    inpainter,
                    current_image,
                    inpaint_mask,
                    region_seed,
                    composite_clip_bbox,
                    verbose,
                )

                if flux_fail_reason is not None:
                    fallback_color_to_use = (
                        fallback_fill_color if fallback_fill_color else (255, 255, 255)
                    )
                    log_message(
                        f"Flux failed for OSB region {i + 1} ({flux_fail_reason})"
                        f"; falling back to CV2 fill ({fallback_color_to_use})",
                        always_print=True,
                    )
                    current_image = simple_fill(
                        current_image, fallback_color_to_use, *region_fill_args
                    )
                    cv2_inpaints += 1
                    continue

                flux_inpaints += 1
                current_image = _composite_to_glyphs(
                    work, current_image, inpainted_image, box_mask
                )

            flush_pending_flux_candidates()

            if grouped_flux_candidates:
                before_grouped = current_image
                current_image, grouped_flux, grouped_cv2 = _run_grouped_flux(
                    current_image,
                    grouped_flux_candidates,
                    inpainter,
                    base_seed,
                    request_coordinator,
                    img_w,
                    img_h,
                    verbose,
                )
                flux_inpaints += grouped_flux
                cv2_inpaints += grouped_cv2
                if grouped_flux:
                    current_image = _composite_to_glyphs(
                        work,
                        before_grouped,
                        current_image,
                        np.logical_or.reduce(
                            [c["fill_mask"] for c in grouped_flux_candidates]
                        ),
                    )

            log_message("Outside text inpainting completed", verbose=verbose)
            parts = [
                f"Flux: {flux_inpaints}",
                f"CV2: {cv2_inpaints}",
            ]
            if none_skips:
                parts.append(f"Skipped (none): {none_skips}")
            log_message(
                f"Inpainted {len(mask_groups)} outside text regions ({', '.join(parts)})",
                always_print=True,
            )

        _apply_inpaint_render_metadata(
            outside_text_data,
            extracted_text_colors,
            none_skipped_clip_bboxes,
        )
        return current_image, outside_text_data

    except CancellationError:
        raise
    except Exception as e:
        log_message(
            f"Error during outside text inpainting: {e}",
            always_print=True,
        )
        return pil_image, outside_text_data


def process_outside_text(
    pil_image: Image.Image,
    config: MangaTranslatorConfig,
    image_path: str | Path,
    image_format: str | None,
    verbose: bool = False,
    bubble_data: list[dict[str, Any]] | None = None,
    text_free_boxes: list[list[float]] | None = None,
    panels: list[tuple[int, int, int, int]] | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """
    Process outside text detection, inpainting, and prepare data for translation.

    This function handles the complete outside text processing pipeline:
    1. Detects text outside speech bubbles using OCR
    2. Inpaints the detected text regions using FluxKontext
    3. Prepares the outside text data for translation API calls

    Args:
        pil_image: The PIL image to process
        config: MangaTranslatorConfig containing all settings
        image_path: Path to the original image file
        image_format: Original image format (PNG, JPEG, etc.)
        processing_scale: The scale factor for image processing
        verbose: Whether to print detailed logging

    Returns:
        Tuple containing:
        - processed_pil_image: The image after outside text inpainting
        - outside_text_data: List of dicts with outside text information for translation
    """
    work = prepare_outside_text_work(
        pil_image,
        config,
        image_path,
        image_format,
        verbose=verbose,
        bubble_data=bubble_data,
        text_free_boxes=text_free_boxes,
        panels=panels,
    )
    if work is None:
        return pil_image, []
    return finish_outside_text_work(work)
