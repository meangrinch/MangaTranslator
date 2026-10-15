"""Developer debug output for reading order: an overlay PNG and the LLM's crops per page.

Set ENABLE_COMPONENT_ORDER_DEBUG to True to write them next to each output image.
"""

import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.logging import log_message

from .image.sorting import sort_panels_by_reading_order

ENABLE_COMPONENT_ORDER_DEBUG = False


def write_component_order_debug(
    sorted_items,
    processed_bubbles_info,
    image_size,
    panels,
    reading_direction,
    image_path,
    output_path,
    verbose=False,
):
    """Write the component-order overlay and the LLM crop images for one page."""
    bubble_masks = {}
    for bubble in sorted_items:
        if bubble.get("is_outside_text", False):
            continue
        bbox = tuple(round(v) for v in bubble.get("bbox", ()))
        if len(bbox) != 4:
            continue
        mask = bubble.get("sam_mask")
        if mask is not None:
            bubble_masks[bbox] = mask

    for info in processed_bubbles_info:
        bbox = tuple(round(v) for v in info.get("bbox", ()))
        if len(bbox) != 4:
            continue
        mask = info.get("mask")
        if mask is None:
            mask = info.get("base_mask")
        if mask is not None:
            bubble_masks[bbox] = mask

    try:
        _write_component_order_debug_image(
            image_size,
            sorted_items,
            panels,
            bubble_masks,
            reading_direction,
            image_path,
            output_path,
            verbose=verbose,
        )
    except Exception as e:
        log_message(
            f"Failed to write component-order debug image: {e}",
            always_print=True,
        )

    try:
        _write_llm_crop_debug_images(
            sorted_items,
            image_path,
            output_path,
            verbose=verbose,
        )
    except Exception as e:
        log_message(
            f"Failed to write LLM crop debug images: {e}",
            always_print=True,
        )


def _debug_mask_bbox(mask):
    """Return full-image bbox for a debug mask, or None when empty/invalid."""
    normalized = (
        _normalize_debug_mask(mask, (mask.shape[1], mask.shape[0]))
        if isinstance(mask, np.ndarray) and mask.ndim >= 2
        else None
    )
    if normalized is None:
        try:
            mask_array = np.asarray(mask)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]
            if mask_array.ndim != 2:
                return None
            normalized = mask_array > 0
        except Exception:
            return None
    coords = np.where(normalized)
    if coords[0].size == 0 or coords[1].size == 0:
        return None
    return [
        int(coords[1].min()),
        int(coords[0].min()),
        int(coords[1].max()) + 1,
        int(coords[0].max()) + 1,
    ]


def _load_debug_font(size: int):
    """Load a bold-ish font for the debug overlay, falling back safely."""
    font_candidates = [
        "arialbd.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_dashed_rectangle(draw, bbox, color, width=2, dash=12, gap=7):
    """Draw a dashed rectangle matching the requested debug style."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return

    def _draw_dashed_line(start, end, horizontal=True):
        if horizontal:
            fixed = start[1]
            pos = start[0]
            limit = end[0]
            while pos < limit:
                seg_end = min(pos + dash, limit)
                draw.line((pos, fixed, seg_end, fixed), fill=color, width=width)
                pos += dash + gap
        else:
            fixed = start[0]
            pos = start[1]
            limit = end[1]
            while pos < limit:
                seg_end = min(pos + dash, limit)
                draw.line((fixed, pos, fixed, seg_end), fill=color, width=width)
                pos += dash + gap

    _draw_dashed_line((x0, y0), (x1, y0), horizontal=True)
    _draw_dashed_line((x0, y1), (x1, y1), horizontal=True)
    _draw_dashed_line((x0, y0), (x0, y1), horizontal=False)
    _draw_dashed_line((x1, y0), (x1, y1), horizontal=False)


def _draw_centered_index(draw, bbox, value, font, color):
    """Draw the index at the visual center of the box."""
    x0, y0, x1, y1 = bbox
    cx = round((x0 + x1) / 2)
    cy = round((y0 + y1) / 2)
    label = str(value)
    try:
        draw.text((cx, cy), label, fill=color, font=font, anchor="mm")
    except TypeError:
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (cx - (right - left) / 2, cy - (bottom - top) / 2),
            label,
            fill=color,
            font=font,
        )


def _normalize_debug_mask(mask, image_size):
    """Normalize a debug mask into a full-image boolean array."""
    if mask is None:
        return None

    try:
        mask_array = np.asarray(mask)
    except Exception:
        return None

    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    if mask_array.ndim != 2:
        return None

    width, height = image_size
    if mask_array.shape != (height, width):
        return None

    return mask_array > 0


def _apply_mask_debug_overlay(canvas, mask, color=(255, 0, 0, 84)):
    """Alpha-composite a semi-transparent mask overlay onto the debug canvas."""
    normalized_mask = _normalize_debug_mask(mask, canvas.size)
    if normalized_mask is None or not np.any(normalized_mask):
        return

    overlay = np.zeros((canvas.size[1], canvas.size[0], 4), dtype=np.uint8)
    overlay[normalized_mask] = color
    canvas.alpha_composite(Image.fromarray(overlay, mode="RGBA"))


def _write_component_order_debug_image(
    image_size,
    sorted_items,
    panels,
    bubble_masks,
    reading_direction,
    image_path,
    output_path,
    verbose=False,
):
    """Write a debug PNG showing panel order and merged text-element order."""
    width, height = image_size
    if width <= 0 or height <= 0:
        return

    canvas = Image.new("RGBA", (width, height), (238, 238, 238, 255))
    draw = ImageDraw.Draw(canvas)

    panel_color = (32, 63, 255)
    osb_color = (255, 0, 255)
    bubble_color = (34, 160, 34)
    index_color = (255, 0, 0)

    font_size = max(14, min(width, height) // 28)
    font = _load_debug_font(font_size)

    panel_order = (
        sort_panels_by_reading_order(panels, reading_direction) if panels else []
    )

    for item in sorted_items:
        if item.get("is_outside_text", False):
            continue
        bbox = tuple(round(v) for v in item.get("bbox", (0, 0, 0, 0)))
        _apply_mask_debug_overlay(
            canvas, bubble_masks.get(bbox) if bubble_masks else None
        )

    for panel_index, panel_id in enumerate(panel_order, start=1):
        panel_bbox = tuple(round(v) for v in panels[panel_id])
        draw.rectangle(panel_bbox, outline=panel_color, width=3)
        _draw_centered_index(draw, panel_bbox, panel_index, font, index_color)

    for item_index, item in enumerate(sorted_items, start=1):
        bbox = tuple(round(v) for v in item.get("bbox", (0, 0, 0, 0)))
        if item.get("is_outside_text", False):
            draw.rectangle(bbox, outline=osb_color, width=2)
            draw_bbox = bbox
        else:
            mask_bbox = (
                _debug_mask_bbox(bubble_masks.get(bbox)) if bubble_masks else None
            )
            draw_bbox = tuple(mask_bbox) if mask_bbox is not None else bbox
            _draw_dashed_rectangle(draw, draw_bbox, bubble_color, width=2)
        _draw_centered_index(draw, draw_bbox, item_index, font, index_color)

    base_path = Path(output_path) if output_path else Path(image_path)
    debug_path = base_path.parent / f"{base_path.stem}.component-order-debug.png"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(debug_path, format="PNG")
    log_message(
        f"Wrote component-order debug image: {debug_path}",
        verbose=verbose,
        always_print=True,
    )


def _write_llm_crop_debug_images(
    sorted_items,
    image_path,
    output_path,
    verbose=False,
):
    """Save the exact image crops the LLM sees to a debug subfolder."""
    base_path = Path(output_path) if output_path else Path(image_path)
    crop_dir = base_path.parent / f"{base_path.stem}.llm-crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, item in enumerate(sorted_items, start=1):
        img_b64 = item.get("image_b64")
        if not img_b64:
            continue
        try:
            img_bytes = base64.b64decode(img_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_cv = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_cv is None:
                continue
            label = "osb" if item.get("is_outside_text", False) else "bubble"
            crop_path = crop_dir / f"{i:03d}_{label}.png"
            cv2.imwrite(str(crop_path), img_cv)
            count += 1
        except Exception:
            pass

    log_message(
        f"Wrote {count} LLM crop debug images to: {crop_dir}",
        verbose=verbose,
        always_print=True,
    )
