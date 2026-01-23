import os
import threading
from typing import Dict, Optional, Tuple

import skia
import uharfbuzz as hb
from PIL import Image

from core.text.font_manager import LRUCache, load_font_data
from core.text.layout_engine import shape_line
from core.text.text_processing import parse_styled_segments
from utils.exceptions import FontError, RenderingError
from utils.logging import log_message

_typeface_cache = LRUCache(max_size=50)
_hb_face_cache = LRUCache(max_size=50)
_font_cache_lock = threading.RLock()


def load_font_resources(
    font_path: str,
) -> Tuple[bytes, skia.Typeface, hb.Face]:
    """
    Loads font data, Skia Typeface, and HarfBuzz Face, using LRU caching.

    Args:
        font_path: Path to the font file

    Returns:
        Tuple of (font_data, skia_typeface, harfbuzz_face)

    Raises:
        FontError: If font data cannot be loaded or Skia/HarfBuzz resources fail to load
    """
    try:
        font_data = load_font_data(font_path)
    except FontError:
        raise

    with _font_cache_lock:
        typeface = _typeface_cache.get(font_path)
        if typeface is None:
            skia_data = skia.Data.MakeWithoutCopy(font_data)
            typeface = skia.Typeface.MakeFromData(skia_data)
            if typeface is None:
                log_message(
                    f"Skia typeface load failed: {os.path.basename(font_path)}",
                    always_print=True,
                )
                raise FontError(
                    f"Failed to create Skia typeface from font: {font_path}"
                )
            _typeface_cache.put(font_path, typeface)

        hb_face = _hb_face_cache.get(font_path)
        if hb_face is None:
            try:
                hb_face = hb.Face(font_data)
                _hb_face_cache.put(font_path, hb_face)
            except Exception as e:
                log_message(
                    f"HarfBuzz face load failed: {os.path.basename(font_path)}: {e}",
                    always_print=True,
                )
                # Clean up Skia cache if HarfBuzz fails to avoid inconsistent state
                if font_path in _typeface_cache:
                    del _typeface_cache[font_path]
                raise FontError(
                    f"Failed to create HarfBuzz face from font: {font_path}"
                ) from e

    return font_data, typeface, hb_face


def pil_to_skia_surface(pil_image: Image.Image) -> skia.Surface:
    """Converts a PIL image to a Skia Surface.

    Raises:
        RenderingError: If conversion fails
    """
    try:
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")
        skia_image = skia.Image.frombytes(
            pil_image.tobytes(), pil_image.size, skia.kRGBA_8888_ColorType
        )
        if skia_image is None:
            log_message("PIL to Skia conversion failed", always_print=True)
            raise RenderingError("Failed to create Skia image from PIL")
        surface = skia.Surface(pil_image.width, pil_image.height)
        with surface as canvas:
            canvas.drawImage(skia_image, 0, 0)
        return surface
    except Exception as e:
        log_message(f"PIL to Skia conversion error: {e}", always_print=True)
        raise RenderingError("PIL to Skia conversion failed") from e


def skia_surface_to_pil(surface: skia.Surface) -> Image.Image:
    """Converts a Skia Surface back to a PIL image.

    Raises:
        RenderingError: If conversion fails
    """
    try:
        skia_image: Optional[skia.Image] = surface.makeImageSnapshot()
        if skia_image is None:
            log_message("Skia surface snapshot failed", always_print=True)
            raise RenderingError("Failed to create Skia image snapshot")

        skia_image = skia_image.convert(
            alphaType=skia.kUnpremul_AlphaType, colorType=skia.kRGBA_8888_ColorType
        )
        pil_image = Image.fromarray(skia_image)
        return pil_image
    except Exception as e:
        log_message(f"Skia to PIL conversion error: {e}", always_print=True)
        raise RenderingError("Skia to PIL conversion failed") from e


def draw_layout(
    surface: skia.Surface,
    layout_data: Dict,
    target_center_x: float,
    target_center_y: float,
    loaded_typefaces: Dict[str, Optional[skia.Typeface]],
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    regular_typeface: skia.Typeface,
    regular_hb_face: hb.Face,
    features_to_enable: Dict[str, bool],
    text_color: int,
    use_subpixel_rendering: bool,
    font_hinting: str,
    outline_width: float = 0.0,
    verbose: bool = False,
    pre_translate_x: float = 0.0,
    pre_translate_y: float = 0.0,
    pre_rotate_deg: float = 0.0,
) -> bool:
    """
    Draws the text layout onto a Skia surface.

    This is a "dumb" renderer - all layout decisions have been made.
    It just executes the drawing plan.

    Args:
        surface: Skia surface to draw on
        layout_data: Layout data from layout_engine (font_size, lines, metrics, etc.)
        target_center_x: X coordinate of the text block center
        target_center_y: Y coordinate of the text block center
        loaded_typefaces: Dictionary of Skia typefaces for each style
        loaded_hb_faces: Dictionary of HarfBuzz faces for each style
        regular_typeface: Regular Skia typeface (fallback)
        regular_hb_face: Regular HarfBuzz face (fallback)
        features_to_enable: HarfBuzz features to enable
        text_color: Skia color for the text
        use_subpixel_rendering: Whether to use subpixel rendering
        font_hinting: Font hinting level ("none", "slight", "normal", "full")
        outline_width: Width of text outline (0 = no outline)
        verbose: Whether to print detailed logs

    Returns:
        True if successful, False otherwise
    """
    try:
        final_font_size = layout_data["font_size"]
        final_lines_data = layout_data["lines"]
        final_metrics = layout_data["metrics"]
        final_max_line_width = layout_data["max_line_width"]
        final_line_height = layout_data["line_height"]

        for line_data in final_lines_data:
            line_data["segments"] = parse_styled_segments(
                line_data["text_with_markers"]
            )

        hinting_map = {
            "none": skia.FontHinting.kNone,
            "slight": skia.FontHinting.kSlight,
            "normal": skia.FontHinting.kNormal,
            "full": skia.FontHinting.kFull,
        }
        skia_hinting = hinting_map.get(font_hinting.lower(), skia.FontHinting.kNone)

        paint = skia.Paint(AntiAlias=True, Color=text_color)

        outline_paint = None
        if outline_width > 0:
            # Use opposite color for outline to ensure visibility
            if text_color == skia.ColorBLACK:
                outline_color = skia.ColorWHITE
            else:
                outline_color = skia.ColorBLACK

            outline_paint = skia.Paint(
                AntiAlias=True,
                Color=outline_color,
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=outline_width,
            )

        block_start_x = target_center_x - final_max_line_width / 2.0

        num_lines = len(final_lines_data)
        if num_lines > 0:
            total_visual_height = (
                (num_lines - 1) * final_line_height
                - final_metrics.fAscent
                + final_metrics.fDescent
            )
            block_top_y = target_center_y - (total_visual_height / 2.0)
            first_baseline_y = block_top_y - final_metrics.fAscent
        else:
            first_baseline_y = (
                target_center_y - (final_metrics.fAscent + final_metrics.fDescent) / 2.0
            )

        log_message(
            f"Rendering at size {final_font_size}, center: ({target_center_x:.0f}, {target_center_y:.0f})",
            verbose=verbose,
        )

        with surface as canvas:
            # Apply optional pre-transform (used for rotated OSB rendering)
            need_transform = (
                abs(pre_translate_x) > 1e-3
                or abs(pre_translate_y) > 1e-3
                or abs(pre_rotate_deg) > 1e-3
            )
            if need_transform:
                canvas.save()
                if abs(pre_translate_x) > 1e-3 or abs(pre_translate_y) > 1e-3:
                    canvas.translate(float(pre_translate_x), float(pre_translate_y))
                if abs(pre_rotate_deg) > 1e-3:
                    canvas.rotate(float(pre_rotate_deg))

            current_baseline_y = first_baseline_y
            for i, line_data in enumerate(final_lines_data):
                line_width_measured = line_data["width"]

                line_start_x = (
                    block_start_x + (final_max_line_width - line_width_measured) / 2.0
                )
                cursor_x = line_start_x

                segments = line_data.get("segments", [])
                log_message(f"Line {i}: {len(segments)} segments", verbose=verbose)

                for segment_text, style_name in segments:
                    typeface_to_use = None
                    hb_face_to_use = None
                    fallback_style_used = None

                    # Font fallback: try exact style first, then degrade gracefully
                    if style_name == "bold_italic":
                        typeface_to_use = loaded_typefaces.get("bold_italic")
                        hb_face_to_use = loaded_hb_faces.get("bold_italic")
                        if not typeface_to_use or not hb_face_to_use:
                            fallback_style_used = "bold"
                            typeface_to_use = loaded_typefaces.get("bold")
                            hb_face_to_use = loaded_hb_faces.get("bold")
                        if not typeface_to_use or not hb_face_to_use:
                            fallback_style_used = "italic"
                            typeface_to_use = loaded_typefaces.get("italic")
                            hb_face_to_use = loaded_hb_faces.get("italic")
                        if not typeface_to_use or not hb_face_to_use:
                            fallback_style_used = "regular"
                            typeface_to_use = regular_typeface
                            hb_face_to_use = regular_hb_face
                    elif style_name == "bold":
                        typeface_to_use = loaded_typefaces.get("bold")
                        hb_face_to_use = loaded_hb_faces.get("bold")
                        if not typeface_to_use or not hb_face_to_use:
                            fallback_style_used = "regular"
                            typeface_to_use = regular_typeface
                            hb_face_to_use = regular_hb_face
                    elif style_name == "italic":
                        typeface_to_use = loaded_typefaces.get("italic")
                        hb_face_to_use = loaded_hb_faces.get("italic")
                        if not typeface_to_use or not hb_face_to_use:
                            fallback_style_used = "regular"
                            typeface_to_use = regular_typeface
                            hb_face_to_use = regular_hb_face
                    else:  # Regular or unknown style
                        typeface_to_use = regular_typeface
                        hb_face_to_use = regular_hb_face

                    if fallback_style_used:
                        log_message(
                            f"Style '{style_name}' -> '{fallback_style_used}'",
                            verbose=verbose,
                        )

                    if not typeface_to_use or not hb_face_to_use:
                        log_message(
                            f"ERROR: No font resources for style '{style_name}' - skipping segment",
                            always_print=True,
                        )
                        continue

                    skia_font_segment = skia.Font(typeface_to_use, final_font_size)
                    skia_font_segment.setSubpixel(use_subpixel_rendering)
                    skia_font_segment.setHinting(skia_hinting)

                    hb_font_segment = hb.Font(hb_face_to_use)
                    hb_font_segment.ptem = float(final_font_size)
                    # Standard HarfBuzz scaling: font_size * 64 (for 26.6 fixed point coordinates)
                    hb_scale = int(final_font_size * 64)
                    hb_font_segment.scale = (hb_scale, hb_scale)

                    try:
                        infos, positions = shape_line(
                            segment_text, hb_font_segment, features_to_enable
                        )
                        if not infos:
                            log_message(
                                f"No glyphs for segment '{segment_text}'",
                                verbose=verbose,
                            )
                            continue
                    except Exception as e:
                        log_message(
                            f"Shaping failed for '{segment_text}': {e}",
                            always_print=True,
                        )
                        continue

                    builder = skia.TextBlobBuilder()
                    glyph_ids = [info.codepoint for info in infos]
                    skia_point_positions = []
                    segment_cursor_x = 0

                    # HarfBuzz uses 26.6 fixed-point format (64 units per pixel)
                    HB_26_6_SCALE_FACTOR = 64.0
                    for _, pos in zip(infos, positions):
                        glyph_x = (
                            cursor_x
                            + segment_cursor_x
                            + (pos.x_offset / HB_26_6_SCALE_FACTOR)
                        )
                        glyph_y = current_baseline_y - (
                            pos.y_offset / HB_26_6_SCALE_FACTOR
                        )
                        skia_point_positions.append(skia.Point(glyph_x, glyph_y))

                        segment_cursor_x += pos.x_advance / HB_26_6_SCALE_FACTOR

                    try:
                        _ = builder.allocRunPos(
                            skia_font_segment, glyph_ids, skia_point_positions
                        )

                        text_blob = builder.make()
                        if text_blob:
                            # Draw outline first (if enabled), then fill
                            if outline_paint:
                                canvas.drawTextBlob(text_blob, 0, 0, outline_paint)
                            canvas.drawTextBlob(text_blob, 0, 0, paint)
                            segment_width = segment_cursor_x
                            cursor_x += segment_width
                            log_message(
                                f"Rendered '{segment_text}' ({style_name}) width={segment_width:.0f}",
                                verbose=verbose,
                            )
                        else:
                            log_message(
                                f"TextBlob build failed for '{segment_text}'",
                                verbose=verbose,
                            )

                    except Exception as e:
                        log_message(
                            f"Skia rendering error for '{segment_text}': {e}",
                            always_print=True,
                        )
                        segment_width = segment_cursor_x
                        cursor_x += segment_width

                current_baseline_y += final_line_height

            if need_transform:
                canvas.restore()

        return True

    except Exception as e:
        log_message(f"Drawing failed: {e}", always_print=True)
        return False
