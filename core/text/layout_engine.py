import math
import re
import unicodedata

import numpy as np
import skia
import uharfbuzz as hb

from core.text.text_processing import (
    NO_SPACE_BEFORE_MARKER,
    STYLE_PATTERN,
    WORD_CORE_PATTERN,
    find_optimal_breaks_dp,
    is_detached_trailing_punctuation,
    is_rtl_script,
    parse_styled_segments,
    split_hangul_word_for_wrapping,
    split_thai_word_for_wrapping,
    strip_no_space_before_marker,
    tokenize_styled_text,
    try_hyphenate_word,
)
from utils.exceptions import RenderingError
from utils.logging import log_message

# Epsilon to guard rounding when converting from HarfBuzz 26.6 fixed-point.
VISUAL_WIDTH_EPSILON = 0.0
HB_26_6_SCALE_FACTOR = 64.0
VERTICAL_ADVANCE_TRACKING = 0.90
VERTICAL_GROUPED_PUNCTUATION = set(".,;:!?…。．！？｡")


def shape_line(
    text_line: str, hb_font: hb.Font, features: dict[str, bool]
) -> tuple[list[hb.GlyphInfo], list[hb.GlyphPosition], str]:
    """Shapes a line of text with HarfBuzz.

    Returns:
        Tuple of (glyph_infos, glyph_positions, direction) where direction
        is "ltr" or "rtl" as detected by HarfBuzz.

    Raises:
        RenderingError: If HarfBuzz shaping fails
    """
    hb_buffer = hb.Buffer()
    hb_buffer.add_str(text_line)
    hb_buffer.guess_segment_properties()
    direction = str(hb_buffer.direction)
    try:
        hb.shape(hb_font, hb_buffer, features)
        return hb_buffer.glyph_infos, hb_buffer.glyph_positions, direction
    except Exception as e:
        log_message(f"HarfBuzz shaping failed: {e}", always_print=True)
        raise RenderingError("HarfBuzz text shaping failed") from e


def calculate_line_width(positions: list[hb.GlyphPosition]) -> float:
    """Calculate visual width using advances and first/last x_offset."""
    if not positions:
        return 0.0
    HB_26_6_SCALE_FACTOR = 64.0

    total_advance_fixed = sum(pos.x_advance for pos in positions)
    first_offset_fixed = positions[0].x_offset
    last_offset_fixed = positions[-1].x_offset

    visual_width_fixed = total_advance_fixed + (last_offset_fixed - first_offset_fixed)
    visual_width = float(visual_width_fixed / HB_26_6_SCALE_FACTOR)
    return visual_width + VISUAL_WIDTH_EPSILON


def calculate_styled_line_width(
    line_with_markers: str,
    font_size: int,
    loaded_hb_faces: dict[str, hb.Face | None],
    features: dict[str, bool],
) -> float:
    """Calculate the width of a line that may contain style markers.

    Uses the appropriate HarfBuzz faces per style segment, falling back to
    the 'regular' face if a style-specific face is missing.
    """
    if not line_with_markers:
        return 0.0

    segments = parse_styled_segments(line_with_markers)
    if not segments:
        return 0.0

    regular_face = loaded_hb_faces.get("regular")
    if regular_face is None:
        return 0.0

    total_advance_fixed_all = 0
    first_offset_fixed_global: int | None = None
    last_offset_fixed_global: int | None = None

    for segment_text, style_name in segments:
        hb_face_to_use = (
            loaded_hb_faces.get(style_name)
            if style_name in ("regular", "italic", "bold", "bold_italic")
            else None
        ) or regular_face

        hb_font_segment = hb.Font(hb_face_to_use)
        hb_font_segment.ptem = float(font_size)
        # Standard HarfBuzz scaling: font_size * 64 (for 26.6 fixed point coordinates)
        hb_scale = int(font_size * 64)
        hb_font_segment.scale = (hb_scale, hb_scale)

        _, positions, _ = shape_line(segment_text, hb_font_segment, features)
        if not positions:
            continue

        total_advance_fixed_all += sum(pos.x_advance for pos in positions)
        if first_offset_fixed_global is None:
            first_offset_fixed_global = positions[0].x_offset
        last_offset_fixed_global = positions[-1].x_offset

    if total_advance_fixed_all == 0 and first_offset_fixed_global is None:
        return 0.0

    HB_26_6_SCALE_FACTOR = 64.0
    offset_delta_fixed = 0
    if first_offset_fixed_global is not None and last_offset_fixed_global is not None:
        offset_delta_fixed = last_offset_fixed_global - first_offset_fixed_global

    visual_width_fixed_all = total_advance_fixed_all + offset_delta_fixed
    visual_width_all = float(visual_width_fixed_all / HB_26_6_SCALE_FACTOR)
    return visual_width_all + VISUAL_WIDTH_EPSILON


class _SkiaPathPen:
    """Pen for HarfBuzz glyph outlines that appends them to a skia.Path in pixels (y down)."""

    def __init__(self, path: skia.Path, origin_x: float, origin_y: float):
        self.path = path
        self.origin_x = origin_x
        self.origin_y = origin_y

    def _point(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            self.origin_x + point[0] / HB_26_6_SCALE_FACTOR,
            self.origin_y - point[1] / HB_26_6_SCALE_FACTOR,
        )

    def moveTo(self, point):
        self.path.moveTo(*self._point(point))

    def lineTo(self, point):
        self.path.lineTo(*self._point(point))

    def qCurveTo(self, control, point):
        self.path.quadTo(*self._point(control), *self._point(point))

    def curveTo(self, control_1, control_2, point):
        self.path.cubicTo(
            *self._point(control_1), *self._point(control_2), *self._point(point)
        )

    def closePath(self):
        self.path.close()


def _styled_line_ink_path(
    line_with_markers: str,
    line_width: float,
    font_size: int,
    loaded_hb_faces: dict[str, hb.Face | None],
    features: dict[str, bool],
) -> skia.Path | None:
    """Glyph outlines of a styled line, positioned as draw_layout positions its glyphs.

    The path is in pixels relative to the line's start x and baseline (y down). Returns
    None when no regular face is loaded, so the caller can fall back to the layout box.
    """
    regular_face = loaded_hb_faces.get("regular")
    if regular_face is None:
        return None

    path = skia.Path()
    is_rtl = is_rtl_script(line_with_markers)
    cursor_x = line_width if is_rtl else 0.0

    for segment_text, style_name in parse_styled_segments(line_with_markers):
        hb_face_to_use = (
            loaded_hb_faces.get(style_name)
            if style_name in ("regular", "italic", "bold", "bold_italic")
            else None
        ) or regular_face
        hb_font = hb.Font(hb_face_to_use)
        hb_font.ptem = float(font_size)
        hb_scale = int(font_size * 64)
        hb_font.scale = (hb_scale, hb_scale)

        infos, positions, _ = shape_line(segment_text, hb_font, features)
        if not infos:
            continue
        segment_width = sum(pos.x_advance for pos in positions) / HB_26_6_SCALE_FACTOR
        if is_rtl:
            cursor_x -= segment_width
        glyph_cursor_x = cursor_x

        for info, pos in zip(infos, positions, strict=False):
            pen = _SkiaPathPen(
                path,
                glyph_cursor_x + pos.x_offset / HB_26_6_SCALE_FACTOR,
                -pos.y_offset / HB_26_6_SCALE_FACTOR,
            )
            hb_font.draw_glyph_with_pen(info.codepoint, pen)
            glyph_cursor_x += pos.x_advance / HB_26_6_SCALE_FACTOR

        if not is_rtl:
            cursor_x += segment_width

    return path


def _ink_leaves_safe_area(
    ink_path: skia.Path,
    origin_x: float,
    origin_y: float,
    safe_area_mask: np.ndarray,
) -> bool:
    """Rasterize a line's glyphs at (origin_x, origin_y) and test them against the mask."""
    bounds = ink_path.computeTightBounds()
    x1 = math.floor(origin_x + bounds.left())
    y1 = math.floor(origin_y + bounds.top())
    x2 = math.ceil(origin_x + bounds.right())
    y2 = math.ceil(origin_y + bounds.bottom())
    mask_h, mask_w = safe_area_mask.shape
    if x1 < 0 or y1 < 0 or x2 > mask_w or y2 > mask_h:
        return True

    surface = skia.Surface(max(1, x2 - x1), max(1, y2 - y1))
    with surface as canvas:
        canvas.translate(origin_x - x1, origin_y - y1)
        canvas.drawPath(ink_path, skia.Paint(AntiAlias=True))
    # Pixels the ink covers by less than half are antialiasing fringe, not text
    ink = surface.toarray()[: y2 - y1, : x2 - x1, 3] >= 128
    return bool(np.any(ink & (safe_area_mask[y1:y2, x1:x2] == 0)))


def _is_separator_or_space(ch: str) -> bool:
    try:
        category = unicodedata.category(ch)
    except Exception:
        return ch.isspace()
    return ch.isspace() or (len(category) > 0 and category[0] == "Z")


def _iter_vertical_units(text: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    for segment_text, style_name in parse_styled_segments(text):
        current = ""
        current_is_punctuation = False
        for ch in segment_text:
            if _is_separator_or_space(ch):
                if current:
                    units.append((current, style_name))
                    current = ""
                    current_is_punctuation = False
                continue
            if unicodedata.combining(ch) and current:
                current += ch
                continue
            ch_is_punctuation = ch in VERTICAL_GROUPED_PUNCTUATION
            if current and current_is_punctuation and ch_is_punctuation:
                current += ch
                continue
            if current:
                units.append((current, style_name))
            current = ch
            current_is_punctuation = ch_is_punctuation
        if current:
            units.append((current, style_name))
    return units


def _measure_vertical_unit(
    text: str,
    style_name: str,
    font_size: int,
    loaded_hb_faces: dict[str, hb.Face | None],
    features: dict[str, bool],
) -> dict | None:
    hb_face = loaded_hb_faces.get(style_name) or loaded_hb_faces.get("regular")
    if hb_face is None:
        return None

    hb_font = hb.Font(hb_face)
    hb_font.ptem = float(font_size)
    hb_scale = int(font_size * 64)
    hb_font.scale = (hb_scale, hb_scale)

    infos, positions, _ = shape_line(text, hb_font, features)
    if not infos or not positions:
        return None

    cursor_x = 0
    left = float("inf")
    right = float("-inf")
    top = float("inf")
    bottom = float("-inf")
    vertical_advances = []
    first_vertical_origin_y: float | None = None

    for info, pos in zip(infos, positions, strict=False):
        extents = hb_font.get_glyph_extents(info.codepoint)
        origin_x = cursor_x + pos.x_offset
        origin_y = -pos.y_offset

        if extents:
            glyph_x1 = origin_x + extents.x_bearing
            glyph_x2 = glyph_x1 + extents.width
            glyph_y1 = origin_y - extents.y_bearing
            glyph_y2 = origin_y - (extents.y_bearing + extents.height)
        else:
            glyph_x1 = origin_x
            glyph_x2 = origin_x + pos.x_advance
            glyph_y1 = origin_y - font_size * 64
            glyph_y2 = origin_y

        left = min(left, glyph_x1, glyph_x2)
        right = max(right, glyph_x1, glyph_x2)
        top = min(top, glyph_y1, glyph_y2)
        bottom = max(bottom, glyph_y1, glyph_y2)
        cursor_x += pos.x_advance

        vertical_advance = hb_font.get_glyph_v_advance(info.codepoint)
        if vertical_advance:
            vertical_advances.append(abs(vertical_advance) / HB_26_6_SCALE_FACTOR)

        if first_vertical_origin_y is None:
            vertical_origin = hb_font.get_glyph_v_origin(info.codepoint)
            if vertical_origin is not None:
                _, vertical_origin_y = vertical_origin
                first_vertical_origin_y = vertical_origin_y / HB_26_6_SCALE_FACTOR

    if left == float("inf") or right == float("-inf"):
        return None

    visual_width = max(1.0, (right - left) / HB_26_6_SCALE_FACTOR)
    visual_height = max(1.0, (bottom - top) / HB_26_6_SCALE_FACTOR)
    advance_height = max(vertical_advances) if vertical_advances else 0.0
    if advance_height <= 0:
        advance_height = max(visual_height, float(font_size))

    if first_vertical_origin_y is None:
        first_vertical_origin_y = -top / HB_26_6_SCALE_FACTOR + max(
            0.0, (advance_height - visual_height) / 2.0
        )

    return {
        "text_with_markers": text,
        "style": style_name,
        "width": visual_width,
        "height": visual_height,
        "advance_height": advance_height,
        "baseline_offset_y": first_vertical_origin_y,
        "left": left / HB_26_6_SCALE_FACTOR,
        "right": right / HB_26_6_SCALE_FACTOR,
        "top": top / HB_26_6_SCALE_FACTOR,
        "bottom": bottom / HB_26_6_SCALE_FACTOR,
    }


def _build_vertical_layout(
    text: str,
    font_size: int,
    max_render_width: float,
    max_render_height: float,
    loaded_hb_faces: dict[str, hb.Face | None],
    features_to_enable: dict[str, bool],
    line_spacing_mult: float,
    metrics,
) -> dict | None:
    lines_data_at_size = []
    for unit_text, style_name in _iter_vertical_units(text):
        unit = _measure_vertical_unit(
            unit_text,
            style_name,
            font_size,
            loaded_hb_faces,
            features_to_enable,
        )
        if unit is None:
            continue
        lines_data_at_size.append(unit)

    if not lines_data_at_size:
        return None

    current_y = 0.0
    for index, unit in enumerate(lines_data_at_size):
        advance_height = max(unit["height"], unit.get("advance_height", 0.0))
        step_height = max(
            unit["height"],
            advance_height * line_spacing_mult * VERTICAL_ADVANCE_TRACKING,
        )

        unit["advance_height"] = advance_height
        unit["origin_y"] = current_y + unit.get("baseline_offset_y", 0.0)
        if index == len(lines_data_at_size) - 1:
            current_y += advance_height
        else:
            current_y += step_height

    visual_top = min(unit["origin_y"] + unit["top"] for unit in lines_data_at_size)
    visual_bottom = max(
        unit["origin_y"] + unit["bottom"] for unit in lines_data_at_size
    )
    content_top = min(0.0, visual_top)
    content_bottom = max(current_y, visual_bottom)
    if content_top < 0.0:
        for unit in lines_data_at_size:
            unit["origin_y"] -= content_top
        content_bottom -= content_top

    block_height = max(1.0, content_bottom)
    max_line_width = max(unit["width"] for unit in lines_data_at_size)

    if max_line_width <= max_render_width and block_height <= max_render_height:
        return {
            "lines": lines_data_at_size,
            "metrics": metrics,
            "max_line_width": max_line_width,
            "line_height": 0.0,
            "block_height": block_height,
            "orientation": "vertical",
        }
    return None


def check_fit(
    font_size: int,
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: dict[str, hb.Face | None],
    features_to_enable: dict[str, bool],
    line_spacing_mult: float,
    hyphenate_before_scaling: bool,
    hyphenation_min_word_length: int,
    badness_exponent: float,
    word_width_cache: dict[tuple[str, int], float] | None = None,
    verbose: bool = False,
    detach_trailing_punctuation: bool = True,
    vertical_stack: bool = False,
    hyphenation_language: str = "",
) -> dict | None:
    """Check if text fits within the given dimensions at the specified font size.

    Args:
        font_size: Font size to test
        text: Text to wrap and measure
        max_render_width: Maximum allowed width
        max_render_height: Maximum allowed height
        regular_hb_face: HarfBuzz face for shaping
        regular_typeface: Skia typeface for metrics
        loaded_hb_faces: Dictionary of HarfBuzz faces for each style
        features_to_enable: HarfBuzz features to enable
        line_spacing_mult: Line spacing multiplier
        hyphenate_before_scaling: Whether to hyphenate before scaling
        hyphenation_min_word_length: Minimum word length for hyphenation
        badness_exponent: Exponent for line breaking badness calculation
        word_width_cache: Optional cache for word widths
        verbose: Whether to print detailed logs
        hyphenation_language: Target language name for syllable-aware hyphenation

    Returns:
        Dict containing fit data if successful, None if doesn't fit
    """
    try:
        hb_font = hb.Font(regular_hb_face)
        hb_font.ptem = float(font_size)

        # Standard HarfBuzz scaling: font_size * 64 (for 26.6 fixed point coordinates)
        hb_scale = int(font_size * 64)
        hb_font.scale = (hb_scale, hb_scale)

        skia_font_test = skia.Font(regular_typeface, font_size)
        try:
            metrics = skia_font_test.getMetrics()
            single_line_height = (
                -metrics.fAscent + metrics.fDescent + metrics.fLeading
            ) * line_spacing_mult
            if single_line_height <= 0:
                single_line_height = font_size * 1.2 * line_spacing_mult
        except Exception as e:
            log_message(
                f"Font metrics unavailable at size {font_size}: {e}",
                verbose=verbose,
            )
            single_line_height = font_size * 1.2 * line_spacing_mult

        if vertical_stack:
            return _build_vertical_layout(
                text,
                font_size,
                max_render_width,
                max_render_height,
                loaded_hb_faces,
                features_to_enable,
                line_spacing_mult,
                metrics,
            )

        # Respect explicit newlines as hard line breaks
        if "\n" in text:
            explicit_lines = text.split("\n")
            current_max_line_width = 0.0
            lines_data_at_size = []
            for line_text in explicit_lines:
                width = calculate_styled_line_width(
                    line_text, font_size, loaded_hb_faces, features_to_enable
                )
                lines_data_at_size.append(
                    {"text_with_markers": line_text, "width": width}
                )
                current_max_line_width = max(current_max_line_width, width)

            total_block_height = (-metrics.fAscent + metrics.fDescent) + (
                len(explicit_lines) - 1
            ) * single_line_height

            if (
                current_max_line_width <= max_render_width
                and total_block_height <= max_render_height
            ):
                return {
                    "lines": lines_data_at_size,
                    "metrics": metrics,
                    "max_line_width": current_max_line_width,
                    "line_height": single_line_height,
                    "block_height": total_block_height,
                }
            return None

        tokens: list[tuple[str, bool]] = tokenize_styled_text(
            text, detach_trailing_punctuation
        )
        augmented_tokens: list[str] = []

        if hyphenate_before_scaling:
            for token_text, is_styled in tokens:
                marker = ""
                core_text = token_text

                if is_styled:
                    styled_match = STYLE_PATTERN.match(token_text)
                    if not styled_match:
                        augmented_tokens.append(token_text)
                        continue
                    marker = styled_match.group(1)
                    core_text = styled_match.group(2)

                match = WORD_CORE_PATTERN.match(core_text)
                core_word_length = len(match.group(2)) if match else len(core_text)

                if core_word_length >= hyphenation_min_word_length:
                    word_width = calculate_styled_line_width(
                        token_text, font_size, loaded_hb_faces, features_to_enable
                    )

                    if word_width > max_render_width:

                        def wrap_part(part: str, m: str = marker) -> str:
                            no_space_before = part.startswith(NO_SPACE_BEFORE_MARKER)
                            clean_part = strip_no_space_before_marker(part)
                            wrapped = f"{m}{clean_part}{m}" if m else clean_part
                            if no_space_before:
                                return f"{NO_SPACE_BEFORE_MARKER}{wrapped}"
                            return wrapped

                        def width_test_func(part: str) -> bool:
                            wrapped = wrap_part(part)
                            w = calculate_styled_line_width(
                                wrapped, font_size, loaded_hb_faces, features_to_enable
                            )
                            return w <= max_render_width

                        split_parts = split_hangul_word_for_wrapping(core_text)
                        if split_parts is None:
                            split_parts = split_thai_word_for_wrapping(core_text)
                        if split_parts is None:
                            split_parts = try_hyphenate_word(
                                core_text,
                                hyphenation_min_word_length,
                                width_test_func,
                                hyphenation_language,
                            )

                        if split_parts:
                            augmented_tokens.extend(wrap_part(p) for p in split_parts)
                        else:
                            augmented_tokens.append(token_text)
                    else:
                        augmented_tokens.append(token_text)
                else:
                    augmented_tokens.append(token_text)
        else:
            augmented_tokens = [t for t, _ in tokens]

        try:
            GLUE_TRAILING_PUNCT_RE = re.compile(r"^[,.;:!?…]+$")
            GLUE_CLOSERS_RE = re.compile(r"^[\)\]\}\u2019\u201D\'\"]+$")

            def _glue_trailing_punctuation(
                tokens_list: list[str], _detach: bool = True
            ) -> list[str]:
                glued: list[str] = []
                for tok in tokens_list:
                    clean_tok = strip_no_space_before_marker(tok)
                    match = STYLE_PATTERN.match(clean_tok)
                    content = match.group(2) if match else clean_tok

                    if _detach and is_detached_trailing_punctuation(content):
                        glued.append(tok)
                        continue

                    if glued and (
                        GLUE_TRAILING_PUNCT_RE.match(content)
                        or GLUE_CLOSERS_RE.match(content)
                    ):
                        glued[-1] = glued[-1] + clean_tok
                    else:
                        glued.append(tok)
                return glued

            augmented_tokens = _glue_trailing_punctuation(
                augmented_tokens, detach_trailing_punctuation
            )
        except Exception:
            pass

        def word_width_func(word: str) -> float:
            if word_width_cache is not None:
                cached_key = (word, font_size)
                if cached_key in word_width_cache:
                    return word_width_cache[cached_key]

            clean_word = strip_no_space_before_marker(word)
            width_val = calculate_styled_line_width(
                clean_word, font_size, loaded_hb_faces, features_to_enable
            )

            if word_width_cache is not None:
                word_width_cache[(word, font_size)] = width_val

            return width_val

        space_width = calculate_styled_line_width(
            " ", font_size, loaded_hb_faces, features_to_enable
        )

        wrapped_lines_text = find_optimal_breaks_dp(
            augmented_tokens,
            max_render_width,
            word_width_func,
            space_width,
            badness_exponent,
            detach_trailing_punctuation,
        )

        if not wrapped_lines_text:
            return None

        current_max_line_width = 0
        lines_data_at_size = []
        for line_text_with_markers in wrapped_lines_text:
            width = calculate_styled_line_width(
                line_text_with_markers, font_size, loaded_hb_faces, features_to_enable
            )
            lines_data_at_size.append(
                {"text_with_markers": line_text_with_markers, "width": width}
            )
            current_max_line_width = max(current_max_line_width, width)

        total_block_height = (-metrics.fAscent + metrics.fDescent) + (
            len(wrapped_lines_text) - 1
        ) * single_line_height

        log_message(
            f"Size {font_size}: {current_max_line_width:.0f}x{total_block_height:.0f} "
            f"(max {max_render_width:.0f}x{max_render_height:.0f})",
            verbose=verbose,
        )

        if (
            current_max_line_width <= max_render_width
            and total_block_height <= max_render_height
        ):
            log_message(f"Size {font_size} fits", verbose=verbose)
            return {
                "lines": lines_data_at_size,
                "metrics": metrics,
                "max_line_width": current_max_line_width,
                "line_height": single_line_height,
                "block_height": total_block_height,
            }

        return None

    except Exception as e:
        log_message(f"Fit check failed at size {font_size}: {e}", verbose=verbose)
        return None


def _check_collision(
    lines_data: list[dict],
    line_ink_paths: list[skia.Path | None],
    box_top_left: tuple[int, int],
    safe_area_mask: np.ndarray,
    line_height: float,
    metrics: skia.FontMetrics,
    box_size: tuple[float, float],
) -> bool:
    """
    Check whether any line's glyph ink leaves the padded safe area.

    Lines are placed as draw_layout places them: each centered on the box's
    center, with the block centered vertically on its visual height.

    Args:
        lines_data: List of dictionaries containing line width and text.
        line_ink_paths: Per line glyph outlines relative to the line's start x and
            baseline; None tests the line's full layout box instead.
        box_top_left: (x, y) coordinates of the layout box top-left corner.
        safe_area_mask: Bubble pixels far enough from the outline (255), else 0.
        line_height: Distance between consecutive baselines.
        metrics: Font metrics providing fAscent (negative) and fDescent.
        box_size: (width, height) of the full layout box, not a squeezed width.

    Returns:
        True if collision detected, False otherwise.
    """
    box_x, box_y = box_top_left
    mask_h, mask_w = safe_area_mask.shape
    box_w, box_h = box_size
    center_x = box_x + box_w / 2.0
    center_y = box_y + box_h / 2.0

    glyph_height = metrics.fDescent - metrics.fAscent
    total_visual_height = (len(lines_data) - 1) * line_height + glyph_height
    baseline_y = center_y - total_visual_height / 2.0 - metrics.fAscent

    for line, ink_path in zip(lines_data, line_ink_paths, strict=True):
        line_left = center_x - line["width"] / 2.0
        if ink_path is None:
            x1, x2 = round(line_left), round(line_left + line["width"])
            y1 = round(baseline_y + metrics.fAscent)
            y2 = round(baseline_y + metrics.fDescent)
            if x1 < 0 or y1 < 0 or x2 > mask_w or y2 > mask_h:
                return True
            if not np.all(safe_area_mask[y1:y2, x1:x2]):
                return True
        elif not ink_path.isEmpty() and _ink_leaves_safe_area(
            ink_path, line_left, baseline_y, safe_area_mask
        ):
            return True
        baseline_y += line_height

    return False


def find_optimal_layout(
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: dict[str, hb.Face | None],
    features_to_enable: dict[str, bool],
    min_font_size: int = 8,
    max_font_size: int = 16,
    line_spacing_mult: float = 1.0,
    hyphenate_before_scaling: bool = True,
    hyphenation_min_word_length: int = 8,
    badness_exponent: float = 3.0,
    verbose: bool = False,
    bubble_id: str | None = None,
    safe_area_mask: np.ndarray | None = None,
    box_top_left: tuple[int, int] | None = None,
    detach_trailing_punctuation: bool = True,
    vertical_stack: bool = False,
    hyphenation_language: str = "",
) -> dict:
    """Find the optimal font size and layout for text within given dimensions.

    Uses binary search to find the largest font size that fits.

    Args:
        text: Text to layout
        max_render_width: Maximum allowed width
        max_render_height: Maximum allowed height
        regular_hb_face: HarfBuzz face for the regular font
        regular_typeface: Skia typeface for the regular font
        loaded_hb_faces: Dictionary of HarfBuzz faces for each style
        features_to_enable: HarfBuzz features to enable
        min_font_size: Minimum font size to try
        max_font_size: Maximum font size to try
        line_spacing_mult: Line spacing multiplier
        hyphenate_before_scaling: Whether to hyphenate before reducing font size
        hyphenation_min_word_length: Minimum word length for hyphenation
        badness_exponent: Exponent for line breaking badness calculation
        verbose: Whether to print detailed logs
        bubble_id: Optional identifier for the bubble (for logging purposes)
        safe_area_mask: Optional padded safe-area mask of the bubble for collision detection
        box_top_left: Optional (x, y) coordinates of the bounding box top-left corner
        hyphenation_language: Target language name for syllable-aware hyphenation

    Returns:
        Dictionary containing layout data (font_size, lines, metrics, etc.)

    Raises:
        RenderingError: If text doesn't fit at minimum font size or layout fails
    """
    # Preserve explicit newlines if present, otherwise collapse whitespace for
    # normal paragraph layout. Vertical layout handles whitespace per segment.
    if "\n" in text or "\r" in text:
        clean_text = text.replace("\r\n", "\n").replace("\r", "\n")
    else:
        clean_text = " ".join(text.split())
    if not clean_text:
        raise RenderingError("Empty text cannot be laid out")

    best_fit_size = -1
    best_fit_lines_data = None
    best_fit_metrics = None
    best_fit_max_line_width = float("inf")
    best_fit_line_height = 0.0
    best_fit_block_height = 0.0

    word_width_cache: dict[tuple[str, int], float] = {}

    low = min_font_size
    high = max_font_size

    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            break

        log_message(f"Testing size {mid}", verbose=verbose)

        succeeded_at_current_size = False
        current_width_attempt = max_render_width
        max_squeezes = 3 if safe_area_mask is not None else 1

        for _ in range(max_squeezes):
            fit_data = check_fit(
                mid,
                clean_text,
                current_width_attempt,
                max_render_height,
                regular_hb_face,
                regular_typeface,
                loaded_hb_faces,
                features_to_enable,
                line_spacing_mult,
                hyphenate_before_scaling,
                hyphenation_min_word_length,
                badness_exponent,
                word_width_cache,
                verbose,
                detach_trailing_punctuation,
                vertical_stack,
                hyphenation_language,
            )

            if fit_data is None:
                # Squeezing narrower won't help (only makes it taller)
                break

            if (
                safe_area_mask is not None
                and box_top_left is not None
                and not vertical_stack
            ):
                line_ink_paths = [
                    _styled_line_ink_path(
                        line.get("text_with_markers", ""),
                        line["width"],
                        mid,
                        loaded_hb_faces,
                        features_to_enable,
                    )
                    for line in fit_data["lines"]
                ]
                has_collision = _check_collision(
                    fit_data["lines"],
                    line_ink_paths,
                    box_top_left,
                    safe_area_mask,
                    fit_data["line_height"],
                    fit_data["metrics"],
                    (max_render_width, max_render_height),
                )

                if not has_collision:
                    best_fit_size = mid
                    best_fit_lines_data = fit_data["lines"]
                    best_fit_metrics = fit_data["metrics"]
                    best_fit_max_line_width = fit_data["max_line_width"]
                    best_fit_line_height = fit_data["line_height"]
                    best_fit_block_height = fit_data.get("block_height", 0.0)

                    succeeded_at_current_size = True
                    break
                else:
                    log_message(
                        f"Collision at size {mid} width {current_width_attempt:.0f}, squeezing...",
                        verbose=verbose,
                    )
                    current_width_attempt *= 0.90
                    continue
            else:
                best_fit_size = mid
                best_fit_lines_data = fit_data["lines"]
                best_fit_metrics = fit_data["metrics"]
                best_fit_max_line_width = fit_data["max_line_width"]
                best_fit_line_height = fit_data["line_height"]
                best_fit_block_height = fit_data.get("block_height", 0.0)
                succeeded_at_current_size = True
                break

        if succeeded_at_current_size:
            low = mid + 1
        else:
            high = mid - 1

    if best_fit_size == -1:
        log_message(
            f"Text too large for bubble at min size {min_font_size}: '{clean_text[:30]}'",
            always_print=True,
        )
        raise RenderingError(
            f"Text too large for bubble at minimum font size {min_font_size}"
        )

    if best_fit_size < max_font_size:
        bubble_desc = f"bubble {bubble_id}" if bubble_id else "bubble"
        log_message(
            f"Shrinking text in {bubble_desc} to size {best_fit_size}",
            verbose=verbose,
        )

    return {
        "font_size": best_fit_size,
        "lines": best_fit_lines_data,
        "metrics": best_fit_metrics,
        "max_line_width": best_fit_max_line_width,
        "line_height": best_fit_line_height,
        "orientation": "vertical" if vertical_stack else "horizontal",
        "block_height": best_fit_block_height,
    }
