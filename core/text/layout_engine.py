import re
from typing import Dict, List, Optional, Tuple

import skia
import uharfbuzz as hb

from core.text.text_processing import (find_optimal_breaks_dp,
                                       parse_styled_segments,
                                       tokenize_styled_text,
                                       try_hyphenate_word)
from utils.exceptions import RenderingError
from utils.logging import log_message

# Epsilon to guard rounding when converting from HarfBuzz 26.6 fixed-point.
VISUAL_WIDTH_EPSILON = 0.0


def shape_line(
    text_line: str, hb_font: hb.Font, features: Dict[str, bool]
) -> Tuple[List[hb.GlyphInfo], List[hb.GlyphPosition]]:
    """Shapes a line of text with HarfBuzz.

    Raises:
        RenderingError: If HarfBuzz shaping fails
    """
    hb_buffer = hb.Buffer()
    hb_buffer.add_str(text_line)
    hb_buffer.guess_segment_properties()
    try:
        hb.shape(hb_font, hb_buffer, features)
        return hb_buffer.glyph_infos, hb_buffer.glyph_positions
    except Exception as e:
        log_message(f"HarfBuzz shaping failed: {e}", always_print=True)
        raise RenderingError("HarfBuzz text shaping failed") from e


def calculate_line_width(positions: List[hb.GlyphPosition]) -> float:
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
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features: Dict[str, bool],
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
    first_offset_fixed_global: Optional[int] = None
    last_offset_fixed_global: Optional[int] = None

    for segment_text, style_name in segments:
        hb_face_to_use = (
            loaded_hb_faces.get(style_name)
            if style_name in ("regular", "italic", "bold", "bold_italic")
            else None
        ) or regular_face

        hb_font_segment = hb.Font(hb_face_to_use)
        hb_font_segment.ptem = float(font_size)
        if hb_face_to_use.upem > 0:
            scale_factor = font_size / hb_face_to_use.upem
            hb_scale = int(scale_factor * (2**16))
            hb_font_segment.scale = (hb_scale, hb_scale)
        else:
            hb_font_segment.scale = (int(font_size * (2**16)), int(font_size * (2**16)))

        _, positions = shape_line(segment_text, hb_font_segment, features)
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


def check_fit(
    font_size: int,
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    line_spacing_mult: float,
    hyphenate_before_scaling: bool,
    hyphen_penalty: float,
    hyphenation_min_word_length: int,
    badness_exponent: float,
    word_width_cache: Optional[Dict[Tuple[str, int], float]] = None,
    verbose: bool = False,
) -> Optional[Dict]:
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
        hyphen_penalty: Penalty for hyphenated lines
        hyphenation_min_word_length: Minimum word length for hyphenation
        badness_exponent: Exponent for line breaking badness calculation
        word_width_cache: Optional cache for word widths
        verbose: Whether to print detailed logs

    Returns:
        Dict containing fit data if successful, None if doesn't fit
    """
    try:
        hb_font = hb.Font(regular_hb_face)
        hb_font.ptem = float(font_size)

        scale_factor = 1.0
        if regular_hb_face.upem > 0:
            scale_factor = font_size / regular_hb_face.upem
        else:
            if verbose:
                log_message("Font upem=0, using scale factor 1.0", verbose=verbose)

        hb_scale = int(scale_factor * (2**16))
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
            if verbose:
                log_message(
                    f"Font metrics unavailable at size {font_size}: {e}",
                    verbose=verbose,
                )
            single_line_height = font_size * 1.2 * line_spacing_mult

        # Respect explicit newlines as hard line breaks (e.g., for vertical stacking)
        if "\n" in text:
            explicit_lines = text.split("\n")
            # Measure each explicit line without further wrapping
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
                }
            return None

        tokens: List[Tuple[str, bool]] = tokenize_styled_text(text)
        augmented_tokens: List[str] = []

        if hyphenate_before_scaling:
            for token_text, is_styled in tokens:
                if not is_styled:
                    # Only hyphenate words that are long enough and exceed width
                    match = re.match(r"^(\W*)([\w\-]+)(\W*)$", token_text)
                    if match:
                        core_word_length = len(match.group(2))
                    else:
                        core_word_length = len(token_text)

                    if core_word_length > hyphenation_min_word_length:
                        word_width = calculate_styled_line_width(
                            token_text, font_size, loaded_hb_faces, features_to_enable
                        )

                        if word_width > max_render_width:

                            def width_test_func(part: str) -> bool:
                                w = calculate_styled_line_width(
                                    part, font_size, loaded_hb_faces, features_to_enable
                                )
                                return w <= max_render_width

                            split_parts = try_hyphenate_word(
                                token_text, hyphenation_min_word_length, width_test_func
                            )
                            if split_parts:
                                augmented_tokens.extend(split_parts)
                            else:
                                augmented_tokens.append(token_text)
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

            def _glue_trailing_punctuation(tokens_list: List[str]) -> List[str]:
                glued: List[str] = []
                for tok in tokens_list:
                    if glued and (
                        GLUE_TRAILING_PUNCT_RE.match(tok) or GLUE_CLOSERS_RE.match(tok)
                    ):
                        glued[-1] = glued[-1] + tok
                    else:
                        glued.append(tok)
                return glued

            augmented_tokens = _glue_trailing_punctuation(augmented_tokens)
        except Exception:
            pass

        def word_width_func(word: str) -> float:
            if word_width_cache is not None:
                cached_key = (word, font_size)
                if cached_key in word_width_cache:
                    return word_width_cache[cached_key]

            width_val = calculate_styled_line_width(
                word, font_size, loaded_hb_faces, features_to_enable
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
            hyphen_penalty,
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

        if verbose:
            log_message(
                f"Size {font_size}: {current_max_line_width:.0f}x{total_block_height:.0f} "
                f"(max {max_render_width:.0f}x{max_render_height:.0f})",
                verbose=verbose,
            )

        if (
            current_max_line_width <= max_render_width
            and total_block_height <= max_render_height
        ):
            if verbose:
                log_message(f"Size {font_size} fits", verbose=verbose)
            return {
                "lines": lines_data_at_size,
                "metrics": metrics,
                "max_line_width": current_max_line_width,
                "line_height": single_line_height,
            }

        return None

    except Exception as e:
        if verbose:
            log_message(f"Fit check failed at size {font_size}: {e}", verbose=verbose)
        return None


def find_optimal_layout(
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    min_font_size: int = 8,
    max_font_size: int = 14,
    line_spacing_mult: float = 1.0,
    hyphenate_before_scaling: bool = True,
    hyphen_penalty: float = 1000.0,
    hyphenation_min_word_length: int = 8,
    badness_exponent: float = 3.0,
    verbose: bool = False,
    bubble_id: Optional[str] = None,
) -> Dict:
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
        hyphen_penalty: Penalty for hyphenated lines
        hyphenation_min_word_length: Minimum word length for hyphenation
        badness_exponent: Exponent for line breaking badness calculation
        verbose: Whether to print detailed logs
        bubble_id: Optional identifier for the bubble (for logging purposes)

    Returns:
        Dictionary containing layout data (font_size, lines, metrics, etc.)

    Raises:
        RenderingError: If text doesn't fit at minimum font size or layout fails
    """
    # Preserve explicit newlines if present (e.g., vertical stacking),
    # otherwise collapse whitespace for normal paragraph layout
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

    word_width_cache: Dict[Tuple[str, int], float] = {}

    low = min_font_size
    high = max_font_size

    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            break

        log_message(f"Testing size {mid}", verbose=verbose)

        fit_data = check_fit(
            mid,
            clean_text,
            max_render_width,
            max_render_height,
            regular_hb_face,
            regular_typeface,
            loaded_hb_faces,
            features_to_enable,
            line_spacing_mult,
            hyphenate_before_scaling,
            hyphen_penalty,
            hyphenation_min_word_length,
            badness_exponent,
            word_width_cache,
            verbose,
        )

        if fit_data is not None:
            best_fit_size = mid
            best_fit_lines_data = fit_data["lines"]
            best_fit_metrics = fit_data["metrics"]
            best_fit_max_line_width = fit_data["max_line_width"]
            best_fit_line_height = fit_data["line_height"]
            low = mid + 1
        else:
            high = mid - 1

    if best_fit_size == -1:
        log_message(
            f"Text too large for bubble at min size {min_font_size}: '{clean_text[:30]}...'",
            verbose=verbose,
        )
        raise RenderingError(
            f"Text too large for bubble at minimum font size {min_font_size}"
        )

    # Log when text is shrunk to fit
    if best_fit_size < max_font_size:
        bubble_desc = f"bubble {bubble_id}" if bubble_id else "bubble"
        log_message(
            f"Shrinking text in {bubble_desc} to size {best_fit_size}",
            always_print=True,
        )

    return {
        "font_size": best_fit_size,
        "lines": best_fit_lines_data,
        "metrics": best_fit_metrics,
        "max_line_width": best_fit_max_line_width,
        "line_height": best_fit_line_height,
    }
