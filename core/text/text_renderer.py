import math

import numpy as np
import skia
from PIL import Image

from core.config import RenderingConfig
from core.image.image_utils import calculate_centroid_expansion_box
from core.text.drawing_engine import (
    draw_layout,
    load_font_resources,
    pil_to_skia_surface,
    resolve_outline_width,
    skia_surface_to_pil,
)
from core.text.font_manager import (
    find_font_variants,
    get_font_features,
    sanitize_text_for_font,
)
from core.text.layout_engine import find_optimal_layout
from core.text.text_processing import is_rtl_script, parse_styled_segments
from utils.exceptions import FontError, ImageProcessingError, RenderingError
from utils.logging import log_message

GRAYSCALE_MIDPOINT = 128  # Threshold for determining text color
FALLBACK_PADDING_RATIO = 0.08  # 8% padding ratio when safe area calculation fails
AUTO_VERTICAL_MIN_ASPECT_RATIO = 1.6
AUTO_VERTICAL_MAX_CHARS = 12
AUTO_VERTICAL_MAX_WORDS = 1
AUTO_VERTICAL_MAX_HORIZONTAL_FILL = 0.45
AUTO_VERTICAL_MIN_FILL_GAIN = 0.20
OSB_TILT_SUPERSAMPLE_MARGIN = 8


def _plain_text_for_layout_policy(text: str) -> str:
    return "".join(segment_text for segment_text, _ in parse_styled_segments(text))


def _should_try_auto_vertical_text(
    text: str,
    max_render_width: float,
    max_render_height: float,
) -> bool:
    if max_render_width <= 0 or max_render_height <= 0:
        return False
    if max_render_height / max_render_width < AUTO_VERTICAL_MIN_ASPECT_RATIO:
        return False

    plain_text = _plain_text_for_layout_policy(text).strip()
    if not plain_text or is_rtl_script(plain_text):
        return False

    non_space_chars = sum(1 for ch in plain_text if not ch.isspace())
    word_count = len(plain_text.split())
    return (
        non_space_chars <= AUTO_VERTICAL_MAX_CHARS
        and word_count <= AUTO_VERTICAL_MAX_WORDS
    )


def _auto_vertical_layout_is_better(
    horizontal_layout: dict,
    vertical_layout: dict,
    max_render_height: float,
) -> bool:
    if max_render_height <= 0:
        return False

    horizontal_block_height = horizontal_layout.get("block_height") or 0.0
    vertical_block_height = vertical_layout.get("block_height") or 0.0
    if horizontal_block_height <= 0 or vertical_block_height <= 0:
        return False

    horizontal_fill = horizontal_block_height / max_render_height
    vertical_fill = vertical_block_height / max_render_height
    return (
        vertical_layout["font_size"] >= horizontal_layout["font_size"]
        and horizontal_fill <= AUTO_VERTICAL_MAX_HORIZONTAL_FILL
        and vertical_fill >= horizontal_fill + AUTO_VERTICAL_MIN_FILL_GAIN
    )


class _ScaledMetrics:
    """Font metrics scaled by the supersampling factor."""

    def __init__(self, original, scale_factor):
        self.fAscent = original.fAscent * scale_factor
        self.fDescent = original.fDescent * scale_factor
        # Preserve other attributes if they exist
        if hasattr(original, "fLeading"):
            self.fLeading = original.fLeading * scale_factor
        if hasattr(original, "fXMin"):
            self.fXMin = original.fXMin * scale_factor
        if hasattr(original, "fXMax"):
            self.fXMax = original.fXMax * scale_factor
        if hasattr(original, "fYMin"):
            self.fYMin = original.fYMin * scale_factor
        if hasattr(original, "fYMax"):
            self.fYMax = original.fYMax * scale_factor


def _scale_layout_data(layout_data: dict, factor: float) -> dict:
    """Copy of `layout_data` with every size and position scaled by `factor`."""
    scaled_layout_data = layout_data.copy()
    scaled_layout_data["lines"] = [
        line_data.copy() for line_data in layout_data["lines"]
    ]
    scaled_layout_data["font_size"] = layout_data["font_size"] * factor
    scaled_layout_data["line_height"] = layout_data["line_height"] * factor
    scaled_layout_data["max_line_width"] = layout_data["max_line_width"] * factor
    if layout_data.get("block_height") is not None:
        scaled_layout_data["block_height"] = layout_data["block_height"] * factor

    # Scale per-line and per-glyph measurements
    for line_data in scaled_layout_data["lines"]:
        for key in (
            "width",
            "height",
            "advance_height",
            "baseline_offset_y",
            "left",
            "right",
            "top",
            "bottom",
            "origin_y",
        ):
            if key in line_data:
                line_data[key] = line_data[key] * factor

    scaled_layout_data["metrics"] = _ScaledMetrics(layout_data["metrics"], factor)
    return scaled_layout_data


def _load_style_variants(
    clean_text: str,
    font_variants: dict,
    regular_typeface,
    regular_hb_face,
    verbose: bool,
) -> tuple[dict, dict]:
    """Load the italic/bold variants the text's style markers use (regular as fallback)."""
    required_styles = {"regular"} | {
        style for _, style in parse_styled_segments(clean_text)
    }
    log_message(f"Required styles: {sorted(required_styles)}", verbose=verbose)

    loaded_typefaces = {"regular": regular_typeface}
    loaded_hb_faces = {"regular": regular_hb_face}

    for style in ["italic", "bold", "bold_italic"]:
        if style in required_styles:
            font_path = font_variants.get(style)
            if font_path:
                log_message(f"Loading {style}: {font_path.name}", verbose=verbose)
                _, typeface, hb_face = load_font_resources(str(font_path), style=style)
                if typeface and hb_face:
                    loaded_typefaces[style] = typeface
                    loaded_hb_faces[style] = hb_face
                else:
                    log_message(
                        f"Failed to load {style} variant, using regular",
                        verbose=verbose,
                    )
            else:
                log_message(
                    f"Style '{style}' not found, using regular",
                    verbose=verbose,
                )
    return loaded_typefaces, loaded_hb_faces


def _resolve_text_color(
    text_color_rgb: tuple[int, int, int] | None,
    bubble_color_bgr: tuple[int, int, int] | None,
):
    """Explicit text color, else black or white by contrast with the bubble background."""
    if text_color_rgb is not None:
        return skia.Color(text_color_rgb[0], text_color_rgb[1], text_color_rgb[2])
    if bubble_color_bgr is None:
        return skia.ColorBLACK
    try:
        # bubble_color_bgr may be a grayscale proxy; treat as BGR
        bg_brightness = (
            bubble_color_bgr[0] + bubble_color_bgr[1] + bubble_color_bgr[2]
        ) / 3.0
        # If background is dark, use white text; if light, use black
        return (
            skia.ColorWHITE if bg_brightness < GRAYSCALE_MIDPOINT else skia.ColorBLACK
        )
    except Exception:
        return skia.ColorBLACK


def _draw_text(
    surface,
    layout_data: dict,
    center: tuple[float, float],
    rotation_deg: float,
    fonts: tuple,
    text_color,
    outline_width: float,
    verbose: bool,
    text_background_color,
) -> None:
    """Draw the layout centered on `center`, rotated about it when tilted.

    `fonts` is (typefaces, hb_faces, regular_typeface, regular_hb_face, features).
    Rotation and translation are delegated to drawing_engine so Skia state stays
    consistent."""
    rotated = bool(rotation_deg and abs(rotation_deg) > 0.01)
    center_x, center_y = center
    success = draw_layout(
        surface,
        layout_data,
        0.0 if rotated else center_x,
        0.0 if rotated else center_y,
        *fonts,
        text_color,
        outline_width,
        verbose,
        pre_translate_x=float(center_x) if rotated else 0.0,
        pre_translate_y=float(center_y) if rotated else 0.0,
        pre_rotate_deg=float(rotation_deg) if rotated else 0.0,
        text_background_color=text_background_color,
    )
    if not success:
        log_message("Drawing failed", always_print=True)
        raise RenderingError("Text drawing failed")


def _render_supersampled(
    pil_image: Image.Image,
    bbox: tuple[int, int, int, int],
    layout_data: dict,
    center: tuple[float, float],
    rotation_deg: float,
    factor: int,
    fonts: tuple,
    text_color,
    outline_width: float,
    verbose: bool,
    text_background_color,
) -> Image.Image:
    """Draw at `factor`x on an upscaled crop of the bbox, then downscale and paste back."""
    x1, y1, x2, y2 = bbox
    # Crop the bbox region (plus room for tilted text), within image bounds
    img_width, img_height = pil_image.size
    tilt_margin = 0
    if rotation_deg and abs(rotation_deg) > 0.01:
        tilt_margin = (
            math.ceil(
                max(x2 - x1, y2 - y1) * math.sin(math.radians(abs(rotation_deg))) / 2
            )
            + OSB_TILT_SUPERSAMPLE_MARGIN
        )
    crop_x1 = max(0, x1 - tilt_margin)
    crop_y1 = max(0, y1 - tilt_margin)
    crop_x2 = min(img_width, x2 + tilt_margin)
    crop_y2 = min(img_height, y2 + tilt_margin)

    cropped_region = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    upscaled_region = cropped_region.resize(
        (int(crop_width * factor), int(crop_height * factor)), Image.Resampling.LANCZOS
    )

    # Scale coordinates relative to the crop origin
    scaled_center = (
        (center[0] - crop_x1) * factor,
        (center[1] - crop_y1) * factor,
    )
    scaled_layout_data = _scale_layout_data(layout_data, factor)

    try:
        scaled_surface = pil_to_skia_surface(upscaled_region)
    except RenderingError as e:
        raise RenderingError(f"Scaled surface preparation failed: {e}") from e

    _draw_text(
        scaled_surface,
        scaled_layout_data,
        scaled_center,
        rotation_deg,
        fonts,
        text_color,
        outline_width * factor,
        verbose,
        text_background_color,
    )

    try:
        scaled_pil_result = skia_surface_to_pil(scaled_surface)
    except RenderingError as e:
        raise RenderingError(f"Scaled conversion failed: {e}") from e

    # Downscale using LANCZOS for high quality
    downscaled_result = scaled_pil_result.resize(
        (crop_width, crop_height), Image.Resampling.LANCZOS
    )
    final_pil_image = pil_image.copy()
    final_pil_image.paste(downscaled_result, (crop_x1, crop_y1))
    return final_pil_image


def _text_area(
    bbox: tuple[int, int, int, int],
    cleaned_mask: np.ndarray | None,
    config: RenderingConfig,
    raise_on_safe_error: bool,
    layout_dimensions: tuple[float, float] | None,
    fallback_padding_pixels: float | None,
    verbose: bool,
):
    """Where text may go: the mask's centered safe box, else a padded bbox.

    Returns (layout box top-left or None, max width, max height, center x, center y).
    """
    x1, y1, x2, y2 = bbox
    bubble_width = x2 - x1
    bubble_height = y2 - y1
    layout_box_top_left = None
    safe_area_result = None
    safe_area_fallback_logged = False
    if cleaned_mask is not None:
        try:
            safe_area_result = calculate_centroid_expansion_box(
                cleaned_mask, padding_pixels=config.padding_pixels, verbose=verbose
            )
        except ImageProcessingError:
            # Safe area calculation failed, will use fallback below
            safe_area_result = None
            if raise_on_safe_error:
                raise
            log_message(
                "Safe area calculation failed, falling back to padded bbox method",
                verbose=verbose,
            )
            safe_area_fallback_logged = True

    if safe_area_result is not None:
        guaranteed_box, _ = safe_area_result
        box_x, box_y, box_w, box_h = guaranteed_box
        layout_box_top_left = (box_x, box_y)
        max_render_width = float(box_w)
        max_render_height = float(box_h)
        target_center_x = box_x + box_w / 2.0
        target_center_y = box_y + box_h / 2.0
        log_message("Using centroid-based safe area calculation", verbose=verbose)
    else:
        # Fallback to padded bbox
        if not safe_area_fallback_logged:
            log_message(
                "Safe area calculation failed, falling back to padded bbox method",
                verbose=verbose,
            )
        if layout_dimensions is not None:
            max_render_width, max_render_height = layout_dimensions
        elif fallback_padding_pixels is not None:
            max_render_width = bubble_width - 2 * fallback_padding_pixels
            max_render_height = bubble_height - 2 * fallback_padding_pixels
        else:
            max_render_width = bubble_width * (1 - 2 * FALLBACK_PADDING_RATIO)
            max_render_height = bubble_height * (1 - 2 * FALLBACK_PADDING_RATIO)

        if max_render_width <= 0 or max_render_height <= 0:
            max_render_width = max(1.0, float(bubble_width))
            max_render_height = max(1.0, float(bubble_height))

        target_center_x = x1 + bubble_width / 2.0
        target_center_y = y1 + bubble_height / 2.0
    return (
        layout_box_top_left,
        max_render_width,
        max_render_height,
        target_center_x,
        target_center_y,
    )


def render_text_skia(
    pil_image: Image.Image,
    text: str,
    bbox: tuple[int, int, int, int],
    font_dir: str,
    cleaned_mask: np.ndarray | None = None,
    bubble_color_bgr: tuple[int, int, int] | None = (255, 255, 255),
    config: RenderingConfig | None = None,
    raise_on_safe_error: bool = False,
    verbose: bool = False,
    bubble_id: str | None = None,
    rotation_deg: float = 0.0,
    vertical_stack: bool = False,
    text_color_rgb: tuple[int, int, int] | None = None,
    text_background_color: tuple[int, int, int] | None = None,
    layout_only: bool = False,
    fallback_padding_pixels: float | None = None,
    layout_dimensions: tuple[float, float] | None = None,
    oriented_dimensions: tuple[float, float] | None = None,
) -> Image.Image:
    """
    Fits and renders text within a bounding box using Skia and HarfBuzz.

    This is the high-level orchestrator that coordinates:
    1. Font loading (font_manager)
    2. Safe area calculation (image_utils)
    3. Layout optimization (layout_engine)
    4. Text rendering (drawing_engine)

    Uses the 5-step Distance Transform Insetting Method for safe area calculation:
    1. Establish Safe Zone: Create safe_area_mask using cv2.distanceTransform()
    2. Find Unbiased Anchor: Calculate centroid of the safe_area_mask
    3. Measure Available Space: Ray cast from centroid to find distances to edges
    4. Calculate Symmetrical Dimensions: Use min distances for width/height
    5. Construct Final Box: Create centered rectangle within safe zone

    This ensures text is perfectly centered and never touches bubble boundaries.

    Args:
        pil_image: PIL Image object to draw onto.
        text: Text to render.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        font_dir: Directory containing font files.
        cleaned_mask: Binary mask of the cleaned bubble (0/255). Used for safe area calculation.
        bubble_color_bgr: Background color of the bubble (BGR tuple). Used to determine text color.
        config: RenderingConfig object containing all rendering parameters. If None, uses defaults.
        fallback_padding_pixels: Pixel inset for the no-mask padded-bbox fallback; None keeps the ratio-based fallback.
        verbose: Whether to print detailed logs.

    Returns:
        Modified PIL Image object with rendered text.

    Raises:
        RenderingError: If rendering fails due to invalid inputs, font issues, or layout problems
        FontError: If font loading fails
    """
    x1, y1, x2, y2 = bbox
    bubble_width = x2 - x1
    bubble_height = y2 - y1

    if bubble_width <= 0 or bubble_height <= 0:
        log_message(f"Invalid bbox dimensions: {bbox}", always_print=True)
        raise RenderingError(f"Invalid bounding box dimensions: {bbox}")

    # em dash can break wrapping
    normalized_text = text.replace("—", "-")

    clean_text = " ".join(normalized_text.split())
    if not clean_text:
        return pil_image

    layout_text = clean_text

    # Initialize config with defaults if not provided
    if config is None:
        config = RenderingConfig()

    (
        layout_box_top_left,
        max_render_width,
        max_render_height,
        target_center_x,
        target_center_y,
    ) = _text_area(
        bbox,
        cleaned_mask,
        config,
        raise_on_safe_error,
        layout_dimensions,
        fallback_padding_pixels,
        verbose,
    )

    try:
        font_variants = find_font_variants(font_dir, verbose=verbose)
        regular_font_path = font_variants.get("regular")
    except FontError as e:
        raise RenderingError(f"Font loading failed: {e}") from e

    layout_text = sanitize_text_for_font(
        layout_text, str(regular_font_path), verbose=verbose
    )
    if not layout_text.strip():
        log_message(
            "All text characters unsupported by font, skipping render",
            always_print=True,
        )
        return pil_image

    try:
        _, regular_typeface, regular_hb_face = load_font_resources(
            str(regular_font_path)
        )
    except FontError as e:
        raise RenderingError(f"Font resource loading failed: {e}") from e

    available_features = get_font_features(str(regular_font_path))
    features_to_enable = {
        "kern": "kern" in available_features["GPOS"],
        "liga": config.use_ligatures and "liga" in available_features["GSUB"],
        "calt": "calt" in available_features["GSUB"],
    }
    log_message(
        f"Font features: {[k for k, v in features_to_enable.items() if v]}",
        verbose=verbose,
    )

    # Pre-load all required font variants for layout engine
    preload_hb_faces = {"regular": regular_hb_face}
    for style_key in ["italic", "bold", "bold_italic"]:
        style_path = font_variants.get(style_key)
        if style_path:
            _, _typeface, _hb_face = load_font_resources(
                str(style_path), style=style_key
            )
            if _hb_face:
                preload_hb_faces[style_key] = _hb_face

    def _find_layout(use_vertical_stack: bool) -> dict:
        line_spacing = (
            config.vertical_line_spacing_mult
            if use_vertical_stack
            else config.line_spacing_mult
        )
        min_font = config.min_font_size
        max_font = config.max_font_size
        if use_vertical_stack and config.vertical_font_size_mult != 1.0:
            min_font = max(1, round(min_font * config.vertical_font_size_mult))
            max_font = max(1, round(max_font * config.vertical_font_size_mult))

        return find_optimal_layout(
            layout_text,
            max_render_width,
            max_render_height,
            regular_hb_face,
            regular_typeface,
            preload_hb_faces,
            features_to_enable,
            min_font,
            max_font,
            line_spacing,
            False if use_vertical_stack else config.hyphenate_before_scaling,
            config.hyphen_penalty,
            config.hyphenation_min_word_length,
            config.badness_exponent,
            verbose,
            bubble_id,
            cleaned_mask,
            layout_box_top_left,
            config.detach_trailing_punctuation,
            use_vertical_stack,
        )

    try:
        layout_data = _find_layout(vertical_stack)
    except RenderingError as e:
        if not (
            config.auto_vertical_text
            and not vertical_stack
            and _should_try_auto_vertical_text(
                layout_text, max_render_width, max_render_height
            )
        ):
            raise RenderingError(f"Layout optimization failed: {e}") from e

        try:
            layout_data = _find_layout(True)
            log_message(
                "Using auto vertical text layout after horizontal layout failed",
                verbose=verbose,
            )
        except RenderingError:
            raise RenderingError(f"Layout optimization failed: {e}") from e
    else:
        if (
            config.auto_vertical_text
            and not vertical_stack
            and _should_try_auto_vertical_text(
                layout_text, max_render_width, max_render_height
            )
        ):
            try:
                vertical_layout_data = _find_layout(True)
                if _auto_vertical_layout_is_better(
                    layout_data, vertical_layout_data, max_render_height
                ):
                    layout_data = vertical_layout_data
                    log_message("Using auto vertical text layout", verbose=verbose)
            except RenderingError:
                pass

    if layout_only:
        log_message(f"Rendered at size {layout_data['font_size']}", verbose=verbose)
        result = Image.new("RGBA", (1, 1))
        result.info["font_size"] = layout_data["font_size"]
        return result

    loaded_typefaces, loaded_hb_faces = _load_style_variants(
        clean_text, font_variants, regular_typeface, regular_hb_face, verbose
    )
    fonts = (
        loaded_typefaces,
        loaded_hb_faces,
        regular_typeface,
        regular_hb_face,
        features_to_enable,
    )

    # Determine text color contrast based on sampled background brightness
    text_color = _resolve_text_color(text_color_rgb, bubble_color_bgr)

    skia_bg_color = None
    if text_background_color is not None:
        skia_bg_color = skia.Color(
            text_background_color[0],
            text_background_color[1],
            text_background_color[2],
        )

    # Apply supersampling if enabled
    outline_px = resolve_outline_width(layout_data["font_size"], config.outline_ratio)
    factor = config.supersampling_factor
    if factor > 1:
        log_message(f"Using supersampling factor {factor}", verbose=verbose)
        final_pil_image = _render_supersampled(
            pil_image,
            bbox,
            layout_data,
            (target_center_x, target_center_y),
            rotation_deg,
            factor,
            fonts,
            text_color,
            outline_px,
            verbose,
            skia_bg_color,
        )
        log_message(
            f"Rendered at size {layout_data['font_size']} with {factor}x supersampling",
            verbose=verbose,
        )
    else:
        try:
            surface = pil_to_skia_surface(pil_image)
        except RenderingError as e:
            raise RenderingError(f"Surface preparation failed: {e}") from e

        _draw_text(
            surface,
            layout_data,
            (target_center_x, target_center_y),
            rotation_deg,
            fonts,
            text_color,
            outline_px,
            verbose,
            skia_bg_color,
        )

        try:
            final_pil_image = skia_surface_to_pil(surface)
        except RenderingError as e:
            raise RenderingError(f"Final conversion failed: {e}") from e

        log_message(f"Rendered at size {layout_data['font_size']}", verbose=verbose)

    final_pil_image.info["font_size"] = layout_data["font_size"]
    return final_pil_image
