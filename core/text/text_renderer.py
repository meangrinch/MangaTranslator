from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import skia
from PIL import Image

from core.image.image_utils import calculate_centroid_expansion_box
from core.text.drawing_engine import (draw_layout, load_font_resources,
                                      pil_to_skia_surface, skia_surface_to_pil)
from core.text.font_manager import find_font_variants, get_font_features
from core.text.layout_engine import find_optimal_layout
from core.text.text_processing import parse_styled_segments
from utils.exceptions import FontError, ImageProcessingError, RenderingError
from utils.logging import log_message

GRAYSCALE_MIDPOINT = 128  # Threshold for determining text color
FALLBACK_PADDING_RATIO = 0.08  # 8% padding ratio when safe area calculation fails


@dataclass
class RenderingConfig:
    """Configuration for text rendering parameters."""

    min_font_size: int = 8
    max_font_size: int = 14
    line_spacing_mult: float = 1.0
    use_subpixel_rendering: bool = False
    font_hinting: str = "none"
    use_ligatures: bool = False
    hyphenate_before_scaling: bool = True
    hyphen_penalty: float = 1000.0
    hyphenation_min_word_length: int = 8
    badness_exponent: float = 3.0
    padding_pixels: float = 5.0
    outline_width: float = 0.0


def render_text_skia(
    pil_image: Image.Image,
    text: str,
    bbox: Tuple[int, int, int, int],
    font_dir: str,
    cleaned_mask: Optional[np.ndarray] = None,
    bubble_color_bgr: Optional[Tuple[int, int, int]] = (255, 255, 255),
    config: Optional[RenderingConfig] = None,
    verbose: bool = False,
    bubble_id: Optional[str] = None,
    rotation_deg: float = 0.0,
    vertical_stack: bool = False,
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
        verbose: Whether to print detailed logs.

    Returns:
        Modified PIL Image object with rendered text.

    Raises:
        RenderingError: If rendering fails due to invalid inputs, font issues, or layout problems
        FontError: If font loading fails
    """
    # --- Step 1: Validate inputs ---
    x1, y1, x2, y2 = bbox
    bubble_width = x2 - x1
    bubble_height = y2 - y1

    if bubble_width <= 0 or bubble_height <= 0:
        log_message(f"Invalid bbox dimensions: {bbox}", always_print=True)
        raise RenderingError(f"Invalid bounding box dimensions: {bbox}")

    clean_text = " ".join(text.split())
    if not clean_text:
        return pil_image

    # Prepare text for layout (vertical stacking removes whitespace to stay single-column)
    if vertical_stack:
        import unicodedata

        def _is_separator_or_space(ch: str) -> bool:
            try:
                cat = unicodedata.category(ch)
            except Exception:
                return ch.isspace()
            return ch.isspace() or (len(cat) > 0 and cat[0] == "Z")

        stacked_chars = [ch for ch in clean_text if not _is_separator_or_space(ch)]
        layout_text = "\n".join(stacked_chars)
    else:
        layout_text = clean_text

    # Initialize config with defaults if not provided
    if config is None:
        config = RenderingConfig()

    # --- Step 2: Calculate Safe Rendering Area ---
    safe_area_result = None
    if cleaned_mask is not None:
        try:
            safe_area_result = calculate_centroid_expansion_box(
                cleaned_mask, padding_pixels=config.padding_pixels, verbose=verbose
            )
        except ImageProcessingError:
            # Safe area calculation failed, will use fallback below
            safe_area_result = None
            log_message(
                "Safe area calculation failed, falling back to padded bbox method",
                verbose=verbose,
            )

    if safe_area_result is not None:
        guaranteed_box, _ = safe_area_result
        box_x, box_y, box_w, box_h = guaranteed_box
        max_render_width = float(box_w)
        max_render_height = float(box_h)
        target_center_x = box_x + box_w / 2.0
        target_center_y = box_y + box_h / 2.0
        log_message("Using centroid-based safe area calculation", verbose=verbose)
    else:
        # Fallback to padded bbox
        log_message(
            "Safe area calculation failed, falling back to padded bbox method",
            verbose=verbose,
        )
        max_render_width = bubble_width * (1 - 2 * FALLBACK_PADDING_RATIO)
        max_render_height = bubble_height * (1 - 2 * FALLBACK_PADDING_RATIO)

        if max_render_width <= 0 or max_render_height <= 0:
            max_render_width = max(1.0, float(bubble_width))
            max_render_height = max(1.0, float(bubble_height))

        target_center_x = x1 + bubble_width / 2.0
        target_center_y = y1 + bubble_height / 2.0

    # --- Step 3: Load Font Family ---
    try:
        font_variants = find_font_variants(font_dir, verbose=verbose)
        regular_font_path = font_variants.get("regular")
    except FontError as e:
        raise RenderingError(f"Font loading failed: {e}") from e

    # Load regular font resources
    try:
        _, regular_typeface, regular_hb_face = load_font_resources(
            str(regular_font_path)
        )
    except FontError as e:
        raise RenderingError(f"Font resource loading failed: {e}") from e

    # Get font features
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
            _, _typeface, _hb_face = load_font_resources(str(style_path))
            if _hb_face:
                preload_hb_faces[style_key] = _hb_face

    # --- Step 4: Find Optimal Layout ---
    try:
        layout_data = find_optimal_layout(
            layout_text,
            max_render_width,
            max_render_height,
            regular_hb_face,
            regular_typeface,
            preload_hb_faces,
            features_to_enable,
            config.min_font_size,
            config.max_font_size,
            config.line_spacing_mult,
            False if vertical_stack else config.hyphenate_before_scaling,
            config.hyphen_penalty,
            config.hyphenation_min_word_length,
            config.badness_exponent,
            verbose,
            bubble_id,
        )
    except RenderingError as e:
        raise RenderingError(f"Layout optimization failed: {e}") from e

    # --- Step 5: Load Font Resources for Drawing ---
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
                _, typeface, hb_face = load_font_resources(str(font_path))
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

    # --- Step 6: Prepare Drawing Surface ---
    try:
        surface = pil_to_skia_surface(pil_image)
    except RenderingError as e:
        raise RenderingError(f"Surface preparation failed: {e}") from e

    # Determine text color contrast based on sampled background brightness
    text_color = skia.ColorBLACK
    if bubble_color_bgr is not None:
        try:
            # bubble_color_bgr may be a grayscale proxy; treat as BGR
            bg_brightness = (
                bubble_color_bgr[0] + bubble_color_bgr[1] + bubble_color_bgr[2]
            ) / 3.0
            # If background is dark, use white text; if light, use black
            text_color = (
                skia.ColorWHITE
                if bg_brightness < GRAYSCALE_MIDPOINT
                else skia.ColorBLACK
            )
        except Exception:
            text_color = skia.ColorBLACK

    # --- Step 7: Draw Layout ---
    # Delegate rotation/translate to drawing_engine so Skia state is consistent
    success = draw_layout(
        surface,
        layout_data,
        0.0 if (rotation_deg and abs(rotation_deg) > 0.01) else target_center_x,
        0.0 if (rotation_deg and abs(rotation_deg) > 0.01) else target_center_y,
        loaded_typefaces,
        loaded_hb_faces,
        regular_typeface,
        regular_hb_face,
        features_to_enable,
        text_color,
        config.use_subpixel_rendering,
        config.font_hinting,
        config.outline_width,
        verbose,
        pre_translate_x=(
            float(target_center_x)
            if (rotation_deg and abs(rotation_deg) > 0.01)
            else 0.0
        ),
        pre_translate_y=(
            float(target_center_y)
            if (rotation_deg and abs(rotation_deg) > 0.01)
            else 0.0
        ),
        pre_rotate_deg=(
            float(rotation_deg) if (rotation_deg and abs(rotation_deg) > 0.01) else 0.0
        ),
    )

    if not success:
        log_message("Drawing failed", always_print=True)
        raise RenderingError("Text drawing failed")

    # --- Step 8: Convert Back to PIL ---
    try:
        final_pil_image = skia_surface_to_pil(surface)
    except RenderingError as e:
        raise RenderingError(f"Final conversion failed: {e}") from e

    log_message(f"Rendered at size {layout_data['font_size']}", verbose=verbose)
    return final_pil_image
