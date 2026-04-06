from typing import Any, Dict, List

from PIL import Image

from core.config import MangaTranslatorConfig, RenderingConfig
from core.text.text_renderer import render_text_skia
from utils.exceptions import FontError, RenderingError
from utils.logging import log_message


def generate_test_placeholders(
    sorted_bubble_data: List[Dict[str, Any]],
    processed_bubbles_info: List[Dict[str, Any]],
    pil_cleaned_image: Image.Image,
    config: MangaTranslatorConfig,
    main_min_font: int,
    main_max_font: int,
    osb_min_font: int,
    osb_max_font: int,
    padding_pixels: float,
    osb_outline_width: float,
    verbose: bool = False,
) -> List[str]:
    """
    Generates test placeholder text by probing the rendering engine.
    Finds the largest text string that fits in the bounding box.
    """
    translated_texts = []
    placeholder_long = "Lorem **ipsum** *dolor* sit amet, consectetur adipiscing elit."
    placeholder_short = "Lorem **ipsum** *dolor* sit amet..."
    placeholder_tiny = "Lorem..."
    placeholder_tiers = [
        placeholder_long,
        placeholder_short,
        placeholder_tiny,
    ]

    log_message(
        f"Test mode: generating placeholders for {len(sorted_bubble_data)} bubbles",
        always_print=True,
    )
    # Map for rendering info used in probe
    bubble_render_info_map_probe = {
        tuple(info["bbox"]): {
            "color": info["color"],
            "mask": info.get("mask"),
        }
        for info in processed_bubbles_info
        if "bbox" in info and "color" in info and "mask" in info
    }

    for i, bubble in enumerate(sorted_bubble_data):
        bbox = bubble["bbox"]
        is_outside_text = bubble.get("is_outside_text", False)

        probe_info = bubble_render_info_map_probe.get(tuple(bbox), {})

        if is_outside_text:
            is_dark_text = bubble.get("is_dark_text", True)
            bubble_color_bgr = (50, 50, 50) if is_dark_text else (255, 255, 255)
            cleaned_mask = None
        else:
            bubble_color_bgr = probe_info.get("color", (255, 255, 255))
            cleaned_mask = probe_info.get("mask")

        min_font = osb_min_font if is_outside_text else main_min_font
        max_font = osb_max_font if is_outside_text else main_max_font
        line_spacing = (
            config.outside_text.osb_line_spacing
            if is_outside_text
            else config.rendering.line_spacing_mult
        )
        use_ligs = (
            config.outside_text.osb_use_ligatures
            if is_outside_text
            else config.rendering.use_ligatures
        )

        probe_config = RenderingConfig(
            min_font_size=min_font,
            max_font_size=max_font,
            line_spacing_mult=line_spacing,
            use_subpixel_rendering=(
                config.outside_text.osb_use_subpixel_rendering
                if is_outside_text
                else config.rendering.use_subpixel_rendering
            ),
            font_hinting=(
                config.outside_text.osb_font_hinting
                if is_outside_text
                else config.rendering.font_hinting
            ),
            use_ligatures=use_ligs,
            hyphenate_before_scaling=config.rendering.hyphenate_before_scaling,
            hyphen_penalty=config.rendering.hyphen_penalty,
            hyphenation_min_word_length=config.rendering.hyphenation_min_word_length,
            badness_exponent=config.rendering.badness_exponent,
            padding_pixels=padding_pixels,
            outline_width=(osb_outline_width if is_outside_text else 0.0),
            supersampling_factor=1,  # No supersampling for probe
        )
        best_fit = (
            placeholder_tiny.rstrip(".") if is_outside_text else placeholder_tiny
        )  # fallback

        placeholder_tiers_to_use = [
            t.rstrip(".") if is_outside_text else t for t in placeholder_tiers
        ]

        best_font_size = -1

        for text_tier in placeholder_tiers_to_use:
            _probe_canvas_iter = pil_cleaned_image.copy()

            test_text = text_tier.upper() if is_outside_text else text_tier

            font_dir = (
                config.outside_text.osb_font_dir
                if is_outside_text and config.outside_text.osb_font_dir
                else config.rendering.font_dir
            )

            try:
                rendered = render_text_skia(
                    pil_image=_probe_canvas_iter,
                    text=test_text,
                    bbox=bbox,
                    font_dir=font_dir,
                    cleaned_mask=cleaned_mask,
                    bubble_color_bgr=bubble_color_bgr,
                    config=probe_config,
                    verbose=verbose,
                    bubble_id=str(i + 1),
                    raise_on_safe_error=False,
                )

                font_size = rendered.info.get("font_size", 0)
                if font_size > best_font_size:
                    best_font_size = font_size
                    best_fit = text_tier

            except (RenderingError, FontError) as e:
                log_message(
                    f"Probe rendering failed for tier '{text_tier}': {e}",
                    verbose=verbose,
                )
            except Exception as e:
                log_message(
                    f"Probe rendering unexpected error: {e}",
                    always_print=True,
                )

        translated_texts.append(best_fit)

    return translated_texts
