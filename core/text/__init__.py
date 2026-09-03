"""
Text processing and rendering modules for MangaTranslator.

This subpackage contains modules for:
- Text processing and tokenization
- Font management and loading
- Layout engine for optimal text placement
- Drawing engine using Skia
- High-level text rendering orchestration
"""

from .drawing_engine import (
    draw_layout,
    load_font_resources,
    pil_to_skia_surface,
    skia_surface_to_pil,
)
from .font_manager import (
    LRUCache,
    find_font_variants,
    get_font_features,
    load_font_data,
)
from .layout_engine import find_optimal_layout, shape_line
from .text_processing import (
    find_optimal_breaks_dp,
    is_cjk_character,
    is_rtl_script,
    parse_styled_segments,
    tokenize_styled_text,
    try_hyphenate_word,
)
from .text_renderer import render_text_skia

__all__ = [
    "LRUCache",
    "draw_layout",
    "find_font_variants",
    "find_optimal_breaks_dp",
    "find_optimal_layout",
    "get_font_features",
    "is_cjk_character",
    "is_rtl_script",
    "load_font_data",
    "load_font_resources",
    "parse_styled_segments",
    "pil_to_skia_surface",
    "render_text_skia",
    "shape_line",
    "skia_surface_to_pil",
    "tokenize_styled_text",
    "try_hyphenate_word",
]
