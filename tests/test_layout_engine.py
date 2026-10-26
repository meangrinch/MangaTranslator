import io
import math

import numpy as np
import pytest
import skia
import uharfbuzz as hb
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from core.text import layout_engine
from utils.exceptions import RenderingError


def test_check_fit_hyphenates_word_at_exactly_min_length(monkeypatch):
    monkeypatch.setattr(
        layout_engine,
        "calculate_styled_line_width",
        lambda text, *args, **kwargs: 10.0 * len(text),
    )
    fit = layout_engine.check_fit(
        font_size=12,
        text="BEAUTIFUL",
        max_render_width=50,
        max_render_height=1000,
        regular_hb_face=hb.Face(b""),
        regular_typeface=skia.Typeface.MakeEmpty(),
        loaded_hb_faces={},
        features_to_enable={},
        line_spacing_mult=1.0,
        hyphenate_before_scaling=True,
        hyphenation_min_word_length=9,
        badness_exponent=3.0,
        hyphenation_language="English",
    )
    assert fit is not None
    assert [line["text_with_markers"] for line in fit["lines"]] == ["BEAU-", "TIFUL"]


class _Metrics:
    fAscent = -15.0
    fDescent = 5.0


def _fixed_ratio_fit(width_ratio: float):
    """check_fit stand-in that lays out one line spanning width_ratio of the offered width."""

    def fit(font_size, text, max_render_width, *args, **kwargs):
        line_width = width_ratio * max_render_width
        return {
            "lines": [{"width": line_width, "text_with_markers": text}],
            "metrics": _Metrics(),
            "max_line_width": line_width,
            "line_height": 20.0,
            "block_height": 20.0,
        }

    return fit


def _layout_in_box(safe_area_mask, box_size=100):
    return layout_engine.find_optimal_layout(
        "Lorem",
        box_size,
        box_size,
        regular_hb_face=hb.Face(b""),
        regular_typeface=skia.Typeface.MakeEmpty(),
        loaded_hb_faces={},
        features_to_enable={},
        min_font_size=12,
        max_font_size=12,
        safe_area_mask=safe_area_mask,
        box_top_left=(0, 0),
    )


def test_layout_rejects_spike_between_line_corners(monkeypatch):
    monkeypatch.setattr(layout_engine, "check_fit", _fixed_ratio_fit(0.8))
    safe_area_mask = np.full((100, 100), 255, dtype=np.uint8)
    # An outline spike reaching into the middle of the line's left edge, clear of its corners
    safe_area_mask[49:52, 0:30] = 0

    with pytest.raises(RenderingError):
        _layout_in_box(safe_area_mask)


def test_layout_checks_squeezed_lines_where_they_are_drawn(monkeypatch):
    monkeypatch.setattr(layout_engine, "check_fit", _fixed_ratio_fit(0.8))
    safe_area_mask = np.zeros((100, 100), dtype=np.uint8)
    safe_area_mask[:, 12:89] = 255

    # Full width (80 px line at x 10-90) collides; the first squeeze (72 px) fits when
    # centered on the box like the drawing does (x 14-86)
    layout = _layout_in_box(safe_area_mask)

    assert layout["lines"][0]["width"] == pytest.approx(72.0)


# Glyph outlines in font units (1000 per em), clockwise from the baseline
BOX_GLYPH = [(50, 0), (50, 700), (550, 700), (550, 0)]
# Fills only the lower-right half of its bounding box, leaving the top-left corner empty
TRIANGLE_GLYPH = [(50, 0), (550, 700), (550, 0)]


def _single_glyph_font_bytes(outline: list[tuple[int, int]]) -> bytes:
    """Font whose only glyph, "A", is the given polygon, with nothing below the baseline."""
    pen = TTGlyphPen(None)
    pen.moveTo(outline[0])
    for point in outline[1:]:
        pen.lineTo(point)
    pen.closePath()

    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "A"])
    fb.setupCharacterMap({ord("A"): "A"})
    fb.setupGlyf({".notdef": TTGlyphPen(None).glyph(), "A": pen.glyph()})
    fb.setupHorizontalMetrics({".notdef": (600, 0), "A": (600, 50)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupOS2(
        sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200
    )
    fb.setupNameTable({"familyName": "Test Glyph", "styleName": "Regular"})
    fb.setupPost()
    buffer = io.BytesIO()
    fb.save(buffer)
    return buffer.getvalue()


def _real_font_layout(font_bytes, text, font_size, safe_area_mask):
    hb_face = hb.Face(font_bytes)
    box_h, box_w = safe_area_mask.shape
    return layout_engine.find_optimal_layout(
        text,
        box_w,
        box_h,
        regular_hb_face=hb_face,
        regular_typeface=_typeface(font_bytes),
        loaded_hb_faces={"regular": hb_face},
        features_to_enable={},
        min_font_size=font_size,
        max_font_size=font_size,
        safe_area_mask=safe_area_mask,
        box_top_left=(0, 0),
    )


def _typeface(font_bytes):
    return skia.Typeface.MakeFromData(skia.Data.MakeWithCopy(font_bytes))


def _single_line_baseline(font_bytes, font_size, box_h):
    """Baseline y of a one-line block centered in a box, as draw_layout places it."""
    metrics = skia.Font(_typeface(font_bytes), font_size).getMetrics()
    line_top = box_h / 2 - (metrics.fDescent - metrics.fAscent) / 2
    return line_top - metrics.fAscent, metrics


def test_layout_allows_empty_descender_strip_inside_padding():
    font_bytes = _single_glyph_font_bytes(BOX_GLYPH)
    font_size = 40
    baseline, metrics = _single_line_baseline(font_bytes, font_size, box_h=100)
    safe_area_mask = np.full((100, 300), 255, dtype=np.uint8)
    # Close the safe area inside the descent strip, below every glyph's ink
    safe_area_mask[math.ceil(baseline + metrics.fDescent / 2) :, :] = 0

    layout = _real_font_layout(font_bytes, "AAA", font_size, safe_area_mask)

    assert layout["font_size"] == font_size


def test_layout_allows_empty_glyph_corner_inside_padding():
    font_bytes = _single_glyph_font_bytes(TRIANGLE_GLYPH)
    font_size = 40
    baseline, _ = _single_line_baseline(font_bytes, font_size, box_h=100)
    # At 40 px the 600-unit advance is 24 px, centered at x=50, so the glyph's
    # 50-550 unit ink spans x 40-60 and its 700-unit height ends 28 px above the baseline
    ink_left, ink_top = 40, math.floor(baseline - 28)
    safe_area_mask = np.full((100, 100), 255, dtype=np.uint8)
    safe_area_mask[ink_top : ink_top + 6, ink_left : ink_left + 6] = 0

    layout = _real_font_layout(font_bytes, "A", font_size, safe_area_mask)

    assert layout["font_size"] == font_size


def test_layout_rejects_glyph_ink_inside_padding():
    font_bytes = _single_glyph_font_bytes(TRIANGLE_GLYPH)
    font_size = 40
    baseline, _ = _single_line_baseline(font_bytes, font_size, box_h=100)
    safe_area_mask = np.full((100, 100), 255, dtype=np.uint8)
    # The triangle's filled bottom-right corner sits at x 60, on the baseline
    safe_area_mask[math.floor(baseline) - 6 : math.floor(baseline), 54:60] = 0

    with pytest.raises(RenderingError):
        _real_font_layout(font_bytes, "A", font_size, safe_area_mask)
