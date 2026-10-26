import skia
import uharfbuzz as hb

from core.text import layout_engine


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
