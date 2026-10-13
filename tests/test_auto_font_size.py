import numpy as np
import pytest
import torch

from core import pipeline
from core.config import MangaTranslatorConfig, RenderingConfig, TranslationConfig
from core.image.image_utils import estimate_glyph_size

CAP_HEIGHT_RATIO = 0.5


def _bubble_page(glyph: int, dark: bool = False, glyph_width: int | None = None):
    """A 200x200 page with one bubble holding a column of glyphs plus small ruby marks."""
    paper, ink = (0, 255) if dark else (255, 0)
    gray = np.full((200, 200), 128, dtype=np.uint8)
    mask = np.zeros_like(gray)
    mask[10:190, 40:160] = 255
    gray[mask == 255] = paper
    for i in range(5):
        y = 20 + i * (glyph + 6)
        gray[y : y + glyph, 60 : 60 + (glyph_width or glyph)] = ink
        for x in (120, 140):
            gray[y : y + glyph // 3, x : x + glyph // 3] = ink
    return gray, (40, 10, 160, 190), mask


def _resolve(
    glyph: int | None,
    input_language: str = "Japanese",
    auto_font_size: bool = True,
    min_font: int = 8,
    max_font: int = 100,
    processing_scale: float = 1.0,
    strip_glyph_size: list[float] | None = None,
) -> int:
    gray, bbox, mask = _bubble_page(glyph or 24)
    if glyph is None:
        gray[mask == 255] = 255
    config = MangaTranslatorConfig(
        yolo_model_path="model.pt",
        device=torch.device("cpu"),
        translation=TranslationConfig(input_language=input_language),
        rendering=RenderingConfig(auto_font_size=auto_font_size),
    )
    return pipeline._resolve_main_max_font(
        config,
        np.dstack([gray] * 3),
        [{"bbox": bbox, "mask": mask}, {"bbox": bbox, "mask": None}],
        min_font,
        max_font,
        processing_scale,
        strip_glyph_size,
        False,
    )


@pytest.fixture(autouse=True)
def fake_font(monkeypatch):
    monkeypatch.setattr(
        pipeline, "find_font_variants", lambda *a, **k: {"regular": "font.ttf"}
    )
    monkeypatch.setattr(pipeline, "get_cap_height_ratio", lambda _: CAP_HEIGHT_RATIO)


@pytest.mark.parametrize("dark", [False, True])
def test_estimate_glyph_size_measures_full_size_glyphs(dark):
    gray, bbox, mask = _bubble_page(24, dark)

    size, parts = estimate_glyph_size(gray, bbox, mask, True)

    assert size == pytest.approx(24, abs=1)
    assert parts == 15


def test_estimate_glyph_size_returns_none_without_text():
    gray, bbox, mask = _bubble_page(24)
    gray[mask == 255] = 255

    assert estimate_glyph_size(gray, bbox, mask, True) is None


def test_estimate_glyph_size_ignores_width_unless_glyphs_are_full_width():
    # Touching letters merge into ink parts much wider than one letter is tall
    gray, bbox, mask = _bubble_page(24, glyph_width=40)

    assert estimate_glyph_size(gray, bbox, mask, False)[0] == pytest.approx(24, abs=1)
    assert estimate_glyph_size(gray, bbox, mask, True)[0] == pytest.approx(40, abs=1)


def _page_glyph_size(monkeypatch, *bubbles: tuple[float, int]) -> float | None:
    measured = iter(bubbles)
    monkeypatch.setattr(pipeline, "estimate_glyph_size", lambda *a: next(measured))
    infos = [{"bbox": (0, 0, 1, 1), "mask": np.zeros((1, 1), np.uint8)}] * len(bubbles)
    return pipeline._page_glyph_size(np.zeros((1, 1), np.uint8), infos, True)


@pytest.mark.parametrize(
    "outlier",
    [(5.0, 3), (169.0, 4)],
    ids=["ellipsis-only bubble", "shout in giant lettering"],
)
def test_page_glyph_size_lets_bubbles_with_more_text_outvote_outliers(
    monkeypatch, outlier
):
    assert _page_glyph_size(monkeypatch, outlier, (27.0, 20)) == 27.0
    assert _page_glyph_size(monkeypatch, (27.0, 20), outlier) == 27.0


def test_page_glyph_size_does_not_favor_bubbles_with_denser_text(monkeypatch):
    # Small print packs in more ink parts but is still one bubble's worth of evidence
    assert _page_glyph_size(monkeypatch, (15.0, 99), (20.0, 28), (26.0, 38)) == 20.0
    assert _page_glyph_size(monkeypatch, (22.0, 64), (30.0, 23)) == 26.0


@pytest.mark.parametrize("lone_bubble", [(5.0, 3), (169.0, 4)])
def test_page_glyph_size_needs_enough_ink_to_trust(monkeypatch, lone_bubble):
    assert _page_glyph_size(monkeypatch, lone_bubble) is None


def test_auto_font_size_follows_original_lettering_within_limits():
    expected = round(pipeline.AUTO_FONT_CJK_GLYPH_RATIO * 24 / CAP_HEIGHT_RATIO)

    assert _resolve(24) == expected
    assert _resolve(24, max_font=20) == 20
    assert _resolve(24, min_font=50) == 50


def test_auto_font_size_matches_non_cjk_lettering_at_full_size():
    assert _resolve(24, input_language="English") == round(24 / CAP_HEIGHT_RATIO)


def test_auto_font_size_disabled_keeps_max():
    assert _resolve(24, auto_font_size=False, max_font=16) == 16


def test_auto_font_size_assumes_typical_lettering_when_nothing_is_measurable():
    expected = round(pipeline.AUTO_FONT_TYPICAL_CAP_HEIGHT * 2.0 / CAP_HEIGHT_RATIO)

    assert _resolve(None, processing_scale=2.0) == expected


def test_auto_font_size_is_shared_by_the_sections_of_a_strip():
    strip_glyph_size: list[float] = []

    assert _resolve(None, strip_glyph_size=strip_glyph_size) == _resolve(None)
    assert strip_glyph_size == []
    first = _resolve(24, strip_glyph_size=strip_glyph_size)
    assert strip_glyph_size == [pytest.approx(24, abs=1)]
    assert _resolve(12, strip_glyph_size=strip_glyph_size) == first
    assert _resolve(None, strip_glyph_size=strip_glyph_size) == first
