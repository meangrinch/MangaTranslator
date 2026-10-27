import numpy as np
import pytest
from PIL import Image

from core.image.glyph_mask import (
    glyph_dilation_radius,
    glyph_scale,
    page_glyph_probability,
    refine_mask_to_glyphs,
)


def _dark_predictor(calls=None):
    """Fake text model: probability 1 wherever the window is dark."""

    def predict(window_rgb):
        if calls is not None:
            calls.append(window_rgb.shape)
        return (window_rgb.mean(axis=2) < 64).astype(np.float32)

    return predict


def _box(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


@pytest.mark.parametrize(
    ("size", "radius"),
    [((830, 1200), 7), ((1660, 2400), 14), ((2400, 1660), 14), ((100, 150), 2)],
)
def test_dilation_radius_scales_with_the_page_short_side(size, radius):
    assert glyph_dilation_radius(size) == radius


def test_scale_shrinks_large_pages_but_never_upscales():
    assert glyph_scale((830, 1200)) == 1.0
    assert glyph_scale((600, 900)) == 1.0
    assert glyph_scale((1660, 2400)) == pytest.approx(0.5)
    assert glyph_scale((800, 12000)) == 1.0  # webtoon strips keep their width


def test_page_probability_maps_strokes_back_on_a_large_page():
    page = np.full((3496, 2480, 3), 255, np.uint8)
    page[2000:2040, 1500:1700] = 0  # a stroke far from the origin
    calls = []

    prob = page_glyph_probability(Image.fromarray(page), _dark_predictor(calls))

    assert prob.shape == (3496, 2480)
    assert calls and all(h <= 1024 and w <= 1024 for h, w, _ in calls)
    assert prob[2020, 1600] > 0.5
    assert prob[1000, 600] < 0.5
    assert prob[1960, 1600] < 0.5


def test_page_probability_pads_a_page_smaller_than_one_window():
    page = np.full((500, 400, 3), 255, np.uint8)
    page[100:110, 50:150] = 0
    calls = []

    prob = page_glyph_probability(Image.fromarray(page), _dark_predictor(calls))

    assert calls == [(500, 400, 3)]
    assert prob.shape == (500, 400)
    assert prob[105, 100] > 0.5


def test_refine_keeps_only_dilated_glyphs_inside_the_box():
    prob = np.zeros((1200, 830), np.float32)
    prob[500:504, 300:400] = 1.0  # text inside the box
    prob[100:104, 300:400] = 1.0  # art/text outside the box
    box = _box(prob.shape, 280, 480, 420, 530)

    mask = refine_mask_to_glyphs(box, prob, radius=7)

    assert mask[502, 350]
    assert mask[496, 350]  # dilated by 7 px above the stroke
    assert not mask[490, 350]
    assert not mask[102, 350]
    assert not mask[~box].any()
    assert not mask[520, 350]  # box pixels away from glyphs are dropped


def test_refine_falls_back_to_the_box_when_no_glyph_is_found():
    prob = np.zeros((300, 300), np.float32)
    box = _box(prob.shape, 50, 50, 150, 120)

    mask = refine_mask_to_glyphs(box, prob, radius=4)

    np.testing.assert_array_equal(mask, box)


def test_refine_handles_each_region_of_a_grouped_mask_separately():
    prob = np.zeros((400, 400), np.float32)
    prob[60:64, 60:120] = 1.0  # glyphs only in the first region
    first = _box(prob.shape, 50, 50, 150, 100)
    second = _box(prob.shape, 200, 250, 350, 330)

    mask = refine_mask_to_glyphs(first | second, prob, radius=3)

    assert mask[62, 90] and not mask[90, 140]
    np.testing.assert_array_equal(mask[second], np.ones(second.sum(), bool))
