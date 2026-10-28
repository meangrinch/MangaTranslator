import numpy as np
import pytest
from PIL import Image

from core.image.glyph_mask import (
    UpscaledGlyphMap,
    composite_glyphs,
    glyph_dilation_radius,
    glyph_scale,
    group_glyphs,
    page_glyph_probability,
    refine_box_mask_to_glyphs,
    refine_mask_to_glyphs,
    texts_sharing_detections,
    verified_glyph_box,
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


def _window_predictor(calls):
    """Fake text model whose output depends on the whole window it sees, so averages
    over overlapping windows differ unless every covering window is run."""

    def predict(window_rgb):
        calls.append(window_rgb.shape)
        dark = (window_rgb.mean(axis=2) < 64).astype(np.float32)
        return 0.8 * dark + 0.2 * window_rgb.mean() / 255

    return predict


def _upscaled_page():
    page = np.full((1200, 830, 3), 255, np.uint8)
    page[300:340, 200:500] = 0
    page[900:910, 100:700] = 40
    return Image.fromarray(page)


def test_upscaled_box_probability_matches_the_whole_upscaled_page_inside_the_box():
    page = _upscaled_page()
    full_calls, box_calls = [], []
    whole = UpscaledGlyphMap(page).box_probability(
        (0, 0, 830, 1200), _window_predictor(full_calls)
    )
    box = (180, 280, 520, 360)

    prob = UpscaledGlyphMap(page).box_probability(box, _window_predictor(box_calls))

    assert prob.shape == (80, 340)
    np.testing.assert_allclose(prob, whole[280:360, 180:520], atol=1e-6)
    # The 830 px page runs at 1660 px: windows of 1024 there, and only those near the box
    assert all(h <= 1024 and w <= 1024 for h, w, _ in full_calls)
    assert 0 < len(box_calls) < len(full_calls)


def test_upscaled_box_probability_runs_each_window_once():
    calls = []
    upscaled = UpscaledGlyphMap(_upscaled_page())
    predict = _window_predictor(calls)

    upscaled.box_probability((180, 280, 520, 360), predict)
    first = len(calls)
    upscaled.box_probability((200, 300, 400, 330), predict)

    assert len(calls) == first


def test_upscaled_box_probability_finds_strokes_on_a_large_page():
    page = np.full((3496, 2480, 3), 255, np.uint8)
    page[2000:2040, 1500:1700] = 0
    calls = []

    prob = UpscaledGlyphMap(Image.fromarray(page)).box_probability(
        (1480, 1980, 1720, 2060), _dark_predictor(calls)
    )

    assert calls and all(h <= 1024 and w <= 1024 for h, w, _ in calls)
    assert prob[40, 120] > 0.5  # page (1600, 2020)
    assert prob[5, 120] < 0.5


def test_box_refine_adds_glyphs_only_the_box_probability_finds():
    prob = np.zeros((300, 300), np.float32)
    prob[100:104, 50:100] = 0.9  # found by the page map
    box_prob = np.zeros((100, 200), np.float32)  # for box (30, 80, 230, 180)
    box_prob[60:64, 120:160] = (
        0.8  # a letter only the box pass finds, at (150-190, 140)
    )
    box = _box(prob.shape, 30, 80, 230, 180)

    mask = refine_box_mask_to_glyphs(box, prob, (30, 80, 230, 180), box_prob, 3)

    assert mask[102, 75] and mask[142, 170]
    assert not mask[120, 120]
    assert not mask[~box].any()
    assert prob[142, 170] == 0  # the page map is left alone


def test_box_refine_matches_the_page_refine_when_the_box_pass_finds_nothing():
    prob = np.zeros((300, 300), np.float32)
    prob[100:104, 25:100] = 0.9  # a stroke crossing the box edge
    box = _box(prob.shape, 30, 80, 230, 180)

    mask = refine_box_mask_to_glyphs(
        box, prob, (30, 80, 230, 180), np.zeros((100, 200), np.float32), 3
    )

    np.testing.assert_array_equal(mask, refine_mask_to_glyphs(box, prob, 3))


def _composite_case():
    """Original page with art and a stroke, and an edit-model redraw of it.

    The redraw erased the stroke but drew the art 1 px off and its paper 6 levels darker.
    """
    before = np.full((200, 300, 3), 255, np.uint8)
    before[60:100, 40:42] = 30  # art line crossing the box
    before[78:82, 100:200] = 0  # text stroke
    after = np.full_like(before, 249)
    after[60:100, 41:43] = 30
    prob = np.zeros((200, 300), np.float32)
    prob[78:82, 100:200] = 1.0
    return before, after, _box((200, 300), 30, 60, 230, 100), prob


def test_composite_keeps_the_original_art_away_from_the_glyphs():
    before, after, box, prob = _composite_case()

    out = composite_glyphs(before, after, box, prob, radius=6)

    np.testing.assert_array_equal(out[60:100, 30:60], before[60:100, 30:60])
    np.testing.assert_array_equal(out[~box], before[~box])


def test_composite_replaces_the_glyphs_with_the_redraw_matched_to_the_paper():
    before, after, box, prob = _composite_case()

    out = composite_glyphs(before, after, box, prob, radius=6)

    # The stroke is gone and the redrawn paper takes the original's tone, so no
    # letter-shaped ghost is left; the misplaced art line doesn't skew the match
    assert (out[76:84, 100:200] >= 254).all()


def test_composite_feathers_the_edge_of_the_dilated_glyphs():
    before, after, box, prob = _composite_case()
    after[:] = 128  # a redraw so far off that tone matching ignores it

    out = composite_glyphs(before, after, box, prob, radius=6)

    column = out[70:90, 150, 0].astype(int)
    assert column[8:12].tolist() == [128] * 4  # the glyph core is fully redrawn
    edge = out[73, 150, 0]  # 5 px out from the stroke, inside the 6 px dilation
    assert 128 < edge < 255


def test_composite_takes_the_whole_box_when_no_glyph_is_found():
    before, after, box, _ = _composite_case()

    out = composite_glyphs(before, after, box, np.zeros((200, 300), np.float32), 6)

    np.testing.assert_array_equal(out[box], after[box])
    np.testing.assert_array_equal(out[~box], before[~box])


def test_refine_low_threshold_adds_faint_parts_connected_to_glyphs():
    prob = np.zeros((300, 300), np.float32)
    prob[100:104, 50:100] = 0.9  # confident stroke
    prob[100:104, 100:140] = 0.3  # its faint outline tail
    prob[200:204, 50:100] = 0.3  # faint marks with no confident part
    box = _box(prob.shape, 30, 80, 250, 230)

    strict = refine_mask_to_glyphs(box, prob, radius=3)
    loose = refine_mask_to_glyphs(box, prob, radius=3, low_threshold=0.15)

    assert not strict[102, 130] and loose[102, 130]
    assert not loose[202, 75]


def _letters(shape=(830, 830)):
    glyph = np.zeros(shape, bool)
    glyph[100:140, 100:130] = True  # three 30x40 letters with 10 px gaps
    glyph[100:140, 140:170] = True
    glyph[100:140, 180:210] = True
    glyph[100:140, 600:630] = True  # a letter far from the others
    glyph[400:402, 400:402] = True  # a speck
    return glyph


def test_group_glyphs_merges_nearby_letters_and_drops_specks():
    groups = group_glyphs(_letters())

    assert sorted(g.box for g in groups) == [(100, 100, 210, 140), (600, 100, 630, 140)]
    (word,) = [g for g in groups if g.box[0] == 100]
    assert sorted(word.components) == [
        (100, 100, 130, 140),
        (140, 100, 170, 140),
        (180, 100, 210, 140),
    ]


def test_group_glyphs_sizes_long_strokes_by_their_width():
    glyph = np.zeros((830, 830), bool)
    glyph[100:104, 100:500] = True  # a long thin line
    glyph[124:164, 300:330] = True  # a letter 20 px below it

    # Measured by its 400 px length, the line would swallow the letter
    assert len(group_glyphs(glyph)) == 2


def test_group_glyphs_scales_the_minimum_area_with_the_page():
    small = np.zeros((830, 830), bool)
    small[100:113, 100:113] = True  # 169 px: a small letter at the reference scale
    large = np.zeros((1660, 1660), bool)
    large[100:113, 100:113] = True  # the same pixels are a speck on a page twice as big

    assert len(group_glyphs(small)) == 1
    assert group_glyphs(large) == []


def test_verified_box_keeps_only_components_inside_a_detection():
    glyph = _letters()
    glyph[100:140, 220:300] = True  # a solid bar right after the word
    (word,) = [g for g in group_glyphs(glyph) if g.box[0] == 100]

    assert word.box == (100, 100, 300, 140)
    assert verified_glyph_box(word, [(95, 95, 212, 145)]) == (100, 100, 210, 140)
    assert verified_glyph_box(word, [(400, 95, 500, 145)]) is None
    assert verified_glyph_box(word, []) is None


def test_boxes_inside_one_detection_form_one_text():
    words = [
        (100, 100, 160, 140),  # "MY"
        (180, 100, 300, 140),  # "VOICE", both on the line one detection covers
        (100, 160, 200, 200),  # the next line, linked to "VOICE" by a second detection
        (600, 100, 660, 140),  # a word no shared detection reaches
    ]
    detections = [(95, 95, 305, 145), (95, 95, 305, 205), (590, 90, 670, 150)]

    assert sorted(texts_sharing_detections(words, detections)) == [[0, 1, 2], [3]]


def test_a_box_mostly_outside_a_detection_stays_apart():
    words = [(100, 100, 160, 140), (180, 100, 300, 140)]

    # The detection holds all of the first word but only a sliver of the second
    assert sorted(texts_sharing_detections(words, [(95, 95, 200, 145)])) == [[0], [1]]
