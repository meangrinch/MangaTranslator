import cv2
import numpy as np
import pytest

from core.image.cleaning import (
    _build_adaptive_shrink_mask,
    clean_speech_bubbles,
    process_single_bubble,
)
from utils.exceptions import CleaningError


@pytest.mark.parametrize("processing_scale", [1.0, 1.04, 2.0])
def test_junction_zone_keeps_roi_edge_at_any_processing_scale(processing_scale):
    roi = np.zeros((100, 100), np.uint8)
    roi[20:80, 20:60] = 255

    shrunk = _build_adaptive_shrink_mask(
        roi, 5.0, (20, 20, 60, 80), [(55, 20, 95, 80)], processing_scale
    )

    assert shrunk[50, 59] == 255  # edge pixel facing the conjoined neighbor
    assert shrunk[50, 20] == 0  # far edge still gets the normal shrink


def test_auto_threshold_scales_fixed_value_to_a_gray_bubble_background():
    gray = np.full((200, 200), 60, np.uint8)
    cv2.ellipse(gray, (100, 100), (70, 80), 0, 0, 360, 170, -1)
    cv2.ellipse(gray, (100, 100), (70, 80), 0, 0, 360, 0, 3)
    cv2.rectangle(gray, (90, 70), (110, 130), 0, -1)
    base = np.zeros_like(gray)
    cv2.ellipse(base, (100, 100), (70, 80), 0, 0, 360, 255, -1)

    with pytest.raises(CleaningError):
        process_single_bubble(base, gray, 200, 200, 200, False, 5, False)

    mask = process_single_bubble(
        base, gray, 200, 200, 200, False, 5, False, auto_threshold=True
    )[0]

    assert mask[100, 100] == 255
    assert mask[100, 50] == 255


def _bubble_page(height, width, ox, oy):
    """Gray page with a lettered, slightly shaded bubble and a conjoined neighbor."""
    gray = np.full((height, width), 120, np.uint8)
    base = np.zeros_like(gray)
    cv2.ellipse(base, (ox + 90, oy + 110), (80, 100), 0, 0, 360, 255, -1)
    cv2.rectangle(base, (ox + 160, oy + 60), (ox + 200, oy + 160), 255, -1)
    gray[base == 255] = 245
    gray[oy + 40 : oy + 180, ox + 40 : ox + 140] -= (
        np.arange(140, dtype=np.uint8)[:, None] // 10
    )
    for i, text in enumerate(("ABC", "DEFG", "HI")):
        cv2.putText(
            gray, text, (ox + 45, oy + 80 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 3
        )
    bbox = (ox + 10, oy + 10, ox + 200, oy + 210)
    neighbors = [(ox + 190, oy + 50, ox + 260, oy + 170)]
    return gray, base, bbox, neighbors


def test_sam_and_yolo_detections_of_one_bubble_clean_the_same():
    from PIL import Image

    gray, base, bbox, _ = _bubble_page(240, 280, 0, 0)
    page = Image.fromarray(gray).convert("RGB")
    contours, _ = cv2.findContours(base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sam = {"bbox": bbox, "sam_mask": base > 0}
    yolo = {"bbox": bbox, "mask_points": contours[0].tolist()}
    bad = {"bbox": (0, 0, 5, 5), "mask_points": [[1, 2, 3]]}

    _, sam_info = clean_speech_bubbles(page, None, pre_computed_detections=[sam])
    cleaned, yolo_info = clean_speech_bubbles(
        page, None, pre_computed_detections=[bad, yolo]
    )

    assert len(sam_info) == len(yolo_info) == 1
    assert sam_info[0]["is_sam"] and not yolo_info[0]["is_sam"]
    assert np.array_equal(sam_info[0]["mask"], yolo_info[0]["mask"])
    assert sam_info[0]["color"] == yolo_info[0]["color"]
    assert cleaned[100, 90].tolist() == [255, 255, 255]


@pytest.mark.parametrize("use_otsu", [False, True])
def test_cleaning_a_bubble_does_not_depend_on_its_position_on_the_page(use_otsu):
    ox, oy = 700, 900
    small = _bubble_page(240, 280, 0, 0)
    large = _bubble_page(1600, 1200, ox, oy)

    def clean(page):
        gray, base, bbox, neighbors = page
        return process_single_bubble(
            base,
            gray,
            *gray.shape,
            200,
            use_otsu,
            4,
            False,
            detection_bbox=bbox,
            classify_colored=True,
            neighbor_bboxes=neighbors,
            image_bgr=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            auto_threshold=True,
        )

    s_mask, *s_rest, s_text_bbox, s_text_color, s_art = clean(small)
    l_mask, *l_rest, l_text_bbox, l_text_color, l_art = clean(large)

    assert s_mask.any()
    assert np.array_equal(l_mask[oy : oy + 240, ox : ox + 280], s_mask)
    assert np.count_nonzero(l_mask) == np.count_nonzero(s_mask)
    assert l_rest == s_rest
    assert l_text_bbox == tuple(
        v + (ox if i % 2 == 0 else oy) for i, v in enumerate(s_text_bbox)
    )
    assert l_text_color == s_text_color
    assert l_art is s_art is None


def _edge_letter_page(touch_edge=True):
    """White bubble with a letter in the middle and a bar that runs into the outline
    (or stops short of it), plus the text map of both."""
    from PIL import Image

    gray = np.full((200, 200), 120, np.uint8)
    base = np.zeros_like(gray)
    cv2.ellipse(base, (100, 100), (70, 80), 0, 0, 360, 255, -1)
    gray[base == 255] = 245
    cv2.ellipse(gray, (100, 100), (70, 80), 0, 0, 360, 0, 3)
    letters = np.zeros_like(gray, dtype=bool)
    letters[90:120, 70:80] = True
    letters[(10 if touch_edge else 40) : 60, 96:104] = True
    gray[letters] = 20
    prob = letters.astype(np.float32)
    return Image.fromarray(gray).convert("RGB"), base, prob


class _Map:
    """Text map stub; bubble cleaning only reads the probabilities."""

    def __init__(self, prob):
        self.prob = prob

    def get(self):
        return self.prob


def test_glyphs_left_by_the_fill_are_cleaned_inside_the_bubble():
    page, base, prob = _edge_letter_page()
    detection = {"bbox": (30, 20, 170, 180), "sam_mask": base > 0}

    plain, _ = clean_speech_bubbles(page, None, pre_computed_detections=[detection])
    cleaned, info = clean_speech_bubbles(
        page, None, pre_computed_detections=[detection], glyph_map=_Map(prob)
    )

    assert plain[40, 100].tolist() == [20, 20, 20]  # the fill misses the edge letter
    assert cleaned[40, 100].tolist() == [255, 255, 255]
    assert cleaned[25, 100].tolist() == [255, 255, 255]  # up to the outline
    assert cleaned[20, 100].tolist() == [20, 20, 20]  # the outline under the letter
    assert cleaned[12, 100].tolist() == [20, 20, 20]  # outside the bubble polygon
    assert cleaned[100, 31].tolist() == [0, 0, 0]  # outline away from the letter
    assert info[0]["mask"][40, 100] == 255


def test_glyphs_the_fill_covers_leave_the_mask_unchanged():
    page, base, prob = _edge_letter_page(touch_edge=False)
    detection = {"bbox": (30, 20, 170, 180), "sam_mask": base > 0}

    _, plain = clean_speech_bubbles(page, None, pre_computed_detections=[detection])
    _, info = clean_speech_bubbles(
        page, None, pre_computed_detections=[detection], glyph_map=_Map(prob)
    )

    assert np.array_equal(plain[0]["mask"], info[0]["mask"])


def test_a_bubble_detected_inside_a_letter_takes_no_glyphs():
    page, base, prob = _edge_letter_page()
    detection = {"bbox": (30, 20, 170, 180), "sam_mask": base > 0}
    # Hollow SFX letters can pass as bubbles: their white inside is the fill
    prob[base > 0] = 1

    _, plain = clean_speech_bubbles(page, None, pre_computed_detections=[detection])
    _, info = clean_speech_bubbles(
        page, None, pre_computed_detections=[detection], glyph_map=_Map(prob)
    )

    assert np.array_equal(plain[0]["mask"], info[0]["mask"])


def _blush_page():
    """White bubble with a letter, a blush hatch far from it, a dot beside it, and a
    speck of dust far from it, plus the text map, which has only the letter."""
    from PIL import Image

    gray = np.full((400, 400), 120, np.uint8)
    base = np.zeros_like(gray)
    cv2.ellipse(base, (200, 200), (150, 170), 0, 0, 360, 255, -1)
    gray[base == 255] = 245
    cv2.ellipse(gray, (200, 200), (150, 170), 0, 0, 360, 0, 3)
    letter = np.zeros_like(gray, dtype=bool)
    letter[150:250, 190:210] = True
    gray[letter] = 20
    for x in (100, 106, 112):
        cv2.line(gray, (x, 215), (x + 8, 200), 60, 2)
    gray[252:255, 199:202] = 20  # 2 px below the letter
    gray[300:302, 120:122] = 60
    return Image.fromarray(gray).convert("RGB"), base, letter.astype(np.float32)


def test_marks_away_from_the_text_are_kept():
    page, base, prob = _blush_page()
    detection = {"bbox": (50, 30, 350, 370), "sam_mask": base > 0}

    plain, _ = clean_speech_bubbles(page, None, pre_computed_detections=[detection])
    cleaned, _ = clean_speech_bubbles(
        page, None, pre_computed_detections=[detection], glyph_map=_Map(prob)
    )

    hatch = (slice(198, 218), slice(98, 122))
    ink = np.asarray(page)[hatch][..., 0] < 200
    assert (plain[hatch][ink] == 255).all()
    assert (cleaned[hatch][ink] < 200).all()
    assert cleaned[200, 200].tolist() == [255, 255, 255]  # the letter
    assert cleaned[253, 200].tolist() == [255, 255, 255]  # the dot beside it
    assert cleaned[301, 121].tolist() == [255, 255, 255]  # the speck


def test_kept_marks_leave_the_text_area_whole():
    page, base, prob = _blush_page()
    detection = {"bbox": (50, 30, 350, 370), "sam_mask": base > 0}

    _, plain = clean_speech_bubbles(page, None, pre_computed_detections=[detection])
    _, info = clean_speech_bubbles(
        page, None, pre_computed_detections=[detection], glyph_map=_Map(prob)
    )

    # Rendering places text in the mask, so holes would push it away from the marks
    assert np.array_equal(plain[0]["mask"], info[0]["mask"])
    assert info[0]["art_mask"][208, 104]
    assert plain[0]["art_mask"] is None


def test_bubbles_without_text_in_the_map_clean_every_mark():
    page, base, prob = _blush_page()
    detection = {"bbox": (50, 30, 350, 370), "sam_mask": base > 0}

    _, plain = clean_speech_bubbles(page, None, pre_computed_detections=[detection])
    _, info = clean_speech_bubbles(
        page,
        None,
        pre_computed_detections=[detection],
        glyph_map=_Map(np.zeros_like(prob)),
    )

    assert np.array_equal(plain[0]["mask"], info[0]["mask"])


def _colored_bubbles_page():
    """Two bubbles with a gradient interior, each with one letter, and their map."""
    from PIL import Image

    rgb = np.full((200, 400, 3), 120, np.uint8)
    prob = np.zeros((200, 400), np.float32)
    detections = []
    for cx in (100, 300):
        base = np.zeros((200, 400), np.uint8)
        cv2.ellipse(base, (cx, 100), (70, 80), 0, 0, 360, 255, -1)
        ys, xs = np.nonzero(base)
        # Red ramps across the bubble and blue down it, so the interior isn't solid
        rgb[ys, xs] = np.stack(
            [
                150 + (xs - cx + 70) * 105 // 140,
                np.full_like(xs, 255),
                150 + (ys - 20) * 100 // 160,
            ],
            axis=1,
        ).astype(np.uint8)
        rgb[85:115, cx - 5 : cx + 5] = 20
        prob[85:115, cx - 5 : cx + 5] = 1
        detections.append({"bbox": (cx - 70, 20, cx + 70, 180), "sam_mask": base > 0})
    return Image.fromarray(rgb), detections, prob


class _WhiteInpainter:
    """Paints the mask white, like an inpainter that redraws the whole region."""

    def __init__(self):
        self.masks = []

    def inpaint_mask(self, image, mask, **kwargs):
        mask = np.asarray(mask, bool)
        self.masks.append(mask.copy())
        out = np.asarray(image).copy()
        out[mask] = 255
        from PIL import Image

        return Image.fromarray(out)


class _InlineCoordinator:
    def map_ordered(self, jobs):
        return [job() for job in jobs]

    def run(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _clean_colored(monkeypatch, method, preserve_art=True, coordinator=None):
    from core.image import cleaning
    from core.image.page_glyph_map import PageGlyphMap

    page, detections, prob = _colored_bubbles_page()
    inpainter = _WhiteInpainter()
    monkeypatch.setattr(cleaning, "make_flux_inpainter", lambda *a, **k: inpainter)
    glyph_map = PageGlyphMap(page, prob=prob)
    cleaned, info = clean_speech_bubbles(
        page,
        None,
        pre_computed_detections=detections,
        inpaint_colored_bubbles=True,
        inpaint_method=method,
        flux_preserve_art=preserve_art,
        glyph_map=glyph_map,
        request_coordinator=coordinator,
    )
    assert all(b["is_colored"] and b["inpainted"] for b in info)
    return np.asarray(page), cleaned[..., ::-1], inpainter.masks


@pytest.mark.parametrize("coordinator", [None, _InlineCoordinator()])
def test_lama_inpaints_only_the_glyphs_of_colored_bubbles(monkeypatch, coordinator):
    _, _, masks = _clean_colored(monkeypatch, "lama_manga", coordinator=coordinator)

    assert len(masks) == 2
    for mask, cx in zip(masks, (100, 300), strict=True):
        assert mask[100, cx] and mask[100, cx + 5]  # the letter and its dilation
        assert not mask[100, cx - 40]  # gradient away from the letter


@pytest.mark.parametrize("coordinator", [None, _InlineCoordinator()])
def test_edit_model_redraws_of_colored_bubbles_keep_the_art_around_letters(
    monkeypatch, coordinator
):
    original, cleaned, masks = _clean_colored(
        monkeypatch, "flux_klein_4b", coordinator=coordinator
    )

    assert masks[0][100, 60]  # the edit model still sees the whole interior
    for cx in (100, 300):
        assert (cleaned[100, cx] > 200).all()  # tone-matched to the gradient
        assert np.array_equal(cleaned[100, cx - 40], original[100, cx - 40])


def test_colored_bubbles_take_the_whole_redraw_without_preserve_art(monkeypatch):
    _, cleaned, _ = _clean_colored(monkeypatch, "flux_klein_4b", preserve_art=False)

    assert cleaned[100, 60].tolist() == [255, 255, 255]
