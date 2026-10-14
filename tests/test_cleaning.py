import cv2
import numpy as np
import pytest

from core.image.cleaning import _build_adaptive_shrink_mask, process_single_bubble
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

    s_mask, *s_rest, s_text_bbox, s_text_color = clean(small)
    l_mask, *l_rest, l_text_bbox, l_text_color = clean(large)

    assert s_mask.any()
    assert np.array_equal(l_mask[oy : oy + 240, ox : ox + 280], s_mask)
    assert np.count_nonzero(l_mask) == np.count_nonzero(s_mask)
    assert l_rest == s_rest
    assert l_text_bbox == tuple(
        v + (ox if i % 2 == 0 else oy) for i, v in enumerate(s_text_bbox)
    )
    assert l_text_color == s_text_color
