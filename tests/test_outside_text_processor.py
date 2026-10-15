import cv2
import numpy as np
import pytest

from core.outside_text_processor import _is_page_number, _outlined_glyph_components


def _full_region_glyph_components(crop_gray, glyph_mask, is_contrasting, min_core):
    """The original per-component scan over the whole region, kept as the reference."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(glyph_mask)
    found = []
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 10:
            continue
        comp = (labels == label).astype(np.uint8) * 255
        core = np.count_nonzero(cv2.erode(comp, np.ones((2, 2), np.uint8)))
        if core == 0 or core / area < min_core:
            continue
        dilated = cv2.dilate(comp, np.ones((5, 5), np.uint8))
        border = crop_gray[cv2.bitwise_and(dilated, cv2.bitwise_not(comp)) == 255]
        if len(border) == 0:
            continue
        ratio = np.count_nonzero(is_contrasting(border)) / len(border)
        if ratio >= 0.40:
            found.append((area, ratio))
    return found


@pytest.mark.parametrize(
    ("threshold_mask", "is_contrasting", "min_core"),
    [
        (lambda g: g < 50, lambda b: b > 150, 0.0),
        (lambda g: g > 200, lambda b: b < 90, 0.05),
    ],
)
def test_glyph_components_match_the_full_region_scan(
    threshold_mask, is_contrasting, min_core
):
    rng = np.random.default_rng(0)
    crop = np.full((90, 120), 128, np.uint8)
    for y in range(2, 90, 7):
        for x in range(2, 120, 7):
            cv2.circle(crop, (x, y), 2, int(rng.choice([10, 240])), -1)
    # Outlined glyphs, including ones touching the region edges
    for x, y in ((0, 0), (60, 40), (110, 80)):
        cv2.rectangle(crop, (x - 6, y - 6), (x + 6, y + 6), 255, -1)
        cv2.rectangle(crop, (x - 3, y - 3), (x + 3, y + 3), 0, -1)
    glyph_mask = threshold_mask(crop).astype(np.uint8) * 255

    assert _outlined_glyph_components(
        crop, glyph_mask, is_contrasting, min_core
    ) == _full_region_glyph_components(crop, glyph_mask, is_contrasting, min_core)


@pytest.mark.parametrize("text", ["１２", "Ｐａｇｅ２０", "ｐ．２０", "－３５－", "20"])
def test_page_numbers_are_recognized_in_full_width_ocr_output(text):
    assert _is_page_number(text)


@pytest.mark.parametrize("text", ["ノア", "これで３日は食えるな！", ""])
def test_ordinary_text_is_not_a_page_number(text):
    assert not _is_page_number(text)
