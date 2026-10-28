import cv2
import numpy as np
import pytest

from core.image.tilt_detection import detect_crop_tilt_angle


def test_steep_ink_in_horizontal_crop_is_not_clamped_into_a_tilt():
    crop = np.full((120, 120, 3), 255, np.uint8)
    t = np.radians(70)
    dx, dy = int(45 * np.cos(t)), int(45 * np.sin(t))
    cv2.line(crop, (60 - dx, 60 - dy), (60 + dx, 60 + dy), (0, 0, 0), 9)

    angle, conf, _, _ = detect_crop_tilt_angle(crop, max_tilt_deg=20.0)

    assert (angle, conf) == (0.0, 0.0)


def _glyph_run(shape, center, direction_deg, count, step, glyph_deg):
    """Mask of `count` glyph-sized blocks spaced along `direction_deg` (0 = rightward)."""
    mask = np.zeros(shape, np.uint8)
    t = np.radians(direction_deg)
    for k in range(count):
        offset = (k - (count - 1) / 2) * step
        cx = center[0] + offset * np.cos(t)
        cy = center[1] + offset * np.sin(t)
        box = cv2.boxPoints(((cx, cy), (11, 15), glyph_deg))
        cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    return mask.astype(bool)


def _page_with_art(shape, text):
    crop = np.full((*shape, 3), 235, np.uint8)
    # Bold strokes of art crossing the crop at another angle
    for x in range(-200, shape[1], 45):
        cv2.line(crop, (x, shape[0]), (x + 160, 0), (20, 20, 20), 7)
    crop[text] = 0
    return crop


def test_ink_mask_gives_the_angle_of_a_tilted_line_among_art():
    text = _glyph_run((130, 240), (120, 65), 12, count=7, step=24, glyph_deg=12)
    crop = _page_with_art((130, 240), text)

    angle, conf, (length, thickness), axis = detect_crop_tilt_angle(
        crop, max_tilt_deg=20.0, ink_mask=text
    )

    assert axis == "horizontal"
    assert angle == pytest.approx(12, abs=2)
    assert conf >= 0.5
    assert length > 3 * thickness


def test_ink_mask_gives_the_angle_of_a_tilted_vertical_column():
    # Vertical tilt is positive when the column slopes down and to the left
    text = _glyph_run((260, 110), (55, 130), 90 + 14, count=8, step=27, glyph_deg=14)
    crop = _page_with_art((260, 110), text)

    angle, conf, _, axis = detect_crop_tilt_angle(
        crop, max_tilt_deg=20.0, ink_mask=text
    )

    assert axis == "vertical"
    assert angle == pytest.approx(14, abs=2.5)
    assert conf >= 0.5


@pytest.mark.parametrize("ink_pixels", [0, 6])
def test_ink_mask_with_too_little_text_falls_back_to_the_heuristic(ink_pixels):
    text = _glyph_run((130, 240), (120, 65), 12, count=7, step=24, glyph_deg=12)
    crop = _page_with_art((130, 240), text)
    sparse = np.zeros_like(text)
    sparse.flat[:ink_pixels] = True

    assert detect_crop_tilt_angle(
        crop, max_tilt_deg=20.0, ink_mask=sparse
    ) == detect_crop_tilt_angle(crop, max_tilt_deg=20.0)


def test_staggered_columns_keep_the_lean_of_each_column():
    # Two columns sloping down and to the right, the left one starting lower, so the
    # ink as a whole runs down and to the left
    shape = (300, 170)
    right = _glyph_run(shape, (120, 95), 90 - 25, count=5, step=26, glyph_deg=-25)
    left = _glyph_run(shape, (55, 200), 90 - 25, count=5, step=26, glyph_deg=-25)
    text = right | left
    crop = np.full((*shape, 3), 255, np.uint8)
    crop[text] = 0

    angle, conf, _, axis = detect_crop_tilt_angle(
        crop, max_tilt_deg=45.0, ink_mask=text
    )

    assert axis == "vertical"
    assert angle == pytest.approx(-25, abs=3)
    assert conf >= 0.5


def test_italic_strokes_in_an_upright_column_are_not_a_tilt():
    # Long strokes slanting like "/", stacked straight down
    shape = (170, 80)
    text = np.zeros(shape, np.uint8)
    dx = round(30 * np.tan(np.radians(16)))
    for cy in (48, 122):
        cv2.line(text, (40 + dx, cy - 30), (40 - dx, cy + 30), 1, 6)
    text = text.astype(bool)
    crop = np.full((*shape, 3), 255, np.uint8)
    crop[text] = 0

    angle, _, _, axis = detect_crop_tilt_angle(crop, max_tilt_deg=20.0, ink_mask=text)

    assert axis == "vertical"
    assert angle == 0.0


def test_confidence_stays_within_one():
    text = _glyph_run((260, 110), (55, 130), 90 + 14, count=8, step=27, glyph_deg=14)
    crop = np.full((260, 110, 3), 255, np.uint8)
    crop[text] = 0

    _, conf, _, _ = detect_crop_tilt_angle(crop, max_tilt_deg=20.0)

    assert 0.0 <= conf <= 1.0


def test_ink_mask_picks_the_column_axis_of_a_near_square_crop():
    # A short column sloping down and to the right, in a box too square to say
    text = _glyph_run((150, 140), (70, 75), 90 - 14, count=5, step=27, glyph_deg=-14)
    crop = np.full((150, 140, 3), 255, np.uint8)
    crop[text] = 0

    angle, _, _, axis = detect_crop_tilt_angle(crop, max_tilt_deg=20.0, ink_mask=text)

    assert axis == "vertical"
    assert angle == pytest.approx(-14, abs=2.5)
    # Heuristic ink keeps the axis from the box shape
    assert detect_crop_tilt_angle(crop, max_tilt_deg=20.0)[3] == "horizontal"
