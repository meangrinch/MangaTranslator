import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from core.outside_text_processor import (
    _border_color,
    _border_pixels,
    _detect_osb_text_color,
    _is_color_page,
    _is_page_number,
    _outlined_glyph_components,
)


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

    found = _outlined_glyph_components(crop, glyph_mask, is_contrasting, min_core)
    assert [(g.area, g.ratio) for g in found] == _full_region_glyph_components(
        crop, glyph_mask, is_contrasting, min_core
    )


@pytest.mark.parametrize("text", ["１２", "Ｐａｇｅ２０", "ｐ．２０", "－３５－", "20"])
def test_page_numbers_are_recognized_in_full_width_ocr_output(text):
    assert _is_page_number(text)


@pytest.mark.parametrize("text", ["ノア", "これで３日は食えるな！", ""])
def test_ordinary_text_is_not_a_page_number(text):
    assert not _is_page_number(text)


def _draw_text(page, text, xy, size, fill, stroke=0, stroke_fill=None, margin=6):
    """Draw `text` on `page` and return a detector-like box around it."""
    font = ImageFont.load_default(size=size)
    draw = ImageDraw.Draw(page)
    draw.text(
        xy, text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill
    )
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=font, stroke_width=stroke)
    return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)


def _colorful_art():
    rng = np.random.default_rng(0)
    page = Image.new("RGB", (400, 300), "black")
    draw = ImageDraw.Draw(page)
    for _ in range(160):
        x, y = int(rng.integers(0, 400)), int(rng.integers(0, 300))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        w, h = int(rng.integers(10, 60)), int(rng.integers(10, 60))
        draw.rectangle((x, y, x + w, y + h), fill=color)
    for _ in range(400):
        x, y = int(rng.integers(0, 400)), int(rng.integers(0, 300))
        r = int(rng.integers(2, 5))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(10, 10, 10))
    return page


def _hollow_thin_outline():
    page = Image.new("RGB", (400, 300), "white")
    box = _draw_text(page, "WHAM", (40, 100), 90, "white", 2, (80, 80, 80))
    return page.filter(ImageFilter.GaussianBlur(0.6)), box


def _soft_gray_digits():
    page = Image.new("RGB", (400, 300), "white")
    box = _draw_text(page, "272", (140, 110), 60, (55, 55, 55))
    return page.filter(ImageFilter.GaussianBlur(0.6)), box


def _framed_caption():
    page = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(page).rectangle((60, 100, 340, 200), outline="black", width=3)
    _draw_text(page, "LATER", (100, 120), 50, "black")
    return page, (55, 95, 346, 206)


def _black_with_white_outline_on_tone():
    page = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(page)
    for y in range(0, 300, 5):
        for x in range(0, 400, 5):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(30, 30, 30))
    return page, _draw_text(page, "ZAP", (80, 100), 100, "black", 5, "white")


def _white_outlined_on_color_art():
    page = _colorful_art()
    return page, _draw_text(page, "FLY", (90, 90), 110, "white", 3, "black")


def _black_on_yellow_banner():
    page = Image.new("RGB", (400, 300), (255, 230, 0))
    return page, _draw_text(page, "PUNCH", (60, 110), 70, "black")


def _text_color(page, box):
    """Detected text color, with the background sampled the way the pipeline does."""
    bg, _ = _border_color(_border_pixels(page, box, *page.size))
    return _detect_osb_text_color(
        page, box, np.array(bg, dtype=np.uint8), bg, _is_color_page(page)
    )


@pytest.mark.parametrize(
    ("make_page", "is_color", "expected"),
    [
        # Thin outlines fail the glyph border test, so the fill must be found directly
        (_hollow_thin_outline, False, (255, 255, 255)),
        # Soft gray strokes leave the page around them looking like an outlined white glyph
        (_soft_gray_digits, False, (0, 0, 0)),
        # The frame's interior holds the text, so it is a container, not a white glyph
        (_framed_caption, False, (0, 0, 0)),
        # A white outline around black letters is not white text
        (_black_with_white_outline_on_tone, False, (0, 0, 0)),
        # Dark specks in colored art outnumber the letters
        (_white_outlined_on_color_art, True, (255, 255, 255)),
        # Yellow clears the gray cutoff but is not white text
        (_black_on_yellow_banner, True, (0, 0, 0)),
    ],
)
def test_osb_text_color(make_page, is_color, expected):
    page, box = make_page()
    assert _is_color_page(page) is is_color
    assert _text_color(page, box) == expected


def test_page_with_a_small_colored_stamp_is_not_a_color_page():
    page = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(page).rectangle((10, 10, 40, 40), fill=(220, 20, 20))
    assert not _is_color_page(page)


def _letter_prob(page, text, xy, size, stroke, grow=0):
    """A text-model map covering whole letters, outlines included, like SAM-TS-L's."""
    letters = Image.new("L", page.size, 0)
    ImageDraw.Draw(letters).text(
        xy,
        text,
        font=ImageFont.load_default(size=size),
        fill=255,
        stroke_width=stroke,
        stroke_fill=255,
    )
    prob = np.asarray(letters, np.float32) / 255
    if grow:
        prob = cv2.dilate(prob, np.ones((2 * grow + 1,) * 2, np.uint8))
    return prob


def _colored_text_color(page, box, prob):
    bg, _ = _border_color(_border_pixels(page, box, *page.size))
    return _detect_osb_text_color(
        page, box, np.array(bg, dtype=np.uint8), bg, True, glyph_prob=prob
    )


def test_colored_letters_on_a_color_page_keep_their_hue():
    blue = (60, 140, 230)
    page = _colorful_art()
    box = _draw_text(page, "ZAP", (80, 90), 110, blue, 2, "black")

    color = _colored_text_color(page, box, _letter_prob(page, "ZAP", (80, 90), 110, 2))

    assert np.abs(np.subtract(color, blue)).max() <= 10


def test_saturated_paper_inside_the_text_map_does_not_tint_black_letters():
    page, box = _black_on_yellow_banner()
    # Text maps of thin strokes spill onto the paper around them
    prob = _letter_prob(page, "PUNCH", (60, 110), 70, 0, grow=2)

    assert _colored_text_color(page, box, prob) == (0, 0, 0)
