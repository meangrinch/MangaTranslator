import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from core import pipeline
from core.config import MangaTranslatorConfig
from core.image.strip_splitting import find_cut_rows

WIDTH = 100


def _noise_strip(height, blank_rows=()):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (height, WIDTH), dtype=np.uint8)
    for start, end in blank_rows:
        pixels[start:end] = 255
    return Image.fromarray(pixels)


def test_image_at_most_eight_times_its_width_is_not_split():
    assert find_cut_rows(_noise_strip(800, [(390, 410)])) == []


def test_cuts_land_mid_gutter_with_page_sized_sections():
    strip = _noise_strip(1400, [(480, 500), (980, 1000)])

    assert find_cut_rows(strip) == [490, 990]


def test_gaps_shorter_than_the_minimum_are_never_cut():
    line_gaps = [(row, row + 3) for row in range(50, 1400, 50)]

    assert find_cut_rows(_noise_strip(1400, line_gaps)) == []


def test_detector_fallback_cuts_between_reported_boxes():
    def detect_rows(crop):
        return [(0, 150), (250, crop.height)]

    assert find_cut_rows(_noise_strip(1400), detect_rows) == [500, 1000]


def test_no_cut_when_detector_reports_every_row_occupied():
    def detect_rows(crop):
        return [(0, crop.height)]

    assert find_cut_rows(_noise_strip(1400), detect_rows) == []


def test_strip_sections_chain_context_share_one_scale_and_stitch_back(
    monkeypatch, tmp_path
):
    strip = _noise_strip(1400).convert("RGB")
    seen = []

    def fake_translate_and_render(
        section_path, config, previous_context_texts_provider, ocr_texts_out, **_
    ):
        seen.append(
            (
                previous_context_texts_provider(),
                config.processing_scale,
                config.preprocessing.auto_scale,
            )
        )
        ocr_texts_out.append(f"text {len(seen)}")
        return Image.open(section_path).copy()

    monkeypatch.setattr(pipeline, "translate_and_render", fake_translate_and_render)
    config = MangaTranslatorConfig(
        yolo_model_path="model.pt", device=torch.device("cpu")
    )
    output_path = tmp_path / "out.png"
    ocr_texts = []

    stitched = pipeline._translate_strip(
        Path("strip.png"),
        strip,
        [500, 1000],
        config,
        output_path,
        "RGB",
        None,
        [],
        [["previous page"]],
        None,
        ocr_texts,
    )

    nominal_scale = math.sqrt(5) * WIDTH / 1000
    assert seen == [
        ([["previous page"]], nominal_scale, False),
        ([["text 1"]], nominal_scale, False),
        ([["text 2"]], nominal_scale, False),
    ]
    assert ocr_texts == ["text 1", "text 2", "text 3"]
    assert np.array_equal(np.asarray(stitched), np.asarray(strip))
    saved = Image.open(output_path).convert("RGB")
    assert np.array_equal(np.asarray(saved), np.asarray(strip))
