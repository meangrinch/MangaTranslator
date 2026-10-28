import numpy as np
import pytest
import torch
from PIL import Image

import core.outside_text_processor as otp
from core import pipeline
from core.config import MangaTranslatorConfig
from utils.exceptions import RenderingError

BOX = (100, 100, 220, 180)  # x0, y0, x1, y1 of the OSB text box
WHITE = 255
CORNERS = [(102, 102), (217, 102), (102, 177), (217, 177)]  # (x, y), inside BOX
CENTER = (160, 140)


def _page():
    rng = np.random.default_rng(0)
    # Mid-gray texture: never white, so filled pixels are easy to tell apart
    return Image.fromarray(rng.integers(110, 190, size=(300, 400, 3), dtype=np.uint8))


def _work(page):
    config = MangaTranslatorConfig(yolo_model_path="unused.pt")
    config.device = torch.device("cpu")
    config.outside_text.enabled = True
    config.outside_text.inpainting_method = "opencv"
    x0, y0, x1, y1 = BOX
    mask = np.zeros((300, 400), bool)
    mask[y0:y1, x0:x1] = True
    result = (np.array(BOX, dtype=np.float32), 0.9)
    item = {
        "bbox": BOX,
        "original_bbox": BOX,
        "is_outside_text": True,
        "is_dark_text": True,
        "tilt_deg": 20.0,
        "tilt_conf": 1.0,
        "tilt_axis": "horizontal",
        "original_crop_pil": page.crop(BOX),
    }
    return otp.OutsideTextWork(
        pil_image=page,
        config=config,
        image_path="page.png",
        image_format="PNG",
        verbose=False,
        outside_text_results=[result],
        raw_outside_text_results=[result],
        original_text_colors={BOX: False},  # light background: white fill
        total_bubble_mask=np.zeros((300, 400), bool),
        outside_detector=None,
        mask_groups=[
            {
                "combined_mask": mask,
                "original_bbox": {
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                },
                "mask_indices": [0],
            }
        ],
        img_w=400,
        img_h=300,
        mime_type="image/png",
        cv2_ext=".png",
        outside_text_data=[item],
    )


@pytest.fixture
def run_page(monkeypatch):
    """Inpaints and renders one tilted OSB box; returns (page, result, render angles)."""
    monkeypatch.setattr(otp, "_detect_osb_text_color", lambda *a, **k: (0, 0, 0))

    def run(text, tilted_fits=True, ocr_text=None):
        angles = []

        def fake_render(pil_image, rotation_deg=0.0, layout_only=False, **kwargs):
            if abs(rotation_deg) > 0.01 and not tilted_fits:
                raise RenderingError("Text too large for bubble at minimum font size")
            if not layout_only:
                angles.append(rotation_deg)
                return pil_image.copy()
            return Image.new("RGBA", (1, 1))

        monkeypatch.setattr(pipeline, "render_text_skia", fake_render)
        page = _page()
        work = _work(page)
        cleaned, data = otp.finish_outside_text_work(work)
        if ocr_text is not None:
            data[0]["ocr_text"] = ocr_text
        result = pipeline._render_translations(
            cleaned,
            data,
            [text],
            [],
            np.asarray(page),
            work.config,
            pipeline._scaled_render_sizes(work.config, 1.0),
            1.0,
            [],
            None,
        )
        return np.asarray(page), np.asarray(result.convert("RGB")), angles

    return run


def _white(out, x, y):
    return (out[y, x] == WHITE).all()


def test_fill_turns_upright_when_only_upright_text_fits(run_page):
    _, out, angles = run_page("HELLO", tilted_fits=False)

    assert angles == [0.0]
    x0, y0, x1, y1 = BOX
    assert (out[y0:y1, x0:x1] == WHITE).all()


def test_fill_follows_the_tilt_when_tilted_text_fits(run_page):
    _, out, angles = run_page("HELLO")

    assert angles == [20.0]
    assert _white(out, *CENTER)
    assert not any(_white(out, x, y) for x, y in CORNERS)


@pytest.mark.parametrize("text", ["HELLO", ""])
def test_tilted_fill_stays_inside_the_text_box(run_page, text):
    page, out, _ = run_page(text)

    x0, y0, x1, y1 = BOX
    outside = np.ones(page.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    np.testing.assert_array_equal(out[outside], page[outside])


def test_unrendered_text_is_still_erased_at_the_detected_tilt(run_page):
    _, out, angles = run_page("")

    assert angles == []
    assert _white(out, *CENTER)
    assert not any(_white(out, x, y) for x, y in CORNERS)


def test_restored_original_patch_gets_no_fill(run_page):
    page, out, angles = run_page("HELLO", ocr_text="HELLO")

    assert angles == []
    np.testing.assert_array_equal(out, page)
