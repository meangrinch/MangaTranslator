import numpy as np
import pytest
import torch
from PIL import Image

from core.image.inpainting import FluxKleinInpainter
from core.image.ocr_detection import OutsideTextDetector


def test_klein_inpaints_masks_wider_than_its_max_resolution(monkeypatch):
    inpainter = FluxKleinInpainter(
        backend="sdcpp", device=torch.device("cpu"), luminance_correction=False
    )
    inpainter.pipeline = inpainter.sdcpp_assets = {}
    monkeypatch.setattr(inpainter, "load_models", lambda: None)
    monkeypatch.setattr(
        inpainter,
        "_run_sdcpp_inference",
        lambda image, width, height, seed, verbose=False: Image.new(
            "RGB", (width, height), "white"
        ),
    )
    page = Image.new("RGB", (2600, 200), "black")
    mask = np.zeros((200, 2600), dtype=bool)
    mask[50:150, 20:2580] = True

    result = np.asarray(inpainter.inpaint_mask(page, mask, seed=7))

    assert result[100, 40].min() > 200
    assert result[100, 2560].min() > 200


@pytest.mark.parametrize("upscale_small_crops", [True, False])
@pytest.mark.parametrize("size", [(3000, 400), (400, 3000), (2600, 200)])
def test_klein_inference_keeps_the_crop_shape(upscale_small_crops, size):
    inpainter = FluxKleinInpainter(
        device=torch.device("cpu"), upscale_small_crops=upscale_small_crops
    )
    width, height = size

    prepared, _, _ = inpainter._prepare_image_for_inference(Image.new("RGB", size))
    new_w, new_h = prepared.size

    long_side, short_side = max(width, height), min(width, height)
    new_long, new_short = max(new_w, new_h), min(new_w, new_h)
    assert new_long <= inpainter.MAX_RESOLUTION
    # The short side is snapped down to a multiple of 16, so allow that much drift
    expected_short = short_side * new_long / long_side
    assert abs(new_short - expected_short) <= inpainter.RESOLUTION_MULTIPLE


def test_osb_groups_keep_results_aligned_after_a_degenerate_box():
    results = [((10, 10, 10, 30), 0.9), ((100, 100, 140, 120), 0.5)]
    groups, _ = OutsideTextDetector(device=torch.device("cpu")).get_text_masks(
        "unused.png",
        image_override=Image.new("RGB", (200, 200), "white"),
        existing_results=results,
    )

    assert len(groups) == 1
    assert groups[0]["mask_indices"] == [1]
    assert groups[0]["confidence"] == 0.5
    assert groups[0]["original_bbox"]["x"] == 100
