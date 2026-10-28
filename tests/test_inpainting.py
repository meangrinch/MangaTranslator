import numpy as np
import pytest
import torch
from PIL import Image

from core.image.inpainting import (
    Fibo2Inpainter,
    Flux3KleinInpainter,
    FluxKleinInpainter,
    FluxKontextInpainter,
    LamaInpainter,
    QwenImage21Inpainter,
    make_flux_inpainter,
)
from core.image.ocr_detection import OutsideTextDetector
from utils.model_metadata import FLUX_SDCPP_INT8_CONVROT


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


def _make(method, **overrides):
    kwargs = {
        "device": torch.device("cpu"),
        "huggingface_token": "",
        "num_inference_steps": 4,
        "backend": "sdcpp",
        "low_vram": True,
        "residual_diff_threshold": 0.15,
        "luminance_correction": False,
        "upscale_small_crops": True,
        "sdcpp_cache_mode": "none",
        "sdcpp_diffusion_quant": "",
        "sdcpp_text_encoder_quant": "",
    }
    kwargs.update(overrides)
    return make_flux_inpainter(method, **kwargs)


@pytest.mark.parametrize(
    ("method", "cls"),
    [
        ("flux_klein_4b", FluxKleinInpainter),
        ("flux3_klein", Flux3KleinInpainter),
        ("fibo2_edit_turbo", Fibo2Inpainter),
        ("qwen_image_2_1_turbo", QwenImage21Inpainter),
        ("lama_manga", LamaInpainter),
        ("flux_kontext", FluxKontextInpainter),
    ],
)
def test_make_flux_inpainter_dispatches_each_method(method, cls):
    assert type(_make(method)) is cls


@pytest.mark.parametrize(
    "method",
    [
        "flux_klein_4b",
        "flux3_klein",
        "fibo2_edit_turbo",
        "qwen_image_2_1_turbo",
        "lama_manga",
        "flux_kontext",
    ],
)
def test_every_model_takes_the_luminance_correction_setting(method):
    assert _make(method, luminance_correction=True).luminance_correction is True
    assert _make(method, luminance_correction=False).luminance_correction is False


def _white_patch_inpainter(method, monkeypatch, luminance_correction):
    """An inpainter whose model paints the whole crop white."""
    inpainter = _make(method, luminance_correction=luminance_correction)
    if method == "lama_manga":

        def fake_model(image, mask):
            return torch.ones_like(image)

        fake_model.device = torch.device("cpu")
        monkeypatch.setattr(
            inpainter, "load_models", lambda: setattr(inpainter, "pipeline", fake_model)
        )
    else:
        inpainter.pipeline = object()
        monkeypatch.setattr(inpainter, "load_models", lambda: None)
        monkeypatch.setattr(
            inpainter,
            "_generate_patch",
            lambda image, width, height, seed, verbose: Image.new(
                "RGB", (width, height), "white"
            ),
        )
    return inpainter


@pytest.mark.parametrize("method", ["lama_manga", "flux_kontext"])
@pytest.mark.parametrize("luminance_correction", [True, False])
def test_luminance_correction_matches_the_patch_to_its_context(
    method, luminance_correction, monkeypatch
):
    inpainter = _white_patch_inpainter(method, monkeypatch, luminance_correction)
    page = Image.new("RGB", (400, 300), (128, 128, 128))
    mask = np.zeros((300, 400), dtype=bool)
    mask[100:140, 150:250] = True

    result = np.asarray(inpainter.inpaint_mask(page, mask, seed=-1))

    if luminance_correction:
        assert abs(int(result[120, 200].mean()) - 128) <= 3
    else:
        assert result[120, 200].min() > 250


def test_fibo_rejects_sdnq_backend():
    with pytest.raises(ValueError):
        _make("fibo2_edit_turbo", backend="sdnq")


def test_new_models_default_to_their_own_quants():
    fibo = _make("fibo2_edit_turbo")
    assert fibo.sdcpp_diffusion_quant == FLUX_SDCPP_INT8_CONVROT
    assert fibo.sdcpp_text_encoder_quant == "Q4_K_XL"


def test_qwen_runs_its_fixed_steps_on_a_32px_grid():
    inpainter = _make("qwen_image_2_1_turbo", num_inference_steps=3)
    assert inpainter.num_inference_steps == 8

    prepared, _, _ = inpainter._prepare_image_for_inference(
        Image.new("RGB", (700, 300))
    )
    assert prepared.size[0] % 32 == 0
    assert prepared.size[1] % 32 == 0


def test_lama_gets_the_crop_mask_at_native_size(monkeypatch):
    inpainter = _make("lama_manga", upscale_small_crops=True)
    seen = {}

    def fake_model(image, mask):
        seen["image"] = tuple(image.shape)
        seen["mask"] = tuple(mask.shape)
        seen["mask_sum"] = float(mask.sum())
        return torch.ones_like(image)

    fake_model.device = torch.device("cpu")
    monkeypatch.setattr(
        inpainter, "load_models", lambda: setattr(inpainter, "pipeline", fake_model)
    )
    page = Image.new("RGB", (400, 300), "black")
    mask = np.zeros((300, 400), dtype=bool)
    mask[100:140, 150:250] = True

    result = np.asarray(inpainter.inpaint_mask(page, mask, seed=-1))

    # No ~1MP upscale: LaMa sees the padded crop itself, mask channel included
    _, _, height, width = seen["image"]
    assert seen["mask"] == (1, 1, height, width)
    assert width * height < 400 * 300 + 1
    assert seen["mask_sum"] == mask.sum()
    assert result[120, 200].min() > 200
    assert result[10, 10].max() == 0


def test_lama_only_erases_inside_the_composite_clip_box(monkeypatch):
    inpainter = _make("lama_manga")
    seen = {}

    def fake_model(image, mask):
        seen["mask_sum"] = float(mask.sum())
        return torch.ones_like(image)

    fake_model.device = torch.device("cpu")
    monkeypatch.setattr(
        inpainter, "load_models", lambda: setattr(inpainter, "pipeline", fake_model)
    )
    page = Image.new("RGB", (400, 300), "black")
    mask = np.zeros((300, 400), dtype=bool)
    mask[100:140, 150:250] = True

    inpainter.inpaint_mask(page, mask, seed=-1, composite_clip_bbox=(170, 90, 230, 150))

    # LaMa fills whatever it is told to erase, so the margin beyond the text box stays
    assert seen["mask_sum"] == 40 * 60


def test_lama_checkpoint_loads_through_spandrel(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from spandrel.architectures.LaMa import LaMa

    from core.ml.model_manager import ModelType, get_model_manager

    # mayocream/lama-manga stores the generator's own keys ("model.1.ffc...")
    state_dict = {
        key.removeprefix("model."): tensor.contiguous()
        for key, tensor in LaMa(in_nc=4, out_nc=3).state_dict().items()
    }
    checkpoint = tmp_path / "lama-manga.safetensors"
    save_file(state_dict, str(checkpoint))
    manager = get_model_manager()
    monkeypatch.setattr(manager, "_ensure_hf_file", lambda *args, **kwargs: checkpoint)
    monkeypatch.setitem(manager.models, ModelType.LAMA_MANGA, None)

    model = manager.load_lama_manga()

    assert model.architecture.id == "LaMa"
    manager.unload_lama_manga()


class _DeviceRecorder:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


def test_parked_text_sam_returns_from_ram_without_reloading(monkeypatch):
    from core.ml.model_manager import ModelType, get_model_manager

    manager = get_model_manager()
    model = _DeviceRecorder()
    monkeypatch.setattr(manager, "device", torch.device("cuda"))
    monkeypatch.setattr(manager, "_ensure_hf_file", lambda *a, **k: pytest.fail())
    monkeypatch.setitem(manager.models, ModelType.TEXT_SAM, model)

    manager.park_text_sam()
    assert manager.load_text_sam() is model

    assert model.moves == ["cpu", "cuda"]


def test_text_sam_is_unloaded_instead_of_parked_on_cpu(monkeypatch):
    from core.ml.model_manager import ModelType, get_model_manager

    manager = get_model_manager()
    monkeypatch.setattr(manager, "device", torch.device("cpu"))
    monkeypatch.setitem(manager.models, ModelType.TEXT_SAM, _DeviceRecorder())
    lama = object()
    monkeypatch.setitem(manager.models, ModelType.LAMA_MANGA, lama)

    manager.park_text_sam()

    assert not manager.is_loaded(ModelType.TEXT_SAM)
    assert manager.models[ModelType.LAMA_MANGA] is lama


@pytest.mark.parametrize(
    ("method", "parks"),
    [
        ("flux_klein_4b", True),
        ("flux_kontext", True),
        ("qwen_image_2_1_turbo", True),
        ("lama_manga", False),
    ],
)
def test_only_diffusion_inpainters_park_the_text_model_before_loading(
    monkeypatch, method, parks
):
    inpainter = _make(method)
    calls = []
    monkeypatch.setattr(
        inpainter.manager, "park_text_sam", lambda verbose=False: calls.append("park")
    )
    monkeypatch.setattr(inpainter, "load_models", lambda: calls.append("load"))
    page = Image.new("RGB", (400, 300), "black")
    mask = np.zeros((300, 400), dtype=bool)
    mask[100:140, 150:250] = True

    inpainter.inpaint_mask(page, mask, seed=-1)

    assert calls == (["park", "load"] if parks else ["load"])


def test_parking_twice_moves_the_text_sam_once(monkeypatch):
    from core.ml.model_manager import ModelType, get_model_manager

    manager = get_model_manager()
    model = _DeviceRecorder()
    monkeypatch.setattr(manager, "device", torch.device("cuda"))
    monkeypatch.setitem(manager.models, ModelType.TEXT_SAM, model)
    monkeypatch.setattr(manager, "_text_sam_parked", False)

    manager.park_text_sam()
    manager.park_text_sam()  # every diffusion job parks it
    manager.load_text_sam()
    manager.park_text_sam()

    assert model.moves == ["cpu", "cuda", "cpu"]
