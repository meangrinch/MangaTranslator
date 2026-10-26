import pytest
import torch

from core.config import (
    MangaTranslatorConfig,
    RenderingConfig,
    TranslationConfig,
)
from core.validation import (
    clamp_settings,
    validate_batch_input_path,
    validate_config,
    validate_core_inputs,
    validate_zip_file,
)
from utils.exceptions import ValidationError


def _config(**kwargs):
    return MangaTranslatorConfig(
        yolo_model_path="model.pt", device=torch.device("cpu"), **kwargs
    )


def test_clamp_settings_dict():
    settings = {
        "temperature": 5.0,
        "top_k": 100,
        "jpeg_quality": 0,
        "max_tokens": None,
        "png_compression": True,
        "unknown": 999,
    }
    clamped = clamp_settings(settings)
    assert clamped == {
        "temperature": 2.0,
        "top_k": 64,
        "jpeg_quality": 1,
        "max_tokens": None,
        "png_compression": True,
        "unknown": 999,
    }
    assert isinstance(clamped["top_k"], int)
    assert settings["temperature"] == 5.0


def test_clamp_settings_config_object():
    config = _config(parallel_requests=100)
    config.translation.temperature = 9.0
    config.rendering.max_font_size = 1
    config.output.image_upscale_factor = 20.0
    config.preprocessing.factor = 0.1

    assert clamp_settings(config) is config
    assert config.translation.temperature == 2.0
    assert config.rendering.max_font_size == 5
    assert config.output.image_upscale_factor == 8.0
    assert config.preprocessing.factor == 1.0
    assert config.parallel_requests == 20


def test_validate_config_rejects_nunchaku_with_klein():
    config = _config()
    config.outside_text.inpainting_method = "flux_klein_4b"
    config.outside_text.flux_backend = "nunchaku"
    with pytest.raises(ValidationError):
        validate_config(config)

    config.outside_text.inpainting_method = "flux_kontext"
    validate_config(config)


def test_validate_config_rejects_sdnq_with_fibo():
    config = _config()
    config.outside_text.inpainting_method = "fibo2_edit_turbo"
    config.outside_text.flux_backend = "sdnq"
    with pytest.raises(ValidationError):
        validate_config(config)

    config.outside_text.flux_backend = "sdcpp"
    validate_config(config)
    # Methods without a backend ignore it
    config.outside_text.inpainting_method = "lama_manga"
    config.outside_text.flux_backend = "nunchaku"
    validate_config(config)


def test_validate_batch_input_path(tmp_path):
    assert validate_batch_input_path(tmp_path) == tmp_path

    path_list = tmp_path / "failed_paths.txt"
    path_list.write_text("")
    assert validate_batch_input_path(path_list) == path_list

    comic = tmp_path / "chapter.CBZ"
    comic.write_bytes(b"")
    assert validate_batch_input_path(comic) == comic
    assert validate_zip_file(comic) == comic

    image = tmp_path / "page.png"
    image.write_bytes(b"")
    with pytest.raises(ValidationError):
        validate_batch_input_path(image)

    with pytest.raises(FileNotFoundError):
        validate_batch_input_path(tmp_path / "missing")


@pytest.fixture
def dirs(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    fonts = tmp_path / "fonts"
    (fonts / "Pack").mkdir(parents=True)
    (fonts / "Pack" / "font.ttf").write_bytes(b"")
    return models, fonts


def test_validate_core_inputs_ok(dirs):
    models, fonts = dirs
    yolo, font_dir = validate_core_inputs(
        TranslationConfig(), RenderingConfig(font_dir="Pack"), models, fonts
    )
    assert yolo == (models / "yolo" / "manga109-segmentation-bubble.pt").resolve()
    assert font_dir == (fonts / "Pack").resolve()


@pytest.mark.parametrize(
    ("translation", "rendering"),
    [
        (TranslationConfig(), RenderingConfig(font_dir="Pack", min_font_size=40)),
        (TranslationConfig(reading_direction="ttb"), RenderingConfig(font_dir="Pack")),
        (TranslationConfig(model_name=""), RenderingConfig(font_dir="Pack")),
    ],
)
def test_validate_core_inputs_rejects_bad_config(dirs, translation, rendering):
    models, fonts = dirs
    with pytest.raises(ValidationError):
        validate_core_inputs(translation, rendering, models, fonts)


def test_validate_core_inputs_rejects_empty_font_pack(dirs):
    models, fonts = dirs
    (fonts / "Empty").mkdir()
    with pytest.raises(ValidationError):
        validate_core_inputs(
            TranslationConfig(), RenderingConfig(font_dir="Empty"), models, fonts
        )
