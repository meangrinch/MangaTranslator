from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from core.caching import UnifiedCache
from core.config import TranslationConfig
from core.ml.model_manager import ModelType, get_model_manager
from ui import callbacks, layout, utils


def test_hayai_and_baberu_model_types_and_repos():
    assert layout.OCR_METHODS == [
        "LLM",
        "baberu-ocr",
        "hayai-ocr-v2.5-nova",
        "manga-ocr",
        "paddleocr-vl-1.6",
    ]
    manager = get_model_manager()
    assert ModelType.HAYAI_OCR.value == "hayai_ocr"
    assert ModelType.BABERU_OCR.value == "baberu_ocr"

    assert (
        manager.model_hf_repos[ModelType.HAYAI_OCR]["repo_id"]
        == "JustANormalTinkerer/hayai-ocr-v2.5-nova"
    )
    assert (
        manager.model_hf_repos[ModelType.BABERU_OCR]["repo_id"]
        == "genshiai-daichi/baberu-ocr"
    )
    assert "onnx/*" in manager.model_hf_repos[ModelType.BABERU_OCR].get(
        "ignore_patterns", []
    )


def test_unload_ocr_models_includes_hayai_and_baberu():
    manager = get_model_manager()
    manager.models[ModelType.HAYAI_OCR] = ("proc", "tok", "model")
    manager.models[ModelType.BABERU_OCR] = ("proc", "tok", "model", {1, 2})
    try:
        manager.unload_ocr_models()
        assert not manager.is_loaded(ModelType.HAYAI_OCR)
        assert not manager.is_loaded(ModelType.BABERU_OCR)
    finally:
        manager.models[ModelType.HAYAI_OCR] = None
        manager.models[ModelType.BABERU_OCR] = None


def test_ocr_cache_key_prefixes_are_distinct():
    cache = UnifiedCache()
    mocr_key = cache.get_manga_ocr_cache_key(["img1"], 1, prefix="mocr_")
    pocr_key = cache.get_manga_ocr_cache_key(["img1"], 1, prefix="pocr16_")
    hayai_key = cache.get_manga_ocr_cache_key(["img1"], 1, prefix="hayai25_")
    baberu_key = cache.get_manga_ocr_cache_key(["img1"], 1, prefix="baberu_")
    assert len({mocr_key, pocr_key, hayai_key, baberu_key}) == 4


def test_extract_text_with_hayai_ocr_normalizes_and_preserves_hangul_spaces():
    from core.image.ocr_detection import extract_text_with_hayai_ocr

    img = Image.new("RGB", (64, 64), (255, 255, 255))

    fake_proc = MagicMock(
        return_value={
            "pixel_values": MagicMock(to=lambda dev: MagicMock(device=dev)),
            "pixel_attention_mask": MagicMock(to=lambda dev: MagicMock()),
            "spatial_shapes": MagicMock(to=lambda dev: MagicMock()),
        }
    )
    fake_tok = MagicMock()
    fake_model = MagicMock()
    fake_model.device = "cpu"
    # Return fullwidth + inter-CJK space and Korean Hangul with space
    fake_model.generate.return_value = ["お は よ う ！ 안녕 하세요"]

    with patch("core.image.ocr_detection.get_model_manager") as mock_mm:
        mock_mm.return_value.get_hayai_ocr.return_value = (
            fake_proc,
            fake_tok,
            fake_model,
        )
        results = extract_text_with_hayai_ocr([img], verbose=False)

    assert results == ["おはよう! 안녕 하세요"]


def test_extract_text_with_baberu_ocr_decodes_tokens():
    import torch

    from core.image.ocr_detection import extract_text_with_baberu_ocr

    img = Image.new("RGB", (64, 64), (255, 255, 255))

    fake_proc = MagicMock(return_value={"pixel_values": torch.zeros((1, 3, 224, 224))})
    fake_tok = MagicMock()
    fake_tok.bos_token_id = 1
    fake_tok.eos_token_id = 2
    fake_tok.pad_token_id = 0
    fake_tok.decode.return_value = "Hello\nWorld"

    fake_model = MagicMock()
    fake_model.device = torch.device("cpu")
    fake_model.generate.return_value = torch.tensor([[1, 10, 11, 2]])

    with patch("core.image.ocr_detection.get_model_manager") as mock_mm:
        mock_mm.return_value.get_baberu_ocr.return_value = (
            fake_proc,
            fake_tok,
            fake_model,
            {10, 11},
        )
        results = extract_text_with_baberu_ocr([img], verbose=False)

    assert results == ["Hello\nWorld"]


def test_run_ocr_step_dispatches_hayai_and_baberu():
    from core.services.translation import _run_ocr_step

    cfg_hayai = TranslationConfig(ocr_method="hayai-ocr-v2.5-nova")
    cfg_baberu = TranslationConfig(ocr_method="baberu-ocr")

    with patch("core.services.translation._perform_hayai_ocr") as mock_hayai:
        mock_hayai.return_value = ["テスト"]
        res = _run_ocr_step(cfg_hayai, ["b64"], ["image/jpeg"], [{}], False, "key")
        assert res == ["テスト"]
        mock_hayai.assert_called_once_with(["b64"], [{}], False)

    with patch("core.services.translation._perform_baberu_ocr") as mock_baberu:
        mock_baberu.return_value = ["テスト2"]
        res = _run_ocr_step(cfg_baberu, ["b64"], ["image/jpeg"], [{}], False, "key")
        assert res == ["テスト2"]
        mock_baberu.assert_called_once_with(["b64"], [{}], False)


@pytest.mark.parametrize(
    ("ocr_method", "expected_langs"),
    [
        (
            "hayai-ocr-v2.5-nova",
            [
                "Japanese",
                "Korean",
                "Chinese (Simplified)",
                "Chinese (Traditional)",
                "English",
            ],
        ),
        (
            "baberu-ocr",
            [
                "Japanese",
                "Chinese (Simplified)",
                "Chinese (Traditional)",
                "English",
            ],
        ),
    ],
)
def test_handle_ocr_method_change_filters_languages_and_disables_image_controls(
    ocr_method, expected_langs
):
    assert hasattr(layout, "HAYAI_OCR_LANGUAGES")
    assert hasattr(layout, "BABERU_OCR_LANGUAGES")

    updates = callbacks.handle_ocr_method_change(
        ocr_method=ocr_method,
        input_language="Spanish",
        original_language_state="Spanish",
        batch_input_language="Korean",
        batch_original_language_state="Korean",
        provider="Google",
        current_model="gemini-3-flash-preview",
        openai_compatible_url="http://localhost:8080/v1",
        openai_compatible_api_key="",
    )

    # Index 1: single input_language dropdown update; Index 2: saved original_language_state
    single_lang_update = updates[1]
    assert single_lang_update["choices"] == expected_langs
    assert single_lang_update["value"] == "Japanese"
    assert single_lang_update["interactive"] is True
    assert updates[2] == "Spanish"

    # Index 3: batch_input_language dropdown update; Index 4: batch_original_language_state
    batch_lang_update = updates[3]
    assert batch_lang_update["choices"] == expected_langs
    if "Korean" in expected_langs:
        assert batch_lang_update["value"] == "Korean"
    else:
        assert batch_lang_update["value"] == "Japanese"
    assert batch_lang_update["interactive"] is True

    # Index 5: send_full_page_context disabled in local OCR mode
    assert updates[5]["value"] is False
    assert updates[5]["interactive"] is False


@pytest.mark.parametrize("ocr_method", ["hayai-ocr-v2.5-nova", "baberu-ocr"])
def test_handle_translation_mode_change_resets_local_ocr_on_one_step(ocr_method):
    ocr_update, corr_update = callbacks.handle_translation_mode_change(
        "one-step", ocr_method
    )
    assert ocr_update["value"] == "LLM"
    assert ocr_update["interactive"] is False
    assert corr_update["visible"] is False


@pytest.mark.parametrize("ocr_method", ["hayai-ocr-v2.5-nova", "baberu-ocr"])
def test_openrouter_models_filter_text_only_for_hayai_and_baberu(ocr_method):
    utils.OPENROUTER_MODEL_CACHE["raw_response"] = [
        {
            "id": "text-only/model",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
        },
        {
            "id": "image-gen/model",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["image"],
            },
        },
    ]
    try:
        update = utils.fetch_and_update_openrouter_models(ocr_method=ocr_method)
        assert update["choices"] == ["text-only/model"]
    finally:
        utils.OPENROUTER_MODEL_CACHE.clear()
