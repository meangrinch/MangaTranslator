from dataclasses import replace

import pytest
from PIL import Image

from core.caching import UnifiedCache
from core.config import TranslationConfig

RANDOM = TranslationConfig(temperature=0.7, top_k=40, top_p=0.9)
DETERMINISTIC = replace(RANDOM, temperature=0.0)


def _key(cache, config=DETERMINISTIC, **kwargs):
    return cache.get_translation_cache_key(["b1", "b2"], "page", config, **kwargs)


@pytest.fixture
def cache():
    return UnifiedCache()


def test_translation_key_none_when_sampling_is_random(cache):
    assert _key(cache, RANDOM) is None
    assert _key(cache, replace(RANDOM, force_cache_translations=True)) is not None


@pytest.mark.parametrize(
    "overrides", [{"temperature": 0.0}, {"top_k": 1}, {"top_p": 0.0}]
)
def test_translation_key_present_when_deterministic(cache, overrides):
    assert _key(cache, replace(RANDOM, **overrides)) is not None


def test_translation_key_is_stable(cache):
    assert _key(cache) == _key(UnifiedCache())
    assert _key(cache, replace(DETERMINISTIC, special_instructions="x")) == _key(
        cache, replace(DETERMINISTIC, special_instructions="  x \n")
    )


@pytest.mark.parametrize(
    "change",
    [
        {"config": replace(DETERMINISTIC, model_name="other-model")},
        {"config": replace(DETERMINISTIC, output_language="French")},
        {"previous_context_texts": [["earlier line"]]},
        {"previous_context_images": [{"data": "img"}]},
    ],
)
def test_translation_key_changes_with_inputs(cache, change):
    assert _key(cache, **change) != _key(cache)


def test_translation_round_trip(cache):
    key = _key(cache)
    cache.set_translation(key, ["hello"], ocr_texts=["こんにちは"])
    assert cache.get_translation(key) == (["hello"], ["こんにちは"])
    assert cache.get_translation(None) == (None, None)


def test_yolo_key_depends_on_pixels_and_confidence(cache):
    white = Image.new("RGB", (8, 8), "white")
    black = Image.new("RGB", (8, 8), "black")
    base = cache.get_yolo_cache_key(white, "m.pt", 0.5)
    assert base == cache.get_yolo_cache_key(white.copy(), "m.pt", 0.5)
    assert base != cache.get_yolo_cache_key(black, "m.pt", 0.5)
    assert base != cache.get_yolo_cache_key(white, "m.pt", 0.6)


def test_set_current_image_clears_caches_on_new_image(cache):
    white = Image.new("RGB", (8, 8), "white")
    cache.set_current_image(white)
    key = _key(cache)
    cache.set_translation(key, ["hello"])

    cache.set_current_image(white.copy())
    assert cache.get_translation(key)[0] == ["hello"]

    cache.set_current_image(Image.new("RGB", (8, 8), "black"))
    assert cache.get_translation(key) == (None, None)
