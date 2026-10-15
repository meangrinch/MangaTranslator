import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from core import pipeline
from core.config import MangaTranslatorConfig
from core.terminology import (
    format_glossary_section,
    load_terms,
    normalize_source,
    record_new_terms,
    split_new_terms,
)


@pytest.mark.parametrize(
    ("response", "head"),
    [
        ("1: a || b\n2: c || d", "1: a || b\n2: c || d"),
        ("1: a || b\nNEW TERMS:\n猫猫 -> Maomao", "1: a || b"),
        ("1: a || b\n\nNEW TERMS:\n猫猫 -> Maomao\n", "1: a || b"),
        ("1: a || b\n**NEW TERMS:**\n猫猫 -> Maomao", "1: a || b"),
        ("```\n1: a || b\nNEW TERMS:\n猫猫 -> Maomao\n```", "```\n1: a || b"),
        ("1: a || b\n```\nNEW TERMS:\n猫猫 -> Maomao\n```", "1: a || b"),
    ],
)
def test_split_new_terms_leaves_numbered_part(response, head):
    assert split_new_terms(response)[0] == head


def test_split_new_terms_parses_pairs():
    response = (
        "1: a || b\nNEW TERMS:\n猫猫 -> Maomao\n- 壬氏 -> Jinshi\n`小蘭 -> Xiaolan`\n"
        "not a pair\n -> missing source\n```"
    )
    assert split_new_terms(response)[1] == [
        ("猫猫", "Maomao"),
        ("壬氏", "Jinshi"),
        ("小蘭", "Xiaolan"),
    ]


def test_split_new_terms_handles_none():
    assert split_new_terms(None) == (None, [])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("猫猫(マオマオ)", "猫猫"),
        ("梨花（リファ）妃", "梨花妃"),
        (" 壬氏 ", "壬氏"),
    ],
)
def test_normalize_source_strips_readings(text, expected):
    assert normalize_source(text) == expected


def test_load_terms_missing_file(tmp_path):
    assert load_terms(tmp_path / "missing.txt") == {}


def test_load_terms_ignores_junk_and_keeps_first(tmp_path):
    path = tmp_path / "g.txt"
    path.write_text(
        "# series notes\n猫猫 -> Maomao\n\n猫猫 -> Mao Mao\n -> x\n小蘭 ->\n",
        encoding="utf-8",
    )
    assert load_terms(path) == {"猫猫": "Maomao"}


def test_format_glossary_section():
    assert format_glossary_section({}) == ""
    assert format_glossary_section({"猫猫": "Maomao", "妃": "Consort"}) == (
        "\n## GLOSSARY\n猫猫 -> Maomao\n妃 -> Consort\n"
    )


def test_record_creates_file_and_parent(tmp_path):
    path = tmp_path / "new" / "g.txt"
    added, rejected = record_new_terms(
        path, [("猫猫(マオマオ)", "Maomao")], ["後宮下女猫猫(マオマオ)"]
    )
    assert added == [("猫猫", "Maomao")]
    assert rejected == []
    assert path.read_text(encoding="utf-8") == "猫猫 -> Maomao\n"


def test_record_first_saved_wins(tmp_path):
    path = tmp_path / "g.txt"
    path.write_text("梨花 -> Lihua\n", encoding="utf-8")
    added, rejected = record_new_terms(path, [("梨花", "Lifa")], ["梨花妃"])
    assert (added, rejected) == ([], [])
    assert load_terms(path) == {"梨花": "Lihua"}


def test_record_rejects_source_not_on_page(tmp_path):
    path = tmp_path / "g.txt"
    added, rejected = record_new_terms(
        path, [("壬氏", "Jinshi"), ("小蘭", "Xiaolan")], ["小蘭(シャオラン)"]
    )
    assert added == [("小蘭", "Xiaolan")]
    assert rejected == [("壬氏", "Jinshi")]


def test_record_preserves_existing_lines_and_skips_duplicates(tmp_path):
    path = tmp_path / "g.txt"
    path.write_text("# notes\n猫猫 -> Maomao", encoding="utf-8")
    record_new_terms(path, [("小蘭", "Xiaolan"), ("小蘭", "Shaoran")], ["小蘭"])
    assert path.read_text(encoding="utf-8") == (
        "# notes\n猫猫 -> Maomao\n小蘭 -> Xiaolan\n"
    )


def test_record_without_new_terms_writes_nothing(tmp_path):
    path = tmp_path / "g.txt"
    assert record_new_terms(path, [], ["a"]) == ([], [])
    assert not path.exists()


def _default_glossary(name):
    return str(Path("output") / ".glossary" / f"{name}_{time.strftime('%Y-%m-%d')}.txt")


def _run_batch(monkeypatch, tmp_path, translation_overrides, mode=None, **kwargs):
    # Default glossaries go under ./output/.glossary, so keep them inside tmp_path
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for name in ("page_10.png", "page_2.png"):
        Image.new("RGB", (8, 8), "white").save(input_dir / name)
    seen_paths = []

    def fake_translate_and_render(image_path, config, output_path, **_):
        path = config.translation.terminology_memory_path
        seen_paths.append(path)
        assert path is None or Path(path).is_file()

    monkeypatch.setattr(pipeline, "translate_and_render", fake_translate_and_render)
    config = MangaTranslatorConfig(
        yolo_model_path="model.pt", device=torch.device("cpu")
    )
    config.translation = replace(config.translation, **translation_overrides)
    if mode:
        setattr(config, mode, True)
    output_dir = tmp_path / "out"
    results = pipeline.batch_translate_images(input_dir, config, output_dir, **kwargs)
    assert results["error_count"] == 0
    return seen_paths, results, config


def test_batch_defaults_glossary_to_input_name(monkeypatch, tmp_path):
    seen, results, config = _run_batch(
        monkeypatch, tmp_path, {"terminology_memory": True}, glossary_name="series_ch1"
    )
    expected = _default_glossary("series_ch1")
    assert seen == [expected, expected]
    assert results["glossary_path"] == expected
    assert config.translation.terminology_memory_path is None


def test_batch_glossary_falls_back_to_first_image(monkeypatch, tmp_path):
    seen, _, _ = _run_batch(monkeypatch, tmp_path, {"terminology_memory": True})
    assert seen[0] == _default_glossary("page_2")


def test_batch_keeps_explicit_glossary_path(monkeypatch, tmp_path):
    explicit = str(tmp_path / "series.txt")
    seen, _, _ = _run_batch(
        monkeypatch,
        tmp_path,
        {"terminology_memory": True, "terminology_memory_path": explicit},
        glossary_name="ignored",
    )
    assert seen == [explicit, explicit]


def test_batch_without_memory_sets_no_path(monkeypatch, tmp_path):
    seen, _, _ = _run_batch(monkeypatch, tmp_path, {}, glossary_name="series_ch1")
    assert seen == [None, None]


@pytest.mark.parametrize("mode", ["test_mode", "cleaning_only", "upscaling_only"])
def test_batch_without_translation_creates_no_glossary(monkeypatch, tmp_path, mode):
    seen, results, _ = _run_batch(
        monkeypatch, tmp_path, {"terminology_memory": True}, mode=mode
    )
    assert seen == [None, None]
    assert "glossary_path" not in results
    assert not (tmp_path / "output" / ".glossary").exists()


def _run_translation(monkeypatch, tmp_path, mode, responses, seed=""):
    from core.config import TranslationConfig
    from core.services import translation

    path = tmp_path / "g.txt"
    if seed:
        path.write_text(seed, encoding="utf-8")
    calls = []

    def fake_call(config, parts, prompt, debug, system_prompt=None, **_):
        calls.append((prompt, system_prompt))
        return responses[len(calls) - 1]

    monkeypatch.setattr(translation, "_call_llm_endpoint", fake_call)
    config = TranslationConfig(
        translation_mode=mode,
        temperature=0.7,
        terminology_memory=True,
        terminology_memory_path=str(path),
    )
    result = translation.call_translation_api_batch(
        config,
        [f"{mode}-crop-1", f"{mode}-crop-2"],
        "",
        ["image/png", "image/png"],
        "image/png",
        [{}, {}],
    )
    return result, calls, path


def test_one_step_records_new_terms_and_injects_glossary(monkeypatch, tmp_path):
    response = (
        "1: 猫猫(マオマオ) || Maomao\n2: 小蘭(シャオラン)だ || It's Xiaolan.\n"
        "NEW TERMS:\n小蘭 -> Xiaolan\n壬氏 -> Jinshi"
    )
    result, calls, path = _run_translation(
        monkeypatch, tmp_path, "one-step", [response], seed="猫猫 -> Maomao\n"
    )
    assert result == ["Maomao", "It's Xiaolan."]
    prompt, system_prompt = calls[0]
    assert "## GLOSSARY\n猫猫 -> Maomao" in prompt
    assert "NEW TERMS:" in system_prompt
    assert path.read_text(encoding="utf-8") == "猫猫 -> Maomao\n小蘭 -> Xiaolan\n"


def test_two_step_records_terms_against_ocr_text(monkeypatch, tmp_path):
    responses = [
        "1: 梨花(リファ)妃\n2: 玉葉妃",
        "1: Consort Lihua\n2: Consort Gyokuyou\nNEW TERMS:\n梨花 -> Lihua\n妃 -> Consort",
    ]
    result, calls, path = _run_translation(monkeypatch, tmp_path, "two-step", responses)
    assert result == ["Consort Lihua", "Consort Gyokuyou"]
    assert "NEW TERMS:" in calls[1][1]
    assert path.read_text(encoding="utf-8") == "梨花 -> Lihua\n妃 -> Consort\n"


def test_ui_maps_terminology_memory_for_batch_only(tmp_path):
    from ui.ui_models import UIConfigState, map_ui_to_backend_config

    state = UIConfigState(batch_glossary_file="series.txt")
    batch = map_ui_to_backend_config(state, tmp_path, None, is_batch=True)
    single = map_ui_to_backend_config(state, tmp_path, None)
    assert batch.translation.terminology_memory is True
    assert batch.translation.terminology_memory_path == "series.txt"
    assert single.translation.terminology_memory is False
    assert single.translation.terminology_memory_path is None

    reloaded = UIConfigState.from_dict(
        {**state.to_save_dict(), "batch_terminology_memory": False}
    )
    assert reloaded.batch_terminology_memory is False
    # A saved choice would carry one series' glossary into the next series' batches
    assert "batch_glossary_file" not in state.to_save_dict()
    assert (
        UIConfigState.from_dict({"batch_glossary_file": "old.txt"}).batch_glossary_file
        == ""
    )


def test_cache_hit_rebuilds_glossary_without_llm_call(monkeypatch, tmp_path):
    from core.config import TranslationConfig
    from core.services import translation

    responses = ["1: 小蘭 || Xiaolan\n2: 猫猫 || Maomao\nNEW TERMS:\n小蘭 -> Xiaolan"]
    calls = []

    def fake_call(config, parts, prompt, debug, system_prompt=None, **_):
        calls.append(prompt)
        return responses[0]

    monkeypatch.setattr(translation, "_call_llm_endpoint", fake_call)

    def run(glossary_path):
        config = TranslationConfig(
            translation_mode="one-step",
            temperature=0.0,
            terminology_memory=True,
            terminology_memory_path=str(glossary_path),
        )
        return translation.call_translation_api_batch(
            config,
            ["cached-crop-1", "cached-crop-2"],
            "",
            ["image/png"] * 2,
            "image/png",
            [{}, {}],
        )

    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    assert run(first) == run(second) == ["Xiaolan", "Maomao"]
    assert len(calls) == 1
    assert second.read_text(encoding="utf-8") == "小蘭 -> Xiaolan\n"


@pytest.mark.parametrize(
    "marker",
    ["New terms:", "NEW TERMS", "### NEW TERMS", "**New Terms:**", "`NEW TERMS:`"],
)
def test_split_new_terms_accepts_marker_variants(marker):
    head, pairs = split_new_terms(f"1: a || b\n{marker}\n猫猫 -> Maomao")
    assert head == "1: a || b"
    assert pairs == [("猫猫", "Maomao")]


def test_split_new_terms_reads_inline_pair_after_marker():
    assert split_new_terms("1: a || b\nNEW TERMS: 猫猫 -> Maomao") == (
        "1: a || b",
        [("猫猫", "Maomao")],
    )


def test_split_new_terms_ignores_marker_inside_an_item():
    response = "1: a || The NEW TERMS: list\n2: c || d"
    assert split_new_terms(response) == (response, [])


def test_split_new_terms_ignores_item_line_starting_with_new_terms():
    response = "1: a || The contract is void.\nNew terms will apply.\n2: c || d"
    assert split_new_terms(response) == (response, [])


def test_split_new_terms_drops_tagged_fence_before_block():
    head, _ = split_new_terms("1: a || b\n```text\nNEW TERMS:\n猫猫 -> Maomao\n```")
    assert head == "1: a || b"


def test_split_new_terms_parses_bullet_and_arrow_variants():
    response = (
        "1: a || b\nNEW TERMS:\n* 猫猫 -> Maomao\n1. 小蘭 → Xiaolan\n"
        "**梨花** -> **Lihua**\n+ 玉葉 -> Gyokuyou"
    )
    assert split_new_terms(response)[1] == [
        ("猫猫", "Maomao"),
        ("小蘭", "Xiaolan"),
        ("梨花", "Lihua"),
        ("玉葉", "Gyokuyou"),
    ]


def test_record_treats_reading_keyed_entry_as_known(tmp_path):
    path = tmp_path / "g.txt"
    path.write_text("猫猫(マオマオ) -> Maomao\n", encoding="utf-8")
    assert record_new_terms(path, [("猫猫", "Mao")], ["猫猫"]) == ([], [])


def test_load_terms_ignores_utf8_bom(tmp_path):
    path = tmp_path / "g.txt"
    path.write_bytes("\ufeff猫猫 -> Maomao\n".encode())
    assert load_terms(path) == {"猫猫": "Maomao"}


def test_record_removes_temp_file_when_replace_fails(monkeypatch, tmp_path):
    from core import terminology

    def fail_replace(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(terminology.os, "replace", fail_replace)
    with pytest.raises(PermissionError):
        record_new_terms(tmp_path / "g.txt", [("猫猫", "Maomao")], ["猫猫"])
    assert list(tmp_path.iterdir()) == []


def test_glossary_write_failure_keeps_translation(monkeypatch, tmp_path):
    from core.services import translation

    def fail_record(*_):
        raise PermissionError("locked by another process")

    monkeypatch.setattr(translation, "record_new_terms", fail_record)
    result, _, _ = _run_translation(
        monkeypatch,
        tmp_path,
        "one-step",
        ["1: 猫猫 || Maomao\n2: a || b\nNEW TERMS:\n猫猫 -> Maomao"],
    )
    assert result == ["Maomao", "b"]


def test_batch_rejects_folder_as_glossary_path(monkeypatch, tmp_path):
    from utils.exceptions import ValidationError

    with pytest.raises(ValidationError, match="folder"):
        _run_batch(
            monkeypatch,
            tmp_path,
            {"terminology_memory": True, "terminology_memory_path": str(tmp_path)},
        )


def test_batch_rejects_unreadable_glossary(monkeypatch, tmp_path):
    from utils.exceptions import ValidationError

    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ValidationError, match="glossary"):
        _run_batch(
            monkeypatch,
            tmp_path,
            {"terminology_memory": True, "terminology_memory_path": str(bad)},
        )


def test_ui_strips_quotes_from_glossary_path(tmp_path):
    from ui.ui_models import UIConfigState, map_ui_to_backend_config

    state = UIConfigState(batch_glossary_file='"D:\\manga\\series.txt"')
    config = map_ui_to_backend_config(state, tmp_path, None, is_batch=True)
    assert config.translation.terminology_memory_path == "D:\\manga\\series.txt"
