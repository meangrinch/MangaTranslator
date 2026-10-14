from types import SimpleNamespace

import pytest

from core.services.translation import (
    _build_rosetta_instruction,
    _build_system_prompt_hy_mt2,
    _build_system_prompt_ocr,
    _build_system_prompt_translation,
    _check_ocr_failure,
    _format_previous_context_texts,
    _join_ocr_lines,
    _pair_context_lines,
    _parse_llm_response_unified,
    _parse_rosetta_response,
    _uses_code_execution,
)
from utils.exceptions import TranslationError


def test_parse_numbered_list():
    response = '1: Hello\n2. "Quoted"\n3 : Spaced'
    assert _parse_llm_response_unified(response, 3, "P") == [
        "Hello",
        "Quoted",
        "Spaced",
    ]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("1: line one\ncontinued\n2: two", ["line one\ncontinued", "two"]),
        # One-step format with the translation wrapped onto the next line.
        ("1: 原文 ||\nHello there\n2: a || b", ["原文 ||\nHello there", "a || b"]),
        ("1: a\n2: b\n\nLet me know if you need changes.", ["a", "b"]),
        ("```\n1: a\n2: b\n```", ["a", "b"]),
        ("```text\n1: a\n2: b\n```", ["a", "b"]),
    ],
)
def test_parse_multiline_items_and_wrappers(response, expected):
    assert _parse_llm_response_unified(response, 2, "P") == expected


def test_parse_marks_missing_and_ignores_out_of_range():
    response = "1: one\n5: extra"
    assert _parse_llm_response_unified(response, 2, "P") == [
        "one",
        "[P: Missing item 2]",
    ]


@pytest.mark.parametrize("response", [None, ""])
def test_parse_raises_on_empty_response(response):
    with pytest.raises(TranslationError):
        _parse_llm_response_unified(response, 1, "P")


def test_parse_rosetta_json():
    assert _parse_rosetta_response('{"1": "a", "2": " b "}', 2, "P") == ["a", "b"]


def test_parse_rosetta_strips_code_fence_and_marks_missing():
    response = '```json\n{"1": "a"}\n```'
    assert _parse_rosetta_response(response, 2, "P") == ["a", "[P: Missing item 2]"]


def test_parse_rosetta_falls_back_to_numbered_list():
    assert _parse_rosetta_response("1: a\n2: b", 2, "P") == ["a", "b"]


@pytest.mark.parametrize("response", [None, ""])
def test_parse_rosetta_raises_on_empty_response(response):
    with pytest.raises(TranslationError):
        _parse_rosetta_response(response, 1, "P")


@pytest.mark.parametrize(
    ("texts", "provider", "expected"),
    [
        ([], None, True),
        (["[OCR FAILED]", "[OCR FAILED]"], None, True),
        (["[OCR FAILED]", "hello"], None, False),
        (["[Gemini-OCR: timeout]"], "Gemini", True),
        (["[Gemini-OCR: timeout]", "hello"], "Gemini", False),
    ],
)
def test_check_ocr_failure(texts, provider, expected):
    assert _check_ocr_failure(texts, provider) is expected


def test_format_previous_context_skips_empty_and_failed_entries():
    result = _format_previous_context_texts([["a", "[OCR FAILED]", "c"], [], ["  "]])
    assert "### Previous page, 3 back\n1: a\n3: c" in result
    assert "2 back" not in result
    assert "1 back" not in result


@pytest.mark.parametrize("texts", [None, [], [[], ["[OCR FAILED]"]]])
def test_format_previous_context_empty(texts):
    assert _format_previous_context_texts(texts) == ""


def test_hy_mt2_system_prompt_adds_optional_rules_only_when_enabled():
    base = _build_system_prompt_hy_mt2("English", input_language="Japanese")
    assert base.startswith("You translate Japanese comics into natural English.")
    assert "(e.g. **THUD**)" in base
    for marker in ("Previous pages", "automatic OCR", "adult comic"):
        assert marker not in base

    full = _build_system_prompt_hy_mt2(
        "French", ero_doujinshi_mode=True, ocr_correction=True, previous_context=True
    )
    for marker in ("Previous pages", "automatic OCR", "adult comic"):
        assert marker in full
    # English examples would leak into other target languages
    assert "(e.g. **THUD**)" not in full
    assert "write yours in French" in full


def test_pair_context_lines_skips_failures():
    ocr = ["こんにちは", "[OCR FAILED]", "ドン", "やあ"]
    translations = ["Hello", "[OCR FAILED]", "[P: Missing item 3]", ""]
    assert _pair_context_lines(ocr, translations) == [
        "こんにちは → Hello",
        "[OCR FAILED]",
        "ドン",
        "やあ",
    ]
    assert _pair_context_lines(ocr, ["Hello"]) == ocr


def test_endpoint_keeps_labels_next_to_their_images(monkeypatch):
    from utils.endpoints import openai_compatible

    sent = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json=None, **kwargs):
        sent.update(json)
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    image = {"inline_data": {"mime_type": "image/png", "data": "AAAA"}}
    openai_compatible.call_openai_compatible_endpoint(
        base_url="http://localhost:1/v1",
        api_key=None,
        model_name="local-model",
        parts=[{"text": "Crop 1:"}, image, {"text": "Crop 2:"}, image, {"text": "go"}],
        generation_config={},
    )
    content = sent["messages"][-1]["content"]
    assert [c.get("text", "<image>") for c in content] == [
        "Crop 1:",
        "<image>",
        "Crop 2:",
        "<image>",
        "go",
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("何やってんだ\n俺は！！？", "何やってんだ俺は！！？"),
        ("Hello\nworld", "Hello world"),
        ("안녕\n하세요", "안녕 하세요"),
    ],
)
def test_join_ocr_lines_adds_spaces_only_outside_cjk(text, expected):
    assert _join_ocr_lines(text) == expected


@pytest.mark.parametrize("mode", ["one-step", "two-step"])
def test_system_prompt_adds_glossary_rules_only_when_enabled(mode):
    off = _build_system_prompt_translation("English", mode)
    on = _build_system_prompt_translation("English", mode, terminology_memory=True)
    assert "## GLOSSARY" not in off and "NEW TERMS:" not in off
    assert "## GLOSSARY" in on
    assert "besides the optional `NEW TERMS:` block" in on
    assert on.index("## GLOSSARY") < on.index("## OUTPUT FORMAT")
    assert "Chunhua" in on and "Imperial Household Department" in on


def test_system_prompt_glossary_rules_drop_english_examples_for_other_languages():
    prompt = _build_system_prompt_translation(
        "French", "one-step", terminology_memory=True
    )
    assert "official French name" in prompt
    assert "Chunhua" not in prompt and "Imperial Household" not in prompt


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("Google", "gemini-3.5-flash-lite", True),
        ("Google", "gemini-4-argon", True),
        ("Google", "gemini-2.5-pro", False),
        ("Google", "gemma-4-31b-it", False),
        ("OpenRouter", "google/gemini-3.8-flash", False),
    ],
)
def test_code_execution_only_reaches_gemini_on_google(provider, model, expected):
    config = SimpleNamespace(
        provider=provider, model_name=model, enable_code_execution=True
    )
    assert _uses_code_execution(config) is expected


def test_code_execution_section_only_where_crops_are_read():
    assert "## CODE EXECUTION" in _build_system_prompt_ocr(None, code_execution=True)
    assert "## CODE EXECUTION" not in _build_system_prompt_ocr(None)
    one_step = _build_system_prompt_translation(
        "English", "one-step", code_execution=True
    )
    assert one_step.index("## TRANSCRIPTION RULES") < one_step.index(
        "## CODE EXECUTION"
    )
    two_step = _build_system_prompt_translation(
        "English", "two-step", code_execution=True
    )
    assert "## CODE EXECUTION" not in two_step


def test_hy_mt2_prompt_mentions_glossary_only_when_given():
    assert "glossary" not in _build_system_prompt_hy_mt2("English").lower()
    assert "glossary" in _build_system_prompt_hy_mt2("English", glossary=True).lower()


def test_rosetta_instruction_merges_glossary_terms():
    instruction = _build_rosetta_instruction(
        "English", "王 -> King", glossary_terms={"猫猫": "Maomao"}
    )
    assert "Glossary:\n- 王 -> King\n- 猫猫 -> Maomao" in instruction
