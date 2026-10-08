import pytest

from core.services.translation import (
    _build_hy_mt2_prompt,
    _check_ocr_failure,
    _format_previous_context_texts,
    _parse_llm_response_unified,
    _parse_rosetta_response,
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
    assert "### Previous Page 1\n1: a\n3: c" in result
    assert "Previous Page 2" not in result
    assert "Previous Page 3" not in result


@pytest.mark.parametrize("texts", [None, [], [[], ["[OCR FAILED]"]]])
def test_format_previous_context_empty(texts):
    assert _format_previous_context_texts(texts) == ""


def test_hy_mt2_prompt_converts_glossary_and_embeds_json():
    prompt = _build_hy_mt2_prompt(
        "English",
        ["こんにちは"],
        "- ナルト -> Naruto\nサスケ => Sasuke\nKeep honorifics",
    )
    assert "ナルト translates to Naruto" in prompt
    assert "サスケ translates to Sasuke" in prompt
    assert "Keep honorifics" in prompt
    assert prompt.endswith('{"1": "こんにちは"}')
