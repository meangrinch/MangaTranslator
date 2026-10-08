import pytest

from core.text.text_processing import (
    NO_SPACE_BEFORE_MARKER,
    find_optimal_breaks_dp,
    is_rtl_script,
    parse_styled_segments,
    split_hangul_word_for_wrapping,
    text_layout_control_interactivity,
    tokenize_styled_text,
    try_hyphenate_word,
)


def test_parse_styled_segments_all_styles():
    text = "Hello *world* and **bold** ***both***"
    assert parse_styled_segments(text) == [
        ("Hello ", "regular"),
        ("world", "italic"),
        (" and ", "regular"),
        ("bold", "bold"),
        (" ", "regular"),
        ("both", "bold_italic"),
    ]


def test_tokenize_styled_text_wraps_each_styled_word():
    assert tokenize_styled_text("Hello **big red** world") == [
        ("Hello", False),
        ("**big**", True),
        ("**red**", True),
        ("world", False),
    ]


@pytest.mark.parametrize(
    ("text", "detach", "expected"),
    [
        ("Wait...", True, ["Wait", "..."]),
        ("What?!", True, ["What", "?!"]),
        ("Wait...", False, ["Wait..."]),
        ("Hi.", True, ["Hi."]),
    ],
)
def test_tokenize_detaches_trailing_punctuation(text, detach, expected):
    tokens = tokenize_styled_text(text, detach_trailing_punctuation=detach)
    assert [t for t, _ in tokens] == expected


def test_tokenize_japanese_splits_chars_and_applies_kinsoku():
    tokens = [t for t, _ in tokenize_styled_text("こんにちは。")]
    # "。" may not start a line, so it sticks to the previous character.
    assert tokens == ["こ", "ん", "に", "ち", "は。"]


def test_tokenize_japanese_keeps_brackets_with_content():
    tokens = [t for t, _ in tokenize_styled_text("「あ」い")]
    assert tokens == ["「あ」", "い"]


def test_tokenize_korean_keeps_word_level_tokens():
    tokens = [t for t, _ in tokenize_styled_text("안녕 세계")]
    assert tokens == ["안녕", "세계"]


def test_split_hangul_glues_particles_to_previous_syllable():
    assert split_hangul_word_for_wrapping("사람이") == [
        "사",
        f"{NO_SPACE_BEFORE_MARKER}람이",
    ]


def test_split_hangul_ignores_non_hangul():
    assert split_hangul_word_for_wrapping("hello") is None


def test_try_hyphenate_word_splits_long_word():
    word = "internationalization"
    parts = try_hyphenate_word(word, 6, lambda s: len(s) <= 12)
    assert parts is not None
    left, right = parts
    assert left.endswith("-")
    assert left[:-1] + right == word
    assert len(left) <= 12 and len(right) <= 12


def test_try_hyphenate_word_prefers_existing_hyphen():
    assert try_hyphenate_word("well-known", 4, lambda s: len(s) <= 6) == [
        "well-",
        "known",
    ]


def test_try_hyphenate_word_keeps_surrounding_punctuation():
    left, right = try_hyphenate_word('"internationalization!"', 6, lambda s: True)
    assert left.startswith('"') and left.endswith("-")
    assert right.endswith('!"')


def test_try_hyphenate_word_rejects_short_or_unfittable_words():
    assert try_hyphenate_word("short", 8, lambda s: True) is None
    assert try_hyphenate_word("internationalization", 6, lambda s: False) is None


def test_find_optimal_breaks_fits_every_line():
    tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    lines = find_optimal_breaks_dp(tokens, 10, len, 1)
    assert lines is not None
    assert all(len(line) <= 10 for line in lines)
    assert " ".join(lines) == " ".join(tokens)


def test_find_optimal_breaks_single_line_when_wide():
    assert find_optimal_breaks_dp(["a", "b", "c"], 100, len, 1) == ["a b c"]


def test_find_optimal_breaks_returns_none_when_token_too_wide():
    assert find_optimal_breaks_dp(["supercalifragilistic"], 5, len, 1) is None


def test_find_optimal_breaks_joins_cjk_and_detached_punctuation_without_spaces():
    assert find_optimal_breaks_dp(["こ", "ん", "に", "ち", "は。"], 100, len, 1) == [
        "こんにちは。"
    ]
    assert find_optimal_breaks_dp(["Wait", "..."], 100, len, 1) == ["Wait..."]


def test_thai_tokens_join_without_spaces():
    text = "สวัสดีครับ"
    tokens = [t for t, _ in tokenize_styled_text(text)]
    assert len(tokens) > 1
    assert find_optimal_breaks_dp(tokens, 1000, len, 1) == [text]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("مرحبا", True), ("שלום", True), ("hello", False), ("hi مرحبا", True)],
)
def test_is_rtl_script(text, expected):
    assert is_rtl_script(text) is expected


@pytest.mark.parametrize(
    ("languages", "expected"),
    [
        (
            ("Japanese", "English"),
            {
                "hyphenate_before_scaling": True,
                "hyphen_penalty": True,
                "hyphenation_min_word_length": True,
            },
        ),
        (
            ("Korean", "Japanese"),
            {
                "hyphenate_before_scaling": True,
                "hyphen_penalty": False,
                "hyphenation_min_word_length": True,
            },
        ),
        (
            ("Japanese", "Chinese"),
            {
                "hyphenate_before_scaling": False,
                "hyphen_penalty": False,
                "hyphenation_min_word_length": False,
            },
        ),
    ],
)
def test_text_layout_control_interactivity(languages, expected):
    assert text_layout_control_interactivity(*languages, True) == expected
