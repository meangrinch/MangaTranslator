import pytest

from core.text.text_processing import (
    NO_SPACE_BEFORE_MARKER,
    find_optimal_breaks_dp,
    is_rtl_script,
    parse_styled_segments,
    split_hangul_word_for_wrapping,
    split_thai_word_for_wrapping,
    supports_long_word_breaking,
    text_layout_control_interactivity,
    tokenize_styled_text,
    try_hyphenate_word,
    uses_true_hyphenation,
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


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("KURTARMAYA...!?", ["KURTAR-", "MAYA...!?"]),
        ("DÜŞÜNDÜĞÜ", ["DÜŞÜN-", "DÜĞÜ"]),
        ("AĞABEYİM", ["AĞA-", "BEYİM"]),
        ("Türkçesi", ["Türk-", "çesi"]),
        ("şiiri", ["şi-", "iri"]),
    ],
)
def test_try_hyphenate_word_turkish_breaks_at_syllables(word, expected):
    assert try_hyphenate_word(word, 4, lambda s: True, "Turkish") == expected


def test_try_hyphenate_word_turkish_never_breaks_inside_syllable():
    # "DÜŞÜ-NDÜĞÜ" would fit, but only syllable breaks are allowed.
    assert try_hyphenate_word("DÜŞÜNDÜĞÜ", 4, lambda s: len(s) <= 5, "Turkish") is None


def test_try_hyphenate_word_turkish_falls_back_without_syllables():
    assert try_hyphenate_word("HMMMMMMMM", 4, lambda s: True, "Turkish") is not None


def test_try_hyphenate_word_turkish_keeps_apostrophe_suffix_together():
    assert try_hyphenate_word("İstanbul'dan", 4, lambda s: True, "Turkish") == [
        "İstan-",
        "bul'dan",
    ]


@pytest.mark.parametrize(
    ("language", "word", "expected"),
    [
        ("English", "UNDERSTANDING", ["UNDER-", "STANDING"]),
        ("German", "Entschuldigung", ["Entschul-", "digung"]),
        ("Spanish", "muchachos", ["mucha-", "chos"]),
        ("Portuguese", "infelizmente", ["infeliz-", "mente"]),
        ("French", "aujourd'hui", ["aujour-", "d'hui"]),
        ("Malay", "menyenangkan", ["menye-", "nangkan"]),
        ("Russian", "ДОСТОПРИМЕЧАТЕЛЬНОСТИ", ["ДОСТОПРИМЕ-", "ЧАТЕЛЬНОСТИ"]),
        ("Serbian (Cyrillic)", "захваљујући", ["захва-", "љујући"]),
        ("Greek", "ΕΥΧΑΡΙΣΤΩ", ["ΕΥΧΑ-", "ΡΙΣΤΩ"]),
        # Breaks that change the spelling
        ("Hungarian", "ASSZONY", ["ASZ-", "SZONY"]),
        ("Catalan", "col·lecció", ["col-", "lecció"]),
        ("Dutch", "ideeën", ["idee-", "en"]),
    ],
)
def test_try_hyphenate_word_uses_dictionary_breaks(language, word, expected):
    assert try_hyphenate_word(word, 4, lambda s: True, language) == expected


def test_try_hyphenate_word_dictionary_never_breaks_inside_syllable():
    # "UNDERS-TANDING" would fit, but the dictionary has no break there.
    assert (
        try_hyphenate_word("UNDERSTANDING", 4, lambda s: len(s) <= 7, "English") is None
    )


def test_try_hyphenate_word_dictionary_has_no_midpoint_fallback():
    # The English dictionary leaves "stranger" unbroken; "STRA-NGER" is wrong.
    assert try_hyphenate_word("STRANGER", 4, lambda s: True, "English") is None


@pytest.mark.parametrize(
    ("language", "word", "expected"),
    [
        ("Polish", "biało-czerwony", ["biało-", "-czerwony"]),
        ("Spanish", "teórico-práctico", ["teórico-", "-práctico"]),
        ("Spanish", "TEÓRICO-PRÁCTICO", ["TEÓRICO-", "-PRÁCTICO"]),
        # RAE: no repeat before a capitalized proper name
        ("Spanish", "Calvo-Sotelo", ["Calvo-", "Sotelo"]),
        ("English", "well-known", ["well-", "known"]),
    ],
)
def test_try_hyphenate_word_repeats_existing_hyphen(language, word, expected):
    assert try_hyphenate_word(word, 4, lambda s: True, language) == expected


def test_try_hyphenate_word_never_splits_at_apostrophe():
    assert try_hyphenate_word("ng'ombe", 4, lambda s: True, "Swahili") == [
        "ng'o-",
        "mbe",
    ]


@pytest.mark.parametrize(
    ("language", "word_break", "true_hyphen"),
    [
        ("Russian", True, True),
        ("Ukrainian", True, True),
        ("Bulgarian", True, True),
        ("Serbian (Cyrillic)", True, True),
        ("Greek", True, True),
        ("Turkish", True, True),
        ("Vietnamese", False, False),
        ("Korean", True, False),
        ("Hindi", False, False),
    ],
)
def test_word_breaking_language_gates(language, word_break, true_hyphen):
    assert supports_long_word_breaking(language) is word_break
    assert uses_true_hyphenation(language) is true_hyphen


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


def test_split_thai_word_breaks_between_syllables():
    assert split_thai_word_for_wrapping("ความรับผิดชอบ!") == [
        "ความ",
        f"{NO_SPACE_BEFORE_MARKER}รับ",
        f"{NO_SPACE_BEFORE_MARKER}ผิด",
        f"{NO_SPACE_BEFORE_MARKER}ชอบ!",
    ]


def test_split_thai_word_keeps_single_syllable_whole():
    assert split_thai_word_for_wrapping("ครับ") is None


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
                "hyphenation_min_word_length": True,
            },
        ),
        (
            ("Korean", "Japanese"),
            {
                "hyphenate_before_scaling": True,
                "hyphenation_min_word_length": True,
            },
        ),
        (
            ("Japanese", "Chinese"),
            {
                "hyphenate_before_scaling": False,
                "hyphenation_min_word_length": False,
            },
        ),
    ],
)
def test_text_layout_control_interactivity(languages, expected):
    assert text_layout_control_interactivity(*languages, True) == expected
