import re
import unicodedata
from collections.abc import Callable
from functools import cache
from itertools import pairwise

import numpy as np
import pyphen

# Markdown-like style pattern: ***bold italic***, **bold**, *italic*
STYLE_PATTERN = re.compile(r"(\*{1,3})(.*?)(\1)")
NO_SPACE_BEFORE_MARKER = "\uf000"
# Hangul syllables that must not start a later emergency-wrap unit.
# Expanded particle/connective set: when a prior unit exists, these glue onto it.
KOREAN_NO_LINE_START_SYLLABLES = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "도",
    "만",
    "로",
    "와",
    "과",
    "랑",
    "께",
    "란",
    "게",
    "서",
    "럼",
    "면",
    "요",
    "뿐",
    "씩",
    "님",
    "죠",
    "며",
    "겠",
    "잖",
}

# Thai script block (Unicode U+0E00–U+0E7F)
THAI_CODEPOINT_MIN = 0x0E00
THAI_CODEPOINT_MAX = 0x0E7F
# PyThaiNLP engines used for wrapping / orphan detection
THAI_WORD_TOKENIZE_ENGINE = "newmm"
THAI_TCC_ENGINE = "tcc_p"
# pyphen dictionary for splitting Thai words wider than the bubble
THAI_HYPHENATION_DICTIONARY = "th_TH"
# Thai tokens at or below this many TCC clusters are short line-start orphans.
# 3 covers ปกติ (3 clusters); 2 would not.
THAI_SHORT_LINE_START_MAX_CLUSTERS = 3
# Default DP cost for a short Thai line-start (scaled by cluster count).
# Tuned to outweigh equal-line packs that orphan compounds like เรื่อง|ปกติ.
DEFAULT_THAI_SHORT_LINE_START_PENALTY = 5000.0


def is_rtl_script(text: str) -> bool:
    """Check if text contains dominant RTL script characters (Arabic, Hebrew, etc.)."""
    rtl_count = 0
    ltr_count = 0
    for ch in text:
        cp = ord(ch)
        if ch.isspace() or ch in ("*",):
            continue
        # Arabic (0600–06FF), Arabic Supplement (0750–077F), Arabic Extended-A (08A0–08FF),
        # Arabic Presentation Forms A/B (FB50–FDFF, FE70–FEFF)
        if (
            (
                0x0600 <= cp <= 0x06FF
                or 0x0750 <= cp <= 0x077F
                or 0x08A0 <= cp <= 0x08FF
                or 0xFB50 <= cp <= 0xFDFF
                or 0xFE70 <= cp <= 0xFEFF
            )
            or 0x0590 <= cp <= 0x05FF
            or 0xFB1D <= cp <= 0xFB4F
            or 0x0780 <= cp <= 0x07BF
            or 0x07C0 <= cp <= 0x07FA
        ):
            rtl_count += 1
        else:
            ltr_count += 1
    return rtl_count > ltr_count


# Target language -> pyphen dictionary. Malay and Bosnian share the Indonesian
# and Croatian standards closely enough to borrow their dictionaries.
HYPHENATION_DICTIONARIES = {
    "afrikaans": "af_ZA",
    "albanian": "sq_AL",
    "bosnian": "hr_HR",
    "bulgarian": "bg_BG",
    "catalan": "ca",
    "croatian": "hr_HR",
    "czech": "cs_CZ",
    "danish": "da_DK",
    "dutch": "nl_NL",
    "english": "en_US",
    "estonian": "et_EE",
    "french": "fr",
    "galician": "gl",
    "german": "de_DE",
    "greek": "el_GR",
    "hungarian": "hu_HU",
    "icelandic": "is",
    "indonesian": "id_ID",
    "italian": "it_IT",
    "latvian": "lv_LV",
    "lithuanian": "lt",
    "malay": "id_ID",
    "norwegian": "nb_NO",
    "polish": "pl_PL",
    "portuguese": "pt_BR",
    "romanian": "ro_RO",
    "russian": "ru_RU",
    "serbian (cyrillic)": "sr",
    "serbian (latin)": "sr_Latn",
    "slovak": "sk_SK",
    "slovenian": "sl_SI",
    "spanish": "es",
    "swedish": "sv",
    "ukrainian": "uk_UA",
}

# Languages that hyphenate long words. Those without a dictionary use
# SYLLABLE_BREAK_RULES or the generic midpoint split.
HYPHENATING_LANGUAGES = frozenset(HYPHENATION_DICTIONARIES) | {
    "filipino (tagalog)",
    "finnish",
    "swahili",
    "tagalog",
    "turkish",
    "uzbek",
    "welsh",
}

# Breaking at an existing hyphen repeats it at the start of the next line
# (biało-/-czerwony, léxico-/-semántico).
REPEAT_HYPHEN_LANGUAGES = frozenset(
    {
        "bosnian",
        "croatian",
        "czech",
        "galician",
        "polish",
        "portuguese",
        "serbian (cyrillic)",
        "serbian (latin)",
        "slovak",
        "slovenian",
        "spanish",
    }
)


def _normalize_language(language_name: str) -> str:
    return (language_name or "").strip().lower()


def supports_long_word_breaking(language_name: str) -> bool:
    """
    Whether long-word emergency breaking is applied for this target language.

    Hyphenating languages may insert hyphens; Korean and Thai use no-hyphen
    emergency splits under the same user setting.
    """
    return uses_true_hyphenation(language_name) or _normalize_language(
        language_name
    ) in ("korean", "thai")


def uses_true_hyphenation(language_name: str) -> bool:
    """Whether wrapping may insert actual hyphen characters for this language."""
    return _normalize_language(language_name) in HYPHENATING_LANGUAGES


def text_layout_control_interactivity(
    output_language: str,
    batch_output_language: str,
    hyphenate_before_scaling: bool,
) -> dict[str, bool]:
    """
    Interactive flags for Text Layout hyphenation controls.

    A control is enabled if it is relevant for either the single-tab or batch
    target language (shared Config panel).
    """
    languages = [output_language or "", batch_output_language or ""]
    word_break = any(supports_long_word_breaking(lang) for lang in languages)

    return {
        "hyphenate_before_scaling": word_break,
        "hyphenation_min_word_length": word_break and bool(hyphenate_before_scaling),
    }


def _is_hangul_character(char: str) -> bool:
    """Check if a character is Korean Hangul (syllables or jamo)."""
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        (0xAC00 <= code <= 0xD7AF)  # Hangul Syllables
        or (0x1100 <= code <= 0x11FF)  # Hangul Jamo
        or (0x3130 <= code <= 0x318F)  # Hangul Compatibility Jamo
    )


def _is_thai_character(char: str) -> bool:
    """Check if a character is in the Thai Unicode block (U+0E00–U+0E7F)."""
    if len(char) != 1:
        return False
    return THAI_CODEPOINT_MIN <= ord(char) <= THAI_CODEPOINT_MAX


def _contains_thai(text: str) -> bool:
    return any(_is_thai_character(ch) for ch in text)


def _thai_word_tokenize(text: str) -> list[str]:
    """Segment Thai text into words via PyThaiNLP (lazy import)."""
    from pythainlp.tokenize import word_tokenize

    return [w for w in word_tokenize(text, engine=THAI_WORD_TOKENIZE_ENGINE) if w]


def strip_no_space_before_marker(token: str) -> str:
    if token.startswith(NO_SPACE_BEFORE_MARKER):
        return token[len(NO_SPACE_BEFORE_MARKER) :]
    return token


def split_hangul_word_for_wrapping(token: str) -> list[str] | None:
    """
    Split a Hangul-containing word into breakable units without adding hyphens.

    The first unit keeps normal spacing. Later units are marked so the wrapper
    can break before them without inserting spaces into the rendered line.
    """
    normalized = unicodedata.normalize("NFC", token)
    match = re.match(r"^(\W*)([\w\-]+)(\W*)$", normalized)
    if match:
        leading_punc, core_word, trailing_punc = match.groups()
    else:
        leading_punc, core_word, trailing_punc = "", normalized, ""

    if not any(_is_hangul_character(ch) for ch in core_word):
        return None

    units: list[str] = []
    current_non_hangul = ""
    for ch in core_word:
        if _is_hangul_character(ch):
            if current_non_hangul:
                units.append(current_non_hangul)
                current_non_hangul = ""
            if units and ch in KOREAN_NO_LINE_START_SYLLABLES:
                units[-1] += ch
            else:
                units.append(ch)
        elif unicodedata.combining(ch) and units:
            units[-1] += ch
        else:
            current_non_hangul += ch

    if current_non_hangul:
        units.append(current_non_hangul)

    if len(units) < 2:
        return None

    units[0] = leading_punc + units[0]
    units[-1] = units[-1] + trailing_punc
    return [units[0]] + [f"{NO_SPACE_BEFORE_MARKER}{unit}" for unit in units[1:]]


def split_thai_word_for_wrapping(token: str) -> list[str] | None:
    """
    Split a Thai-containing word into syllables without adding hyphens.

    Used as an emergency break when a single Thai word is wider than the
    bubble. Later units are marked so the wrapper can break before them
    without inserting spaces. Words the dictionary can't split stay whole.
    """
    normalized = unicodedata.normalize("NFC", token)
    match = re.match(r"^(\W*)(.+?)(\W*)$", normalized, flags=re.UNICODE)
    if match:
        leading_punc, core_word, trailing_punc = match.groups()
    else:
        leading_punc, core_word, trailing_punc = "", normalized, ""

    if not _contains_thai(core_word):
        return None

    breaks = [0, *_pyphen_dictionary(THAI_HYPHENATION_DICTIONARY).positions(core_word)]
    if len(breaks) < 2:
        return None
    units = [core_word[start:end] for start, end in pairwise([*breaks, len(core_word)])]

    units[0] = leading_punc + units[0]
    units[-1] = units[-1] + trailing_punc
    return [units[0]] + [f"{NO_SPACE_BEFORE_MARKER}{unit}" for unit in units[1:]]


def is_cjk_character(char: str) -> bool:
    """Check if a character is CJK (Chinese/Japanese/Korean)."""
    if len(char) != 1:
        return False
    code = ord(char)
    return (
        (0x4E00 <= code <= 0x9FFF)  # CJK Unified Ideographs
        or (0x3400 <= code <= 0x4DBF)  # CJK Extension A
        or (0x20000 <= code <= 0x2CEAF)  # CJK Extension B-F
        or (0xF900 <= code <= 0xFAFF)  # CJK Compatibility
        or (0x3040 <= code <= 0x309F)  # Hiragana
        or (0x30A0 <= code <= 0x30FF)  # Katakana
        or (0x31F0 <= code <= 0x31FF)  # Katakana Extensions
        or (0xAC00 <= code <= 0xD7AF)  # Hangul Syllables
        or (0x1100 <= code <= 0x11FF)  # Hangul Jamo
        or (0x3130 <= code <= 0x318F)  # Hangul Compatibility Jamo
        or (0x3000 <= code <= 0x303F)  # CJK Symbols/Punctuation
        or (0xFF00 <= code <= 0xFFEF)  # Fullwidth Forms
    )


def parse_styled_segments(text: str) -> list[tuple[str, str]]:
    """
    Parses text with markdown-like style markers into segments.

    Args:
        text (str): Input text potentially containing ***bold italic***, **bold**, *italic*.

    Returns:
        List[Tuple[str, str]]: List of (segment_text, style_name) tuples.
                               style_name is one of "regular", "italic", "bold", "bold_italic".
    """
    segments = []
    last_end = 0
    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        marker = match.group(1)
        content = match.group(2)

        if start > last_end:
            segments.append((text[last_end:start], "regular"))

        style = "regular"
        if len(marker) == 3:
            style = "bold_italic"
        elif len(marker) == 2:
            style = "bold"
        elif len(marker) == 1:
            style = "italic"

        segments.append((content, style))
        last_end = end

    if last_end < len(text):
        segments.append((text[last_end:], "regular"))

    return [(txt, style) for txt, style in segments if txt]


# Kinsoku Shori (禁則処理) - CJK line-breaking rules
KINSOKU_NOT_AT_START = set(  # Cannot start a line
    "、。，．！？）】」』〕〉》，．！？）］｝,.)!?;:…‥ー"
    "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ"
)
KINSOKU_NOT_AT_END = set("（【「『〔〈《（［｛([")  # Cannot end a line

TRAILING_PUNCT_CLOSERS = r"\)\]\}\u2019\u201D'\""
DETACHABLE_TRAILING_PUNCT_CORE = r"[.!?]{2,}"
DETACHABLE_TRAILING_PUNCT_RE = re.compile(
    rf"^(.*?)({DETACHABLE_TRAILING_PUNCT_CORE}[{TRAILING_PUNCT_CLOSERS}]*)$"
)
DETACHED_TRAILING_PUNCT_RE = re.compile(
    rf"^{DETACHABLE_TRAILING_PUNCT_CORE}[{TRAILING_PUNCT_CLOSERS}]*$"
)


def is_detached_trailing_punctuation(token: str) -> bool:
    return bool(DETACHED_TRAILING_PUNCT_RE.match(token))


def _is_detached_ellipsis(token: str) -> bool:
    return is_detached_trailing_punctuation(token) and token.startswith("..")


def _append_breakable_token(token: str, tokens: list[str]) -> None:
    """Append a word-level token, segmenting Thai runs with PyThaiNLP."""
    if not token:
        return
    if _contains_thai(token):
        tokens.extend(_thai_word_tokenize(token))
    else:
        tokens.append(token)


def _split_with_cjk_awareness(
    text: str, detach_trailing_punctuation: bool = True
) -> list[str]:
    """Split text into tokens. Each CJK char is a token; kinsoku rules apply.

    Hangul (Korean) is excluded from per-character splitting because Korean
    uses spaces between words — syllables accumulate into word-level tokens
    like Latin characters, preserving inter-word spacing.

    Thai runs are dictionary-segmented via PyThaiNLP so line breaks fall on
    word boundaries without inserting spaces between Thai words.
    """
    tokens: list[str] = []
    current_token = ""

    for char in text:
        if char.isspace():
            if current_token:
                _append_breakable_token(current_token, tokens)
                current_token = ""
        elif is_cjk_character(char) and not _is_hangul_character(char):
            if char in KINSOKU_NOT_AT_START:
                if current_token:
                    current_token += char
                elif tokens:
                    tokens[-1] += char
                else:
                    current_token = char
            elif char in KINSOKU_NOT_AT_END:
                if current_token:
                    _append_breakable_token(current_token, tokens)
                current_token = char
            else:
                if current_token:
                    if current_token[-1] in KINSOKU_NOT_AT_END:
                        current_token += char
                        tokens.append(current_token)
                        current_token = ""
                    else:
                        _append_breakable_token(current_token, tokens)
                        current_token = ""
                        tokens.append(char)
                else:
                    tokens.append(char)
        else:
            current_token += char

    if current_token:
        _append_breakable_token(current_token, tokens)

    if detach_trailing_punctuation:
        final_tokens = []
        for t in tokens:
            m = DETACHABLE_TRAILING_PUNCT_RE.match(t)
            if m and m.group(1):
                final_tokens.append(m.group(1))
                final_tokens.append(m.group(2))
            else:
                final_tokens.append(t)
        return final_tokens

    return tokens


def tokenize_styled_text(
    text: str, detach_trailing_punctuation: bool = True
) -> list[tuple[str, bool]]:
    """
    Tokenizes text into atomic units for wrapping where styled blocks are
    preserved as single, unbreakable tokens.

    Returns: List[Tuple[str, bool]] where each tuple is (token_text, is_styled).
    - Styled tokens are split into per-word tokens (CJK-aware), each wrapped
      with the same markers, to allow wrapping at word/character boundaries
      while preserving style.
    - Plain text outside markers is split with CJK awareness into word/character tokens.
    """
    tokens: list[tuple[str, bool]] = []
    last_end = 0
    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            preceding = text[last_end:start]
            for w in _split_with_cjk_awareness(preceding, detach_trailing_punctuation):
                tokens.append((w, False))

        marker = match.group(1)
        content = match.group(2)
        if content:
            for w in _split_with_cjk_awareness(content, detach_trailing_punctuation):
                tokens.append((f"{marker}{w}{marker}", True))

        last_end = end

    if last_end < len(text):
        trailing = text[last_end:]
        for w in _split_with_cjk_awareness(trailing, detach_trailing_punctuation):
            tokens.append((w, False))

    return tokens


TURKISH_VOWELS = frozenset("aeıioöuüâîûAEIİOÖUÜÂÎÛ")


def _turkish_syllable_breaks(word: str) -> list[int]:
    """
    Indices where a Turkish word may break between syllables.

    Every syllable holds exactly one vowel, and the last consonant before a
    vowel starts that vowel's syllable (kur-tar-ma-ya, Türk-çe, sa-at).
    """
    vowel_positions = [i for i, ch in enumerate(word) if ch in TURKISH_VOWELS]
    breaks: list[int] = []
    for prev, nxt in pairwise(vowel_positions):
        if all(ch.isalpha() for ch in word[prev : nxt + 1]):
            breaks.append(nxt if nxt - prev == 1 else nxt - 1)
    return breaks


SYLLABLE_BREAK_RULES: dict[str, Callable[[str], list[int]]] = {
    "turkish": _turkish_syllable_breaks,
}

# A word's core may hold hyphens and internal apostrophes or Catalan middle
# dots (aujourd'hui, İstanbul'dan, col·lecció); outer punctuation splits off.
WORD_CORE_PATTERN = re.compile(r"^(\W*)([\w\-]+(?:['’‘·][\w\-]+)*)(\W*)$")
_BREAK_SEPARATORS = "-'’‘·"


@cache
def _pyphen_dictionary(code: str) -> pyphen.Pyphen:
    return pyphen.Pyphen(lang=code)


def _hyphenation_dictionary(language: str) -> pyphen.Pyphen | None:
    code = HYPHENATION_DICTIONARIES.get(language)
    return _pyphen_dictionary(code) if code else None


def _drop_leading_trema(text: str) -> str:
    """Dutch drops the trema that starts the line after a break (idee-en)."""
    if not text:
        return text
    first = unicodedata.normalize("NFD", text[0]).replace("\u0308", "")
    return unicodedata.normalize("NFC", first) + text[1:]


def _break_pairs(core_word: str, language: str) -> list[tuple[str, str]]:
    """
    Candidate (left, right) splits of a word, most balanced first.

    Dictionary languages only break where the dictionary allows, since its
    patterns leave some words unbroken on purpose (stranger). Syllable-rule
    words that yield nothing (e.g. "HMMMMMMMM") and other languages fall back
    to every position near the middle. Breaks touching an existing hyphen or
    apostrophe are skipped; existing hyphens are tried separately.
    """
    dictionary = _hyphenation_dictionary(language)
    syllable_rule = SYLLABLE_BREAK_RULES.get(language)
    pairs: list[tuple[str, str]] = []
    if dictionary is not None:
        # iterate() applies spelling changes such as Hungarian asz-szony.
        pairs = list(dictionary.iterate(core_word))
    elif syllable_rule is not None:
        pairs = [(core_word[:i], core_word[i:]) for i in syllable_rule(core_word)]
    pairs = [
        (left, right)
        for left, right in pairs
        if len(left) >= 2
        and len(right) >= 2
        and left[-1] not in _BREAK_SEPARATORS
        and right[0] not in _BREAK_SEPARATORS
    ]

    if not pairs and dictionary is None:
        pairs = [
            (core_word[:i], core_word[i:])
            for i in range(2, len(core_word) - 2)
            if core_word[i - 1] not in _BREAK_SEPARATORS
            and core_word[i] not in _BREAK_SEPARATORS
        ]

    # Ties favor the left split.
    mid = len(core_word) // 2
    return sorted(pairs, key=lambda p: (abs(len(p[0]) - mid), len(p[0])))


def _repeats_hyphen(language: str, core_word: str, right_part: str) -> bool:
    if language not in REPEAT_HYPHEN_LANGUAGES:
        return False
    # The RAE skips the repeat before a capitalized proper name (Calvo-/Sotelo).
    return not (
        language == "spanish" and right_part[:1].isupper() and not core_word.isupper()
    )


def try_hyphenate_word(
    word_str: str,
    min_word_length: int,
    width_test_func: Callable[[str], bool],
    language: str = "",
) -> list[str] | None:
    """
    Attempts to split a word into two parts with a hyphen such that each part passes the width test.

    This is a generic hyphenation function that doesn't know about fonts or rendering.
    It uses a callback function to test if each part fits.

    Args:
        word_str: The word to potentially hyphenate
        min_word_length: Minimum word length to attempt hyphenation
        width_test_func: Function that takes a string and returns True if it fits
        language: Target language name; languages with a hyphenation
            dictionary or syllable rule only break between syllables

    Returns:
        List of two strings (the split parts) if successful, None otherwise
    """
    match = WORD_CORE_PATTERN.match(word_str)
    if not match:
        return None

    leading_punc, core_word, trailing_punc = match.groups()

    if len(core_word) < min_word_length:
        return None

    lang = _normalize_language(language)

    def _fitting_split(left_part: str, right_part: str) -> list[str] | None:
        final_left_part = leading_punc + left_part
        final_right_part = right_part + trailing_punc
        if width_test_func(final_left_part) and width_test_func(final_right_part):
            return [final_left_part, final_right_part]
        return None

    # Try splitting at existing hyphens first
    mid = len(core_word) // 2
    hyphen_positions = sorted(
        (
            i
            for i, ch in enumerate(core_word)
            if ch == "-" and 0 < i < len(core_word) - 1
        ),
        key=lambda i: abs(i - mid),
    )
    for pos in hyphen_positions:
        right_part = core_word[pos + 1 :]
        if _repeats_hyphen(lang, core_word, right_part):
            right_part = "-" + right_part
        split = _fitting_split(core_word[: pos + 1], right_part)
        if split:
            return split

    for left_part, right_part in _break_pairs(core_word, lang):
        if lang == "dutch":
            right_part = _drop_leading_trema(right_part)
        split = _fitting_split(left_part + "-", right_part)
        if split:
            return split

    return None


def _is_cjk_token(token: str) -> bool:
    """Check if token consists entirely of spaceless CJK (Chinese/Japanese, not Hangul)."""
    token = strip_no_space_before_marker(token)
    match = STYLE_PATTERN.match(token)
    content = match.group(2) if match else token
    return len(content) > 0 and all(
        is_cjk_character(c) and not _is_hangul_character(c) for c in content
    )


def _is_thai_token(token: str) -> bool:
    """Check if token is Thai-script content (may include punctuation)."""
    token = strip_no_space_before_marker(token)
    match = STYLE_PATTERN.match(token)
    content = match.group(2) if match else token
    if not content:
        return False
    return _contains_thai(content) and not any(
        ch.isascii() and ch.isalpha() for ch in content
    )


def _token_plain_content(token: str) -> str:
    token = strip_no_space_before_marker(token)
    match = STYLE_PATTERN.match(token)
    return match.group(2) if match else token


def _thai_tcc_cluster_count(text: str) -> int:
    """Count Thai Character Clusters via PyThaiNLP tcc_p."""
    if not text:
        return 0
    from pythainlp.tokenize import subword_tokenize

    units = subword_tokenize(text, engine=THAI_TCC_ENGINE)
    return len([u for u in units if u])


def _thai_short_line_start_cost(
    token: str,
    penalty: float,
    max_clusters: int = THAI_SHORT_LINE_START_MAX_CLUSTERS,
    cluster_count_cache: dict | None = None,
) -> float:
    """Extra DP cost when a continuation line would start with a short Thai token.

    Length is TCC cluster count (tcc_p), not Unicode code points.

    Cost scales with cluster count inside the short band (penalty * n). That
    charges 1-cluster particles and, importantly, charges medium short openers
    such as ปกติ (3) more than shorter heads such as เรื่อง (2). Pure
    "shorter is worse" would prefer starting a line with ปกติ over เรื่อง and
    re-introduce awkward compound splits after dictionary segmentation.
    """
    if penalty <= 0 or max_clusters <= 0:
        return 0.0
    if not _is_thai_token(token):
        return 0.0
    content = _token_plain_content(token)
    if not content:
        return 0.0

    if cluster_count_cache is not None and content in cluster_count_cache:
        n = cluster_count_cache[content]
    else:
        n = _thai_tcc_cluster_count(content)
        if cluster_count_cache is not None:
            cluster_count_cache[content] = n

    if n == 0 or n > max_clusters:
        return 0.0
    return penalty * float(n)


def _needs_space_between(
    left_token: str, right_token: str, detach_trailing_punctuation: bool = True
) -> bool:
    """No space needed between adjacent CJK/Thai tokens or before separated punctuation."""
    if right_token.startswith(NO_SPACE_BEFORE_MARKER):
        return False

    left_token = strip_no_space_before_marker(left_token)
    right_token = strip_no_space_before_marker(right_token)

    if _is_cjk_token(left_token) and _is_cjk_token(right_token):
        return False

    if _is_thai_token(left_token) and _is_thai_token(right_token):
        return False

    if detach_trailing_punctuation:
        match = STYLE_PATTERN.match(right_token)
        r_content = match.group(2) if match else right_token
        if is_detached_trailing_punctuation(r_content):
            return False

    return True


def _join_tokens_smart(
    tokens: list[str], detach_trailing_punctuation: bool = True
) -> str:
    """Join tokens with smart spacing (no space between adjacent CJK tokens)."""
    if not tokens:
        return ""
    result = strip_no_space_before_marker(tokens[0])
    for i in range(1, len(tokens)):
        starts_with_detached_punctuation = False
        if detach_trailing_punctuation and i == 1:
            first_token = strip_no_space_before_marker(tokens[0])
            match = STYLE_PATTERN.match(first_token)
            content = match.group(2) if match else first_token
            starts_with_detached_punctuation = _is_detached_ellipsis(content)

        clean_token = strip_no_space_before_marker(tokens[i])
        if starts_with_detached_punctuation:
            result += clean_token
        elif _needs_space_between(
            tokens[i - 1], tokens[i], detach_trailing_punctuation
        ):
            result += " " + clean_token
        else:
            result += clean_token
    return result


def find_optimal_breaks_dp(
    tokens: list[str],
    max_width: float,
    word_width_func: Callable[[str], float],
    space_width: float,
    badness_exponent: float = 3.0,
    detach_trailing_punctuation: bool = True,
    thai_short_line_start_penalty: float = DEFAULT_THAI_SHORT_LINE_START_PENALTY,
    thai_short_line_start_max_clusters: int = THAI_SHORT_LINE_START_MAX_CLUSTERS,
) -> list[str] | None:
    """
    Pragmatic Knuth-Plass style DP to find globally optimal line breaks.

    This is a pure algorithm that doesn't know about fonts or rendering.
    It uses callback functions to get widths.

    Args:
        tokens: List of word tokens
        max_width: Maximum allowed line width
        word_width_func: Function that takes a word and returns its width
        space_width: Width of a space character
        badness_exponent: Exponent for badness calculation (higher = prefer tighter lines)
        thai_short_line_start_penalty: Extra cost when a continuation line would
            start with a short Thai token (orphan avoidance). 0 disables.
        thai_short_line_start_max_clusters: Thai tokens at or below this many
            TCC clusters (tcc_p) count as short line-start orphans.

    Returns:
        List of lines (strings) if successful, None if impossible to fit
    """
    try:
        if not tokens:
            return []

        # Calculate widths for all tokens
        token_w: list[float] = [word_width_func(t) for t in tokens]
        thai_cluster_cache: dict = {}

        N = len(tokens)
        min_cost: list[float] = [float("inf")] * (N + 1)
        path: list[int] = [0] * (N + 1)
        min_cost[0] = 0.0

        for i in range(1, N + 1):
            line_width = 0.0
            for j in range(i - 1, -1, -1):
                # Add space only if needed between this token and the previous one on the line
                if j < i - 1 and _needs_space_between(
                    tokens[j], tokens[j + 1], detach_trailing_punctuation
                ):
                    line_width += space_width
                line_width += token_w[j]

                if line_width > max_width:
                    break

                slack = max_width - line_width
                badness = pow(slack, badness_exponent)

                # Avoid orphaning short Thai tokens at the start of a new line.
                if j > 0:
                    badness += _thai_short_line_start_cost(
                        tokens[j],
                        thai_short_line_start_penalty,
                        thai_short_line_start_max_clusters,
                        thai_cluster_cache,
                    )

                total_cost = min_cost[j] + badness
                if total_cost < min_cost[i]:
                    min_cost[i] = total_cost
                    path[i] = j

        if not np.isfinite(min_cost[N]):
            return None

        lines: list[str] = []
        current_break = N
        while current_break > 0:
            prev_break = path[current_break]
            line = _join_tokens_smart(
                tokens[prev_break:current_break], detach_trailing_punctuation
            )
            lines.insert(0, line)
            current_break = prev_break

        return lines

    except Exception:
        return None
