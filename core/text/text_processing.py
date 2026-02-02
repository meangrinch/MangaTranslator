import re
from typing import Callable, List, Optional, Tuple

import numpy as np

# Markdown-like style pattern: ***bold italic***, **bold**, *italic*
STYLE_PATTERN = re.compile(r"(\*{1,3})(.*?)(\1)")


def is_latin_style_language(language_name: str) -> bool:
    """
    Determines if a language typically uses Latin script and hyphenation.
    This is used to decide whether to apply automatic hyphenation logic.
    """
    latin_style_languages = {
        "english",
        "french",
        "spanish",
        "german",
        "italian",
        "portuguese",
        "dutch",
        "polish",
        "czech",
        "swedish",
        "danish",
        "norwegian",
        "finnish",
        "hungarian",
        "romanian",
        "turkish",
        "vietnamese",
        "indonesian",
        "malay",
        "tagalog",
    }
    return language_name.lower() in latin_style_languages


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


def parse_styled_segments(text: str) -> List[Tuple[str, str]]:
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


def _split_with_cjk_awareness(text: str) -> List[str]:
    """Split text into tokens. Each CJK char is a token; kinsoku rules apply."""
    tokens: List[str] = []
    current_token = ""

    for char in text:
        if char.isspace():
            if current_token:
                tokens.append(current_token)
                current_token = ""
        elif is_cjk_character(char):
            if char in KINSOKU_NOT_AT_START:
                if current_token:
                    current_token += char
                elif tokens:
                    tokens[-1] += char
                else:
                    current_token = char
            elif char in KINSOKU_NOT_AT_END:
                if current_token:
                    tokens.append(current_token)
                current_token = char
            else:
                if current_token:
                    if current_token[-1] in KINSOKU_NOT_AT_END:
                        current_token += char
                        tokens.append(current_token)
                        current_token = ""
                    else:
                        tokens.append(current_token)
                        current_token = ""
                        tokens.append(char)
                else:
                    tokens.append(char)
        else:
            current_token += char

    if current_token:
        tokens.append(current_token)

    return tokens


def tokenize_styled_text(text: str) -> List[Tuple[str, bool]]:
    """
    Tokenizes text into atomic units for wrapping where styled blocks are
    preserved as single, unbreakable tokens.

    Returns: List[Tuple[str, bool]] where each tuple is (token_text, is_styled).
    - Styled tokens are split into per-word tokens (CJK-aware), each wrapped
      with the same markers, to allow wrapping at word/character boundaries
      while preserving style.
    - Plain text outside markers is split with CJK awareness into word/character tokens.
    """
    tokens: List[Tuple[str, bool]] = []
    last_end = 0
    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            preceding = text[last_end:start]
            for w in _split_with_cjk_awareness(preceding):
                tokens.append((w, False))

        marker = match.group(1)
        content = match.group(2)
        if content:
            for w in _split_with_cjk_awareness(content):
                tokens.append((f"{marker}{w}{marker}", True))

        last_end = end

    if last_end < len(text):
        trailing = text[last_end:]
        for w in _split_with_cjk_awareness(trailing):
            tokens.append((w, False))

    return tokens


def try_hyphenate_word(
    word_str: str,
    min_word_length: int,
    width_test_func: Callable[[str], bool],
) -> Optional[List[str]]:
    """
    Attempts to split a word into two parts with a hyphen such that each part passes the width test.

    This is a generic hyphenation function that doesn't know about fonts or rendering.
    It uses a callback function to test if each part fits.

    Args:
        word_str: The word to potentially hyphenate
        min_word_length: Minimum word length to attempt hyphenation
        width_test_func: Function that takes a string and returns True if it fits

    Returns:
        List of two strings (the split parts) if successful, None otherwise
    """
    match = re.match(r"^(\W*)([\w\-]+)(\W*)$", word_str)
    if not match:
        return None

    leading_punc, core_word, trailing_punc = match.groups()

    if len(core_word) < min_word_length:
        return None

    def _split_with_single_hyphen(base: str, idx: int) -> Tuple[str, str]:
        ch_before = base[idx - 1] if idx > 0 else ""
        ch_at = base[idx] if idx < len(base) else ""
        if ch_at == "-":
            left = base[: idx + 1]
            right = base[idx + 1 :]
        elif ch_before == "-":
            left = base[:idx]
            right = base[idx:]
        else:
            left = base[:idx] + "-"
            right = base[idx:]
        if left.endswith("-") and right.startswith("-"):
            right = right[1:]
        return left, right

    # Try splitting at existing hyphens first
    if "-" in core_word:
        hyphen_positions = [i for i, ch in enumerate(core_word) if ch == "-"]
        mid = len(core_word) // 2
        hyphen_positions.sort(key=lambda i: abs(i - mid))
        for pos in hyphen_positions:
            if pos <= 0 or pos >= len(core_word) - 1:
                continue
            left_part = core_word[: pos + 1]
            right_part = core_word[pos + 1 :]

            final_left_part = leading_punc + left_part
            final_right_part = right_part + trailing_punc

            if width_test_func(final_left_part) and width_test_func(final_right_part):
                return [final_left_part, final_right_part]

    # Try splitting at various positions
    mid = len(core_word) // 2
    candidate_indices: List[int] = []
    max_d = max(mid, len(core_word) - mid)
    for d in range(0, max_d):
        left_idx = mid - d
        right_idx = mid + d
        if 2 <= left_idx < len(core_word) - 2:
            candidate_indices.append(left_idx)
        if 2 <= right_idx < len(core_word) - 2 and right_idx != left_idx:
            candidate_indices.append(right_idx)

    for idx in candidate_indices:
        left_part, right_part = _split_with_single_hyphen(core_word, idx)

        final_left_part = leading_punc + left_part
        final_right_part = right_part + trailing_punc

        if width_test_func(final_left_part) and width_test_func(final_right_part):
            return [final_left_part, final_right_part]

    return None


def _is_cjk_token(token: str) -> bool:
    """Check if token consists entirely of CJK characters."""
    match = STYLE_PATTERN.match(token)
    content = match.group(2) if match else token
    return len(content) > 0 and all(is_cjk_character(c) for c in content)


def _needs_space_between(left_token: str, right_token: str) -> bool:
    """No space needed between adjacent CJK tokens."""
    return not (_is_cjk_token(left_token) and _is_cjk_token(right_token))


def _join_tokens_smart(tokens: List[str]) -> str:
    """Join tokens with smart spacing (no space between adjacent CJK tokens)."""
    if not tokens:
        return ""
    result = tokens[0]
    for i in range(1, len(tokens)):
        if _needs_space_between(tokens[i - 1], tokens[i]):
            result += " " + tokens[i]
        else:
            result += tokens[i]
    return result


def find_optimal_breaks_dp(
    tokens: List[str],
    max_width: float,
    word_width_func: Callable[[str], float],
    space_width: float,
    badness_exponent: float = 3.0,
    hyphen_penalty: float = 1000.0,
) -> Optional[List[str]]:
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
        hyphen_penalty: Penalty for lines ending with hyphens

    Returns:
        List of lines (strings) if successful, None if impossible to fit
    """
    try:
        if not tokens:
            return []

        # Calculate widths for all tokens
        token_w: List[float] = [word_width_func(t) for t in tokens]

        N = len(tokens)
        min_cost: List[float] = [float("inf")] * (N + 1)
        path: List[int] = [0] * (N + 1)
        min_cost[0] = 0.0

        for i in range(1, N + 1):
            line_width = 0.0
            for j in range(i - 1, -1, -1):
                # Add space only if needed between this token and the previous one on the line
                if j < i - 1:
                    # Check if we need space between tokens[j] and tokens[j+1]
                    if _needs_space_between(tokens[j], tokens[j + 1]):
                        line_width += space_width
                line_width += token_w[j]

                if line_width > max_width:
                    break

                slack = max_width - line_width
                badness = pow(slack, badness_exponent)

                # Add hyphen penalty if line ends with hyphen (support styled markers)
                last_token = tokens[i - 1] if i > 0 else ""
                ends_with_hyphen = last_token.endswith("-")
                if not ends_with_hyphen:
                    styled_match = STYLE_PATTERN.match(last_token)
                    if styled_match:
                        ends_with_hyphen = styled_match.group(2).endswith("-")
                if ends_with_hyphen:
                    badness += hyphen_penalty

                total_cost = min_cost[j] + badness
                if total_cost < min_cost[i]:
                    min_cost[i] = total_cost
                    path[i] = j

        if not np.isfinite(min_cost[N]):
            return None

        lines: List[str] = []
        current_break = N
        while current_break > 0:
            prev_break = path[current_break]
            line = _join_tokens_smart(tokens[prev_break:current_break])
            lines.insert(0, line)
            current_break = prev_break

        return lines

    except Exception:
        return None
