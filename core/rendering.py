import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import skia
import uharfbuzz as hb
from PIL import Image

from utils.logging import log_message


# --- LRU Cache Implementation ---
class LRUCache:
    """Simple LRU cache implementation to prevent unbounded memory growth."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

    def put(self, key, value):
        if key in self.cache:
            # Update existing key
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used (first item)
            self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

    def __delitem__(self, key):
        if key in self.cache:
            del self.cache[key]


# Cache loaded font data with LRU management
_font_data_cache = LRUCache(max_size=50)  # Store up to 50 font files
_typeface_cache = LRUCache(max_size=50)
_hb_face_cache = LRUCache(max_size=50)

# Cache font features
_font_features_cache = LRUCache(max_size=50)

# Cache font variants
_font_variants_cache: Dict[str, Dict[str, Optional[Path]]] = {}
FONT_KEYWORDS = {
    "bold": {"bold", "heavy", "black"},
    "italic": {"italic", "oblique", "slanted", "inclined"},
    "regular": {"regular", "normal", "roman", "medium"},
}

# --- Style Parsing Helper ---
STYLE_PATTERN = re.compile(r"(\*{1,3})(.*?)(\1)")  # Matches *text*, **text**, ***text***


def get_font_features(font_path: str) -> Dict[str, List[str]]:
    """
    Uses fontTools to list GSUB and GPOS features in a font file. Caches results.

    Args:
        font_path (str): Path to the font file.

    Returns:
        Dict[str, List[str]]: Dictionary with 'GSUB' and 'GPOS' keys,
                              each containing a list of feature tags.
    """
    cached_features = _font_features_cache.get(font_path)
    if cached_features is not None:
        return cached_features

    features = {"GSUB": [], "GPOS": []}
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(font_path, fontNumber=0)

        if "GSUB" in font and hasattr(font["GSUB"].table, "FeatureList") and font["GSUB"].table.FeatureList:
            features["GSUB"] = sorted([fr.FeatureTag for fr in font["GSUB"].table.FeatureList.FeatureRecord])

        if "GPOS" in font and hasattr(font["GPOS"].table, "FeatureList") and font["GPOS"].table.FeatureList:
            features["GPOS"] = sorted([fr.FeatureTag for fr in font["GPOS"].table.FeatureList.FeatureRecord])

    except ImportError:
        log_message("fontTools not installed, cannot inspect font features.", always_print=True)
    except Exception as e:
        log_message(f"Could not inspect font features for {os.path.basename(font_path)}: {e}", always_print=True)

    _font_features_cache.put(font_path, features)
    return features


def _find_font_variants(font_dir: str, verbose: bool = False) -> Dict[str, Optional[Path]]:
    """
    Finds regular, italic, bold, and bold-italic font variants (.ttf, .otf)
    in a directory based on filename keywords. Caches results per directory.

    Args:
        font_dir (str): Directory containing font files.
        verbose (bool): Whether to print detailed logs.

    Returns:
        Dict[str, Optional[Path]]: Dictionary mapping style names
                                   ("regular", "italic", "bold", "bold_italic")
                                   to their respective Path objects, or None if not found.
    """
    resolved_dir = str(Path(font_dir).resolve())
    if resolved_dir in _font_variants_cache:
        return _font_variants_cache[resolved_dir]

    log_message(f"Scanning font directory: {resolved_dir}", verbose=verbose)
    font_files: List[Path] = []
    font_variants: Dict[str, Optional[Path]] = {"regular": None, "italic": None, "bold": None, "bold_italic": None}
    identified_files: set[Path] = set()

    try:
        font_dir_path = Path(resolved_dir)
        if font_dir_path.exists() and font_dir_path.is_dir():
            font_files = list(font_dir_path.glob("*.ttf")) + list(font_dir_path.glob("*.otf"))
        else:
            log_message(f"Font directory '{font_dir_path}' does not exist or is not a directory.", always_print=True)
            _font_variants_cache[resolved_dir] = font_variants
            return font_variants
    except Exception as e:
        log_message(f"Error accessing font directory '{font_dir}': {e}", always_print=True)
        _font_variants_cache[resolved_dir] = font_variants
        return font_variants

    if not font_files:
        log_message(f"No font files (.ttf, .otf) found in '{resolved_dir}'", always_print=True)
        _font_variants_cache[resolved_dir] = font_variants
        return font_variants

    # Sort by name length (desc) to potentially prioritize more specific names like "BoldItalic" over "Bold"
    font_files.sort(key=lambda x: len(x.name), reverse=True)

    # --- Font File Detection (Multi-pass) ---
    # Pass 1: Exact matches for combined styles first
    for font_file in font_files:
        if font_file in identified_files:
            continue
        stem_lower = font_file.stem.lower()
        is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
        is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
        assigned = False
        if is_bold and is_italic:
            if not font_variants["bold_italic"]:
                font_variants["bold_italic"] = font_file
                assigned = True
                log_message(f"Found Bold Italic: {font_file.name}", verbose=verbose)
        if assigned:
            identified_files.add(font_file)

    # Pass 2: Exact matches for single styles
    for font_file in font_files:
        if font_file in identified_files:
            continue
        stem_lower = font_file.stem.lower()
        is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
        is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
        assigned = False
        if is_bold and not is_italic:  # Only bold
            if not font_variants["bold"]:
                font_variants["bold"] = font_file
                assigned = True
                log_message(f"Found Bold: {font_file.name}", verbose=verbose)
        elif is_italic and not is_bold:  # Only italic
            if not font_variants["italic"]:
                font_variants["italic"] = font_file
                assigned = True
                log_message(f"Found Italic: {font_file.name}", verbose=verbose)
        if assigned:
            identified_files.add(font_file)

    # Pass 3: Explicit regular matches
    for font_file in font_files:
        if font_file in identified_files:
            continue
        stem_lower = font_file.stem.lower()
        is_regular = any(kw in stem_lower for kw in FONT_KEYWORDS["regular"])
        is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
        is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
        assigned = False
        if is_regular and not is_bold and not is_italic:
            if not font_variants["regular"]:
                font_variants["regular"] = font_file
                assigned = True
                log_message(f"Found Regular (explicit): {font_file.name}", verbose=verbose)
        if assigned:
            identified_files.add(font_file)

    # Pass 4: Infer regular (no style keywords found)
    if not font_variants["regular"]:
        for font_file in font_files:
            if font_file in identified_files:
                continue
            stem_lower = font_file.stem.lower()
            is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
            is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
            if not is_bold and not is_italic and not any(kw in stem_lower for kw in FONT_KEYWORDS["regular"]):
                font_name_lower = font_file.name.lower()
                is_likely_specific = any(
                    spec in font_name_lower
                    for spec in [
                        "light",
                        "thin",
                        "condensed",
                        "expanded",
                        "semi",
                        "demi",
                        "extra",
                        "ultra",
                        "book",
                        "medium",
                        "black",
                        "heavy",
                    ]
                )
                if not is_likely_specific:
                    font_variants["regular"] = font_file
                    identified_files.add(font_file)
                    log_message(f"Inferred Regular (no keywords): {font_file.name}", verbose=verbose)
                    break

    # Pass 5: Fallback regular (use first available unidentified font)
    if not font_variants["regular"]:
        first_available = next((f for f in font_files if f not in identified_files), None)
        if first_available:
            font_variants["regular"] = first_available
            if first_available not in identified_files:
                identified_files.add(first_available)
            log_message(f"Fallback Regular (first unidentified): {first_available.name}", verbose=verbose)

    # Pass 6: Final fallback (use *any* identified font if regular is still missing)
    if not font_variants["regular"]:
        backup_regular = next(
            (
                f
                for f in [font_variants.get("bold"), font_variants.get("italic"), font_variants.get("bold_italic")]
                if f
            ),
            None,
        )
        if backup_regular:
            font_variants["regular"] = backup_regular
            log_message(f"Fallback Regular (using existing variant): {backup_regular.name}", verbose=verbose)
        elif font_files:  # Absolute last resort: just grab the first font file found
            font_variants["regular"] = font_files[0]
            log_message(f"Fallback Regular (absolute first file): {font_files[0].name}", verbose=verbose)

    if not font_variants["regular"]:
        log_message(
            (
                f"CRITICAL: Could not identify or fallback to any regular font file in '{resolved_dir}'. "
                "Rendering will likely fail."
            ),
            always_print=True,
        )
        # Still cache the (lack of) results
    else:
        log_message(f"Final Font Variants Found in {resolved_dir}:", verbose=verbose)
        for style, path in font_variants.items():
            log_message(f"  - {style}: {path.name if path else 'None'}", verbose=verbose)

    _font_variants_cache[resolved_dir] = font_variants
    return font_variants


def _parse_styled_segments(text: str) -> List[Tuple[str, str]]:
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

        # Add preceding regular text
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


def _tokenize_styled_text(text: str) -> List[Tuple[str, bool]]:
    """
    Tokenizes text into atomic units for wrapping where styled blocks are
    preserved as single, unbreakable tokens.

    Returns: List[Tuple[str, bool]] where each tuple is (token_text, is_styled).
    - Styled tokens include their surrounding markers (e.g., **bold text**)
      and are split into per-word tokens, each wrapped with the same markers,
      to allow wrapping at word boundaries while preserving style.
    - Plain text outside markers is split on whitespace into word tokens.
    """
    tokens: List[Tuple[str, bool]] = []
    last_end = 0
    for match in STYLE_PATTERN.finditer(text):
        start, end = match.span()
        # Preceding plain text -> split by whitespace into words
        if start > last_end:
            preceding = text[last_end:start]
            for w in preceding.split():
                tokens.append((w, False))

        # Styled block -> split content into words but preserve markers per word
        marker = match.group(1)
        content = match.group(2)
        if content:
            for w in content.split():
                tokens.append((f"{marker}{w}{marker}", True))

        last_end = end

    # Trailing plain text
    if last_end < len(text):
        trailing = text[last_end:]
        for w in trailing.split():
            tokens.append((w, False))

    return tokens


# --- Skia/HarfBuzz Rendering ---
def _load_font_resources(font_path: str) -> Tuple[Optional[bytes], Optional[skia.Typeface], Optional[hb.Face]]:
    """Loads font data, Skia Typeface, and HarfBuzz Face, using LRU caching."""
    # Check cache for font data
    font_data = _font_data_cache.get(font_path)
    if font_data is None:
        try:
            with open(font_path, "rb") as f:
                font_data = f.read()
            _font_data_cache.put(font_path, font_data)
        except Exception as e:
            log_message(f"ERROR: Failed to read font file {font_path}: {e}", always_print=True)
            return None, None, None

    # Check cache for typeface
    typeface = _typeface_cache.get(font_path)
    if typeface is None:
        skia_data = skia.Data.MakeWithoutCopy(font_data)
        typeface = skia.Typeface.MakeFromData(skia_data)
        if typeface is None:
            log_message(f"ERROR: Skia could not load typeface from {font_path}", always_print=True)
            # Remove from font data cache if typeface creation failed
            if font_path in _font_data_cache:
                del _font_data_cache[font_path]
            return None, None, None
        _typeface_cache.put(font_path, typeface)

    # Check cache for HarfBuzz face
    hb_face = _hb_face_cache.get(font_path)
    if hb_face is None:
        try:
            hb_face = hb.Face(font_data)
            _hb_face_cache.put(font_path, hb_face)
        except Exception as e:
            log_message(f"ERROR: HarfBuzz could not load face from {font_path}: {e}", always_print=True)
            # Remove from other caches if HarfBuzz face creation failed
            if font_path in _typeface_cache:
                del _typeface_cache[font_path]
            if font_path in _font_data_cache:
                del _font_data_cache[font_path]
            return None, None, None

    return font_data, typeface, hb_face


def _shape_line(
    text_line: str, hb_font: hb.Font, features: Dict[str, bool]
) -> Tuple[List[hb.GlyphInfo], List[hb.GlyphPosition]]:
    """Shapes a line of text with HarfBuzz."""
    hb_buffer = hb.Buffer()
    hb_buffer.add_str(text_line)
    hb_buffer.guess_segment_properties()
    try:
        hb.shape(hb_font, hb_buffer, features)
        return hb_buffer.glyph_infos, hb_buffer.glyph_positions
    except Exception as e:
        log_message(f"ERROR: HarfBuzz shaping failed for line '{text_line[:30]}...': {e}", always_print=True)
        return [], []


def _calculate_line_width(positions: List[hb.GlyphPosition], scale_factor: float) -> float:
    """
    Calculates the width of a shaped line from HarfBuzz positions.
    Assumes positions are in 26.6 fixed point if hb_font.ptem was set before shaping.
    """
    if not positions:
        return 0.0
    # Sum advances (which are in 26.6 fixed point)
    total_advance_fixed = sum(pos.x_advance for pos in positions)
    # Convert from 26.6 fixed point to pixels by dividing by 64.0
    HB_26_6_SCALE_FACTOR = 64.0
    return float(total_advance_fixed / HB_26_6_SCALE_FACTOR)


def _calculate_styled_line_width(
    line_with_markers: str,
    font_size: int,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features: Dict[str, bool],
) -> float:
    """
    Calculates the width of a line that may contain style markers using
    the appropriate HarfBuzz faces per style segment.

    Falls back to the 'regular' face if a style-specific face is missing.
    """
    if not line_with_markers:
        return 0.0

    segments = _parse_styled_segments(line_with_markers)
    if not segments:
        return 0.0

    regular_face = loaded_hb_faces.get("regular")
    if regular_face is None:
        return 0.0

    total_width = 0.0
    for segment_text, style_name in segments:
        hb_face_to_use = (
            loaded_hb_faces.get(style_name)
            if style_name in ("regular", "italic", "bold", "bold_italic")
            else None
        ) or regular_face

        hb_font_segment = hb.Font(hb_face_to_use)
        hb_font_segment.ptem = float(font_size)
        if hb_face_to_use.upem > 0:
            scale_factor = font_size / hb_face_to_use.upem
            hb_scale = int(scale_factor * (2**16))
            hb_font_segment.scale = (hb_scale, hb_scale)
        else:
            hb_font_segment.scale = (int(font_size * (2**16)), int(font_size * (2**16)))

        infos, positions = _shape_line(segment_text, hb_font_segment, features)
        total_width += _calculate_line_width(positions, 1.0)

    return total_width


def _find_optimal_breaks_dp(
    tokens: List[str],
    max_render_width: float,
    font_size: int,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    persistent_word_width_cache: Optional[Dict[Tuple[str, int], float]] = None,
    badness_exponent: float = 3.0,
    hyphen_penalty: float = 1000.0,
) -> Optional[List[str]]:
    """
    Pragmatic Knuth-Plass style DP to find globally optimal line breaks.

    - Uses cached per-token widths and a single space width.
    - Approximates line width as sum(token_widths) + spaces * space_width.
    - Adds a fixed penalty when a line ends with a hyphenated token.
    """
    try:
        if not tokens:
            return []

        # Step 1.1: Setup and Caching
        token_width_cache: Dict[str, float] = {}
        unique_tokens = set(tokens)
        unique_tokens.add(" ")

        for t in unique_tokens:
            width_val: Optional[float] = None
            if persistent_word_width_cache is not None:
                cached_key = (t, font_size)
                if cached_key in persistent_word_width_cache:
                    width_val = persistent_word_width_cache[cached_key]
            if width_val is None:
                width_val = _calculate_styled_line_width(
                    t, font_size, loaded_hb_faces, features_to_enable
                )
                if persistent_word_width_cache is not None:
                    persistent_word_width_cache[(t, font_size)] = width_val
            token_width_cache[t] = width_val

        space_w = token_width_cache.get(" ", 0.0)

        # Precompute token widths array for quick access
        token_w: List[float] = [token_width_cache.get(t, 0.0) for t in tokens]

        # Step 1.2: Initialize DP arrays
        N = len(tokens)
        min_cost: List[float] = [float("inf")] * (N + 1)
        path: List[int] = [0] * (N + 1)
        min_cost[0] = 0.0

        # Step 1.4: DP loops
        for i in range(1, N + 1):
            # Build line backwards from i-1 down to 0, accumulating width
            line_width = 0.0
            num_spaces = 0
            for j in range(i - 1, -1, -1):
                # Add token j
                if num_spaces > 0:
                    line_width += space_w
                line_width += token_w[j]
                num_spaces += 1

                if line_width > max_render_width:
                    # Further extending will only increase width
                    break

                # Compute badness for line tokens[j..i-1]
                slack = max_render_width - line_width
                badness = pow(slack, badness_exponent)

                # Hyphen penalty if last token ends with a hyphen
                last_token = tokens[i - 1] if i > 0 else ""
                if last_token.endswith("-"):
                    badness += hyphen_penalty

                total_cost = min_cost[j] + badness
                if total_cost < min_cost[i]:
                    min_cost[i] = total_cost
                    path[i] = j

        if not np.isfinite(min_cost[N]):
            return None

        # Step 1.5: Backtrack to reconstruct lines
        lines: List[str] = []
        current_break = N
        while current_break > 0:
            prev_break = path[current_break]
            line = " ".join(tokens[prev_break:current_break])
            lines.insert(0, line)
            current_break = prev_break

        return lines

    except Exception:
        return None


def _pil_to_skia_surface(pil_image: Image.Image) -> Optional[skia.Surface]:
    """Converts a PIL image to a Skia Surface."""
    try:
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")
        skia_image = skia.Image.frombytes(pil_image.tobytes(), pil_image.size, skia.kRGBA_8888_ColorType)
        if skia_image is None:
            log_message("Failed to create Skia image from PIL bytes", always_print=True)
            return None
        surface = skia.Surface(pil_image.width, pil_image.height)
        with surface as canvas:
            canvas.drawImage(skia_image, 0, 0)
        return surface
    except Exception as e:
        log_message(f"Error converting PIL to Skia Surface: {e}", always_print=True)
        return None


def _skia_surface_to_pil(surface: skia.Surface) -> Optional[Image.Image]:
    """Converts a Skia Surface back to a PIL image."""
    try:
        skia_image: Optional[skia.Image] = surface.makeImageSnapshot()
        if skia_image is None:
            log_message("Failed to create Skia snapshot from surface", always_print=True)
            return None

        skia_image = skia_image.convert(alphaType=skia.kUnpremul_AlphaType, colorType=skia.kRGBA_8888_ColorType)
        pil_image = Image.fromarray(skia_image)
        return pil_image
    except Exception as e:
        log_message(f"Error converting Skia Surface to PIL: {e}", always_print=True)
        return None


# --- Distance Transform Insetting Method ---
def _check_fit(
    font_size: int,
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    line_spacing_mult: float,
    hyphenate_before_scaling: bool,
    hyphenation_min_word_length: int,
    badness_exponent: float,
    word_width_cache: Optional[Dict[Tuple[str, int], float]] = None,
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Checks if text fits within the given dimensions at the specified font size.

    Args:
        font_size: Font size to test
        text: Text to wrap and measure
        max_render_width: Maximum allowed width
        max_render_height: Maximum allowed height
        regular_hb_face: HarfBuzz face for shaping
        regular_typeface: Skia typeface for metrics
        features_to_enable: HarfBuzz features to enable
        line_spacing_mult: Line spacing multiplier
        hyphenate_before_scaling: Whether to hyphenate before scaling
        word_width_cache: Optional cache for word widths
        verbose: Whether to print detailed logs

    Returns:
        Dict containing fit data if successful, None if doesn't fit
    """
    try:
        # Setup HarfBuzz font
        hb_font = hb.Font(regular_hb_face)
        hb_font.ptem = float(font_size)

        # Scale factor to convert font units to pixels
        scale_factor = 1.0
        if regular_hb_face.upem > 0:
            scale_factor = font_size / regular_hb_face.upem
        else:
            if verbose:
                log_message("Warning: Regular font has upem=0. Using scale factor 1.0.", verbose=verbose)

        # Convert scale factor to 16.16 fixed-point integer for HarfBuzz
        hb_scale = int(scale_factor * (2**16))
        hb_font.scale = (hb_scale, hb_scale)

        # Setup Skia font for metrics
        skia_font_test = skia.Font(regular_typeface, font_size)
        try:
            metrics = skia_font_test.getMetrics()
            single_line_height = (-metrics.fAscent + metrics.fDescent + metrics.fLeading) * line_spacing_mult
            if single_line_height <= 0:
                single_line_height = font_size * 1.2 * line_spacing_mult
        except Exception as e:
            if verbose:
                log_message(f"Could not get font metrics at size {font_size}: {e}", verbose=verbose)
            single_line_height = font_size * 1.2 * line_spacing_mult

        # Text wrapping logic (token-aware; preserves styled blocks as atomic tokens)
        tokens: List[Tuple[str, bool]] = _tokenize_styled_text(text)
        wrapped_lines_text: List[str] = []
        augmented_tokens: List[str] = []

        def _try_hyphenate_word(word_str: str, hb_font_local: hb.Font) -> Optional[List[str]]:
            """Attempt to split a word into two parts with a hyphen such that each part fits."""
            # 1) Isolate leading/trailing punctuation from the core word
            match = re.match(r'^(\W*)([\w\-]+)(\W*)$', word_str)
            if not match:
                return None

            leading_punc, core_word, trailing_punc = match.groups()

            if len(core_word) < 6:
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

            # Prefer splitting at existing hyphen(s) first (operate on core_word)
            if "-" in core_word:
                hyphen_positions = [i for i, ch in enumerate(core_word) if ch == "-"]
                mid = len(core_word) // 2
                hyphen_positions.sort(key=lambda i: abs(i - mid))
                for pos in hyphen_positions:
                    if pos <= 0 or pos >= len(core_word) - 1:
                        continue
                    left_part = core_word[: pos + 1]
                    right_part = core_word[pos + 1 :]

                    # Re-attach punctuation for measurement
                    final_left_part = leading_punc + left_part
                    final_right_part = right_part + trailing_punc

                    _, pos_left = _shape_line(final_left_part, hb_font_local, features_to_enable)
                    _, pos_right = _shape_line(final_right_part, hb_font_local, features_to_enable)
                    w_left = _calculate_line_width(pos_left, 1.0)
                    w_right = _calculate_line_width(pos_right, 1.0)
                    if w_left <= max_render_width and w_right <= max_render_width:
                        return [final_left_part, final_right_part]

            # Try split positions around the middle (operate on core_word)
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

                # Re-attach punctuation for measurement
                final_left_part = leading_punc + left_part
                final_right_part = right_part + trailing_punc

                _, pos_left = _shape_line(final_left_part, hb_font_local, features_to_enable)
                _, pos_right = _shape_line(final_right_part, hb_font_local, features_to_enable)
                w_left = _calculate_line_width(pos_left, 1.0)
                w_right = _calculate_line_width(pos_right, 1.0)
                if w_left <= max_render_width and w_right <= max_render_width:
                    return [final_left_part, final_right_part]
            return None

        # Phase 2: Preprocess tokens with pragmatic hyphenation
        for token_text, is_styled in tokens:
            if (
                not is_styled
                and hyphenate_before_scaling
                and len(token_text) > hyphenation_min_word_length
            ):
                split_parts = _try_hyphenate_word(token_text, hb_font)
                if split_parts:
                    augmented_tokens.extend(split_parts)
                else:
                    augmented_tokens.append(token_text)
            else:
                augmented_tokens.append(token_text)

        # Phase 3: DP-based optimal line breaking
        wrapped_lines_text = _find_optimal_breaks_dp(
            augmented_tokens,
            max_render_width,
            font_size,
            loaded_hb_faces,
            features_to_enable,
            word_width_cache,
            badness_exponent,
        ) or []

        if not wrapped_lines_text:
            return None

        # Measure wrapped block
        current_max_line_width = 0
        lines_data_at_size = []
        for line_text_with_markers in wrapped_lines_text:
            width = _calculate_styled_line_width(
                line_text_with_markers, font_size, loaded_hb_faces, features_to_enable
            )
            lines_data_at_size.append({"text_with_markers": line_text_with_markers, "width": width})
            current_max_line_width = max(current_max_line_width, width)

        total_block_height = (-metrics.fAscent + metrics.fDescent) + (len(wrapped_lines_text) - 1) * single_line_height

        if verbose:
            log_message(
                f"Size {font_size}: Block W={current_max_line_width:.1f} (Max W={max_render_width:.1f}), "
                f"H={total_block_height:.1f} (Max H={max_render_height:.1f})",
                verbose=verbose,
            )

        # Check if it fits
        if current_max_line_width <= max_render_width and total_block_height <= max_render_height:
            if verbose:
                log_message(f"Size {font_size} fits!", verbose=verbose)
            return {
                'lines': lines_data_at_size,
                'metrics': metrics,
                'max_line_width': current_max_line_width,
                'line_height': single_line_height
            }

        return None

    except Exception as e:
        if verbose:
            log_message(f"Error in _check_fit for size {font_size}: {e}", verbose=verbose)
        return None


def _calculate_centroid_expansion_box(
    cleaned_mask: np.ndarray, padding_pixels: float = 8.0, verbose: bool = False
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[float, float]]]:
    """
    Calculates guaranteed safe rendering box using the Distance Transform approach:

    1. Establish the Safe Zone: Create safe_area_mask using cv2.distanceTransform()
    2. Find the Unbiased Anchor: Calculate centroid (cx, cy) of the safe_area_mask
    3. Measure Available Space: Ray cast from centroid to find distances to edges
    4. Calculate Symmetrical Dimensions: Use min distances for width/height
    5. Construct Final Box: Create centered rectangle within safe zone

    Args:
        cleaned_mask: The binary (0/255) mask of the cleaned bubble.
        padding_pixels: The desired padding distance in pixels from the bubble edges.
        verbose: Whether to print detailed logs.

    Returns:
        Tuple containing:
        - Tuple[int, int, int, int]: Guaranteed safe box coordinates [x, y, w, h], or None if calculation fails.
        - Tuple[float, float]: True geometric center (centroid) of the safe area, or None if calculation fails.
    """
    if cleaned_mask is None or not np.any(cleaned_mask):
        return None

    try:
        # Step 1: Establish the Safe Zone
        distance_map = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        safe_area_mask = (distance_map >= padding_pixels).astype(np.uint8) * 255

        if not np.any(safe_area_mask):
            log_message(
                f"Padding of {padding_pixels:.1f}px resulted in an empty safe area. Calculation failed.",
                verbose=verbose,
                always_print=True
            )
            return None

        # Step 2: Find the Unbiased Anchor (centroid)
        moments = cv2.moments(safe_area_mask)
        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]
        centroid = (float(centroid_x), float(centroid_y))

        cx, cy = int(round(centroid_x)), int(round(centroid_y))
        mask_h, mask_w = safe_area_mask.shape

        # Step 3: Measure Available Space from centroid to edges
        # Find the first zero pixel from the centroid outwards in each direction
        left_zeros = np.where(safe_area_mask[cy, 0:cx] == 0)[0]
        dist_to_left_edge = cx - (left_zeros.max() if left_zeros.size > 0 else 0)

        right_zeros = np.where(safe_area_mask[cy, cx:] == 0)[0]
        dist_to_right_edge = right_zeros.min() if right_zeros.size > 0 else mask_w - cx

        up_zeros = np.where(safe_area_mask[0:cy, cx] == 0)[0]
        dist_to_top_edge = cy - (up_zeros.max() if up_zeros.size > 0 else 0)

        down_zeros = np.where(safe_area_mask[cy:, cx] == 0)[0]
        dist_to_bottom_edge = down_zeros.min() if down_zeros.size > 0 else mask_h - cy

        # Step 4: Calculate Symmetrical Dimensions
        # Subtract 1 to avoid landing on the boundary
        max_safe_width = 2 * max(0, min(dist_to_left_edge, dist_to_right_edge) - 1)
        max_safe_height = 2 * max(0, min(dist_to_top_edge, dist_to_bottom_edge) - 1)

        if max_safe_width <= 0 or max_safe_height <= 0:
            log_message(
                f"Calculated dimensions invalid: width={max_safe_width}, height={max_safe_height}",
                verbose=verbose,
                always_print=True
            )
            return None

        # Step 5: Construct the Final Box (centered on true geometric center)
        # Use the float centroid for the calculation
        box_x_float = centroid_x - max_safe_width / 2.0
        box_y_float = centroid_y - max_safe_height / 2.0

        # Convert to integer only at the very end
        box_x = int(round(box_x_float))
        box_y = int(round(box_y_float))

        guaranteed_box = (box_x, box_y, max_safe_width, max_safe_height)

        # Basic bounds checking
        if box_x >= 0 and box_y >= 0 and box_x + max_safe_width <= mask_w and box_y + max_safe_height <= mask_h:
            log_message(
                f"Safe area calculated: centroid=({centroid_x:.1f}, {centroid_y:.1f}), "
                f"box=({box_x}, {box_y}, {max_safe_width}, {max_safe_height})",
                verbose=verbose
            )
            return guaranteed_box, centroid
        else:
            log_message(
                f"Validation failed: box=({box_x}, {box_y}, {max_safe_width}, {max_safe_height}) "
                f"exceeds mask bounds=({mask_w}, {mask_h})",
                verbose=verbose,
                always_print=True
            )
            return None

    except (cv2.error, ValueError, IndexError, ZeroDivisionError, OverflowError) as e:
        log_message(f"Error calculating safe area: {e}", verbose=verbose, always_print=True)
    except Exception as e:
        log_message(f"Unexpected error calculating safe area: {e}", verbose=verbose, always_print=True)

    return None


def render_text_skia(
    pil_image: Image.Image,
    text: str,
    bbox: Tuple[int, int, int, int],
    font_dir: str,
    cleaned_mask: Optional[np.ndarray] = None,
    bubble_color_bgr: Optional[Tuple[int, int, int]] = (255, 255, 255),
    min_font_size: int = 8,
    max_font_size: int = 14,
    line_spacing_mult: float = 1.0,
    use_subpixel_rendering: bool = False,
    font_hinting: str = "none",
    use_ligatures: bool = False,
    hyphenate_before_scaling: bool = True,
    hyphenation_min_word_length: int = 8,
    badness_exponent: float = 3.0,
    verbose: bool = False,
) -> Tuple[Optional[Image.Image], bool]:
    """
    Fits and renders text within a bounding box using Skia and HarfBuzz.

    Uses the 5-step Distance Transform Insetting Method:
    1. Establish Safe Zone: Create safe_area_mask using cv2.distanceTransform()
    2. Find Unbiased Anchor: Calculate centroid of the safe_area_mask
    3. Measure Available Space: Ray cast from centroid to find distances to edges
    4. Calculate Symmetrical Dimensions: Use min distances for width/height
    5. Construct Final Box: Create centered rectangle within safe zone

    This ensures text is perfectly centered and never touches bubble boundaries.

    Args:
        pil_image: PIL Image object to draw onto.
        text: Text to render.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        font_dir: Directory containing font files.
        cleaned_mask: Binary mask of the cleaned bubble (0/255). Used for safe area calculation.
        bubble_color_bgr: Background color of the bubble (BGR tuple). Used to determine text color.
        min_font_size: Minimum font size to try.
        max_font_size: Starting font size to try.
        line_spacing_mult: Multiplier for line height (based on font metrics).
        use_subpixel_rendering: Whether to use subpixel anti-aliasing.
        font_hinting: Font hinting level ("none", "slight", "normal", "full").
        use_ligatures: Whether to enable standard ligatures ('liga').
        hyphenate_before_scaling: Whether to hyphenate long words before reducing font size.
        verbose: Whether to print detailed logs.

    Returns:
        Tuple containing:
        - Modified PIL Image object (or original if failed).
        - Boolean indicating success.
    """
    x1, y1, x2, y2 = bbox
    bubble_width = x2 - x1
    bubble_height = y2 - y1

    if bubble_width <= 0 or bubble_height <= 0:
        log_message(f"Invalid original bbox dimensions: {bbox}", always_print=True)
        return pil_image, False

    clean_text = " ".join(text.split())
    if not clean_text:
        return pil_image, True

    # --- Calculate Safe Rendering Area ---
    safe_area_result = None
    if cleaned_mask is not None:
        safe_area_result = _calculate_centroid_expansion_box(cleaned_mask, verbose=verbose)

    if safe_area_result is not None:
        guaranteed_box, _ = safe_area_result
        box_x, box_y, box_w, box_h = guaranteed_box
        max_render_width = float(box_w)
        max_render_height = float(box_h)
        target_center_x = box_x + box_w / 2.0
        target_center_y = box_y + box_h / 2.0

    if safe_area_result is None:
        # Fallback to padded bbox
        padding_ratio = 0.08
        max_render_width = bubble_width * (1 - 2 * padding_ratio)
        max_render_height = bubble_height * (1 - 2 * padding_ratio)

        if max_render_width <= 0 or max_render_height <= 0:
            max_render_width = max(1.0, float(bubble_width))
            max_render_height = max(1.0, float(bubble_height))

        target_center_x = x1 + bubble_width / 2.0
        target_center_y = y1 + bubble_height / 2.0

    # --- Find and Load Font Variants ---
    font_variants = _find_font_variants(font_dir, verbose=verbose)
    regular_font_path = font_variants.get("regular")

    if not regular_font_path:
        log_message(f"CRITICAL: Regular font variant not found in '{font_dir}'. Cannot render text.", always_print=True)
        return pil_image, False

    _, regular_typeface, regular_hb_face = _load_font_resources(str(regular_font_path))

    if not regular_typeface or not regular_hb_face:
        log_message(f"CRITICAL: Failed to load regular font resources for {regular_font_path.name}.", always_print=True)
        return pil_image, False

    # --- Determine HarfBuzz Features ---
    available_features = get_font_features(str(regular_font_path))
    features_to_enable = {
        # Enable kerning if available (GPOS 'kern' or legacy 'kern' table)
        "kern": "kern" in available_features["GPOS"],
        # Enable ligatures based on config and availability
        "liga": use_ligatures and "liga" in available_features["GSUB"],
        # Add other features as needed, e.g., "dlig", "calt"
        "calt": "calt" in available_features["GSUB"],
    }
    log_message(f"HarfBuzz features to enable: {features_to_enable}", verbose=verbose)

    # --- Preload style faces for measurement during fitting ---
    preload_hb_faces: Dict[str, Optional[hb.Face]] = {"regular": regular_hb_face}
    for style_key in ["italic", "bold", "bold_italic"]:
        style_path = font_variants.get(style_key)
        if style_path:
            _, _typeface, _hb_face = _load_font_resources(str(style_path))
            if _hb_face:
                preload_hb_faces[style_key] = _hb_face

    # --- Font Size Iteration (Binary Search) ---
    best_fit_size = -1
    best_fit_lines_data = None
    best_fit_metrics = None
    best_fit_max_line_width = float("inf")

    word_width_cache: Dict[Tuple[str, int], float] = {}

    low = min_font_size
    high = max_font_size

    while low <= high:
        mid = (low + high) // 2
        if mid == 0:  # Avoid infinite loop if min_font_size is 0 or 1
            break

        log_message(f"Trying font size: {mid}", verbose=verbose)

        fit_data = _check_fit(
            mid,
            clean_text,
            max_render_width,
            max_render_height,
            regular_hb_face,
            regular_typeface,
            preload_hb_faces,
            features_to_enable,
            line_spacing_mult,
            hyphenate_before_scaling,
            hyphenation_min_word_length,
            badness_exponent,
            word_width_cache,
            verbose,
        )

        if fit_data is not None:
            best_fit_size = mid
            best_fit_lines_data = fit_data['lines']
            best_fit_metrics = fit_data['metrics']
            best_fit_max_line_width = fit_data['max_line_width']
            low = mid + 1
        else:
            high = mid - 1

    # --- Check if any size fit ---
    if best_fit_size == -1:
        log_message(
            f"Could not fit text in bubble even at min size {min_font_size}: '{clean_text[:30]}...'", always_print=True
        )
        return pil_image, False

    # --- Prepare for Rendering ---
    final_font_size = best_fit_size
    final_lines_data = best_fit_lines_data
    final_metrics = best_fit_metrics
    final_max_line_width = best_fit_max_line_width
    final_line_height = (-final_metrics.fAscent + final_metrics.fDescent + final_metrics.fLeading) * line_spacing_mult

    # Parse styled segments once after finding optimal font size to avoid repeated parsing
    if final_lines_data:
        for line_data in final_lines_data:
            line_data['segments'] = _parse_styled_segments(line_data['text_with_markers'])

    # --- Load Needed Font Resources for Rendering ---
    # Determine which styles are actually present in the text
    required_styles = {"regular"} | {style for _, style in _parse_styled_segments(clean_text)}
    log_message(f"Required font styles: {required_styles}", verbose=verbose)

    # Initialize dictionaries to hold loaded resources, starting with regular
    loaded_typefaces: Dict[str, Optional[skia.Typeface]] = {"regular": regular_typeface}
    loaded_hb_faces: Dict[str, Optional[hb.Face]] = {"regular": regular_hb_face}

    for style in ["italic", "bold", "bold_italic"]:
        if style in required_styles:
            font_path = font_variants.get(style)
            if font_path:
                log_message(f"Loading {style} font variant: {font_path.name}", verbose=verbose)
                _, typeface, hb_face = _load_font_resources(str(font_path))
                if typeface and hb_face:
                    loaded_typefaces[style] = typeface
                    loaded_hb_faces[style] = hb_face
                else:
                    log_message(
                        f"Warning: Failed to load {style} font variant {font_path.name}. Will fallback to regular.",
                        verbose=verbose,
                    )
            else:
                log_message(
                    (
                        f"Warning: {style} style requested by text, but font variant not found in {font_dir}. "
                        "Will fallback to regular."
                    ),
                    verbose=verbose,
                )

    surface = _pil_to_skia_surface(pil_image)
    if surface is None:
        log_message("Failed to create Skia surface for rendering.", always_print=True)
        return pil_image, False

    # --- Skia Font Hinting Setup (Used per segment) ---
    hinting_map = {
        "none": skia.FontHinting.kNone,
        "slight": skia.FontHinting.kSlight,
        "normal": skia.FontHinting.kNormal,
        "full": skia.FontHinting.kFull,
    }
    skia_hinting = hinting_map.get(font_hinting.lower(), skia.FontHinting.kNone)

    # --- Determine Text Color based on Bubble Background ---
    text_color = skia.ColorBLACK
    if bubble_color_bgr and (sum(bubble_color_bgr) / 3) < 128:
        text_color = skia.ColorWHITE
    paint = skia.Paint(AntiAlias=True, Color=text_color)

    # --- Calculate Starting Position (Centering the visual block around the target center) ---
    # Center the max line width
    block_start_x = target_center_x - final_max_line_width / 2.0

    # Vertical centering
    num_lines = len(final_lines_data)
    if num_lines > 0:
        total_visual_height = (
            (num_lines - 1) * final_line_height
            - final_metrics.fAscent
            + final_metrics.fDescent
        )
        block_top_y = target_center_y - (total_visual_height / 2.0)
        first_baseline_y = block_top_y - final_metrics.fAscent
    else:
        first_baseline_y = target_center_y - (final_metrics.fAscent + final_metrics.fDescent) / 2.0

    log_message(
        (
            f"Rendering at size {final_font_size}. Centering target: "
            f"({target_center_x:.1f}, {target_center_y:.1f}). Block Start X: {block_start_x:.1f}. "
            f"First Baseline Y: {first_baseline_y:.1f}"
        ),
        verbose=verbose,
    )

    # --- Render Line by Line, Segment by Segment ---
    with surface as canvas:
        current_baseline_y = first_baseline_y
        for i, line_data in enumerate(final_lines_data):
            line_width_measured = line_data["width"]  # Width measured using regular font

            # Horizontal alignment for this specific line, centered within the block's max width
            line_start_x = block_start_x + (final_max_line_width - line_width_measured) / 2.0
            cursor_x = line_start_x

            segments = line_data.get('segments', [])
            log_message(f"Line {i} Segments: {segments}", verbose=verbose)

            for segment_text, style_name in segments:

                # --- Select Font Resources for Segment ---
                typeface_to_use = None
                hb_face_to_use = None
                fallback_style_used = None

                if style_name == "bold_italic":
                    typeface_to_use = loaded_typefaces.get("bold_italic")
                    hb_face_to_use = loaded_hb_faces.get("bold_italic")
                    if not typeface_to_use or not hb_face_to_use:
                        fallback_style_used = "bold"
                        typeface_to_use = loaded_typefaces.get("bold")
                        hb_face_to_use = loaded_hb_faces.get("bold")
                    if not typeface_to_use or not hb_face_to_use:
                        fallback_style_used = "italic"
                        typeface_to_use = loaded_typefaces.get("italic")
                        hb_face_to_use = loaded_hb_faces.get("italic")
                    if not typeface_to_use or not hb_face_to_use:
                        fallback_style_used = "regular"
                        typeface_to_use = regular_typeface
                        hb_face_to_use = regular_hb_face
                elif style_name == "bold":
                    typeface_to_use = loaded_typefaces.get("bold")
                    hb_face_to_use = loaded_hb_faces.get("bold")
                    if not typeface_to_use or not hb_face_to_use:
                        fallback_style_used = "regular"
                        typeface_to_use = regular_typeface
                        hb_face_to_use = regular_hb_face
                elif style_name == "italic":
                    typeface_to_use = loaded_typefaces.get("italic")
                    hb_face_to_use = loaded_hb_faces.get("italic")
                    if not typeface_to_use or not hb_face_to_use:
                        fallback_style_used = "regular"
                        typeface_to_use = regular_typeface
                        hb_face_to_use = regular_hb_face
                else:  # Regular or unknown style
                    typeface_to_use = regular_typeface
                    hb_face_to_use = regular_hb_face

                if fallback_style_used:
                    log_message(
                        f"  Style '{style_name}' not found, falling back to '{fallback_style_used}'.", verbose=verbose
                    )

                # Ensure we actually got *some* font resource (should always get regular at least)
                if not typeface_to_use or not hb_face_to_use:
                    log_message(
                        (
                            f"ERROR: Could not get any valid font resources (including regular) "
                            f"for style '{style_name}'. Skipping segment."
                        ),
                        always_print=True,
                    )
                    continue

                # --- Setup Skia Font for Segment ---
                skia_font_segment = skia.Font(typeface_to_use, final_font_size)
                skia_font_segment.setSubpixel(use_subpixel_rendering)
                skia_font_segment.setHinting(skia_hinting)

                # --- Setup HarfBuzz Font for Segment ---
                hb_font_segment = hb.Font(hb_face_to_use)
                hb_font_segment.ptem = float(final_font_size)
                if hb_face_to_use.upem > 0:
                    scale_factor = final_font_size / hb_face_to_use.upem
                    hb_scale = int(scale_factor * (2**16))
                    hb_font_segment.scale = (hb_scale, hb_scale)
                else:
                    hb_font_segment.scale = (int(final_font_size * (2**16)), int(final_font_size * (2**16)))

                # --- Shape Segment ---
                try:
                    infos, positions = _shape_line(segment_text, hb_font_segment, features_to_enable)
                    if not infos:
                        log_message(
                            f"Warning: HarfBuzz returned no glyphs for segment '{segment_text}'", verbose=verbose
                        )
                        continue
                except Exception as e:
                    log_message(f"ERROR: HarfBuzz shaping failed for segment '{segment_text}': {e}", always_print=True)
                    continue

                # --- Build and Draw Skia TextBlob for Segment ---
                builder = skia.TextBlobBuilder()
                glyph_ids = [info.codepoint for info in infos]
                skia_point_positions = []
                segment_cursor_x = 0

                HB_26_6_SCALE_FACTOR = 64.0
                for _, pos in zip(infos, positions):
                    glyph_x = cursor_x + segment_cursor_x + (pos.x_offset / HB_26_6_SCALE_FACTOR)
                    glyph_y = current_baseline_y - (pos.y_offset / HB_26_6_SCALE_FACTOR)  # Y flipped
                    skia_point_positions.append(skia.Point(glyph_x, glyph_y))

                    segment_cursor_x += pos.x_advance / HB_26_6_SCALE_FACTOR

                try:
                    _ = builder.allocRunPos(skia_font_segment, glyph_ids, skia_point_positions)

                    text_blob = builder.make()
                    if text_blob:
                        canvas.drawTextBlob(text_blob, 0, 0, paint)
                        segment_width = segment_cursor_x
                        cursor_x += segment_width
                        log_message(
                            (
                                f"  Rendered segment '{segment_text}' ({style_name}), width={segment_width:.1f}, "
                                f"new cursor_x={cursor_x:.1f}"
                            ),
                            verbose=verbose,
                        )
                    else:
                        log_message(f"Warning: Failed to build TextBlob for segment '{segment_text}'", verbose=verbose)

                except Exception as e:
                    log_message(f"ERROR during Skia rendering for segment '{segment_text}': {e}", always_print=True)
                    segment_width = segment_cursor_x
                    cursor_x += segment_width

            current_baseline_y += final_line_height

    # --- Convert back to PIL ---
    final_pil_image = _skia_surface_to_pil(surface)
    if final_pil_image is None:
        log_message("Failed to convert final Skia surface back to PIL.", always_print=True)
        return pil_image, False

    log_message(f"Successfully rendered text at size {final_font_size}", verbose=verbose)
    return final_pil_image, True
