import io
import os
from collections import OrderedDict
from pathlib import Path

from fontTools.ttLib import TTFont

from utils.exceptions import FontError
from utils.logging import log_message


class LRUCache:
    """Simple LRU cache implementation to prevent unbounded memory growth."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache

    def __delitem__(self, key):
        if key in self.cache:
            del self.cache[key]


_font_data_cache = LRUCache(max_size=50)
_font_features_cache = LRUCache(max_size=50)
_font_cmap_cache = LRUCache(max_size=50)
_font_variants_cache: dict[str, dict[str, Path | None]] = {}

# Font style detection keywords
FONT_KEYWORDS = {
    "bold": {"bold", "heavy", "black"},
    "italic": {"italic", "oblique", "slanted", "inclined"},
    "regular": {"regular", "normal", "roman", "medium"},
}


def get_font_features(font_path: str) -> dict[str, list[str]]:
    """
    Uses fontTools to list GSUB and GPOS features in a font file. Caches results.

    Args:
        font_path: Path to the font file.

    Returns:
        Dictionary with 'GSUB' and 'GPOS' keys, each containing a list of feature tags.
    """
    cached_features = _font_features_cache.get(font_path)
    if cached_features is not None:
        return cached_features

    features = {"GSUB": [], "GPOS": []}
    try:
        font = TTFont(font_path, fontNumber=0)

        if (
            "GSUB" in font
            and hasattr(font["GSUB"].table, "FeatureList")
            and font["GSUB"].table.FeatureList
        ):
            features["GSUB"] = sorted(
                [fr.FeatureTag for fr in font["GSUB"].table.FeatureList.FeatureRecord]
            )

        if (
            "GPOS" in font
            and hasattr(font["GPOS"].table, "FeatureList")
            and font["GPOS"].table.FeatureList
        ):
            features["GPOS"] = sorted(
                [fr.FeatureTag for fr in font["GPOS"].table.FeatureList.FeatureRecord]
            )

    except ImportError:
        log_message(
            "fontTools not available - font features disabled", always_print=True
        )
    except Exception as e:
        log_message(
            f"Font feature inspection failed for {os.path.basename(font_path)}: {e}",
            always_print=True,
        )

    _font_features_cache.put(font_path, features)
    return features


def get_font_cmap(font_path: str) -> set:
    """
    Returns the set of Unicode codepoints supported by the font.

    Uses fontTools to extract the best cmap table from the font file.
    Results are cached to avoid repeated font parsing.

    Args:
        font_path: Path to the font file.

    Returns:
        Set of integer codepoints (Unicode code points) supported by the font.
        Returns an empty set if the font cannot be read or has no cmap.
    """
    cached_cmap = _font_cmap_cache.get(font_path)
    if cached_cmap is not None:
        return cached_cmap

    supported_codepoints: set = set()
    try:
        font = TTFont(font_path, fontNumber=0)
        cmap = font.getBestCmap()
        if cmap:
            supported_codepoints = set(cmap.keys())
    except Exception as e:
        log_message(
            f"Failed to extract cmap from {os.path.basename(font_path)}: {e}",
            always_print=True,
        )

    _font_cmap_cache.put(font_path, supported_codepoints)
    return supported_codepoints


def sanitize_text_for_font(text: str, font_path: str, verbose: bool = False) -> str:
    """
    Removes characters from text that are not supported by the font's cmap.

    This prevents "tofu" characters (▯) from appearing in rendered text.
    Style markers (*, **, ***) are preserved even if asterisk is not in the font,
    since they are stripped during text processing and never actually rendered.

    Args:
        text: The text to sanitize.
        font_path: Path to the font file to check against.
        verbose: Whether to print detailed logs.

    Returns:
        Sanitized text with unsupported characters removed.
    """
    if not text:
        return text

    supported_codepoints = get_font_cmap(font_path)

    if not supported_codepoints:
        log_message(
            f"Could not get cmap for {os.path.basename(font_path)}, skipping sanitization",
            verbose=verbose,
        )
        return text

    # Characters to always preserve (style markers used in markdown-like formatting)
    STYLE_MARKER_CHARS = {"*"}
    WHITESPACE_CHARS = {" ", "\t", "\n", "\r"}

    removed_chars: list[str] = []
    sanitized_chars: list[str] = []

    for char in text:
        codepoint = ord(char)

        if (
            char in STYLE_MARKER_CHARS
            or char in WHITESPACE_CHARS
            or codepoint in supported_codepoints
        ):
            sanitized_chars.append(char)
        else:
            removed_chars.append(char)

    if removed_chars:
        unique_removed = sorted(set(removed_chars), key=lambda c: ord(c))
        char_descriptions = [f"'{c}' (U+{ord(c):04X})" for c in unique_removed[:10]]
        if len(unique_removed) > 10:
            char_descriptions.append(f"... and {len(unique_removed) - 10} more")

        log_message(
            f"Removed {len(removed_chars)} unsupported character(s) from text: "
            f"{', '.join(char_descriptions)}",
            always_print=True,
        )

    return "".join(sanitized_chars)


def _validate_font_file(font_file: Path, verbose: bool = False) -> bool:
    """
    Validate that a font file is not corrupt by attempting to load it with TTFont.

    Args:
        font_file: Path to the font file to validate
        verbose: Whether to print detailed logs

    Returns:
        True if font is valid, False if corrupt or invalid
    """
    try:
        # Try to load the font with fontTools to check integrity
        font = TTFont(font_file, fontNumber=0)
        # Basic validation - check if it has required tables
        if "cmap" not in font or "head" not in font:
            log_message(
                f"Font file {font_file.name} appears to be missing required tables",
                verbose=verbose,
                always_print=True,
            )
            return False
        return True
    except Exception as e:
        log_message(
            f"Font file {font_file.name} appears to be corrupt: {e}",
            verbose=verbose,
            always_print=True,
        )
        return False


_var_font_info_cache: dict[str, dict | None] = {}
_var_font_style_cache: dict[str, dict | None] = {}


def inspect_variable_font(font_path: Path | str) -> dict | None:
    """
    Inspects a font file to determine if it is a variable font (contains an 'fvar' table).
    Returns axis definitions and named instances, or None if it is a static font or invalid.
    """
    path_str = str(Path(font_path).resolve())
    if path_str in _var_font_info_cache:
        return _var_font_info_cache[path_str]

    try:
        tt = TTFont(path_str, fontNumber=0, lazy=True)
        if "fvar" not in tt:
            _var_font_info_cache[path_str] = None
            return None

        fvar = tt["fvar"]
        axes = {
            axis.axisTag: (axis.minValue, axis.defaultValue, axis.maxValue)
            for axis in fvar.axes
        }

        instances = []
        name_table = tt.get("name")
        for i, inst in enumerate(fvar.instances):
            name_str = ""
            if name_table:
                record = name_table.getName(
                    inst.subfamilyNameID, 3, 1
                ) or name_table.getName(inst.subfamilyNameID, 1, 0)
                if record:
                    name_str = record.toStr()
            instances.append(
                {
                    "index": i + 1,  # 1-based index for HarfBuzz
                    "name": name_str,
                    "coords": dict(inst.coordinates),
                }
            )

        info = {
            "path": Path(path_str),
            "axes": axes,
            "instances": instances,
        }
        _var_font_info_cache[path_str] = info
        return info
    except Exception as e:
        log_message(f"Variable font check failed for {os.path.basename(path_str)}: {e}")
        _var_font_info_cache[path_str] = None
        return None


def resolve_variable_font_style(
    vf_info: dict, target_weight: float = 400.0, target_italic: bool = False
) -> dict:
    """
    Finds the best named instance or coordinates in a variable font for a target weight and italic state.
    """
    best_inst = None
    min_dist = float("inf")

    for inst in vf_info["instances"]:
        coords = inst["coords"]
        w = coords.get("wght", 400.0)
        if "ital" in coords:
            if target_italic and coords["ital"] < 0.5:
                continue
            if not target_italic and coords["ital"] > 0.5:
                continue
        if "slnt" in coords:
            if target_italic and coords["slnt"] >= 0:
                continue
            if not target_italic and coords["slnt"] < 0:
                continue

        dist = abs(w - target_weight)
        if dist < min_dist:
            min_dist = dist
            best_inst = inst

    if best_inst:
        return {
            "instance_index": best_inst["index"],
            "coords": dict(best_inst["coords"]),
        }

    coords = {}
    if "wght" in vf_info["axes"]:
        min_w, _def_w, max_w = vf_info["axes"]["wght"]
        coords["wght"] = max(min_w, min(max_w, target_weight))
    if target_italic:
        if "ital" in vf_info["axes"]:
            coords["ital"] = 1.0
        elif "slnt" in vf_info["axes"]:
            coords["slnt"] = vf_info["axes"]["slnt"][0]

    return {
        "instance_index": None,
        "coords": coords,
    }


def get_variable_font_style_info(
    font_path: Path | str, style: str = "regular"
) -> dict | None:
    """
    Returns the variable font style parameters (instance_index and coords) for a given font file and style.
    Returns None if the font is static.
    """
    path_str = str(Path(font_path).resolve())
    cache_key = f"{path_str}::{style}"
    if cache_key in _var_font_style_cache:
        return _var_font_style_cache[cache_key]

    vf_info = inspect_variable_font(path_str)
    if not vf_info:
        _var_font_style_cache[cache_key] = None
        return None

    target_weight = 700.0 if "bold" in style else 400.0
    target_italic = "italic" in style

    stem_lower = Path(path_str).stem.lower()
    is_italic_file = "italic" in stem_lower or "oblique" in stem_lower

    style_info = resolve_variable_font_style(
        vf_info, target_weight, target_italic=target_italic or is_italic_file
    )
    _var_font_style_cache[cache_key] = style_info
    return style_info


def auto_map_variable_fonts(
    font_files: list[Path], verbose: bool = False
) -> dict[str, Path | None] | None:
    """
    Auto-maps styles (regular, bold, italic, bold_italic) from variable font files.
    Returns mapping if variable fonts are found, or None if no variable fonts exist.
    """
    var_fonts = []
    for f in font_files:
        info = inspect_variable_font(f)
        if info:
            var_fonts.append(info)

    if not var_fonts:
        return None

    roman_font = None
    italic_font = None

    for vf in var_fonts:
        stem = vf["path"].stem.lower()
        is_italic = "italic" in stem or "oblique" in stem
        if is_italic:
            if not italic_font:
                italic_font = vf
        else:
            if not roman_font:
                roman_font = vf

    if not roman_font and var_fonts:
        roman_font = var_fonts[0]

    variants: dict[str, Path | None] = {
        "regular": None,
        "italic": None,
        "bold": None,
        "bold_italic": None,
    }

    if roman_font:
        variants["regular"] = roman_font["path"]
        get_variable_font_style_info(roman_font["path"], "regular")

        has_bold_axis = (
            "wght" in roman_font["axes"] and roman_font["axes"]["wght"][2] >= 600
        )
        has_bold_instance = any(
            inst["coords"].get("wght", 0) >= 600 for inst in roman_font["instances"]
        )
        if has_bold_axis or has_bold_instance:
            variants["bold"] = roman_font["path"]
            get_variable_font_style_info(roman_font["path"], "bold")

    if italic_font:
        variants["italic"] = italic_font["path"]
        get_variable_font_style_info(italic_font["path"], "italic")
        has_bold_italic_axis = (
            "wght" in italic_font["axes"] and italic_font["axes"]["wght"][2] >= 600
        )
        has_bold_italic_instance = any(
            inst["coords"].get("wght", 0) >= 600 for inst in italic_font["instances"]
        )
        if has_bold_italic_axis or has_bold_italic_instance:
            variants["bold_italic"] = italic_font["path"]
            get_variable_font_style_info(italic_font["path"], "bold_italic")
    elif roman_font and ("ital" in roman_font["axes"] or "slnt" in roman_font["axes"]):
        variants["italic"] = roman_font["path"]
        get_variable_font_style_info(roman_font["path"], "italic")
        if variants["bold"]:
            variants["bold_italic"] = roman_font["path"]
            get_variable_font_style_info(roman_font["path"], "bold_italic")

    found_styles = [f"{s}: {p.name}" for s, p in variants.items() if p]
    log_message(
        f"Auto-mapped variable font variants: {', '.join(found_styles)}",
        verbose=verbose,
    )
    return variants


def find_font_variants(font_dir: str, verbose: bool = False) -> dict[str, Path | None]:
    """
    Finds regular, italic, bold, and bold-italic font variants (.ttf, .otf)
    in a directory based on filename keywords. Caches results per directory.

    Args:
        font_dir: Directory containing font files.
        verbose: Whether to print detailed logs.

    Returns:
        Dictionary mapping style names ("regular", "italic", "bold", "bold_italic")
        to their respective Path objects, or None if not found.
    """
    resolved_dir = str(Path(font_dir).resolve())
    if resolved_dir in _font_variants_cache:
        return _font_variants_cache[resolved_dir]

    log_message(f"Scanning fonts in {os.path.basename(resolved_dir)}", verbose=verbose)
    font_files: list[Path] = []
    font_variants: dict[str, Path | None] = {
        "regular": None,
        "italic": None,
        "bold": None,
        "bold_italic": None,
    }
    identified_files: set[Path] = set()

    try:
        font_dir_path = Path(resolved_dir)
        if font_dir_path.exists() and font_dir_path.is_dir():
            font_files = list(font_dir_path.glob("*.ttf")) + list(
                font_dir_path.glob("*.otf")
            )
        else:
            log_message(f"Font directory not found: {font_dir_path}", always_print=True)
            _font_variants_cache[resolved_dir] = font_variants
            return font_variants
    except Exception as e:
        log_message(f"Font directory access error: {e}", always_print=True)
        _font_variants_cache[resolved_dir] = font_variants
        return font_variants

    if not font_files:
        log_message(
            f"No font files found in {os.path.basename(resolved_dir)}",
            always_print=True,
        )
        _font_variants_cache[resolved_dir] = font_variants
        return font_variants

    # Check for variable fonts and auto-map if present
    var_variants = auto_map_variable_fonts(font_files, verbose=verbose)
    if var_variants and var_variants.get("regular"):
        # Check if any missing variants can be provided by static files
        for style in ["italic", "bold", "bold_italic"]:
            if var_variants[style] is None:
                for font_file in font_files:
                    if not _validate_font_file(font_file, verbose=verbose):
                        continue
                    stem_lower = font_file.stem.lower()
                    is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
                    is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
                    if style == "bold" and is_bold and not is_italic:
                        var_variants["bold"] = font_file
                        break
                    elif style == "italic" and is_italic and not is_bold:
                        var_variants["italic"] = font_file
                        break
                    elif style == "bold_italic" and is_bold and is_italic:
                        var_variants["bold_italic"] = font_file
                        break
        _font_variants_cache[resolved_dir] = var_variants
        return var_variants

    # Sort by name length (desc) to prioritize more specific names like "BoldItalic" over "Bold"
    font_files.sort(key=lambda x: len(x.name), reverse=True)

    # Pass 1: Combined styles first
    for font_file in font_files:
        if font_file in identified_files:
            continue

        # Validate font file integrity before processing
        if not _validate_font_file(font_file, verbose=verbose):
            continue

        stem_lower = font_file.stem.lower()
        is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
        is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
        assigned = False
        if is_bold and is_italic and not font_variants["bold_italic"]:
            font_variants["bold_italic"] = font_file
            assigned = True
            log_message(f"Found bold-italic: {font_file.name}", verbose=verbose)
        if assigned:
            identified_files.add(font_file)

    # Pass 2: Single styles
    for font_file in font_files:
        if font_file in identified_files:
            continue

        # Validate font file integrity before processing
        if not _validate_font_file(font_file, verbose=verbose):
            continue

        stem_lower = font_file.stem.lower()
        is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
        is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
        assigned = False
        if is_bold and not is_italic:
            if not font_variants["bold"]:
                font_variants["bold"] = font_file
                assigned = True
                log_message(f"Found bold: {font_file.name}", verbose=verbose)
        elif is_italic and not is_bold and not font_variants["italic"]:
            font_variants["italic"] = font_file
            assigned = True
            log_message(f"Found italic: {font_file.name}", verbose=verbose)
        if assigned:
            identified_files.add(font_file)

    # Pass 3: Explicit regular matches
    for font_file in font_files:
        if font_file in identified_files:
            continue

        # Validate font file integrity before processing
        if not _validate_font_file(font_file, verbose=verbose):
            continue

        stem_lower = font_file.stem.lower()
        is_regular = any(kw in stem_lower for kw in FONT_KEYWORDS["regular"])
        is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
        is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
        assigned = False
        if (
            is_regular
            and not is_bold
            and not is_italic
            and not font_variants["regular"]
        ):
            font_variants["regular"] = font_file
            assigned = True
            log_message(f"Found regular: {font_file.name}", verbose=verbose)
        if assigned:
            identified_files.add(font_file)

    # Pass 4: Infer regular from files without style keywords
    if not font_variants["regular"]:
        for font_file in font_files:
            if font_file in identified_files:
                continue

            # Validate font file integrity before processing
            if not _validate_font_file(font_file, verbose=verbose):
                continue

            stem_lower = font_file.stem.lower()
            is_bold = any(kw in stem_lower for kw in FONT_KEYWORDS["bold"])
            is_italic = any(kw in stem_lower for kw in FONT_KEYWORDS["italic"])
            if (
                not is_bold
                and not is_italic
                and not any(kw in stem_lower for kw in FONT_KEYWORDS["regular"])
            ):
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
                    log_message(f"Inferred regular: {font_file.name}", verbose=verbose)
                    break

    # Pass 5: Fallback to first available
    if not font_variants["regular"]:
        first_available = next(
            (f for f in font_files if f not in identified_files), None
        )
        if first_available:
            font_variants["regular"] = first_available
            if first_available not in identified_files:
                identified_files.add(first_available)
            log_message(f"Fallback regular: {first_available.name}", verbose=verbose)

    # Pass 6: Final fallback to any variant
    if not font_variants["regular"]:
        backup_regular = next(
            (
                f
                for f in [
                    font_variants.get("bold"),
                    font_variants.get("italic"),
                    font_variants.get("bold_italic"),
                ]
                if f
            ),
            None,
        )
        if backup_regular:
            font_variants["regular"] = backup_regular
            log_message(f"Fallback regular: {backup_regular.name}", verbose=verbose)
        elif font_files:
            font_variants["regular"] = font_files[0]
            log_message(f"Fallback regular: {font_files[0].name}", verbose=verbose)

    if not font_variants["regular"]:
        log_message(
            f"CRITICAL: No regular font found in {os.path.basename(resolved_dir)} - rendering will fail",
            always_print=True,
        )
        raise FontError(f"No regular font found in directory: {resolved_dir}")
    else:
        found_variants = [
            f"{style}: {path.name}" for style, path in font_variants.items() if path
        ]
        log_message(f"Font variants: {', '.join(found_variants)}", verbose=verbose)

    _font_variants_cache[resolved_dir] = font_variants
    return font_variants


def sanitize_font_data(font_path: str, font_data: bytes) -> bytes:
    """
    Analyzes font data for known issues (bad UPM, corrupt kern table) and
    returns a sanitized version of the font data.

    Args:
        font_path: Path to the font file (for logging purposes)
        font_data: Raw font data bytes

    Returns:
        Sanitized font data bytes
    """
    try:
        font_file = io.BytesIO(font_data)
        font = TTFont(font_file, fontNumber=0)

        data_was_modified = False

        if "kern" in font:
            try:
                _ = font["kern"].tables[0].kernTable
            except Exception:
                msg = f"Detected corrupt kern table in {os.path.basename(font_path)}. Removing it."
                log_message(msg, always_print=True)
                del font["kern"]
                data_was_modified = True

        test_glyph_name = None
        cmap = font.getBestCmap()
        if cmap and ord("M") in cmap:
            test_glyph_name = cmap[ord("M")]

        if test_glyph_name and "glyf" in font and "hmtx" in font:
            glyph = font["glyf"][test_glyph_name]
            advance_width = font["hmtx"][test_glyph_name][0]

            # Check for italic style (Bit 1 of macStyle) and allow 30% overhang
            is_italic = (font["head"].macStyle & 0b10) != 0 if "head" in font else False
            tolerance = 1.3 if is_italic else 1.0

            if hasattr(glyph, "xMax") and glyph.xMax > (advance_width * tolerance):
                msg = f"Font {os.path.basename(font_path)} has unreliable metrics. Overriding UPM to 1000."
                log_message(msg, always_print=True)
                font["head"].unitsPerEm = 1000
                data_was_modified = True

        if data_was_modified:
            output_bytes = io.BytesIO()
            font.save(output_bytes)
            return output_bytes.getvalue()
        else:
            return font_data

    except Exception as e:
        log_message(
            f"Font sanitization failed for {os.path.basename(font_path)}: {e}",
            always_print=True,
        )
        return font_data


def load_font_data(font_path: str) -> bytes:
    """
    Loads and sanitizes font data from a file. Uses caching to avoid repeated reads.

    Args:
        font_path: Path to the font file

    Returns:
        Sanitized font data bytes

    Raises:
        FontError: If font file cannot be loaded or is invalid
    """
    font_data = _font_data_cache.get(font_path)
    if font_data is None:
        try:
            with open(font_path, "rb") as f:
                original_font_data = f.read()

            font_data = sanitize_font_data(font_path, original_font_data)
            _font_data_cache.put(font_path, font_data)
        except Exception as e:
            log_message(f"Font file read error: {e}", always_print=True)
            raise FontError(f"Failed to load font file: {font_path}") from e

    return font_data


def load_font_family(font_dir: str, verbose: bool = False) -> dict[str, str | None]:
    """
    High-level function to load a complete font family from a directory.

    Args:
        font_dir: Directory containing font files
        verbose: Whether to print detailed logs

    Returns:
        Dictionary mapping style names to font file paths (as strings)
    """
    variants = find_font_variants(font_dir, verbose=verbose)
    return {style: str(path) if path else None for style, path in variants.items()}
