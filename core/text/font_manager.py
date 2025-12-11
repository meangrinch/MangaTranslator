import io
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

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
_font_variants_cache: Dict[str, Dict[str, Optional[Path]]] = {}

# Font style detection keywords
FONT_KEYWORDS = {
    "bold": {"bold", "heavy", "black"},
    "italic": {"italic", "oblique", "slanted", "inclined"},
    "regular": {"regular", "normal", "roman", "medium"},
}


def get_font_features(font_path: str) -> Dict[str, List[str]]:
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


def find_font_variants(
    font_dir: str, verbose: bool = False
) -> Dict[str, Optional[Path]]:
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
    font_files: List[Path] = []
    font_variants: Dict[str, Optional[Path]] = {
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
        if is_bold and is_italic:
            if not font_variants["bold_italic"]:
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
        elif is_italic and not is_bold:
            if not font_variants["italic"]:
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
        if is_regular and not is_bold and not is_italic:
            if not font_variants["regular"]:
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


def load_font_family(font_dir: str, verbose: bool = False) -> Dict[str, Optional[str]]:
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
