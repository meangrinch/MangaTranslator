import re
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import skia
import uharfbuzz as hb
from PIL import Image

from .common_utils import log_message

# --- Font Feature Inspection (Optional but useful for debugging) ---
_font_features_cache = {}

def get_font_features(font_path: str) -> Dict[str, List[str]]:
    """
    Uses fontTools to list GSUB and GPOS features in a font file. Caches results.

    Args:
        font_path (str): Path to the font file.

    Returns:
        Dict[str, List[str]]: Dictionary with 'GSUB' and 'GPOS' keys,
                              each containing a list of feature tags.
    """
    if font_path in _font_features_cache:
        return _font_features_cache[font_path]

    features = {'GSUB': [], 'GPOS': []}
    try:
        from fontTools.ttLib import TTFont
        font = TTFont(font_path, fontNumber=0)

        if "GSUB" in font and hasattr(font['GSUB'].table, 'FeatureList') and font['GSUB'].table.FeatureList:
            features['GSUB'] = sorted([fr.FeatureTag for fr in font['GSUB'].table.FeatureList.FeatureRecord])

        if "GPOS" in font and hasattr(font['GPOS'].table, 'FeatureList') and font['GPOS'].table.FeatureList:
            features['GPOS'] = sorted([fr.FeatureTag for fr in font['GPOS'].table.FeatureList.FeatureRecord])

    except ImportError:
        log_message("fontTools not installed, cannot inspect font features.", always_print=True)
    except Exception as e:
        log_message(f"Could not inspect font features for {os.path.basename(font_path)}: {e}", always_print=True)

    _font_features_cache[font_path] = features
    return features

# --- Font Choice Logic (Keep existing) ---
def choose_font(translated_text, font_dir="./fonts", verbose=False):
    """
    Choose the most appropriate font file path and its intended style
    based on semantic interpretation of the text content, common comic/manga
    conventions, and available font files.

    Args:
        translated_text (str): The translated text
        font_dir (str): Directory containing font files
        verbose (bool): Whether to print more detailed logging information

    Returns:
        str | None: Font path or None if no font is found.
    """
    FONT_KEYWORDS = {
        "bold": {"bold", "heavy", "black", "extrabold", "semibold", "demibold", "ultrabold"},
        "italic": {"italic", "oblique", "slanted", "inclined"},
        "regular": {"regular", "normal", "roman", "book", "medium"}
    }

    font_files = []
    selected_font_path_obj = None

    try:
        font_dir_path = Path(font_dir).resolve()
        if font_dir_path.exists() and font_dir_path.is_dir():
            font_files = list(font_dir_path.glob("*.ttf")) + list(font_dir_path.glob("*.otf"))
        else:
            log_message(f"Font directory '{font_dir_path}' does not exist or is not a directory.", always_print=True)
            return None
    except Exception as e:
        log_message(f"Error accessing font directory '{font_dir}': {e}", always_print=True)
        return None

    if not font_files:
        log_message(f"No font files (.ttf, .otf) found in '{font_dir_path}'", always_print=True)
        return None

    font_variants = {"regular": None, "italic": None, "bold": None, "bold_italic": None}
    identified_files = set()

    def get_filename_parts(filepath):
        name = filepath.stem.lower()
        name = re.sub(r'[-_]', ' ', name)
        return set(name.split())

    font_files.sort(key=lambda x: len(x.name), reverse=True)

    # --- Font File Detection (Multi-pass) ---
    # Pass 1: Exact matches
    for font_file in font_files:
        if font_file in identified_files: continue
        parts = get_filename_parts(font_file)
        is_bold = any(kw in parts for kw in FONT_KEYWORDS["bold"])
        is_italic = any(kw in parts for kw in FONT_KEYWORDS["italic"])
        assigned = False
        if is_bold and is_italic:
            if not font_variants["bold_italic"]: font_variants["bold_italic"] = font_file; assigned = True; log_message(f"Found Bold Italic: {font_file.name}", verbose=verbose)
        elif is_bold:
             if not font_variants["bold"]: font_variants["bold"] = font_file; assigned = True; log_message(f"Found Bold: {font_file.name}", verbose=verbose)
        elif is_italic:
             if not font_variants["italic"]: font_variants["italic"] = font_file; assigned = True; log_message(f"Found Italic: {font_file.name}", verbose=verbose)
        elif any(kw in parts for kw in FONT_KEYWORDS["regular"]):
             if not font_variants["regular"]: font_variants["regular"] = font_file; assigned = True; log_message(f"Found Regular: {font_file.name}", verbose=verbose)
        if assigned: identified_files.add(font_file)

    # Pass 2: Infer regular
    if not font_variants["regular"]:
        for font_file in font_files:
            if font_file in identified_files: continue
            parts = get_filename_parts(font_file)
            is_bold = any(kw in parts for kw in FONT_KEYWORDS["bold"])
            is_italic = any(kw in parts for kw in FONT_KEYWORDS["italic"])
            font_name_lower = font_file.name.lower()
            is_specific = any(spec in font_name_lower for spec in ["italic", "bold", "roman", "oblique", "slanted", "heavy", "black", "light", "thin"])
            if not is_bold and not is_italic and not is_specific:
                font_variants["regular"] = font_file
                identified_files.add(font_file)
                log_message(f"Inferred Regular: {font_file.name}", verbose=verbose)
                break

    # Pass 3: Fallback regular
    if not font_variants["regular"]:
        first_available = next((f for f in font_files if f not in identified_files), font_files[0] if font_files else None)
        if first_available:
            font_variants["regular"] = first_available
            if first_available not in identified_files: identified_files.add(first_available)
            log_message(f"Fallback Regular: {first_available.name}", verbose=verbose)

    if not font_variants["regular"] and any(f for f in font_variants.values() if f):
         backup_regular = next(f for f in [font_variants.get("bold"), font_variants.get("italic"), font_variants.get("bold_italic")] if f)
         if backup_regular:
             font_variants["regular"] = backup_regular
             log_message(f"Using existing variant {backup_regular.name} as fallback Regular", verbose=verbose)

    if not font_variants["regular"]:
        log_message(f"Could not identify or fallback to any font file in '{font_dir_path}'. Cannot proceed.", always_print=True)
        return None

    # --- Semantic Rule-Based Style Selection ---
    text = translated_text.strip()
    text_lower = text.lower()
    text_stripped = text.strip('.,?!()[]{}<>~*_-"\'`')

    intended_style = "regular"
    rule_reason = "Default"

    # Rule checking
    if text_stripped.isupper() and (text.count('!') >= 2 or text.count('?') >= 2 or (len(text) > 2 and text[0]=='(' and text[-1]==')' and text[1:-1].strip().isupper())):
        intended_style = "bold_italic"; rule_reason = "Intense Shouting/Scream"
    elif text_stripped.isupper() and len(text_stripped) <= 15 and any(sfx in text_lower for sfx in ["kaboom", "krakoom", "thooom", "craaash", "baam!", "krakaboom"]):
         intended_style = "bold_italic"; rule_reason = "High-Impact SFX"
    elif intended_style == "regular" and text_stripped.isupper() and len(text_stripped) > 1 and len(text_stripped) < 60:
        intended_style = "bold"; rule_reason = "Standard Shouting / Loud Voice"
    elif intended_style == "regular" and not text.isupper() and not text.islower():
        words = re.findall(r'\b\w+\b', text)
        uppercase_word_count = sum(1 for word in words if word.isupper() and len(word) > 1 and word.isalpha())
        if uppercase_word_count > 0 and uppercase_word_count <= 2 and len(words) > uppercase_word_count:
            intended_style = "bold"; rule_reason = "Single Word Emphasis (All Caps)"
    elif intended_style == "regular" and text.startswith("(") and text.endswith(")") and len(text) > 2:
        intended_style = "italic"; rule_reason = "Thought/Aside (Parentheses)"
    elif intended_style == "regular" and ((text.startswith("*") and text.endswith("*")) or (text.startswith("~") and text.endswith("~"))) and len(text) > 2:
        intended_style = "italic"; rule_reason = "Emphasis/Whisper (Asterisks/Tildes)"
    elif intended_style == "regular" and ((text.startswith("[") and text.endswith("]")) or (text.startswith("<<") and text.endswith(">>"))) and len(text) > 4:
         intended_style = "italic"; rule_reason = "Radio/Speaker Text"
    elif intended_style == "regular" and len(text_stripped) <= 25:
        common_onomatopoeias = {"boom", "bang", "crash", "whoosh", "thud", "slam", "pow", "bam", "splash", "drip", "crack", "crunch", "buzz", "hiss", "click", "clack", "thump", "whack", "ring", "swish", "zap", "vroom", "beep", "boop", "ding", "dong", "fwoosh", "kerplunk", "murmur", "pitter-patter", "rat-a-tat", "screech", "thwip", "whirr", "clank", "creak", "gulp", "sigh"}
        sfx_candidate = text_stripped.lower().replace('!','').replace('?','').replace('.','')
        if sfx_candidate in common_onomatopoeias:
             intended_style = "italic"; rule_reason = f"Onomatopoeia ({sfx_candidate})"
        elif (re.search(r"([a-zA-Z])\1{2,}", text_stripped, re.IGNORECASE) or re.search(r"([ HhAa ]{4,})", text) or (text.endswith("...") and len(text) > 3)):
             if not (text_stripped.isupper() and ("!" in text or "?" in text)):
                 intended_style = "italic"; rule_reason = "Repetitive Sound/Laughter/Trailing Off"
        elif '-' in text and any(part and part[0].isupper() for part in text.split('-')):
             intended_style = "italic"; rule_reason = "Hyphenated SFX/Emphasis"

    log_message(f"Text: '{text[:30]}...' -> Intended Style: {intended_style} (Reason: {rule_reason})", verbose=verbose)

    # --- Font Path Selection  ---
    selected_font_path_obj = None
    if font_variants.get(intended_style):
        selected_font_path_obj = font_variants[intended_style]
        log_message(f"Selected exact font file for {intended_style}: {selected_font_path_obj.name}", verbose=verbose)
    else:
        log_message(f"Font file for intended style '{intended_style}' not found. Applying fallback.", verbose=verbose)
        final_variant_used_for_log = intended_style
        if intended_style == "bold_italic":
            if font_variants.get("bold"): selected_font_path_obj = font_variants["bold"]; final_variant_used_for_log = "bold"
            elif font_variants.get("italic"): selected_font_path_obj = font_variants["italic"]; final_variant_used_for_log = "italic"
            else: selected_font_path_obj = font_variants.get("regular"); final_variant_used_for_log = "regular"
        elif intended_style == "bold":
            selected_font_path_obj = font_variants.get("regular"); final_variant_used_for_log = "regular"
        elif intended_style == "italic":
            selected_font_path_obj = font_variants.get("regular"); final_variant_used_for_log = "regular"
        else:
            selected_font_path_obj = font_variants.get("regular"); final_variant_used_for_log = "regular"

        if selected_font_path_obj:
             log_message(f"Using fallback font: {selected_font_path_obj.name} (as {final_variant_used_for_log})", verbose=verbose)

    if selected_font_path_obj is None:
        log_message(f"Critical: No font file could be selected for '{text[:15]}...'. Check font directory and logic.", always_print=True)
        return None

    try:
        final_path_str = str(selected_font_path_obj.resolve())
        log_message(f"Selected Path: {final_path_str} for text starting with '{text[:15]}...'", verbose=verbose)
        return final_path_str
    except Exception as e:
        log_message(f"Error resolving font path for {selected_font_path_obj.name}: {e}", always_print=True)
        return None


# --- Skia/HarfBuzz Rendering Implementation ---
# Cache loaded font data to avoid repeated disk reads
_font_data_cache = {}
_typeface_cache = {}
_hb_face_cache = {}

def _load_font_resources(font_path: str) -> Tuple[Optional[bytes], Optional[skia.Typeface], Optional[hb.Face]]:
    """Loads font data, Skia Typeface, and HarfBuzz Face, using caching."""
    if font_path not in _font_data_cache:
        try:
            with open(font_path, 'rb') as f:
                _font_data_cache[font_path] = f.read()
        except Exception as e:
            log_message(f"ERROR: Failed to read font file {font_path}: {e}", always_print=True)
            return None, None, None
    font_data = _font_data_cache[font_path]

    if font_path not in _typeface_cache:
        skia_data = skia.Data.MakeWithoutCopy(font_data)
        _typeface_cache[font_path] = skia.Typeface.MakeFromData(skia_data)
        if _typeface_cache[font_path] is None:
            log_message(f"ERROR: Skia could not load typeface from {font_path}", always_print=True)
            # Clear bad cache entry
            del _typeface_cache[font_path]
            del _font_data_cache[font_path]
            return None, None, None
    typeface = _typeface_cache[font_path]

    if font_path not in _hb_face_cache:
        try:
            _hb_face_cache[font_path] = hb.Face(font_data)
        except Exception as e:
             log_message(f"ERROR: HarfBuzz could not load face from {font_path}: {e}", always_print=True)
             if font_path in _typeface_cache: del _typeface_cache[font_path]
             if font_path in _font_data_cache: del _font_data_cache[font_path]
             return None, None, None
    hb_face = _hb_face_cache[font_path]

    return font_data, typeface, hb_face


def _shape_line(text_line: str, hb_font: hb.Font, features: Dict[str, bool]) -> Tuple[List[hb.GlyphInfo], List[hb.GlyphPosition]]:
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

def _pil_to_skia_surface(pil_image: Image.Image) -> Optional[skia.Surface]:
    """Converts a PIL image to a Skia Surface."""
    try:
        if pil_image.mode != 'RGBA':
             pil_image = pil_image.convert('RGBA')
        skia_image = skia.Image.frombytes(
            pil_image.tobytes(),
            pil_image.size,
            skia.kRGBA_8888_ColorType
        )
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
        skia_image = surface.makeImageSnapshot()
        if skia_image is None:
            log_message("Failed to create Skia snapshot from surface", always_print=True)
            return None
        skia_bytes = skia_image.tobytes()
        if skia_bytes is None:
             log_message("Failed to get bytes from Skia snapshot", always_print=True)
             return None
        pil_image = Image.frombytes(
            "RGBA",
            (surface.width(), surface.height()),
            skia_bytes
        )
        return pil_image
    except Exception as e:
        log_message(f"Error converting Skia Surface to PIL: {e}", always_print=True)
        return None

def render_text_skia(
    pil_image: Image.Image,
    text: str,
    bbox: Tuple[int, int, int, int],
    font_path: str,
    min_font_size: int = 8,
    max_font_size: int = 14,
    line_spacing_mult: float = 1.0,
    use_subpixel_rendering: bool = False,
    font_hinting: str = "none",
    use_ligatures: bool = False,
    verbose: bool = False
) -> Tuple[Optional[Image.Image], bool]:
    """
    Fits and renders text within a bounding box using Skia and HarfBuzz.

    Args:
        pil_image: PIL Image object to draw onto.
        text: Text to render.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        font_path: Path to the font file.
        min_font_size: Minimum font size to try.
        max_font_size: Starting font size to try.
        line_spacing_mult: Multiplier for line height (based on font metrics).
        use_subpixel_rendering: Whether to use subpixel anti-aliasing.
        font_hinting: Font hinting level ("none", "slight", "normal", "full").
        use_ligatures: Whether to enable standard ligatures ('liga').
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
        log_message(f"Invalid bubble dimensions: {bbox}", always_print=True)
        return pil_image, False

    clean_text = ' '.join(text.split())
    if not clean_text:
        return pil_image, True

    # --- Padding ---
    padding_ratio = 0.08
    max_render_width = bubble_width * (1 - 2 * padding_ratio)
    max_render_height = bubble_height * (1 - 2 * padding_ratio)

    if max_render_width <= 0 or max_render_height <= 0:
         log_message(f"Bubble too small for padding: {bbox}", verbose=verbose)
         max_render_width = max(1, bubble_width)
         max_render_height = max(1, bubble_height)

    # --- Load Font Resources ---
    _, typeface, hb_face = _load_font_resources(font_path)
    if not typeface or not hb_face:
        return pil_image, False

    # --- Determine HarfBuzz Features ---
    available_features = get_font_features(font_path)
    features_to_enable = {
        # Enable kerning if available (GPOS 'kern' or legacy 'kern' table)
        "kern": "kern" in available_features['GPOS'],
        # Enable ligatures based on config and availability
        "liga": use_ligatures and "liga" in available_features['GSUB'],
        # Add other features as needed, e.g., "dlig", "calt"
        "calt": "calt" in available_features['GSUB'],
    }
    log_message(f"HarfBuzz features to enable: {features_to_enable}", verbose=verbose)

    # --- Font Size Iteration ---
    best_fit_size = -1
    best_fit_lines_data = None
    best_fit_metrics = None
    best_fit_block_height = float('inf')
    best_fit_max_line_width = float('inf')

    current_size = max_font_size
    while current_size >= min_font_size:
        log_message(f"Trying font size: {current_size}", verbose=verbose)
        hb_font = hb.Font(hb_face)
        hb_font.ptem = float(current_size)

        # Scale factor to convert font units to pixels
        scale_factor = 1.0
        if hb_face.upem > 0:
            scale_factor = current_size / hb_face.upem
        else:
            log_message(f"Warning: Font {font_path} has upem=0. Using scale factor 1.0.", verbose=verbose)

        # Convert scale factor to 16.16 fixed-point integer for HarfBuzz
        hb_scale = int(scale_factor * (2**16))
        hb_font.scale = (hb_scale, hb_scale)

        skia_font_test = skia.Font(typeface, current_size)
        try:
            metrics = skia_font_test.getMetrics()
            single_line_height = (-metrics.fAscent + metrics.fDescent + metrics.fLeading) * line_spacing_mult
            if single_line_height <= 0: single_line_height = current_size * 1.2 * line_spacing_mult
        except Exception as e:
             log_message(f"Could not get font metrics at size {current_size}: {e}", verbose=verbose)
             single_line_height = current_size * 1.2 * line_spacing_mult

        # --- Text Wrapping at current_size ---
        words = clean_text.split()
        wrapped_lines_text = []
        current_line_words = []
        longest_word_width_at_size = 0

        for word in words:
            _, w_positions = _shape_line(word, hb_font, features_to_enable)
            word_width = _calculate_line_width(w_positions, 1.0)
            longest_word_width_at_size = max(longest_word_width_at_size, word_width)

            if word_width > max_render_width:
                 log_message(f"Size {current_size}: Word '{word}' ({word_width:.1f}px) wider than max width ({max_render_width:.1f}px). Trying smaller size.", verbose=verbose)
                 current_line_words = None
                 break

            test_line_words = current_line_words + [word]
            test_line_text = " ".join(test_line_words)
            _, test_positions = _shape_line(test_line_text, hb_font, features_to_enable)
            tentative_width = _calculate_line_width(test_positions, 1.0)

            if tentative_width <= max_render_width:
                current_line_words = test_line_words
            else:
                if current_line_words:
                    wrapped_lines_text.append(" ".join(current_line_words))
                current_line_words = [word]

        if current_line_words is None:
            current_size -= 1
            continue

        if current_line_words:
            wrapped_lines_text.append(" ".join(current_line_words))

        if not wrapped_lines_text:
            log_message(f"Size {current_size}: Wrapping failed, no lines generated.", verbose=verbose)
            current_size -= 1
            continue

        # --- Measure Wrapped Block at current_size ---
        current_max_line_width = 0
        lines_data_at_size = []
        for line_text in wrapped_lines_text:
            infos, positions = _shape_line(line_text, hb_font, features_to_enable)
            width = _calculate_line_width(positions, 1.0)
            lines_data_at_size.append({"text": line_text, "infos": infos, "positions": positions, "width": width})
            current_max_line_width = max(current_max_line_width, width)

        total_block_height = (-metrics.fAscent + metrics.fDescent) + (len(wrapped_lines_text) - 1) * single_line_height

        log_message(f"Size {current_size}: Block W={current_max_line_width:.1f} (Max W={max_render_width:.1f}), H={total_block_height:.1f} (Max H={max_render_height:.1f})", verbose=verbose)

        # --- Check Fit ---
        if current_max_line_width <= max_render_width and total_block_height <= max_render_height:
            log_message(f"Size {current_size} fits!", verbose=verbose)
            best_fit_size = current_size
            best_fit_lines_data = lines_data_at_size
            best_fit_metrics = metrics
            best_fit_block_height = total_block_height
            best_fit_max_line_width = current_max_line_width
            break

        current_size -= 1

    # --- Check if any size fit ---
    if best_fit_size == -1:
        log_message(f"Could not fit text in bubble even at min size {min_font_size}: '{clean_text[:30]}...'", always_print=True)
        return pil_image, False

    # --- Prepare for Rendering ---
    final_font_size = best_fit_size
    final_lines_data = best_fit_lines_data
    final_metrics = best_fit_metrics
    final_block_height = best_fit_block_height
    final_max_line_width = best_fit_max_line_width
    final_line_height = (-final_metrics.fAscent + final_metrics.fDescent + final_metrics.fLeading) * line_spacing_mult
    if final_line_height <= 0: final_line_height = final_font_size * 1.2 * line_spacing_mult

    surface = _pil_to_skia_surface(pil_image)
    if surface is None:
        log_message("Failed to create Skia surface for rendering.", always_print=True)
        return pil_image, False

    # --- Skia Font and Paint Setup ---
    skia_font_final = skia.Font(typeface, final_font_size)
    skia_font_final.setSubpixel(use_subpixel_rendering)

    hinting_map = {
        "none": skia.FontHinting.kNone,
        "slight": skia.FontHinting.kSlight,
        "normal": skia.FontHinting.kNormal,
        "full": skia.FontHinting.kFull,
    }
    skia_hinting = hinting_map.get(font_hinting.lower(), skia.FontHinting.kNone)
    skia_font_final.setHinting(skia_hinting)

    paint = skia.Paint(AntiAlias=True, Color=skia.ColorBLACK) # Use black text

    # --- Calculate Centered Starting Position ---
    # Top-left corner of the rendering area within the bubble
    render_area_x = x1 + (bubble_width - max_render_width) / 2
    render_area_y = y1 + (bubble_height - max_render_height) / 2

    # Top-left corner of the text block within the rendering area
    block_start_x = render_area_x + (max_render_width - final_max_line_width) / 2
    block_start_y = render_area_y + (max_render_height - final_block_height) / 2

    # Y coordinate for the baseline of the *first* line
    first_baseline_y = block_start_y - final_metrics.fAscent # Offset by ascent

    log_message(f"Rendering at size {final_font_size}. Block Start (X,Y): ({block_start_x:.1f}, {block_start_y:.1f}). First Baseline Y: {first_baseline_y:.1f}", verbose=verbose)

    # --- Build and Draw TextBlobs ---
    with surface as canvas:
        current_baseline_y = first_baseline_y
        for i, line_data in enumerate(final_lines_data):
            infos = line_data["infos"]
            positions = line_data["positions"]
            line_width = line_data["width"]

            if not infos:
                current_baseline_y += final_line_height
                continue

            # Horizontal alignment for this specific line within the block width
            line_start_x = block_start_x + (final_max_line_width - line_width) / 2

            builder = skia.TextBlobBuilder()
            glyph_ids = [info.codepoint for info in infos]
            skia_point_positions = []
            cursor_x = line_start_x

            # HarfBuzz positions are in 26.6 fixed point, convert to pixels
            HB_26_6_SCALE_FACTOR = 64.0
            for _, pos in zip(infos, positions):
                # Apply offsets (converting from 26.6 fixed point)
                glyph_x = cursor_x + (pos.x_offset / HB_26_6_SCALE_FACTOR)
                glyph_y = current_baseline_y - (pos.y_offset / HB_26_6_SCALE_FACTOR) # Y flipped
                skia_point_positions.append(skia.Point(glyph_x, glyph_y))

                # Advance cursor (converting from 26.6 fixed point)
                cursor_x += pos.x_advance / HB_26_6_SCALE_FACTOR

            try:
                builder.allocRunPos(skia_font_final, glyph_ids, skia_point_positions)

                text_blob = builder.make()
                if text_blob:
                    canvas.drawTextBlob(text_blob, 0, 0, paint)
                else:
                    log_message(f"Warning: Failed to build TextBlob for line {i}", verbose=verbose)
            except Exception as e:
                log_message(f"ERROR during Skia rendering for line {i}: {e}", always_print=True)

            current_baseline_y += final_line_height

    # --- Convert back to PIL ---
    final_pil_image = _skia_surface_to_pil(surface)
    if final_pil_image is None:
        log_message("Failed to convert final Skia surface back to PIL.", always_print=True)
        return pil_image, False

    log_message(f"Successfully rendered text at size {final_font_size}", verbose=verbose)
    return final_pil_image, True
