import re
import textwrap
from pathlib import Path
from PIL import ImageFont

from .common_utils import log_message

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

def fit_text_to_bubble(draw, text, bbox, font_path, max_font_size=14, min_font_size=8, line_spacing=1.0, verbose=False):
    """
    Fit and render text within the speech bubble by adjusting font size and line wrapping.
    Includes a pre-check for the longest word to potentially skip iterations.

    Args:
        draw: PIL ImageDraw object
        text: Text to render
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        font_path: Path to the font file
        max_font_size: Starting font size
        min_font_size: Minimum font size before giving up
        line_spacing: Multiplier for line spacing
        verbose (bool): Whether to print verbose logging for this function

    Returns:
        bool: True if text was successfully rendered, False otherwise
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        log_message(f"Invalid bubble dimensions: {bbox}", always_print=True)
        return False

    clean_text = ' '.join(text.split())
    if not clean_text:
        return True

    clean_text = clean_text.replace("... ", "...\n")

    padding_ratio = 0.10
    max_bubble_width = width * (1 - padding_ratio)
    max_bubble_height = height * (1 - padding_ratio)

    current_size = max_font_size
    while current_size >= min_font_size:
        try:
            current_font = ImageFont.truetype(font_path, current_size)
        except IOError:
             log_message(f"Could not load font {font_path} at size {current_size}. Skipping size.", always_print=True)
             current_size -= 1
             continue
        
        words = clean_text.split()
        if words:
            longest_word = max(words, key=len)
            try:
                 lw_bbox = current_font.getbbox(longest_word)
                 longest_word_width = lw_bbox[2] - lw_bbox[0]
            except AttributeError:
                 lw_bbox = draw.textbbox((0, 0), longest_word, font=current_font)
                 longest_word_width = lw_bbox[2] - lw_bbox[0]

            if longest_word_width > max_bubble_width:
                log_message(f"Font size {current_size}: Longest word '{longest_word}' ({longest_word_width:.0f}px) > bubble width ({max_bubble_width:.0f}px). Skipping.", verbose=verbose)
                current_size -= 1
                continue

        test_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        avg_char_width = 10 # Default fallback
        try:
            if hasattr(current_font, 'getbbox'):
                avg_bbox = current_font.getbbox(test_text)
                avg_width = avg_bbox[2] - avg_bbox[0]
            else:
                avg_bbox = draw.textbbox((0, 0), test_text, font=current_font)
                avg_width = avg_bbox[2] - avg_bbox[0]

            if len(test_text) > 0 and avg_width > 0:
                avg_char_width = avg_width / len(test_text)
        except Exception as e:
            log_message(f"Could not calculate avg char width for size {current_size}: {e}", verbose=verbose)

        if avg_char_width <= 0:
            avg_char_width = 10 # Ensure avg_char_width is positive
            log_message(f"Calculated avg_char_width was <= 0 for font size {current_size}. Using default.", verbose=verbose)

        chars_per_line = max(1, int((max_bubble_width * 0.98) / avg_char_width))

        wrapped_text = textwrap.fill(
            clean_text,
            width=chars_per_line,
            break_long_words=False,
            break_on_hyphens=True,
            replace_whitespace=True
        )
        lines = wrapped_text.split('\n')

        total_text_width = 0
        total_text_height = 0
        single_line_height = 0
        line_widths = []

        try:
            ascent, descent = current_font.getmetrics()
            single_line_height = ascent + descent
            total_text_height = (len(lines) * single_line_height * line_spacing) - (single_line_height * (line_spacing - 1.0) if len(lines) > 0 else 0)

            for line in lines:
                line_width = 0 # Default width
                try:
                    if hasattr(current_font, 'getbbox'):
                        line_bbox = current_font.getbbox(line)
                        line_width = line_bbox[2] - line_bbox[0]
                    else:
                        line_bbox = draw.textbbox((0, 0), line, font=current_font)
                        line_width = line_bbox[2] - line_bbox[0]
                except Exception as e_inner:
                    log_message(f"Could not get width for line '{line[:20]}...' at size {current_size}: {e_inner}", verbose=verbose)
                line_widths.append(line_width)
                total_text_width = max(total_text_width, line_width)

        except Exception as e:
             log_message(f"Error calculating text dimensions for size {current_size}: {e}", verbose=verbose)
             current_size -= 1
             continue

        if total_text_width <= max_bubble_width and total_text_height <= max_bubble_height:
            log_message(f"Font size {current_size}: Fits (W: {total_text_width:.0f}<={max_bubble_width:.0f}, H: {total_text_height:.0f}<={max_bubble_height:.0f})", verbose=verbose)

            text_block_y_start = y1 + (height - total_text_height) / 2
            current_y = text_block_y_start

            for i, line in enumerate(lines):
                 line_width = line_widths[i]
                 line_x = x1 + (width - line_width) / 2
                 draw_y = current_y

                 draw.text((line_x, draw_y), line, fill="black", font=current_font)

                 current_y += single_line_height * line_spacing
            
            return True

        else:
             log_message(f"Font size {current_size}: Does not fit (W: {total_text_width:.0f} vs {max_bubble_width:.0f}, H: {total_text_height:.0f} vs {max_bubble_height:.0f})", verbose=verbose)

        current_size -= 1

    log_message(f"Could not fit text in bubble even at min size {min_font_size}: '{text[:30]}...'", always_print=True)
    return False