import json
import re
import textwrap
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFont
import requests
import torch
from ultralytics import YOLO

# Global cache
_model_cache = {}

def log_message(message, verbose=False, always_print=False):
    """
    Print a formatted log message
    
    Args:
        message (str): The message to print
        verbose (bool): Whether to print detailed logs
        always_print (bool): Whether to print regardless of verbose setting
    """
    if not verbose and not always_print:
        return
    
    print(f"{message}")

# ----------------------
# Mask Generation Utils
# ----------------------

@lru_cache(maxsize=1)
def get_yolo_model(model_path):
    """
    Get a cached YOLO model or load a new one.
    
    Args:
        model_path (str): Path to the YOLO model file
    
    Returns:
        YOLO: Loaded model instance
    """
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)
    
    return _model_cache[model_path]

def detect_speech_bubbles(image_path, model_path, confidence=0.35, verbose=False, device=None):
    """
    Detect speech bubbles in the given image using YOLO model.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the YOLO segmentation model
        confidence (float): Confidence threshold for detections
        verbose (bool): Whether to show detailed YOLO output
        device (torch.device, optional): The device to run the model on. Autodetects if None.
    
    Returns:
        list: List of dictionaries containing detection information (bbox, class, confidence)
    """
    detections = []
    
    _device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = get_yolo_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading image: {e}")
    
    results = model(image, conf=confidence, device=_device, verbose=verbose)
    
    if len(results) == 0:
        return detections
    
    for i, result in enumerate(results[0]):
        boxes = result.boxes
        
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            conf = float(box.conf[0])
            
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            detection = {
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "class": cls_name,
                "width": x2 - x1,
                "height": y2 - y1
            }
            
            detections.append(detection)
            
            if hasattr(result, 'masks') and result.masks is not None:
                try:
                    masks = result.masks
                    if len(masks) > j:
                        mask_points = masks[j].xy[0]
                        detection["mask_points"] = mask_points.tolist() if hasattr(mask_points, 'tolist') else mask_points
                except (IndexError, AttributeError) as e:
                    pass
    
    return detections

# ----------------------
# Cleaner Utils
# ----------------------

def pil_to_cv2(pil_image):
    """
    Convert PIL Image to OpenCV format (numpy array)
    
    Args:
        pil_image (PIL.Image): PIL Image object
    
    Returns:
        numpy.ndarray: OpenCV image in BGR format
    """
    rgb_image = np.array(pil_image)
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return rgb_image

def cv2_to_pil(cv2_image):
    """
    Convert OpenCV image to PIL Image
    
    Args:
        cv2_image (numpy.ndarray): OpenCV image in BGR format
    
    Returns:
        PIL.Image: PIL Image object
    """
    if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    return Image.fromarray(cv2_image)

def clean_speech_bubbles(
    image_path,
    model_path,
    confidence=0.35,
    pre_computed_detections=None,
    device=None,
    dilation_kernel_size=7,
    dilation_iterations=1,
    threshold_value=210,
    min_contour_area=50,
    closing_kernel_size=7,
    closing_iterations=1,
    closing_kernel_shape=cv2.MORPH_ELLIPSE,
    constraint_erosion_kernel_size=5,
    constraint_erosion_iterations=1
):
    """
    Clean speech bubbles in the given image using YOLO detection and refined masking.

    Args:
        image_path (str): Path to input image.
        model_path (str): Path to YOLO model.
        confidence (float): Confidence threshold for detections.
        pre_computed_detections (list, optional): Pre-computed detections from previous call.
        device (torch.device, optional): The device to run detection model on if needed.
        dilation_kernel_size (int): Kernel size for dilating initial YOLO mask for ROI.
        dilation_iterations (int): Iterations for dilation.
        threshold_value (int or None): Fixed threshold for isolating bright bubble interior. None uses Otsu.
        min_contour_area (int): Minimum area for a contour to be considered part of the bubble interior.
        closing_kernel_size (int): Kernel size for morphological closing to smooth mask and include text.
        closing_iterations (int): Iterations for closing.
        closing_kernel_shape (int): Shape for the closing kernel (e.g., cv2.MORPH_RECT, cv2.MORPH_ELLIPSE).
        constraint_erosion_kernel_size (int): Kernel size for eroding the original YOLO mask for edge constraint.
        constraint_erosion_iterations (int): Iterations for constraint erosion.

    Returns:
        numpy.ndarray: Cleaned image with white bubbles.

    Raises:
        ValueError: If the image cannot be loaded.
        RuntimeError: If model loading or bubble detection fails.
    """
    try:
        pil_image = Image.open(image_path)
        image = pil_to_cv2(pil_image)
        img_height, img_width = image.shape[:2]
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cleaned_image = image.copy()

        detections = pre_computed_detections if pre_computed_detections is not None else detect_speech_bubbles(
            image_path, model_path, confidence, device=device
        )

        bubble_masks = []

        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size)
        )
        closing_kernel = cv2.getStructuringElement(
            closing_kernel_shape, (closing_kernel_size, closing_kernel_size)
        )
        constraint_erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (constraint_erosion_kernel_size, constraint_erosion_kernel_size)
        )

        for detection in detections:
            if "mask_points" not in detection or not detection["mask_points"]:
                print(f"Warning: Skipping detection without mask points: {detection.get('bbox')}")
                continue

            try:
                points_list = detection["mask_points"]
                points = np.array(points_list, dtype=np.float32)

                if len(points.shape) == 3 and points.shape[1] == 1:
                    points_int = np.round(points).astype(int)
                elif len(points.shape) == 2 and points.shape[1] == 2:
                    points_int = np.round(points).astype(int).reshape((-1, 1, 2))
                else:
                    print(f"Warning: Unexpected mask points format for detection {detection.get('bbox')}. Skipping.")
                    continue

                original_yolo_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.fillPoly(original_yolo_mask, [points_int], 255)

                roi_mask = cv2.dilate(original_yolo_mask, dilation_kernel, iterations=dilation_iterations)

                roi_gray = np.zeros_like(img_gray)
                roi_indices = roi_mask == 255
                if not np.any(roi_indices):
                    continue
                roi_gray[roi_indices] = img_gray[roi_indices]

                if threshold_value == -1:
                    if np.count_nonzero(roi_gray) > 0:
                        thresh_val, thresholded_roi = cv2.threshold(roi_gray[roi_indices], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        thresholded_full = np.zeros_like(img_gray)
                        thresholded_full[roi_indices] = (roi_gray[roi_indices] > thresh_val).astype(np.uint8) * 255
                        thresholded_roi = thresholded_full
                    else:
                        thresholded_roi = np.zeros_like(img_gray)
                else:
                    _, thresholded_roi = cv2.threshold(roi_gray, threshold_value, 255, cv2.THRESH_BINARY)

                thresholded_roi_refined = cv2.bitwise_and(thresholded_roi, original_yolo_mask)

                contours, _ = cv2.findContours(thresholded_roi_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                final_mask = None
                if contours:
                    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                    if valid_contours:
                        largest_contour = max(valid_contours, key=cv2.contourArea)

                        interior_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                        cv2.drawContours(interior_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

                        closed_mask = cv2.morphologyEx(interior_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations)

                        eroded_constraint_mask = cv2.erode(original_yolo_mask, constraint_erosion_kernel, iterations=constraint_erosion_iterations)

                        final_mask = cv2.bitwise_and(closed_mask, eroded_constraint_mask)

                if final_mask is not None:
                    bubble_masks.append(final_mask)

            except Exception as e:
                print(f"Error processing mask for detection {detection.get('bbox')}: {e}")

        if bubble_masks:
            combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for mask in bubble_masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            white_fill = np.ones_like(image) * 255
            cleaned_image = np.where(
                np.expand_dims(combined_mask, axis=2) == 255,
                white_fill,
                cleaned_image
            )

        return cleaned_image

    except IOError as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error cleaning speech bubbles: {str(e)}")

# ----------------------
# Translation Utils
# ----------------------

def sort_bubbles_manga_order(detections, image_width):
    """
    Sort speech bubbles in Japanese manga reading order (top-right to bottom-left).
    
    Args:
        detections (list): List of detection dictionaries
        image_width (int): Width of the full image for reference
    
    Returns:
        list: Sorted list of detections
    """
    grid_rows = 3
    grid_cols = 2
    
    sorted_detections = detections.copy()
    
    for detection in sorted_detections:
        x1, y1, x2, y2 = detection["bbox"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        col_idx = grid_cols - 1 - int((center_x / image_width) * grid_cols)
        row_idx = int((center_y / image_width) * grid_rows)
        
        detection["grid_pos"] = (row_idx, col_idx)
        detection["center_y"] = center_y
    
    return sorted(
        sorted_detections,
        key=lambda d: (d["grid_pos"][0], d["grid_pos"][1], d["center_y"])
    )

def call_gemini_api_batch(api_key, images_b64, input_language="Japanese", output_language="English", 
                          model_name="gemini-2.0-flash", temperature=0.1, top_p=0.95, top_k=1, debug=False):
    """
    Call Gemini API to translate text from all speech bubbles at once.
    
    Args:
        api_key (str): Gemini API key
        images_b64 (list): List of base64 encoded images of all bubbles
        input_language (str): Source language of the text in the image
        output_language (str): Target language for translation
        model_name (str): Gemini model to use for inference
        temperature (float): Controls randomness in the output
        top_p (float): Nucleus sampling parameter
        top_k (int): Controls diversity by limiting to top k tokens
        debug (bool): Whether to print debugging information
        
    Returns:
        list: List of translated strings, one for each input image.

    Raises:
        ValueError: If API key is missing.
        RuntimeError: If API call fails after retries for rate-limited errors
                      or response processing fails.
    """
    if not api_key:
        raise ValueError("API key is required")
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    prompt_text = f"""Analyze the {len(images_b64)} speech bubble images provided below, which are in reading order from a manga page.
For each image, extract the {input_language} text and translate it to {output_language}. 
Provide your response exactly in this format: "1: [translation] 2: [translation]..."
Make sure the dialogue flows naturally in {output_language}, while still being faithful to the original {input_language} text.
Respond with ONLY the formatted translations without any additional text or explanations."""
    
    parts = [{"text": prompt_text}]
    for i, img_b64 in enumerate(images_b64):
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
    
    safety_settings_config = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]
    
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_output_tokens": 2048},
        "safetySettings": safety_settings_config
    }
    
    max_retries = 5
    base_delay = 1.0
    
    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2 ** attempt), 16.0) # Double delay, max 16s
        try:
            if debug:
                print(f"Making batch API request (Attempt {attempt + 1}/{max_retries + 1})...")

            response = requests.post(url, json=payload, timeout=90)

            response.raise_for_status()

            if debug:
                print("API response OK (200), processing result...")
            try:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    response_text = result["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

                    if not response_text and result["candidates"][0].get("finishReason") == "SAFETY":
                         block_reason = result.get("promptFeedback", {}).get("blockReason", "Unknown Safety Reason")
                         print(f"API Error: Content blocked by safety settings ({block_reason})")
                         return [f"[Blocked: {block_reason}]"] * len(images_b64)

                    translations = []
                    translation_pattern = re.compile(r'(\d+):\s*(.*?)(?=\s*\d+:|\s*$)', re.DOTALL)
                    matches = translation_pattern.findall(response_text)
                    
                    if debug:
                        print(f"Raw response text: {response_text}")
                        print(f"Regex matches: {matches}")
                    
                    translations_dict = {int(num): text.strip() for num, text in matches}
                    for i in range(1, len(images_b64) + 1):
                        translations.append(translations_dict.get(i, f"[No translation for bubble {i}]"))

                    if len(translations) != len(images_b64) and debug:
                         print(f"Warning: Expected {len(images_b64)} translations, but parsed {len(translations)}. Check raw response.")

                    return translations
                else:
                    block_reason = result.get("promptFeedback", {}).get("blockReason")
                    if block_reason:
                         print(f"API Warning: Request blocked. Reason: {block_reason}")
                         return [f"[Blocked: {block_reason}]"] * len(images_b64)
                    else:
                         print("API Warning: No candidates found in successful response.")
                         return ["No text detected or translation failed"] * len(images_b64)

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                print(f"Error processing successful API response: {str(e)}")
                raise RuntimeError(f"Error processing successful API response: {str(e)}") from e

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_text = e.response.text[:500]
            
            if status_code == 429 and attempt < max_retries:
                if debug:
                    print(f"API Error: 429 Rate Limit Exceeded. Retrying in {current_delay:.1f} seconds...")
                time.sleep(current_delay)
                continue

            else:
                error_reason = f"Status Code: {status_code}, Response: {error_text}"
                if status_code == 429 and attempt == max_retries:
                     error_reason = f"Persistent rate limiting after {max_retries + 1} attempts. Last error: {error_text}"

                print(f"API Error: {error_reason}")
                raise RuntimeError(f"API Error: {error_reason}") from e 

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                if debug:
                    print(f"API Connection Error: {str(e)}. Retrying in {current_delay:.1f} seconds...")
                time.sleep(current_delay)
                continue
            else:
                print(f"API Connection Error after {max_retries + 1} attempts: {str(e)}")
                raise RuntimeError(f"API Connection Error after retries: {str(e)}") from e

    print(f"Failed to get translation after {max_retries + 1} attempts due to repeated errors.")
    raise RuntimeError(f"Failed to get translation after {max_retries + 1} attempts.")

# ----------------------
# Text Rendering Utils
# ----------------------

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
        avg_char_width = 10 
        try:
            avg_bbox = current_font.getbbox(test_text)
            avg_char_width = (avg_bbox[2] - avg_bbox[0]) / len(test_text)
        except AttributeError:
            try:
                avg_bbox = draw.textbbox((0, 0), test_text, font=current_font)
                avg_width = avg_bbox[2] - avg_bbox[0]
                if len(test_text) > 0:
                     avg_char_width = avg_width / len(test_text)
            except Exception as e:
                 log_message(f"Could not calculate avg char width for size {current_size} using draw.textbbox: {e}", verbose=verbose)

        if avg_char_width <= 0:
            avg_char_width = 10
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
        line_height_metric = 0
        line_widths = []
        
        try:
            ascent, descent = current_font.getmetrics()
            line_height_metric = ascent + descent
            total_text_height = line_height_metric * len(lines) * line_spacing - (line_height_metric * (line_spacing - 1) * (len(lines)>0))

            for line in lines:
                line_width = 0
                try:
                     line_bbox = current_font.getbbox(line)
                     line_width = line_bbox[2] - line_bbox[0]
                except AttributeError:
                     try:
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

                 current_y += line_height_metric * line_spacing
            
            return True

        else:
             log_message(f"Font size {current_size}: Does not fit (W: {total_text_width:.0f} vs {max_bubble_width:.0f}, H: {total_text_height:.0f} vs {max_bubble_height:.0f})", verbose=verbose)

        current_size -= 1

    log_message(f"Could not fit text in bubble even at min size {min_font_size}: '{text[:30]}...'", always_print=True)
    return False
