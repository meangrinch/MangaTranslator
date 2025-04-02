import json
import re
import time
import requests

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
                    translation_pattern = re.compile(r'\s*(\d+)\s*:\s*(.*?)(?=\s*\d+\s*:|\s*$)', re.DOTALL)
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