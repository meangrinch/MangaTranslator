import re
from .config import TranslationConfig
from scripts.endpoints import call_gemini_endpoint, call_openai_endpoint, call_anthropic_endpoint, call_openrouter_endpoint, call_openai_compatible_endpoint

def sort_bubbles_by_reading_order(detections, image_height, image_width, reading_direction="rtl"):
    """
    Sort speech bubbles based on specified reading order.

    Args:
        detections (list): List of detection dictionaries, each must have a "bbox" key.
        image_height (int): Height of the full image for reference.
        image_width (int): Width of the full image for reference (used for column calculation).
        reading_direction (str): "rtl" (right-to-left, default) or "ltr" (left-to-right).

    Returns:
        list: Sorted list of detections.
    """
    grid_rows = 3
    grid_cols = 2

    sorted_detections = detections.copy()

    for detection in sorted_detections:
        x1, y1, x2, y2 = detection["bbox"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        row_idx = int((center_y / image_height) * grid_rows)

        if reading_direction == "ltr":
            col_idx = int((center_x / image_width) * grid_cols)
            sort_key = lambda d: (d["grid_pos"][0], d["grid_pos"][1], d["center_y"])
        else: # Default to "rtl" (right-to-left)
            col_idx = grid_cols - 1 - int((center_x / image_width) * grid_cols)
            sort_key = lambda d: (d["grid_pos"][0], d["grid_pos"][1], d["center_y"])

        detection["grid_pos"] = (row_idx, col_idx)
        detection["center_y"] = center_y

    return sorted(sorted_detections, key=sort_key)

def call_translation_api_batch(config: TranslationConfig, images_b64: list, full_image_b64: str, debug=False):
    """
    Generates a prompt and calls the appropriate translation API endpoint based on the provider
    specified in the configuration, translating text from speech bubbles.

    Uses the full page for context and requests font style information via markdown markers.
    Handles parsing the structured response from the API.

    Args:
        config (TranslationConfig): Configuration object containing provider, API keys, model, language settings, etc.
        images_b64 (list): List of base64 encoded images of all bubbles, in reading order.
        full_image_b64 (str): Base64 encoded image of the full manga page.
        debug (bool): Whether to print debugging information.

    Returns:
        list: List of translated strings (potentially with style markers), one for each input bubble image.
              Returns placeholder messages like "[Blocked]" or "[No translation...]" on errors or empty responses.

    Raises:
        ValueError: If the required API key for the selected provider is missing or if the provider is unknown.
        RuntimeError: If the API call fails irrecoverably (raised by the endpoint function)
                      or if parsing the successful API response fails.
    """
    # Extract parameters from config
    provider = config.provider
    api_key = ""
    model_name = config.model_name
    temperature = config.temperature
    top_p = config.top_p
    top_k = config.top_k
    input_language = config.input_language
    output_language = config.output_language
    reading_direction = config.reading_direction

    # --- Prepare Prompt (consistent across providers) ---
    reading_order_desc = "right-to-left, top-to-bottom" if reading_direction == "rtl" else "left-to-right, top-to-bottom"
    prompt_text = f"""Analyze the full manga/comic page image provided, followed by the {len(images_b64)} individual speech bubble images extracted from it in reading order ({reading_order_desc}).
For each individual speech bubble image:
1. Extract the {input_language} text and translate it to {output_language}, making *necessary* adjustments to ensure it flows naturally in the target language and fits the context of the page.
2. Decide if styling should be applied to the translated text using these markdown-like markers:
    - Italic (`*text*`): Use for internal thoughts/monologues, dialogue heard through a device or from a distance, or dialogue/narration in a flashback.
    - Bold (`**text**`): Use for sound effects (SFX), shouting/yelling, or location/time stamps (e.g., '**Three Days Later**').
    - Bold Italic (`***text***`): Use for loud dialogue/SFX in flashbacks or heard through devices/distance.
    - Use regular text otherwise. You can apply styling to parts of the text within a bubble (e.g., "He said **'Stop!'** and then thought *why?*").

Provide your response in this *exact* format, with each translation on a new line:
1: [Translation for bubble 1]
...
{len(images_b64)}: [Translation for bubble {len(images_b64)}]

Do not include any other text or explanations in your response."""

    # --- Prepare API Input Parts (consistent structure) ---
    parts = [
        {"text": prompt_text},
        {"inline_data": {"mime_type": "image/jpeg", "data": full_image_b64}} # Assuming JPEG
    ]
    for i, img_b64 in enumerate(images_b64):
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

    # --- Provider Dispatch Logic ---
    response_text = None
    try:
        if provider == "Gemini":
            api_key = config.gemini_api_key
            if not api_key:
                raise ValueError("Gemini API key is missing in configuration and environment variables.")
            generation_config = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_output_tokens": 2048}
            response_text = call_gemini_endpoint(
                api_key=api_key, model_name=model_name, parts=parts,
                generation_config=generation_config, debug=debug
            )
        elif provider == "OpenAI":
            api_key = config.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key is missing in configuration and environment variables.")
            # OpenAI Chat Completions API ignores top_k
            generation_config = {"temperature": temperature, "top_p": top_p, "max_output_tokens": 2048}
            response_text = call_openai_endpoint(
                api_key=api_key, model_name=model_name, parts=parts,
                generation_config=generation_config, debug=debug
            )
        elif provider == "Anthropic":
            api_key = config.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key is missing in configuration and environment variables.")
            # Clamp temperature to 1.0 for Anthropic
            clamped_temp = min(temperature, 1.0)
            generation_config = {"temperature": clamped_temp, "top_p": top_p, "top_k": top_k, "max_output_tokens": 2048}
            response_text = call_anthropic_endpoint(
                api_key=api_key, model_name=model_name, parts=parts,
                generation_config=generation_config, debug=debug
            )
        elif provider == "OpenRouter":
            api_key = config.openrouter_api_key
            if not api_key:
                raise ValueError("OpenRouter API key is missing in configuration and environment variables.")
            # Parameter restrictions (temp clamp, no top_k for OpenAI/Anthropic models) are handled *inside* call_openrouter_endpoint
            generation_config = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_output_tokens": 2048}
            response_text = call_openrouter_endpoint(
                api_key=api_key, model_name=model_name, parts=parts,
                generation_config=generation_config, debug=debug
            )
        elif provider == "OpenAI-Compatible":
            base_url = config.openai_compatible_url
            api_key = config.openai_compatible_api_key # Can be None or empty string
            if not base_url:
                raise ValueError("OpenAI-Compatible URL is missing in configuration.")
            # Compatible endpoints generally follow OpenAI's parameter structure (no top_k)
            generation_config = {"temperature": temperature, "top_p": top_p, "max_output_tokens": 2048}
            response_text = call_openai_compatible_endpoint(
                base_url=base_url, api_key=api_key, model_name=model_name, parts=parts,
                generation_config=generation_config, debug=debug
            )
        else:
            raise ValueError(f"Unknown translation provider specified: {provider}")

        # --- Process Response (consistent parsing logic) ---
        if response_text is None:
            return [f"[{provider}: Blocked or No Response]"] * len(images_b64) # Placeholder for error
        elif response_text == "":
             if debug: print(f"Received empty response from {provider} API, likely no text detected.")
             return [f"[{provider}: No text detected or translation failed]"] * len(images_b64) # Placeholder for empty

        try:
            translations = []
            # Regex to find numbered lines
            translation_pattern = re.compile(r'^\s*(\d+)\s*:\s*(.*?)(?=\s*\n\s*\d+\s*:|\s*$)', re.MULTILINE | re.DOTALL)
            matches = translation_pattern.findall(response_text)

            if debug:
                print(f"Raw response text received from {provider}:\n---\n{response_text}\n---")
                print(f"Regex matches found: {len(matches)}")

            translations_dict = {}
            for num_str, text in matches:
                 try:
                     num = int(num_str)
                     translations_dict[num] = text.strip()
                 except ValueError:
                     if debug: print(f"Warning: Could not parse number '{num_str}' in response line.")

            # Ensure we have a result for every input bubble, even if missing from response
            for i in range(1, len(images_b64) + 1):
                translations.append(translations_dict.get(i, f"[{provider}: No translation for bubble {i}]")) # Placeholder for missing bubble

            if len(translations) != len(images_b64) and debug:
                 print(f"Warning: Expected {len(images_b64)} translations, but constructed {len(translations)}. Check raw response and parsing logic.")

            return translations

        except Exception as e:
            print(f"Error processing successful {provider} API response content: {str(e)}")
            raise RuntimeError(f"Error processing {provider} API response content: {str(e)}\nRaw Response:\n{response_text}") from e

    except (ValueError, RuntimeError) as e:
        print(f"Translation failed using {provider}: {e}")
        raise
