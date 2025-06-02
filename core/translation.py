import re
from typing import List, Dict, Any, Optional
from utils.logging import log_message

from core.models import TranslationConfig
from utils.endpoints import (
    call_gemini_endpoint,
    call_openai_endpoint,
    call_anthropic_endpoint,
    call_openrouter_endpoint,
    call_openai_compatible_endpoint,
)

# Regex to find numbered lines in LLM responses
TRANSLATION_PATTERN = re.compile(
    r'^\s*(\d+)\s*:\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*:|\s*$)', re.MULTILINE | re.DOTALL
)


# Helper functions for sorting keys
def _sort_key_ltr(d):
    """Sort key for left-to-right reading order."""
    return (d["grid_pos"][0], d["grid_pos"][1], d["center_y"])


def _sort_key_rtl(d):
    """Sort key for right-to-left reading order."""
    return (d["grid_pos"][0], d["grid_pos"][1], d["center_y"])


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
            sort_key = _sort_key_ltr
        else:  # Default to "rtl" (right-to-left)
            col_idx = grid_cols - 1 - int((center_x / image_width) * grid_cols)
            sort_key = _sort_key_rtl

        detection["grid_pos"] = (row_idx, col_idx)
        detection["center_y"] = center_y

    return sorted(sorted_detections, key=sort_key)


def _call_llm_endpoint(
    config: TranslationConfig, parts: List[Dict[str, Any]], prompt_text: str, debug: bool = False
) -> Optional[str]:
    """Internal helper to dispatch API calls based on provider."""
    provider = config.provider
    model_name = config.model_name
    temperature = config.temperature
    top_p = config.top_p
    top_k = config.top_k
    api_key = ""

    api_parts = parts + [{"text": prompt_text}]

    try:
        if provider == "Gemini":
            api_key = config.gemini_api_key
            if not api_key:
                raise ValueError("Gemini API key is missing.")
            generation_config = {"temperature": temperature, "topP": top_p, "topK": top_k, "maxOutputTokens": 2048}
            # Conditionally add thinkingConfig for specific Gemini models
            if "gemini-2.5-flash" in model_name and config.enable_thinking:
                log_message(f"Including thoughts for {model_name} (thinkingBudget: default)", verbose=debug)
            elif "gemini-2.5-flash" in model_name and not config.enable_thinking:
                generation_config["thinkingConfig"] = {"thinkingBudget": 0}
                log_message(f"Not including thoughts for {model_name} (thinkingBudget: 0)", verbose=debug)

            return call_gemini_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                debug=debug,
            )
        elif provider == "OpenAI":
            api_key = config.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key is missing.")
            generation_config = {"temperature": temperature, "top_p": top_p, "max_output_tokens": 2048}  # top_k ignored
            return call_openai_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                debug=debug,
            )
        elif provider == "Anthropic":
            api_key = config.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key is missing.")
            clamped_temp = min(temperature, 1.0)  # Clamp temp
            generation_config = {"temperature": clamped_temp, "top_p": top_p, "top_k": top_k, "max_tokens": 2048}
            return call_anthropic_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                debug=debug,
            )
        elif provider == "OpenRouter":
            api_key = config.openrouter_api_key
            if not api_key:
                raise ValueError("OpenRouter API key is missing.")
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": 2048,
            }  # Restrictions handled inside endpoint
            return call_openrouter_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                debug=debug,
            )
        elif provider == "OpenAI-Compatible":
            base_url = config.openai_compatible_url
            api_key = config.openai_compatible_api_key  # Optional
            if not base_url:
                raise ValueError("OpenAI-Compatible URL is missing.")
            generation_config = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": 2048}
            return call_openai_compatible_endpoint(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                debug=debug,
            )
        else:
            raise ValueError(f"Unknown translation provider specified: {provider}")

    except (ValueError, RuntimeError):
        raise


def _parse_llm_response(
    response_text: Optional[str], expected_count: int, provider: str, debug: bool = False
) -> Optional[List[str]]:
    """Internal helper to parse numbered list responses from LLM."""
    if response_text is None:
        log_message(f"Parsing response: Received None from {provider} API call.", verbose=debug)
        return None
    elif response_text == "":
        log_message(
            f"Parsing response: Received empty string from {provider} API,"
            f" likely no text detected or processing failed.",
            verbose=debug,
        )
        return [f"[{provider}: No text/empty response]" for _ in range(expected_count)]

    try:
        log_message(f"Parsing response: Raw text received from {provider}:\n---\n{response_text}\n---", verbose=debug)

        matches = TRANSLATION_PATTERN.findall(response_text)
        log_message(
            f"Parsing response: Regex matches found: {len(matches)} vs expected {expected_count}", verbose=debug
        )

        # Fallback for non-numbered lists
        if len(matches) < expected_count:
            log_message(
                f"Parsing response warning: Regex found fewer items ({len(matches)}) than expected ({expected_count}). "
                f"Attempting newline split fallback.",
                verbose=debug,
                always_print=True,
            )
            lines = [line.strip() for line in response_text.split("\n") if line.strip()]
            # Remove potential leading/trailing markdown code blocks
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            if lines and lines[-1].endswith("```"):
                lines.pop(-1)

            # Basic check: does the number of lines match expected bubbles?
            if len(lines) == expected_count:
                log_message(
                    f"Parsing response: Fallback successful. Using {len(lines)} lines as translations.",
                    verbose=debug,
                    always_print=True,
                )
                return lines
            else:
                log_message(
                    f"Parsing response warning: Fallback failed. Newline count ({len(lines)}) "
                    f"still doesn't match expected count ({expected_count}). "
                    f"Proceeding with regex matches found ({len(matches)}).",
                    verbose=debug,
                    always_print=True,
                )

        # Original logic if regex worked or fallback failed
        parsed_dict = {}
        for num_str, text in matches:
            try:
                num = int(num_str)
                if 1 <= num <= expected_count:
                    parsed_dict[num] = text.strip()
                else:
                    log_message(
                        f"Parsing response warning: "
                        f"Parsed number {num} is out of expected range (1-{expected_count}).",
                        verbose=debug,
                    )
            except ValueError:
                log_message(
                    f"Parsing response warning: Could not parse number '{num_str}' in response line.", verbose=debug
                )

        final_list = []
        for i in range(1, expected_count + 1):
            final_list.append(parsed_dict.get(i, f"[{provider}: No text for bubble {i}]"))

        if len(final_list) != expected_count:
            log_message(
                f"Parsing response warning: Expected {expected_count} items, but constructed {len(final_list)}."
                f" Check raw response and parsing logic.",
                verbose=debug,
            )

        return final_list

    except Exception as e:
        log_message(f"Error parsing successful {provider} API response content: {str(e)}", always_print=True)
        # Treat parsing error as a failure for all bubbles
        return [f"[{provider}: Error parsing response]" for _ in range(expected_count)]


def call_translation_api_batch(
    config: TranslationConfig, images_b64: List[str], full_image_b64: str, debug: bool = False
) -> List[str]:
    """
    Generates prompts and calls the appropriate LLM API endpoint based on the provider and mode
    specified in the configuration, translating text from speech bubbles.

    Supports "one-step" (OCR+Translate+Style) and "two-step" (OCR then Translate+Style) modes.

    Args:
        config (TranslationConfig): Configuration object.
        images_b64 (list): List of base64 encoded images of all bubbles, in reading order.
        full_image_b64 (str): Base64 encoded image of the full manga page.
        debug (bool): Whether to print debugging information.

    Returns:
        list: List of translated strings (potentially with style markers), one for each input bubble image.
              Returns placeholder messages on errors or empty responses.

    Raises:
        ValueError: If required config (API key, provider, URL) is missing or invalid.
        RuntimeError: If an API call fails irrecoverably after retries (raised by endpoint functions).
    """
    provider = config.provider
    input_language = config.input_language
    output_language = config.output_language
    reading_direction = config.reading_direction
    translation_mode = config.translation_mode
    num_bubbles = len(images_b64)
    reading_order_desc = (
        "right-to-left, top-to-bottom" if reading_direction == "rtl" else "left-to-right, top-to-bottom"
    )

    # --- Prepare common API input parts (images) ---
    base_parts = [{"inline_data": {"mime_type": "image/jpeg", "data": full_image_b64}}]
    for img_b64 in images_b64:
        base_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

    try:
        if translation_mode == "two-step":
            # --- Step 1: OCR ---
            ocr_prompt = f"""Analyze the {num_bubbles} individual speech bubble images extracted from a manga/comic page in reading order ({reading_order_desc}).
For each individual speech bubble image, *only* extract the original {input_language} text.
Provide your response in this *exact* format, with each extraction on a new line:
1: [Extracted text for bubble 1]
...
{num_bubbles}: [Extracted text for bubble {num_bubbles}]

Do not include translations, explanations, or any other text in your response."""  # noqa

            log_message("--- Starting Two-Step Translation: Step 1 (OCR) ---", verbose=debug)
            # Prepare parts specifically for OCR (only bubble images)
            ocr_parts = []
            for img_b64 in images_b64:
                ocr_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

            ocr_response_text = _call_llm_endpoint(config, ocr_parts, ocr_prompt, debug)
            extracted_texts = _parse_llm_response(ocr_response_text, num_bubbles, provider + "-OCR", debug)

            if extracted_texts is None:
                log_message("OCR API call failed or was blocked. Returning placeholders.", verbose=debug)
                return [f"[{provider}: OCR call failed/blocked]"] * num_bubbles

            is_ocr_failed = all(f"[{provider}-OCR:" in text for text in extracted_texts)
            if is_ocr_failed:
                log_message("OCR parsing resulted in only placeholders. Returning them.", verbose=debug)
                return extracted_texts

            # --- Step 2: Translation & Styling ---
            log_message("--- Starting Two-Step Translation: Step 2 (Translate & Style) ---", verbose=debug)
            formatted_texts_for_prompt = []
            ocr_failed_indices = set()
            for i, text in enumerate(extracted_texts):
                if f"[{provider}-OCR:" in text:
                    formatted_texts_for_prompt.append(f"{i+1}: [OCR FAILED]")
                    ocr_failed_indices.add(i)
                else:
                    formatted_texts_for_prompt.append(f"{i+1}: {text}")
            extracted_text_block = "\n".join(formatted_texts_for_prompt)

            translation_prompt = f"""Analyze the full manga/comic page image provided for context ({reading_order_desc} reading direction).
Then, translate the following extracted speech bubble texts (numbered {1} to {num_bubbles}) from {input_language} to {output_language}, choosing words and sentence structures that accurately convey the original's tone and sound authentic for that specific context in the target language.
Decide if styling should be applied using these markdown-like markers:
- Italic (`*text*`): Use for thoughts, flashbacks, distant sounds, or dialogue via devices.
- Bold (`**text**`): Use for SFX, shouting, and timestamps.
- Bold Italic (`***text***`): Use for loud SFX/dialogue in flashbacks or via devices.
- Use regular text otherwise. You can style parts of text if needed (e.g., "He said **'Stop!'**").

Extracted {input_language} Text:
---
{extracted_text_block}
---

Provide your response in this *exact* format, with each translation on a new line:
1: [Translation for bubble 1]
2: [Translation for bubble 2]
...
{num_bubbles}: [Translation for bubble {num_bubbles}]

For any extraction labeled as "[OCR FAILED]", output exactly "[OCR FAILED]" for that number. Do not attempt to translate it.
Do not include any other text or explanations in your response."""  # noqa

            # Prepare parts for translation call (only full image + prompt)
            translation_parts = [{"inline_data": {"mime_type": "image/jpeg", "data": full_image_b64}}]

            translation_response_text = _call_llm_endpoint(config, translation_parts, translation_prompt, debug)
            final_translations = _parse_llm_response(
                translation_response_text, num_bubbles, provider + "-Translate", debug
            )

            if final_translations is None:
                log_message("Translation API call failed or was blocked.", verbose=debug)
                combined_results = []
                for i in range(num_bubbles):
                    if i in ocr_failed_indices:
                        combined_results.append(f"[{provider}: OCR Failed]")
                    else:
                        combined_results.append(f"[{provider}: Translation call failed/blocked]")
                return combined_results

            combined_results = []
            for i in range(num_bubbles):
                if i in ocr_failed_indices:
                    if final_translations[i] == "[OCR FAILED]":
                        combined_results.append("[OCR FAILED]")
                    else:
                        log_message(
                            f"Warning: LLM translated bubble {i+1} despite [OCR FAILED] instruction."
                            f" Using placeholder.",
                            verbose=debug,
                        )
                        combined_results.append("[OCR FAILED]")

                else:
                    combined_results.append(final_translations[i])
            return combined_results

        elif translation_mode == "one-step":
            # --- One-Step Logic ---
            log_message("--- Starting One-Step Translation ---", verbose=debug)
            one_step_prompt = f"""Analyze the full manga/comic page image provided, followed by the {num_bubbles} individual speech bubble images extracted from it in reading order ({reading_order_desc}).
For each individual speech bubble image:
1. Extract the {input_language} text and translate it to {output_language}, choosing words and sentence structures that accurately convey the original's tone and sound authentic for that specific context in the target language.
2. Decide if styling should be applied to the translated text using these markdown-like markers:
    - Italic (`*text*`): Use for thoughts, flashbacks, distant sounds, or dialogue via devices.
    - Bold (`**text**`): Use for SFX, shouting, and timestamps.
    - Bold Italic (`***text***`): Use for loud SFX/dialogue in flashbacks or via devices.
    - Use regular text otherwise. You can style parts of text if needed (e.g., "He said **'Stop!'**").

Provide your response in this *exact* format, with each translation on a new line:
1: [Translation for bubble 1]
2: [Translation for bubble 2]
...
{num_bubbles}: [Translation for bubble {num_bubbles}]

Do not include any other text or explanations in your response."""  # noqa

            response_text = _call_llm_endpoint(config, base_parts, one_step_prompt, debug)
            translations = _parse_llm_response(response_text, num_bubbles, provider, debug)

            if translations is None:
                log_message("One-step API call failed or was blocked. Returning placeholders.", verbose=debug)
                return [f"[{provider}: API call failed/blocked]"] * num_bubbles
            else:
                return translations
        else:
            raise ValueError(f"Unknown translation_mode specified in config: {translation_mode}")
    except (ValueError, RuntimeError) as e:
        log_message(f"Translation failed: {e}", always_print=True)  # Catch errors raised during config validation
        return [f"[Translation Error: {e}]"] * num_bubbles
