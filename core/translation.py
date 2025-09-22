import re
from typing import Any, Dict, List, Optional

from core.models import TranslationConfig
from utils.endpoints import (
    call_gemini_endpoint,
    call_openai_endpoint,
    call_anthropic_endpoint,
    call_openrouter_endpoint,
    call_openai_compatible_endpoint,
    openrouter_is_reasoning_model,
)
from utils.logging import log_message

# Regex to find numbered lines in LLM responses
TRANSLATION_PATTERN = re.compile(
    r'^\s*(\d+)\s*:\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*:|\s*$)', re.MULTILINE | re.DOTALL
)


def _build_system_prompt_ocr(input_language: Optional[str], reading_direction: str) -> str:
    """System prompt for OCR-only extraction.

    Emphasizes deterministic, structured output and domain rules from the prompt-engineering guide.
    """
    lang_label = (
        f"{input_language} " if input_language and input_language.lower() != "auto" else ""
    )
    direction = (
        "right-to-left" if (reading_direction or "rtl").lower() == "rtl" else "left-to-right"
    )
    return (
        "You are an expert manga/comic OCR transcriber.\n"
        "- Task: For each speech bubble image, transcribe the visible text only; do not translate.\n"
        "- Layout context: Reading order is "
        + direction
        + ".\n"
        "- Text policy: Preserve punctuation, ellipses, and casing. "
        "Collapse multi-line text into a single line with spaces.\n"
        "- Ignore: borders/tails, watermarks, page numbers, and decorations outside the bubble.\n"
        "- "
        + ("Language: Focus on the original " + lang_label + "text.\n" if lang_label else "")
        + "- Ruby/furigana: If present, ignore small ruby readings and keep the main base text.\n"
        "- If a bubble has no legible text, return [NO TEXT]. If the text is unreadable, return [OCR FAILED].\n"
        "- Output format: Return ONLY a numbered list with exactly one line per bubble, "
        "in the form `i: <text>` for i = 1..N, and nothing else."
    )


def _build_system_prompt_translation(output_language: str) -> str:
    """System prompt for translation and styling.

    Applies role prompting and strict output schema per the prompt-engineering guide.
    """
    return (
        "You are a professional manga localization translator and editor.\n"
        "- Goal: Produce natural-sounding "
        + (output_language or "target language")
        + " while staying faithful to meaning and tone.\n"
        "- Style markers (markdown-like):\n"
        "  * Italic (`*text*`): thoughts, flashbacks, distant sounds, device-mediated dialogue.\n"
        "  * Bold (`**text**`): SFX, shouting, timestamps, emphatic words.\n"
        "  * Bold Italic (`***text***`): very loud SFX/dialogue in flashbacks or via devices.\n"
        "- Typography awareness: If the source text appears visually emphasized (bold or heavy weight, "
        "slanted or italic, ALL-CAPS SFX), consider mirroring that emphasis with the markers above "
        "when doing so conveys meaning. Avoid copying styling that is merely decorative.\n"
        "- Do not add content or explanations. Keep translations concise and idiomatic.\n"
        "- Preserve intended emphasis; you may style only the relevant words within a line.\n"
        "- Use ASCII quotes and punctuation; retain ellipses when meaningful.\n"
        "- If a source line is [OCR FAILED] or [NO TEXT], output it unchanged.\n"
        "- Output format: Return ONLY a numbered list with exactly one line per bubble, "
        "in the form `i: <text>` for i = 1..N and nothing else."
    )


def sort_bubbles_by_reading_order(detections, image_height, image_width, reading_direction="rtl"):
    """
    Sort speech bubbles into a robust reading order.

    Strategy:
    - Group into vertical "row bands" when bubbles vertically overlap enough or their centers are close.
    - Within each band, group into horizontal "columns" using horizontal overlap/center proximity.
      • Order columns horizontally by reading direction (RTL: right→left, LTR: left→right).
      • Within each column, order bubbles top→bottom.
    - Across different bands, order bands top→bottom.

    This avoids coarse fixed grids and is resilient to small vertical misalignments.

    Args:
        detections (list): List of detection dicts with a "bbox" key: (x1, y1, x2, y2).
        image_height (int): Full image height (unused, reserved for future heuristics).
        image_width (int): Full image width (unused, reserved for future heuristics).
        reading_direction (str): "rtl" or "ltr".

    Returns:
        list: New list with the same detection dicts, sorted in reading order.
    """

    if not detections:
        return []

    def _features(d: Dict[str, Any]):
        x1, y1, x2, y2 = d["bbox"]
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return x1, y1, x2, y2, w, h, cx, cy

    # Heuristics
    y_overlap_ratio_threshold = 0.3  # Consider same row if >= 30% vertical overlap
    y_center_band_factor = 0.35      # Or if centers are within 35% of the smaller height
    x_overlap_ratio_threshold = 0.3  # Consider same column if >= 30% horizontal overlap
    x_center_band_factor = 0.35      # Or if centers are within 35% of the smaller width
    x_tie_epsilon = 1e-3
    y_tie_epsilon = 1e-3

    rtl = (reading_direction or "rtl").lower() == "rtl"

    def _compare(a: Dict[str, Any], b: Dict[str, Any]) -> int:
        ax1, ay1, ax2, ay2, aw, ah, acx, acy = _features(a)
        bx1, by1, bx2, by2, bw, bh, bcx, bcy = _features(b)

        # Determine if on the same row band
        overlap_v = max(0.0, min(ay2, by2) - max(ay1, by1))
        min_h = max(1.0, min(ah, bh))
        overlap_ratio = overlap_v / min_h
        center_delta_y = abs(acy - bcy)
        same_row = (overlap_ratio >= y_overlap_ratio_threshold) or (center_delta_y <= y_center_band_factor * min_h)

        if same_row:
            if abs(acx - bcx) <= x_tie_epsilon:
                # If horizontally tied, fall back to top→bottom within the row
                if abs(acy - bcy) <= y_tie_epsilon:
                    # Absolute tie: break by leftmost/rightmost to keep deterministic order
                    return -1 if (ax1 + bx1) <= (ax2 + bx2) else 1
                return -1 if acy < bcy else 1
            if rtl:
                return -1 if acx > bcx else 1
            else:
                return -1 if acx < bcx else 1

        # Different rows: top first
        if abs(acy - bcy) <= y_tie_epsilon:
            # If vertical centers are practically equal but not considered same_row, use direction on x
            if rtl:
                return -1 if acx > bcx else 1
            else:
                return -1 if acx < bcx else 1
        return -1 if acy < bcy else 1

    # Deterministic row-banding approach to ensure transitive ordering
    enriched = []
    for d in detections:
        x1, y1, x2, y2, w, h, cx, cy = _features(d)
        enriched.append({
            "det": d,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
        })

    # Sort by vertical center for stable band construction
    enriched.sort(key=lambda e: e["cy"])  # stable vertical ordering

    bands = []  # Each band: {y_min, y_max, items: [enriched indices]}

    for e in enriched:
        y1 = e["y1"]
        y2 = e["y2"]
        h = e["h"]

        best_band_idx = -1
        best_score = -1.0
        for i, band in enumerate(bands):
            band_h = max(1.0, float(band["y_max"] - band["y_min"]))
            overlap_v = max(0.0, min(y2, band["y_max"]) - max(y1, band["y_min"]))
            overlap_ratio = overlap_v / min(h, band_h)
            center_delta_y = abs(e["cy"] - (band["y_min"] + band["y_max"]) / 2.0)
            same_row = (overlap_ratio >= y_overlap_ratio_threshold) or (
                center_delta_y <= y_center_band_factor * min(h, band_h)
            )
            if same_row:
                # Prefer the band with greater overlap
                score = overlap_ratio - (center_delta_y / (h + band_h)) * 0.1
                if score > best_score:
                    best_score = score
                    best_band_idx = i

        if best_band_idx == -1:
            bands.append({"y_min": y1, "y_max": y2, "items": [e]})
        else:
            band = bands[best_band_idx]
            band["items"].append(e)
            band["y_min"] = min(band["y_min"], y1)
            band["y_max"] = max(band["y_max"], y2)

    # Sort bands by top-most y
    bands.sort(key=lambda b: b["y_min"])  # top rows first

    # Sort items within each band using column clustering, then vertical order inside columns
    ordered_enriched = []
    for band in bands:
        items = band["items"]

        # Build columns within this band
        columns = []  # Each: {x_min, x_max, items: [enriched]}
        for e in items:
            x1 = e["x1"]
            x2 = e["x2"]
            w = e["w"]

            best_col_idx = -1
            best_score = -1.0
            for i, col in enumerate(columns):
                col_w = max(1.0, float(col["x_max"] - col["x_min"]))
                overlap_h = max(0.0, min(x2, col["x_max"]) - max(x1, col["x_min"]))
                overlap_ratio = overlap_h / min(w, col_w)
                col_center_x = (col["x_min"] + col["x_max"]) / 2.0
                center_delta_x = abs(e["cx"] - col_center_x)
                same_col = (overlap_ratio >= x_overlap_ratio_threshold) or (
                    center_delta_x <= x_center_band_factor * min(w, col_w)
                )
                if same_col:
                    score = overlap_ratio - (center_delta_x / (w + col_w)) * 0.1
                    if score > best_score:
                        best_score = score
                        best_col_idx = i

            if best_col_idx == -1:
                columns.append({"x_min": x1, "x_max": x2, "items": [e]})
            else:
                col = columns[best_col_idx]
                col["items"].append(e)
                col["x_min"] = min(col["x_min"], x1)
                col["x_max"] = max(col["x_max"], x2)

        # Sort columns by horizontal position according to reading direction
        if rtl:
            columns.sort(key=lambda c: -((c["x_min"] + c["x_max"]) / 2.0))
        else:
            columns.sort(key=lambda c: ((c["x_min"] + c["x_max"]) / 2.0))

        # Within each column, sort by vertical center (top→bottom)
        for col in columns:
            col["items"].sort(key=lambda e: e["cy"])  # top first
            ordered_enriched.extend(col["items"])

    return [e["det"] for e in ordered_enriched]


def _call_llm_endpoint(
    config: TranslationConfig,
    parts: List[Dict[str, Any]],
    prompt_text: str,
    debug: bool = False,
    system_prompt: Optional[str] = None,
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
            # Reasoning models need higher token limits
            is_gemini_25_series = (model_name or "").startswith("gemini-2.5") or "gemini-2.5" in (model_name or "")
            max_output_tokens = 8192 if is_gemini_25_series else 2048
            generation_config = {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_output_tokens,
            }
            # Enable/disable thinking for Gemini 2.5 models
            if "gemini-2.5-flash" in model_name and config.enable_thinking:
                log_message(f"Using thinking mode for {model_name}", verbose=debug)
            elif "gemini-2.5-flash" in model_name and not config.enable_thinking:
                generation_config["thinkingConfig"] = {"thinkingBudget": 0}
                log_message(f"Disabled thinking mode for {model_name}", verbose=debug)

            return call_gemini_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
            )
        elif provider == "OpenAI":
            api_key = config.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key is missing.")
            # Reasoning models need higher token limits
            lm = (model_name or "").lower()
            is_gpt5 = lm.startswith("gpt-5")
            is_reasoning_capable = is_gpt5 or lm.startswith("o1") or lm.startswith("o3") or lm.startswith("o4-mini")
            max_output_tokens = 8192 if is_reasoning_capable else 2048
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
            }  # top_k ignored
            if config.reasoning_effort:
                generation_config["reasoning_effort"] = config.reasoning_effort
            return call_openai_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
            )
        elif provider == "Anthropic":
            api_key = config.anthropic_api_key
            if not api_key:
                raise ValueError("Anthropic API key is missing.")
            clamped_temp = min(temperature, 1.0)  # Anthropic caps temperature at 1.0
            lm = (model_name or "").lower()
            anthropic_reasoning_prefixes = [
                "claude-opus-4-1",
                "claude-opus-4",
                "claude-sonnet-4",
                "claude-3-7-sonnet",
            ]
            is_anthropic_reasoning_model = any(lm.startswith(p) for p in anthropic_reasoning_prefixes)
            max_tokens = 8192 if is_anthropic_reasoning_model else 2048
            generation_config = {
                "temperature": clamped_temp,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "anthropic_thinking": bool(config.enable_thinking and is_anthropic_reasoning_model),
            }
            return call_anthropic_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
            )
        elif provider == "OpenRouter":
            api_key = config.openrouter_api_key
            if not api_key:
                raise ValueError("OpenRouter API key is missing.")
            # Reasoning models need higher token limits
            is_reasoning_model = (
                config.openrouter_reasoning_override
                or openrouter_is_reasoning_model(model_name, debug)
            )
            max_tokens = 8192 if is_reasoning_model else 2048

            # Determine if this is a reasoning-capable model for OpenRouter
            model_lower = (model_name or "").lower()
            is_openai_model = "openai/" in model_lower or model_lower.startswith("gpt-")
            is_anthropic_model = "anthropic/" in model_lower or model_lower.startswith("claude-")
            is_grok_model = "grok" in model_lower
            is_gemini_model = "gemini" in model_lower

            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
            }

            # Add reasoning parameters for OpenRouter
            if is_openai_model and config.reasoning_effort:
                generation_config["reasoning_effort"] = config.reasoning_effort
            elif ((is_anthropic_model or is_gemini_model or (is_grok_model and "fast" in model_lower))
                  and config.enable_thinking):
                generation_config["enable_thinking"] = config.enable_thinking

            return call_openrouter_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
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
                system_prompt=system_prompt,
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
        log_message(f"API call failed: {provider} returned None", always_print=True)
        return None
    elif response_text == "":
        log_message(f"API call returned empty response: {provider}", always_print=True)
        return [f"[{provider}: Empty response]" for _ in range(expected_count)]

    try:
        log_message(f"Parsing {provider} response: {len(response_text)} chars", verbose=debug)
        log_message(f"Raw response:\n---\n{response_text}\n---", verbose=debug)

        matches = TRANSLATION_PATTERN.findall(response_text)
        log_message(f"Found {len(matches)}/{expected_count} numbered items", verbose=debug)

        # Fallback when regex fails to find numbered format
        if len(matches) < expected_count:
            log_message(
                f"Regex parsing incomplete ({len(matches)}/{expected_count}), trying fallback",
                verbose=debug,
            )
            lines = [line.strip() for line in response_text.split("\n") if line.strip()]
            # Clean up markdown code blocks
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            if lines and lines[-1].endswith("```"):
                lines.pop(-1)

            if len(lines) == expected_count:
                log_message(f"Fallback parsing successful: {len(lines)} lines", verbose=debug)
                return lines
            else:
                log_message(
                    f"Fallback failed ({len(lines)}/{expected_count}), using {len(matches)} regex matches",
                    verbose=debug,
                )

        parsed_dict = {}
        for num_str, text in matches:
            try:
                num = int(num_str)
                if 1 <= num <= expected_count:
                    parsed_dict[num] = text.strip()
                else:
                    log_message(f"Number {num} out of range (1-{expected_count})", verbose=debug)
            except ValueError:
                log_message(f"Invalid number format: '{num_str}'", verbose=debug)

        final_list = []
        for i in range(1, expected_count + 1):
            final_list.append(parsed_dict.get(i, f"[{provider}: Incomplete response for bubble {i}]"))

        if len(final_list) != expected_count:
            log_message(f"Item count mismatch: {len(final_list)}/{expected_count}", verbose=debug)

        return final_list

    except Exception as e:
        log_message(f"Failed to parse {provider} response: {str(e)}", always_print=True)
        return [f"[{provider}: Parse error]" for _ in range(expected_count)]


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

    base_parts = []
    if config.send_full_page_context and full_image_b64:
        base_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": full_image_b64}})
    for img_b64 in images_b64:
        base_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

    try:
        if translation_mode == "two-step":
            # --- Step 1: OCR ---
            lang_part = (
                f"the original {input_language} text"
                if input_language and input_language.lower() != "auto"
                else "the original text"
            )
            ocr_prompt = f"""Analyze the {num_bubbles} individual speech bubble images extracted from a manga/comic page in reading order ({reading_order_desc}).
For each bubble image, transcribe only {lang_part}. Collapse multi-line text into one line with spaces.
If a bubble has no legible text, output exactly [NO TEXT]. If the text is unreadable, output exactly [OCR FAILED].

Return ONLY the following numbered lines, one per bubble:
1: <text>
...
{num_bubbles}: <text>"""  # noqa

            log_message("Starting OCR step", verbose=debug)
            ocr_parts = []
            for img_b64 in images_b64:
                ocr_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

            ocr_system = _build_system_prompt_ocr(input_language, reading_direction)
            ocr_response_text = _call_llm_endpoint(
                config,
                ocr_parts,
                ocr_prompt,
                debug,
                system_prompt=ocr_system,
            )
            extracted_texts = _parse_llm_response(ocr_response_text, num_bubbles, provider + "-OCR", debug)

            if extracted_texts is None:
                log_message("OCR API call failed", always_print=True)
                return [f"[{provider}: OCR failed]"] * num_bubbles

            is_ocr_failed = all(f"[{provider}-OCR:" in text for text in extracted_texts)
            if is_ocr_failed:
                log_message("OCR returned only placeholders", verbose=debug)
                return extracted_texts

            # --- Step 2: Translation & Styling ---
            log_message("Starting translation step", verbose=debug)
            formatted_texts_for_prompt = []
            ocr_failed_indices = set()
            for i, text in enumerate(extracted_texts):
                if f"[{provider}-OCR:" in text:
                    formatted_texts_for_prompt.append(f"{i + 1}: [OCR FAILED]")
                    ocr_failed_indices.add(i)
                else:
                    formatted_texts_for_prompt.append(f"{i + 1}: {text}")
            extracted_text_block = "\n".join(formatted_texts_for_prompt)

            preface = (
                "Analyze the full manga/comic page image provided for context "
                f"({reading_order_desc} reading direction)."
                if config.send_full_page_context
                else ""
            )
            from_part = (
                f"from {input_language} to {output_language}"
                if input_language and input_language.lower() != "auto"
                else f"to {output_language}"
            )
            header_label = (
                "Extracted Text:"
                if not input_language or input_language.lower() == "auto"
                else f"Extracted {input_language} Text:"
            )

            translation_prompt = f"""{preface}
Translate the following extracted speech-bubble texts (numbered 1 to {num_bubbles}) {from_part}. Choose wording that preserves meaning, tone, and naturalness in the target language.
Apply the style markers as defined in your instructions.

{header_label}
---
{extracted_text_block}
---

Return ONLY the following numbered lines, one per bubble:
1: <translation>
2: <translation>
...
{num_bubbles}: <translation>

If an input line is exactly "[OCR FAILED]" or "[NO TEXT]", output it unchanged for that number. Do not add any other text."""  # noqa

            translation_parts = []
            if config.send_full_page_context and full_image_b64:
                translation_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": full_image_b64}})

            translation_system = _build_system_prompt_translation(output_language)
            translation_response_text = _call_llm_endpoint(
                config,
                translation_parts,
                translation_prompt,
                debug,
                system_prompt=translation_system,
            )
            final_translations = _parse_llm_response(
                translation_response_text, num_bubbles, provider + "-Translate", debug
            )

            if final_translations is None:
                log_message("Translation API call failed", always_print=True)
                combined_results = []
                for i in range(num_bubbles):
                    if i in ocr_failed_indices:
                        combined_results.append(f"[{provider}: OCR Failed]")
                    else:
                        combined_results.append(f"[{provider}: Translation failed]")
                return combined_results

            combined_results = []
            for i in range(num_bubbles):
                if i in ocr_failed_indices:
                    if final_translations[i] == "[OCR FAILED]":
                        combined_results.append("[OCR FAILED]")
                    else:
                        log_message(f"Bubble {i + 1}: LLM ignored OCR failure instruction", verbose=debug)
                        combined_results.append("[OCR FAILED]")
                else:
                    combined_results.append(final_translations[i])

            return combined_results

        elif translation_mode == "one-step":
            # --- One-Step Logic ---
            log_message("Starting one-step translation", verbose=debug)
            # Build context-aware prompt pieces
            lang_part = (
                f"the original {input_language} text"
                if input_language and input_language.lower() != "auto"
                else "the original text"
            )
            from_part = (
                f"from {input_language} to {output_language}"
                if input_language and input_language.lower() != "auto"
                else f"to {output_language}"
            )
            preface_full = (
                f"Analyze the full manga/comic page image provided, followed by the {num_bubbles} "
                f"individual speech bubble images extracted from it in reading order "
                f"({reading_order_desc})."
            )
            preface_no_full = (
                f"Analyze the {num_bubbles} individual speech bubble images extracted from a manga/comic page "
                f"in reading order ({reading_order_desc})."
            )
            preface = preface_full if config.send_full_page_context else preface_no_full

            one_step_prompt = f"""{preface}
For each speech bubble image:
1) Extract {lang_part}. Collapse multi-line text into one line with spaces.
   - If a bubble has no legible text, use [NO TEXT]. If the text is unreadable, use [OCR FAILED].
2) Translate {from_part} with wording that preserves meaning, tone, and naturalness.
3) Apply the style markers as defined in your instructions.

Return ONLY the following numbered lines, one per bubble:
1: <translation>
2: <translation>
...
{num_bubbles}: <translation>

If the bubble is [NO TEXT] or [OCR FAILED], output that exact tag unchanged for that number. Do not include any explanations."""  # noqa

            one_step_system = _build_system_prompt_translation(output_language)
            response_text = _call_llm_endpoint(
                config,
                base_parts,
                one_step_prompt,
                debug,
                system_prompt=one_step_system,
            )
            translations = _parse_llm_response(response_text, num_bubbles, provider, debug)

            if translations is None:
                log_message("One-step API call failed", always_print=True)
                return [f"[{provider}: API failed]"] * num_bubbles
            else:
                return translations
        else:
            raise ValueError(f"Unknown translation_mode specified in config: {translation_mode}")
    except (ValueError, RuntimeError) as e:
        log_message(f"Translation error: {e}", always_print=True)
        return [f"[Translation Error: {e}]"] * num_bubbles
