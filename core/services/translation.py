import base64
import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from core.caching import get_cache
from core.config import TranslationConfig
from core.image.image_utils import (cv2_to_pil, pil_to_cv2,
                                    process_bubble_image_cached)
from utils.endpoints import (call_anthropic_endpoint, call_gemini_endpoint,
                             call_openai_compatible_endpoint,
                             call_openai_endpoint, call_openrouter_endpoint,
                             call_xai_endpoint, openrouter_is_reasoning_model)
from utils.exceptions import TranslationError
from utils.logging import log_message

# Regex to find numbered lines in LLM responses
TRANSLATION_PATTERN = re.compile(
    r'^\s*(\d+)\s*:\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*:|\s*$)',
    re.MULTILINE | re.DOTALL,
)


def _build_system_prompt_ocr(
    input_language: Optional[str],
    reading_direction: str,
    num_speech_bubbles: int = 0,
    num_osb_text: int = 0,
) -> str:
    lang_label = f"{input_language} " if input_language else ""
    direction = (
        "right-to-left"
        if (reading_direction or "rtl").lower() == "rtl"
        else "left-to-right"
    )

    return f"""
## ROLE
You are an expert manga/comic OCR transcriber.

## OBJECTIVE
Your sole purpose is to accurately transcribe the original text from a series of provided images. You must not translate, interpret, or add commentary. The images may contain either dialogue text (speech bubbles) or non-dialogue text (text outside speech bubbles).

## CORE RULES
- **Reading Context:** The images are presented in a {direction} reading order.
- **Transcription Policy:** Preserve all original punctuation, ellipses, and casing. Collapse multi-line text into a single line, separated by a single space.
- **Ignore Policy:** You must ignore image borders, speech bubble tails, watermarks, page numbers, and any decorative elements outside the text itself.
- **Language Focus:** Transcribe only the original {lang_label}text.
- **Ruby/Furigana Policy:** If small phonetic characters (ruby/furigana) are present, you must ignore them and transcribe only the main, larger base text.
- **Visual Emphasis Policy:** If the source text is visually emphasized (bold, slanted, etc.), you must mirror that emphasis in your transcription using markdown-style markers: `*italic*` for slanted text, `**bold**` for bold text, `***bold-italic***` for both.
- **Edge Cases:**
  - If an image contains no legible text, you must return the exact token: `[NO TEXT]`.
  - If text is present but indecipherable, you must return the exact token: `[OCR FAILED]`.

## OUTPUT SCHEMA
- You must return your response in separate sections{f': `## SPEECH BUBBLES`' if num_speech_bubbles > 0 else ''}{f' followed by `## NON-DIALOGUE TEXT`' if num_osb_text > 0 else ''}.
- Each section must be a numbered list with exactly one line per image of that type.
- Use S-prefixed numbering for speech bubbles (S1, S2, S3...){f' and N-prefixed numbering for non-dialogue text (N1, N2, N3...)' if num_osb_text > 0 else ''}.
- The format must be `Si: <transcribed {lang_label}text>` for speech bubbles{f' and `Ni: <transcribed {lang_label}text>` for non-dialogue text' if num_osb_text > 0 else ''}.
- If a section has no images, omit that section entirely.
- Do not include any other text, explanations, or formatting outside of these sections.
"""  # noqa


def _build_system_prompt_translation(
    output_language: str, mode: str, num_speech_bubbles: int = 0, num_osb_text: int = 0
) -> str:
    # --- Shared Components ---
    shared_components = f"""
## ROLE
You are a professional manga localization translator and editor.

## OBJECTIVE
Your goal is to produce natural-sounding, high-quality translations in {output_language} that are faithful to the original source's meaning, tone, and visual emphasis. You will handle both dialogue text (speech bubbles) and non-dialogue text (text outside speech bubbles).

## STYLING GUIDE
You must use the following markdown-style markers to convey emphasis:
- `*italic*`: Used for onomatopoeias, thoughts, flashbacks, distant sounds, or dialogue mediated by a device (e.g., phone, radio).
- `**bold**`: Used for sound effects (SFX), shouting, timestamps, or individual emphatic words.
- `***bold-italic***`: Used for extremely loud sounds or dialogue that also meets the criteria for italics (e.g., shouting over a radio).
- **Non-Dialogue Text Styling:** For text outside speech bubbles, use minimal styling and focus on clarity and impact.

## CORE RULES
- **Fidelity:** Stay faithful to the source. Do not add content, explanations, or cultural notes.
- **Conciseness:** Keep translations idiomatic and concise.
- **Emphasis:** If the source text is visually emphasized (bold, slanted, etc.), you must mirror that emphasis using the STYLING GUIDE. Avoid styling text that is merely decorative.
- **Punctuation:** Use standard ASCII quotes and punctuation. Retain ellipses where meaningful.
- **Non-Dialogue Text Translation:** For text categorized as non-dialogue, focus on capturing the feeling and visual metaphor relative to the surrounding panel rather than literal phonetic sounds. Keep these translations concise and impactful.
- **Edge Cases:** If an input line is `[OCR FAILED]` or `[NO TEXT]`, you must output it unchanged.
"""  # noqa

    # --- Mode-Specific Output Schemas ---
    if mode == "one-step":
        # Build the numbering instruction
        numbering_instruction = (
            "Use S-prefixed numbering for speech bubbles (S1, S2, S3...)"
        )
        if num_osb_text > 0:
            numbering_instruction += (
                " and N-prefixed numbering for non-dialogue text (N1, N2, N3...)"
            )
        numbering_instruction += "."

        # Build the sections list
        sections_list = [
            "- `## SPEECH BUBBLES TRANSCRIPTION`",
            "- `## SPEECH BUBBLES TRANSLATION`",
        ]
        if num_osb_text > 0:
            sections_list.extend(
                [
                    "- `## NON-DIALOGUE TEXT TRANSCRIPTION` (if present)",
                    "- `## NON-DIALOGUE TEXT TRANSLATION` (if present)",
                ]
            )

        sections_text = "\n".join(sections_list)

        output_schema = f"""
## OUTPUT SCHEMA
- You must return your response using the following section headings in this exact order:
{sections_text}
- Each section must be a numbered list with exactly one line per image of that type.
- {numbering_instruction}
- If a section has no images, omit that section entirely.
- Do not include any other text, explanations, or formatting outside of these sections.
"""
    elif mode == "two-step":
        # Build conditional sections
        sections = ["`## SPEECH BUBBLES`"]
        if num_osb_text > 0:
            sections.append("`## NON-DIALOGUE TEXT`")

        sections_text = " and ".join(sections) if len(sections) > 1 else sections[0]

        # Build the numbering instruction
        numbering_instruction = (
            "Use S-prefixed numbering for speech bubbles (S1, S2, S3...)"
        )
        if num_osb_text > 0:
            numbering_instruction += (
                " and N-prefixed numbering for non-dialogue text (N1, N2, N3...)"
            )
        numbering_instruction += "."

        # Build the format instruction
        format_instruction = (
            f"`Si: <translated {output_language} text>` for speech bubbles"
        )
        if num_osb_text > 0:
            format_instruction += (
                f" and `Ni: <translated {output_language} text>` for non-dialogue text"
            )
        format_instruction += "."

        output_schema = f"""
## OUTPUT SCHEMA
- You must return your response in separate sections: {sections_text}.
- Each section must be a numbered list with exactly one line per input text of that type.
- {numbering_instruction}
- The format must be {format_instruction}
- If a section has no text, omit that section entirely.
- Do not include any other text, explanations, or formatting.
"""  # noqa
    else:
        raise ValueError(
            f"Invalid mode '{mode}' specified for translation system prompt."
        )

    return shared_components + output_schema


def _is_reasoning_model_google(model_name: str) -> bool:
    """Check if a Google model is reasoning-capable."""
    name = model_name or ""
    return (
        name.startswith("gemini-2.5")
        or "gemini-2.5" in name
        or "gemini-3" in name.lower()
    )


def _is_reasoning_model_openai(model_name: str) -> bool:
    """Check if an OpenAI model is reasoning-capable."""
    lm = (model_name or "").lower()
    return (
        lm.startswith("gpt-5")
        or lm.startswith("o1")
        or lm.startswith("o3")
        or lm.startswith("o4-mini")
    )


def _is_reasoning_model_anthropic(model_name: str) -> bool:
    """Check if an Anthropic model is reasoning-capable."""
    lm = (model_name or "").lower()
    reasoning_prefixes = [
        "claude-opus-4-1",
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3-7-sonnet",
    ]
    return any(lm.startswith(p) for p in reasoning_prefixes)


def _is_reasoning_model_xai(model_name: str) -> bool:
    """Check if an xAI model is reasoning-capable."""
    lm = (model_name or "").lower()
    if not lm:
        return False
    if "non-reasoning" in lm:
        return False
    return "reasoning" in lm


def _is_reasoning_model_openai_compatible(model_name: str) -> bool:
    """Check if an OpenAI-Compatible model is reasoning-capable."""
    lm = (model_name or "").lower()
    return "thinking" in lm or "reasoning" in lm


def _add_media_resolution_to_part(
    part: Dict[str, Any],
    media_resolution_ui: str,
    is_gemini_3: bool,
) -> Dict[str, Any]:
    """
    Add media_resolution to an inline_data part for Gemini 3 models.

    Args:
        part: Part dictionary with inline_data
        media_resolution_ui: UI format media resolution ("auto"/"high"/"medium"/"low")
        is_gemini_3: Whether the model is Gemini 3

    Returns:
        Part dictionary with media_resolution added if Gemini 3, otherwise unchanged
    """
    if not is_gemini_3 or "inline_data" not in part:
        return part

    media_resolution_mapping = {
        "auto": "MEDIA_RESOLUTION_UNSPECIFIED",
        "high": "MEDIA_RESOLUTION_HIGH",
        "medium": "MEDIA_RESOLUTION_MEDIUM",
        "low": "MEDIA_RESOLUTION_LOW",
    }
    backend_media_resolution = media_resolution_mapping.get(
        media_resolution_ui.lower(), "MEDIA_RESOLUTION_UNSPECIFIED"
    )

    # Create a copy to avoid mutating the original
    result = part.copy()
    result["media_resolution"] = {"level": backend_media_resolution}
    return result


def _build_generation_config(
    provider: str,
    model_name: str,
    config: TranslationConfig,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Build provider-specific generation config dictionary.

    Centralizes logic for:
    - Base parameters (temperature, top_p, top_k)
    - Provider-specific parameter names and constraints
    - Reasoning model detection and token limits
    - Special features (thinking, reasoning_effort, etc.)

    Args:
        provider: Provider name (Google, OpenAI, Anthropic, xAI, OpenRouter, OpenAI-Compatible)
        model_name: Model identifier
        config: TranslationConfig with all settings
        debug: Whether to log debug messages

    Returns:
        Dictionary with generation config parameters for the specific provider
    """
    temperature = config.temperature
    top_p = config.top_p
    top_k = config.top_k

    # Determine max_tokens: use config value if provided, otherwise use default logic
    if config.max_tokens is not None:
        max_tokens_value = config.max_tokens
    else:
        # Default logic: 16384 for reasoning models, 4096 otherwise
        is_reasoning = False
        if provider == "Google":
            is_reasoning = _is_reasoning_model_google(model_name)
        elif provider == "OpenAI":
            is_reasoning = _is_reasoning_model_openai(model_name)
        elif provider == "Anthropic":
            is_reasoning = _is_reasoning_model_anthropic(model_name)
        elif provider == "xAI":
            is_reasoning = _is_reasoning_model_xai(model_name)
        elif provider == "OpenRouter":
            is_reasoning = openrouter_is_reasoning_model(model_name, debug)
        elif provider == "OpenAI-Compatible":
            is_reasoning = _is_reasoning_model_openai_compatible(model_name)
        max_tokens_value = 16384 if is_reasoning else 4096

    if provider == "Google":
        is_gemini_3 = "gemini-3" in model_name.lower()
        generation_config = {
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "maxOutputTokens": max_tokens_value,
        }
        # For Gemini 3, media_resolution can be set per-part
        # For other Gemini models, media_resolution is set globally in generation_config
        if not is_gemini_3:
            # Convert UI format to backend format for API
            media_resolution_mapping = {
                "auto": "MEDIA_RESOLUTION_UNSPECIFIED",
                "high": "MEDIA_RESOLUTION_HIGH",
                "medium": "MEDIA_RESOLUTION_MEDIUM",
                "low": "MEDIA_RESOLUTION_LOW",
            }
            backend_media_resolution = media_resolution_mapping.get(
                config.media_resolution.lower(), "MEDIA_RESOLUTION_UNSPECIFIED"
            )
            generation_config["media_resolution"] = backend_media_resolution
        # Handle thinking config for Gemini 3 models
        if is_gemini_3:
            generation_config["thinkingConfig"] = {
                "thinkingLevel": config.thinking_level
            }
            log_message(
                f"Using thinking level '{config.thinking_level}' for {model_name}",
                verbose=debug,
            )
        # Handle thinking config for Gemini 2.5 Flash
        elif "gemini-2.5-flash" in model_name:
            if config.enable_thinking:
                log_message(f"Using thinking mode for {model_name}", verbose=debug)
            else:
                generation_config["thinkingConfig"] = {"thinkingBudget": 0}
                log_message(f"Disabled thinking mode for {model_name}", verbose=debug)
        return generation_config

    elif provider == "OpenAI":
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens_value,
        }  # top_k not supported by OpenAI
        if config.reasoning_effort:
            generation_config["reasoning_effort"] = config.reasoning_effort
        return generation_config

    elif provider == "Anthropic":
        is_reasoning = _is_reasoning_model_anthropic(model_name)
        clamped_temp = min(temperature, 1.0)  # Anthropic caps at 1.0
        return {
            "temperature": clamped_temp,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens_value,
            "anthropic_thinking": bool(config.enable_thinking and is_reasoning),
        }

    elif provider == "xAI":
        is_reasoning = _is_reasoning_model_xai(model_name)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens_value,
        }
        if is_reasoning:
            generation_config["reasoning_tokens"] = max(16384, max_tokens_value)
        return generation_config

    elif provider == "OpenRouter":
        model_lower = (model_name or "").lower()
        is_openai_model = "openai/" in model_lower or model_lower.startswith("gpt-")
        is_anthropic_model = "anthropic/" in model_lower or model_lower.startswith(
            "claude-"
        )
        is_grok_model = "grok" in model_lower
        is_gemini_model = "gemini" in model_lower
        is_gemini_3 = "gemini-3" in model_lower

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens_value,
        }

        # Add reasoning parameters based on underlying model
        if is_openai_model and config.reasoning_effort:
            generation_config["reasoning_effort"] = config.reasoning_effort
        elif is_gemini_3:
            generation_config["thinking_level"] = config.thinking_level
        elif (
            is_anthropic_model
            or is_gemini_model
            or (is_grok_model and "fast" in model_lower)
        ) and config.enable_thinking:
            generation_config["enable_thinking"] = config.enable_thinking

        return generation_config

    elif provider == "OpenAI-Compatible":
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens_value,
        }

    else:
        raise TranslationError(f"Unknown provider for generation config: {provider}")


def sort_bubbles_by_reading_order(detections, reading_direction="rtl"):
    """
    Sort text elements (speech bubbles and OSB text) into a robust reading order.

    Strategy:
    - Group into vertical "row bands" when elements vertically overlap enough or their centers are close.
    - Within each band, group into horizontal "columns" using horizontal overlap/center proximity.
      • Order columns horizontally by reading direction (RTL: right→left, LTR: left→right).
      • Within each column, order elements top→bottom.
    - Across different bands, order bands top→bottom.

    This avoids coarse fixed grids and is resilient to small vertical misalignments.

    Args:
        detections (list): List of detection dicts with a "bbox" key: (x1, y1, x2, y2).
                          Can include both speech bubbles and OSB text elements.
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

    # Heuristics for grouping bubbles into rows and columns
    y_overlap_ratio_threshold = 0.3
    y_center_band_factor = 0.35
    x_overlap_ratio_threshold = 0.3
    x_center_band_factor = 0.35

    rtl = (reading_direction or "rtl").lower() == "rtl"

    # Deterministic row-banding approach to ensure transitive ordering
    enriched = []
    for d in detections:
        x1, y1, x2, y2, w, h, cx, cy = _features(d)
        enriched.append(
            {
                "det": d,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
            }
        )

    enriched.sort(key=lambda e: e["cy"])

    bands = []

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

    bands.sort(key=lambda b: b["y_min"])

    # Sort items within each band using column clustering, then vertical order inside columns
    ordered_enriched = []
    for band in bands:
        items = band["items"]

        columns = []
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

        for col in columns:
            col["items"].sort(key=lambda e: e["cy"])
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
    api_parts = parts + [{"text": prompt_text}]

    try:
        if provider == "Google":
            api_key = config.google_api_key
            if not api_key:
                raise TranslationError("Google API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_gemini_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_grounding=config.enable_grounding,
            )
        elif provider == "OpenAI":
            api_key = config.openai_api_key
            if not api_key:
                raise TranslationError("OpenAI API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
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
                raise TranslationError("Anthropic API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_anthropic_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
            )
        elif provider == "xAI":
            api_key = config.xai_api_key
            if not api_key:
                raise TranslationError("xAI API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_xai_endpoint(
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
                raise TranslationError("OpenRouter API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_openrouter_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_grounding=config.enable_grounding,
            )
        elif provider == "OpenAI-Compatible":
            base_url = config.openai_compatible_url
            api_key = config.openai_compatible_api_key  # Optional
            if not base_url:
                raise TranslationError("OpenAI-Compatible URL is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
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
            raise TranslationError(
                f"Unknown translation provider specified: {provider}"
            )

    except (ValueError, RuntimeError):
        raise


def _parse_llm_response_with_sections(
    response_text: Optional[str],
    speech_bubble_indices: List[int],
    osb_text_indices: List[int],
    provider: str,
    debug: bool = False,
) -> List[str]:
    """Parse LLM response with separate S/N prefixed sections and merge in original order."""
    if response_text is None:
        log_message(f"API call failed: {provider} returned None", always_print=True)
        total_count = len(speech_bubble_indices) + len(osb_text_indices)
        return [f"[{provider}: API failed]"] * total_count
    elif response_text == "":
        log_message(f"API call returned empty response: {provider}", always_print=True)
        total_count = len(speech_bubble_indices) + len(osb_text_indices)
        return [f"[{provider}: Empty response]"] * total_count

    try:
        log_message(
            f"Parsing {provider} response with sections: {len(response_text)} chars",
            verbose=debug,
        )
        log_message(f"Raw response:\n---\n{response_text}\n---", verbose=debug)

        # Parse S-prefixed items (speech bubbles)
        speech_bubble_pattern = re.compile(
            r'^\s*S(\d+)\s*:\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*S\d+\s*:|\s*$)',
            re.MULTILINE | re.DOTALL,
        )
        speech_matches = speech_bubble_pattern.findall(response_text)
        speech_dict = {}
        for num_str, text in speech_matches:
            try:
                num = int(num_str)
                if 1 <= num <= len(speech_bubble_indices):
                    speech_dict[num] = text.strip()
            except ValueError:
                log_message(
                    f"Invalid speech bubble number format: '{num_str}'", verbose=debug
                )

        # Parse N-prefixed items (OSB text)
        osb_pattern = re.compile(
            r'^\s*N(\d+)\s*:\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*N\d+\s*:|\s*$)',
            re.MULTILINE | re.DOTALL,
        )
        osb_matches = osb_pattern.findall(response_text)
        osb_dict = {}
        for num_str, text in osb_matches:
            try:
                num = int(num_str)
                if 1 <= num <= len(osb_text_indices):
                    osb_dict[num] = text.strip()
            except ValueError:
                log_message(
                    f"Invalid OSB text number format: '{num_str}'", verbose=debug
                )

        # Merge results in original order
        final_list = []
        for i in range(len(speech_bubble_indices) + len(osb_text_indices)):
            if i in speech_bubble_indices:
                # This is a speech bubble
                speech_idx = speech_bubble_indices.index(i) + 1
                final_list.append(
                    speech_dict.get(
                        speech_idx, f"[{provider}: Missing speech bubble {speech_idx}]"
                    )
                )
            else:
                # This is OSB text
                osb_idx = osb_text_indices.index(i) + 1
                final_list.append(
                    osb_dict.get(osb_idx, f"[{provider}: Missing OSB text {osb_idx}]")
                )

        log_message(
            f"Parsed {len(speech_matches)} speech bubbles and {len(osb_matches)} OSB texts",
            verbose=debug,
        )
        return final_list

    except Exception as e:
        log_message(
            f"Failed to parse {provider} response with sections: {str(e)}",
            always_print=True,
        )
        total_count = len(speech_bubble_indices) + len(osb_text_indices)
        return [f"[{provider}: Parse error]"] * total_count


def call_translation_api_batch(
    config: TranslationConfig,
    images_b64: List[str],
    full_image_b64: str,
    mime_types: List[str],
    full_image_mime_type: str,
    bubble_metadata: List[Dict[str, Any]],
    debug: bool = False,
) -> List[str]:
    """
    Generates prompts and calls the appropriate LLM API endpoint based on the provider and mode
    specified in the configuration, translating text from speech bubbles and non-dialogue text.

    Supports "one-step" (OCR+Translate+Style) and "two-step" (OCR then Translate+Style) modes.

    Args:
        config (TranslationConfig): Configuration object.
        images_b64 (list): List of base64 encoded images of all text elements, in reading order.
        full_image_b64 (str): Base64 encoded image of the full manga page.
        mime_types (List[str]): List of MIME types for each text element image.
        full_image_mime_type (str): MIME type of the full page image.
        bubble_metadata (List[Dict]): List of metadata dicts with 'is_outside_text' flags for each image.
        debug (bool): Whether to print debugging information.

    Returns:
        list: List of translated strings (potentially with style markers), one for each input text element.
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

    # Separate speech bubbles from OSB text
    speech_bubble_indices = []
    osb_text_indices = []
    for i, metadata in enumerate(bubble_metadata):
        if metadata.get("is_outside_text", False):
            osb_text_indices.append(i)
        else:
            speech_bubble_indices.append(i)

    num_speech_bubbles = len(speech_bubble_indices)
    num_osb_text = len(osb_text_indices)
    total_elements = len(images_b64)

    reading_order_desc = (
        "right-to-left, top-to-bottom"
        if reading_direction == "rtl"
        else "left-to-right, top-to-bottom"
    )

    # Check translation cache first (if deterministic)
    cache = get_cache()
    cache_key = cache.get_translation_cache_key(images_b64, full_image_b64, config)
    cached_translation = cache.get_translation(cache_key)
    if cached_translation is not None:
        log_message("  - Using cached translation", verbose=debug)
        return cached_translation

    # Check if model is Gemini 3 for per-part media_resolution
    model_name = config.model_name
    is_gemini_3 = provider == "Google" and "gemini-3" in model_name.lower()

    base_parts = []
    if config.send_full_page_context and full_image_b64:
        context_part = {
            "inline_data": {
                "mime_type": full_image_mime_type,
                "data": full_image_b64,
            }
        }
        if is_gemini_3:
            context_part = _add_media_resolution_to_part(
                context_part, config.media_resolution_context, is_gemini_3
            )
        base_parts.append(context_part)
    for i, img_b64 in enumerate(images_b64):
        mime_type = mime_types[i] if i < len(mime_types) else "image/jpeg"
        bubble_part = {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        if is_gemini_3:
            bubble_part = _add_media_resolution_to_part(
                bubble_part, config.media_resolution_bubbles, is_gemini_3
            )
        base_parts.append(bubble_part)

    try:
        if translation_mode == "two-step":
            # --- Step 1: OCR ---
            # Build separate sections for speech bubbles and OSB text
            speech_bubble_section = ""
            osb_text_section = ""
            osb_text_context = ""

            if num_speech_bubbles > 0:
                speech_bubble_section = f"""
## SPEECH BUBBLES
You have been provided with {num_speech_bubbles} speech bubble images. These contain dialogue text."""

            if num_osb_text > 0:
                osb_text_section = f"""
## NON-DIALOGUE TEXT
You have been provided with {num_osb_text} non-dialogue text images.
These contain text outside speech bubbles such as sound effects, environmental text, or narrative elements."""
                osb_text_context = f" and {num_osb_text} non-dialogue text images,"

            special_instructions_section = ""
            if config.special_instructions and config.special_instructions.strip():
                special_instructions_section = f"""

## SPECIAL INSTRUCTIONS
{config.special_instructions.strip()}
"""

            ocr_prompt = f"""
## CONTEXT
You have been provided with {num_speech_bubbles} individual dialogue text images{osb_text_context} from a manga page. They are presented in their natural reading order ({reading_order_desc}).

{speech_bubble_section}
{osb_text_section}

## TASK
Apply your OCR transcription rules to each image provided.{special_instructions_section}
"""  # noqa

            log_message("Starting OCR step", verbose=debug)
            ocr_parts = []
            for i, img_b64 in enumerate(images_b64):
                mime_type = mime_types[i] if i < len(mime_types) else "image/jpeg"
                bubble_part = {"inline_data": {"mime_type": mime_type, "data": img_b64}}
                if is_gemini_3:
                    bubble_part = _add_media_resolution_to_part(
                        bubble_part, config.media_resolution_bubbles, is_gemini_3
                    )
                ocr_parts.append(bubble_part)

            ocr_system = _build_system_prompt_ocr(
                input_language, reading_direction, num_speech_bubbles, num_osb_text
            )
            ocr_response_text = _call_llm_endpoint(
                config,
                ocr_parts,
                ocr_prompt,
                debug,
                system_prompt=ocr_system,
            )
            extracted_texts = _parse_llm_response_with_sections(
                ocr_response_text,
                speech_bubble_indices,
                osb_text_indices,
                provider + "-OCR",
                debug,
            )

            if extracted_texts is None:
                log_message("OCR API call failed", always_print=True)
                return [f"[{provider}: OCR failed]"] * total_elements

            is_ocr_failed = all(f"[{provider}-OCR:" in text for text in extracted_texts)
            if is_ocr_failed:
                log_message("OCR returned only placeholders", verbose=debug)
                return extracted_texts

            # --- Step 2: Translation & Styling ---
            log_message("Starting translation step", verbose=debug)

            # Separate speech bubble and OSB text transcriptions
            speech_bubble_texts = []
            osb_texts = []
            ocr_failed_indices = set()

            for i, text in enumerate(extracted_texts):
                if f"[{provider}-OCR:" in text:
                    ocr_failed_indices.add(i)
                    if i in speech_bubble_indices:
                        speech_bubble_texts.append(
                            f"S{speech_bubble_indices.index(i) + 1}: [OCR FAILED]"
                        )
                    else:
                        osb_texts.append(
                            f"N{osb_text_indices.index(i) + 1}: [OCR FAILED]"
                        )
                else:
                    if i in speech_bubble_indices:
                        speech_bubble_texts.append(
                            f"S{speech_bubble_indices.index(i) + 1}: {text}"
                        )
                    else:
                        osb_texts.append(f"N{osb_text_indices.index(i) + 1}: {text}")

            # Build separate input sections
            speech_bubble_section = ""
            osb_text_section = ""
            osb_text_context = ""

            if speech_bubble_texts:
                speech_bubble_section = f"""
### Speech Bubbles
---
{chr(10).join(speech_bubble_texts)}
---"""

            if osb_texts:
                osb_text_section = f"""
### Non-Dialogue Text
---
{chr(10).join(osb_texts)}
---"""
                osb_text_context = f" and {num_osb_text} transcribed non-dialogue texts"

            full_page_context = (
                "A full-page image is also provided for visual and narrative context."
                if config.send_full_page_context
                else ""
            )

            special_instructions_section = ""
            if config.special_instructions and config.special_instructions.strip():
                special_instructions_section = f"""

## SPECIAL INSTRUCTIONS
{config.special_instructions.strip()}
"""

            translation_prompt = f"""
## CONTEXT
You have been provided with a list of {num_speech_bubbles} transcribed dialogue texts{osb_text_context} from a manga page. {full_page_context}

## INPUT DATA
{speech_bubble_section}
{osb_text_section}

## TASK
Apply your translation and styling rules to the text in the `## INPUT DATA` section. 
The target language is {output_language}. Use the appropriate translation approach for each text type.{special_instructions_section}
"""  # noqa

            translation_parts = []
            if config.send_full_page_context and full_image_b64:
                context_part = {
                    "inline_data": {
                        "mime_type": full_image_mime_type,
                        "data": full_image_b64,
                    }
                }
                if is_gemini_3:
                    context_part = _add_media_resolution_to_part(
                        context_part, config.media_resolution_context, is_gemini_3
                    )
                translation_parts.append(context_part)

            translation_system = _build_system_prompt_translation(
                output_language,
                mode="two-step",
                num_speech_bubbles=num_speech_bubbles,
                num_osb_text=num_osb_text,
            )
            translation_response_text = _call_llm_endpoint(
                config,
                translation_parts,
                translation_prompt,
                debug,
                system_prompt=translation_system,
            )
            final_translations = _parse_llm_response_with_sections(
                translation_response_text,
                speech_bubble_indices,
                osb_text_indices,
                provider + "-Translate",
                debug,
            )

            if final_translations is None:
                log_message("Translation API call failed", always_print=True)
                combined_results = []
                for i in range(total_elements):
                    if i in ocr_failed_indices:
                        combined_results.append(f"[{provider}: OCR Failed]")
                    else:
                        combined_results.append(f"[{provider}: Translation failed]")
                return combined_results

            combined_results = []
            for i in range(total_elements):
                if i in ocr_failed_indices:
                    if final_translations[i] == "[OCR FAILED]":
                        combined_results.append("[OCR FAILED]")
                    else:
                        log_message(
                            f"Element {i + 1}: LLM ignored OCR failure instruction",
                            verbose=debug,
                        )
                        combined_results.append("[OCR FAILED]")
                else:
                    combined_results.append(final_translations[i])

            # Cache the result if deterministic
            cache.set_translation(cache_key, combined_results)
            return combined_results

        elif translation_mode == "one-step":
            # --- One-Step Logic ---
            log_message("Starting one-step translation", verbose=debug)
            # Build context-aware prompt pieces
            full_page_context = (
                "A full-page image is also provided for visual and narrative context."
                if config.send_full_page_context
                else ""
            )

            # Build separate sections for speech bubbles and OSB text
            speech_bubble_section = ""
            osb_text_section = ""
            osb_text_context = ""

            if num_speech_bubbles > 0:
                speech_bubble_section = f"""
## SPEECH BUBBLES
You have been provided with {num_speech_bubbles} speech bubble images. These contain dialogue text."""

            if num_osb_text > 0:
                osb_text_section = f"""
## NON-DIALOGUE TEXT
You have been provided with {num_osb_text} non-dialogue text images.
These contain text outside speech bubbles such as sound effects, environmental text, or narrative elements."""
                osb_text_context = f" and {num_osb_text} transcribed non-dialogue texts"

            special_instructions_section = ""
            if config.special_instructions and config.special_instructions.strip():
                special_instructions_section = f"""

## SPECIAL INSTRUCTIONS
{config.special_instructions.strip()}
"""

            one_step_prompt = f"""
## CONTEXT
You have been provided with {num_speech_bubbles} individual dialogue text images{osb_text_context} from a manga page. {full_page_context}

{speech_bubble_section}
{osb_text_section}

## TASK
For each image, you must perform two steps:
1.  **Transcribe:** Extract the original text exactly as it appears.
2.  **Translate:** Translate the text you just transcribed into {output_language}, applying your translation and styling rules.{special_instructions_section}

## OUTPUT FORMAT
You must return your response in separate sections for each text type. Each section must have 
both transcription and translation subsections.

## SPEECH BUBBLES TRANSCRIPTION
S1: <original {input_language} text for speech bubble 1>
S2: <original {input_language} text for speech bubble 2>
...{f"""

## NON-DIALOGUE TEXT TRANSCRIPTION
N1: <original {input_language} text for non-dialogue text 1>
N2: <original {input_language} text for non-dialogue text 2>
...""" if num_osb_text > 0 else ""}

---

## SPEECH BUBBLES TRANSLATION
S1: <{output_language} translation for speech bubble 1>
S2: <{output_language} translation for speech bubble 2>
...{f"""

## NON-DIALOGUE TEXT TRANSLATION
N1: <{output_language} translation for non-dialogue text 1>
N2: <{output_language} translation for non-dialogue text 2>
...""" if num_osb_text > 0 else ""}
"""  # noqa

            one_step_system = _build_system_prompt_translation(
                output_language,
                mode="one-step",
                num_speech_bubbles=num_speech_bubbles,
                num_osb_text=num_osb_text,
            )
            response_text = _call_llm_endpoint(
                config,
                base_parts,
                one_step_prompt,
                debug,
                system_prompt=one_step_system,
            )
            # Prefer parsing only the TRANSLATION section if present
            translation_block = None
            try:
                section_pattern = re.compile(
                    r"^\s*##\s*TRANSLATION\s*$([\s\S]*?)(?=^\s*##\s+|\Z)", re.MULTILINE
                )
                m = section_pattern.search(response_text or "")
                if m:
                    translation_block = m.group(1).strip()
            except Exception:
                translation_block = None

            parse_text = (
                translation_block if translation_block else (response_text or "")
            )
            translations = _parse_llm_response_with_sections(
                parse_text, speech_bubble_indices, osb_text_indices, provider, debug
            )

            if translations is None:
                log_message("One-step API call failed", always_print=True)
                return [f"[{provider}: API failed]"] * total_elements
            else:
                # Cache the result if deterministic
                cache.set_translation(cache_key, translations)
                return translations
        else:
            raise TranslationError(
                f"Unknown translation_mode specified in config: {translation_mode}"
            )
    except (ValueError, RuntimeError) as e:
        log_message(f"Translation error: {e}", always_print=True)
        return [f"[Translation Error: {e}]"] * total_elements


def prepare_bubble_images_for_translation(
    bubble_data: List[Dict[str, Any]],
    original_cv_image: np.ndarray,
    upscale_model: Any,
    device: Any,
    mime_type: str,
    bubble_min_side_pixels: int,
    upscale_method: str = "model",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Prepare bubble images for translation by cropping, upscaling, color matching, and encoding.

    This function processes each speech bubble to prepare it for the translation API:
    1. Crops the bubble from the original image
    2. Upscales the bubble to meet minimum size requirements (based on upscale_method)
    3. Matches colors to preserve visual consistency (only for model upscaling)
    4. Encodes the processed bubble as base64 for API transmission

    Args:
        bubble_data: List of bubble detection dicts with 'bbox' keys
        original_cv_image: OpenCV image array of the original image
        upscale_model: Loaded upscaling model
        device: PyTorch device for model inference
        mime_type: MIME type for image encoding
        upscale_method: Method for upscaling - "model", "lanczos", or "none"
        verbose: Whether to print detailed logging

    Returns:
        List of bubble dicts with added 'image_b64' and 'mime_type' keys
        (immutable approach - returns new list without mutating input)
    """
    # Determine CV2 extension from MIME type
    cv2_ext = ".png" if mime_type == "image/png" else ".jpg"

    prepared_bubbles = []

    if upscale_method == "model":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with 2x-AnimeSharpV4_RCAN",
            always_print=True,
        )
    elif upscale_method == "lanczos":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with LANCZOS",
            always_print=True,
        )
    else:  # upscale_method == "none"
        log_message(
            f"Processing {len(bubble_data)} bubble images without upscaling",
            always_print=True,
        )

    for bubble in bubble_data:
        # Create a copy of the bubble dict to avoid mutating the original
        prepared_bubble = bubble.copy()
        x1, y1, x2, y2 = bubble["bbox"]

        bubble_image_cv = original_cv_image[y1:y2, x1:x2].copy()
        bubble_image_pil = cv2_to_pil(bubble_image_cv)

        # Apply upscaling based on method
        if upscale_method == "model":
            final_bubble_pil = process_bubble_image_cached(
                bubble_image_pil,
                upscale_model,
                device,
                bubble_min_side_pixels,
                "min",
                verbose,
            )
        elif upscale_method == "lanczos":
            w, h = bubble_image_pil.size
            min_side = min(w, h)
            if min_side < bubble_min_side_pixels:
                scale_factor = bubble_min_side_pixels / min_side
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                resized_bubble = bubble_image_pil.resize((new_w, new_h), Image.LANCZOS)
            else:
                resized_bubble = bubble_image_pil
            final_bubble_pil = resized_bubble
        else:  # upscale_method == "none"
            final_bubble_pil = bubble_image_pil

        final_bubble_cv = pil_to_cv2(final_bubble_pil)

        # Encode as base64 for API transmission
        try:
            is_success, buffer = cv2.imencode(cv2_ext, final_bubble_cv)
            if is_success:
                image_b64 = base64.b64encode(buffer).decode("utf-8")
                prepared_bubble["image_b64"] = image_b64
                prepared_bubble["mime_type"] = mime_type
                log_message(f"Bubble {x1},{y1} ({x2 - x1}x{y2 - y1})", verbose=verbose)
            else:
                log_message(
                    f"Failed to encode bubble {bubble['bbox']}", verbose=verbose
                )
                prepared_bubble["image_b64"] = None
        except Exception as e:
            log_message(f"Error encoding bubble {bubble['bbox']}: {e}", verbose=verbose)
            prepared_bubble["image_b64"] = None

        prepared_bubbles.append(prepared_bubble)

    return prepared_bubbles
