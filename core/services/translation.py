import base64
import json
import re
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from core.caching import get_cache
from core.config import TranslationConfig, calculate_reasoning_budget
from core.image.image_utils import cv2_to_pil, pil_to_cv2, process_bubble_image_cached
from core.image.ocr_detection import (
    extract_text_with_manga_ocr,
    extract_text_with_paddle_ocr_vl,
)
from utils.endpoints import (
    call_anthropic_endpoint,
    call_deepseek_endpoint,
    call_gemini_endpoint,
    call_meta_model_endpoint,
    call_mimo_endpoint,
    call_moonshot_endpoint,
    call_openai_compatible_endpoint,
    call_openai_endpoint,
    call_opencode_endpoint,
    call_openrouter_endpoint,
    call_qwencloud_endpoint,
    call_xai_endpoint,
    call_zai_endpoint,
    openrouter_is_reasoning_model,
)
from utils.exceptions import TranslationError
from utils.logging import log_message
from utils.model_metadata import (
    anthropic_model_flags,
    get_gpt5_generation,
    get_max_tokens_cap,
    is_anthropic_model_family,
    is_anthropic_reasoning_model,
    is_azure_url,
    is_deepseek_reasoning_model,
    is_gemini_3_model,
    is_gemini_25_flash_model,
    is_gemini_25_pro_model,
    is_gemini_37_flash_model,
    is_gemini_38_flash_model,
    is_gemini_no_sampling_model,
    is_gemma_model,
    is_google_model_family,
    is_google_reasoning_model,
    is_gpt5_chat_variant,
    is_gpt5_series,
    is_gpt6_astra,
    is_hy_mt2_model,
    is_meta_reasoning_model,
    is_mimo_reasoning_model,
    is_moonshot_reasoning_model,
    is_openai_compatible_reasoning_model,
    is_openai_model_family,
    is_openai_reasoning_model,
    is_openai_virtual_pro,
    is_qwencloud_reasoning_model,
    is_rosetta_model,
    is_xai_reasoning_model,
    is_zai_reasoning_model,
    supports_gpt5_max_effort,
    supports_gpt5_xhigh_effort,
    supports_meta_reasoning_effort,
    supports_moonshot_reasoning_effort,
    supports_openai_original_image_detail,
    supports_openai_verbosity,
    supports_qwencloud_reasoning_effort,
    supports_xai_reasoning_parameter,
    supports_zai_reasoning_effort,
)
from utils.model_metadata import is_openai_reasoning_model as _is_openai_reasoning_meta

TRANSLATION_PATTERN = re.compile(
    r'^\s*(\d+)\s*:\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*:|\s*$)',
    re.MULTILINE | re.DOTALL,
)


def _build_system_prompt_ocr(
    input_language: str | None,
) -> str:
    lang_label = f"{input_language} " if input_language else ""
    lang_clean = (input_language or "").strip().lower()
    primary_code = lang_clean.split("-")[0].split("_")[0]
    no_space_keywords = ("japanese", "chinese", "mandarin", "cantonese")
    no_space_codes = {"ja", "zh", "jpn", "chi", "zho"}
    is_no_space_lang = bool(
        any(kw in lang_clean for kw in no_space_keywords)
        or primary_code in no_space_codes
    )

    spacing_rule = (
        "Do not insert spaces between collapsed lines unless an explicit space existed in the original text."
        if is_no_space_lang
        else "Separate collapsed lines with a single space."
    )

    return f"""
## ROLE
You are an expert comic and manga OCR transcriber specializing in comic typography and text extraction.

## OBJECTIVE
Accurately transcribe all original {lang_label}text from the provided cropped images into a single numbered list. Do not translate, interpret, summarize, or add commentary.

## CORE RULES
- **Sequence & Reading Order:** The crops are pre-sorted in narrative reading order (1, 2, 3...). Transcribe each crop in that order. Read horizontal text left-to-right; read vertical text top-to-bottom, right-to-left.
- **Transcription Integrity:** Preserve original punctuation, casing, and pauses. Collapse multi-line text into a single continuous line. {spacing_rule}
- **Furigana/Ruby Policy:** Transcribe only the main, large base characters (Kanji/Hanzi/Hanja). Completely ignore phonetic ruby/furigana characters.
- **Visual Emphasis:** If the source text is visually emphasized, mirror that styling using markdown markers: `*italic*` for slanted or italicized text, `**bold**` for bold text, and `***bold-italic***` for both. Do not invent emphasis for standard upright dialogue.
- **Negative Constraints:**
  - Ignore non-text visual elements (borders, tails, background artwork, watermarks).
  - Do not enclose transcriptions in quotation marks unless they are explicitly present in the image.
- **Edge Cases:**
  - If a bubble contains pauses/ellipses, preserve the pause length using consecutive periods (e.g., single "…" -> "...", double "……" -> "......").
  - If the text in a crop is completely unreadable or contains no text, output the exact token: `[OCR FAILED]`.

## OUTPUT SCHEMA
Output a single numbered list matching the exact number of input crops. No markdown codeblocks, no commentary, no intro/outro text.

Example format:
1: First transcribed line
2: Second transcribed line
3: [OCR FAILED]
"""


def _format_previous_context_prompt_note(
    previous_context_image_count: int,
    previous_context_text_count: int,
    image_order: str,
) -> str:
    has_images = previous_context_image_count > 0
    has_text = previous_context_text_count > 0

    if has_images and has_text:
        return (
            f" {previous_context_image_count} previous source page image(s) are "
            "attached as visual reference, and transcribed text from "
            f"{previous_context_text_count} previous source page(s) is provided "
            "in `## PREVIOUS PAGE TRANSCRIPTS (REFERENCE ONLY)` as narrative reference only — "
            f"do not transcribe, translate, or renumber them. Image order: {image_order}."
        )

    if has_images:
        return (
            f" {previous_context_image_count} previous source page image(s) "
            "are attached as visual reference only — do not transcribe, translate, or renumber them. "
            f"Image order: {image_order}."
        )

    if has_text:
        return (
            f" Transcribed text from {previous_context_text_count} previous "
            "source page(s) is provided in `## PREVIOUS PAGE TRANSCRIPTS (REFERENCE ONLY)` "
            "as narrative reference only — do not translate or renumber it."
        )

    return ""


def _build_system_prompt_translation(
    output_language: str,
    mode: str,
    full_page_context: bool = False,
) -> str:
    input_type = "transcribed text lines" if mode == "two-step" else "cropped images"
    visual_ref_note = (
        " Refer to the full-page context image to resolve visual context, speaker identities, and off-bubble SFX."
        if full_page_context
        else ""
    )

    if mode == "one-step":
        output_schema = f"""
## OUTPUT SCHEMA
Output a single numbered list with exactly one entry per input image. Each entry must provide the transcription and translation separated by the double-pipe delimiter (` || `).
Do not output markdown codeblocks, notes, or explanations.

Example format:
1: Original source text || Translated {output_language} text
2: Second source text || Translated {output_language} text
3: [OCR FAILED] || [OCR FAILED]
"""
    elif mode == "two-step":
        output_schema = f"""
## OUTPUT SCHEMA
Output a single numbered list matching the input numbers (1 to N). Each entry must contain only the translated text.
Do not output markdown codeblocks, notes, or explanations.

Example format:
1: Translated {output_language} text
2: Translated {output_language} text
3: [OCR FAILED]
"""
    else:
        raise ValueError(
            f"Invalid mode '{mode}' specified for translation system prompt."
        )

    edge_cases = (
        "- **Edge Cases:** If the text in an image crop is completely unreadable or contains no text, output `[OCR FAILED] || [OCR FAILED]`."
        if mode == "one-step"
        else "- **Edge Cases:** If an item is marked `[OCR FAILED]`, output `[OCR FAILED]`."
    )

    return f"""
## ROLE
You are a professional comic, manga, and manhwa localization editor translating dialogue, narration, and sound effects into natural, idiomatic {output_language}.

## OBJECTIVE
Deliver natural, character-accurate translations faithful to the source tone, meaning, and subtext while preserving narrative flow.{visual_ref_note}

## LOCALIZATION & TRANSLATION RULES
- **Continuous Narrative:** Treat the {input_type} as a continuous, chronological narrative. Translate functionally rather than literally to preserve natural character voices and narrative flow.
- **Conciseness:** Keep dialogue idiomatic and concise to fit comic lettering constraints.
- **Split Bubbles:** Comic panels frequently split a single sentence across multiple consecutive bubbles. Keep each fragment in its corresponding numbered slot—never merge two numbers into one or leave a slot blank. Do not add ellipses or trailing dots between split fragments unless they were present in the source text.
- **Visual Emphasis:** If the source text or transcription is visually emphasized (bold, slanted/italic, shouting), preserve that emphasis in your translation using the styling tags below. Reflect stressed words in dialogue using `*italic*` or `**bold**` as appropriate.
- **Punctuation & Formatting:**
  - Standardize all ellipses to consecutive periods (`...` or `......`). Never output unicode single-character ellipses (`…`).
  - If a bubble contains only pauses/ellipses, preserve the relative pause length using standard periods (`...` or `......`).
  - Do not wrap dialogue in quotation marks unless quotation marks were explicitly drawn in the original speech.
- **Text Classification:**
  - **Spoken Dialogue:** Translate naturally, matching each character's voice, personality, and social register.
  - **Narration:** Use neutral, descriptive localization without special styling.
  - **Audible Sound Effects (Onomatopoeia / Giongo):** Translate physical sounds into standard target-language onomatopoeia (e.g., **THUD**, **RUMBLE**, **CLANG**).
  - **Atmospheric / Mimetic Effects (Gitaigo):** Translate atmospheric states, moods, or silent actions into concise descriptive verbs or adjectives (e.g., *stare*, *glare*, *twitch*). Do not place a period at the end.

## STYLING GUIDE
Apply the following markdown tags to indicate dialogue delivery and audio type:
- `*italic*`: Used for stressed words in dialogue, internal monologues/thoughts, flashbacks, distant voices, phone/radio transmissions, and atmospheric/mimetic effects (Gitaigo).
- `**bold**`: Used for audible sound effects (Giongo / onomatopoeia), screaming/shouting, timestamps, or heavily emphasized words.
- `***bold-italic***`: Used for screams transmitted over radio/phone, or deafening, climactic sound effects.
- Plain text (no tags): Standard spoken dialogue and narration.

## CONTINUITY & CONTEXT RULES
- **Previous-Page Context (when provided):** Earlier page images or transcripts in the user prompt are reference-only. Never translate or renumber them. Use them to ensure:
  - Character voice, speech registers, pronoun choices, and terminology remain consistent with established usage.
  - Ambiguous pronouns or call-backs are correctly resolved using prior visuals and dialogue.
{edge_cases}

{output_schema.strip()}
"""


def _is_reasoning_model_google(model_name: str) -> bool:
    """Check if a Google model is reasoning-capable."""
    return is_google_reasoning_model(model_name)


def _is_reasoning_model_openai(model_name: str) -> bool:
    """Check if an OpenAI model is reasoning-capable."""
    return _is_openai_reasoning_meta(model_name)


def _is_reasoning_model_anthropic(model_name: str) -> bool:
    """Check if an Anthropic model is reasoning-capable."""
    return is_anthropic_reasoning_model(model_name)


def _add_media_resolution_to_part(
    part: dict[str, Any],
    media_resolution_ui: str,
) -> dict[str, Any]:
    """
    Add media_resolution to an inline_data part.

    Args:
        part: Part dictionary with inline_data
        media_resolution_ui: UI format media resolution ("auto"/"high"/"medium"/"low")

    Returns:
        Part dictionary with media_resolution added
    """
    if "inline_data" not in part:
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

    result = part.copy()
    result["media_resolution"] = {"level": backend_media_resolution}
    return result


def _build_generation_config(
    provider: str,
    model_name: str,
    config: TranslationConfig,
    debug: bool = False,
    prompt_cache_key: str | None = None,
) -> dict[str, Any]:
    """
    Build provider-specific generation config dictionary.

    Centralizes logic for:
    - Base parameters (temperature, top_p, top_k)
    - Provider-specific parameter names and constraints
    - Reasoning model detection and token limits
    - Special features (thinking, reasoning_effort, etc.)

    Args:
        provider: Provider name (Google, OpenAI, Anthropic, SpaceXAI, OpenRouter, OpenAI-Compatible)
        model_name: Model identifier
        config: TranslationConfig with all settings
        debug: Whether to log debug messages

    Returns:
        Dictionary with generation config parameters for the specific provider
    """
    temperature = config.temperature
    top_p = config.top_p
    top_k = config.top_k
    use_sampling = config.use_custom_sampling

    def normalize_image_detail() -> str:
        image_detail = (config.image_detail or "auto").lower()
        if image_detail not in ("auto", "original", "high", "low"):
            image_detail = "auto"
        if image_detail == "original" and not supports_openai_original_image_detail(
            model_name
        ):
            image_detail = "high"
        return image_detail

    if config.max_tokens is not None:
        max_tokens_value = config.max_tokens
    else:
        is_reasoning = False
        if provider == "Google":
            is_reasoning = _is_reasoning_model_google(model_name)
        elif provider == "OpenAI":
            is_reasoning = _is_reasoning_model_openai(model_name)
        elif provider == "Anthropic":
            is_reasoning = _is_reasoning_model_anthropic(model_name)
        elif provider == "SpaceXAI":
            is_reasoning = is_xai_reasoning_model(model_name)
        elif provider == "Meta Model":
            is_reasoning = is_meta_reasoning_model(model_name)
        elif provider == "OpenRouter":
            is_reasoning = openrouter_is_reasoning_model(model_name, debug)
        elif provider == "OpenAI-Compatible":
            is_reasoning = is_openai_compatible_reasoning_model(model_name)
        elif provider == "DeepSeek":
            is_reasoning = is_deepseek_reasoning_model(model_name)
        elif provider == "Z.ai":
            is_reasoning = is_zai_reasoning_model(model_name)
        elif provider == "Moonshot AI":
            is_reasoning = is_moonshot_reasoning_model(model_name)
        elif provider == "Xiaomi MiMo":
            is_reasoning = is_mimo_reasoning_model(model_name)
        elif provider == "QwenCloud":
            is_reasoning = is_qwencloud_reasoning_model(model_name)
        elif provider == "OpenCode":
            is_reasoning = (
                is_deepseek_reasoning_model(model_name)
                or is_qwencloud_reasoning_model(model_name)
                or is_zai_reasoning_model(model_name)
                or is_openai_reasoning_model(model_name)
                or is_moonshot_reasoning_model(model_name)
                or is_mimo_reasoning_model(model_name)
                or is_xai_reasoning_model(model_name)
                or is_meta_reasoning_model(model_name)
                or is_anthropic_reasoning_model(model_name)
            )
        max_tokens_value = 16384 if is_reasoning else 4096

    max_tokens_cap = get_max_tokens_cap(provider, model_name)
    if max_tokens_cap is not None and max_tokens_value > max_tokens_cap:
        max_tokens_value = max_tokens_cap

    if provider == "Google":
        is_gemini_3 = is_gemini_3_model(model_name)
        is_gemma = is_gemma_model(model_name)
        generation_config = {
            "maxOutputTokens": max_tokens_value,
        }
        if use_sampling and not is_gemini_no_sampling_model(model_name):
            generation_config.update(
                {
                    "temperature": temperature,
                    "topP": top_p,
                    "topK": top_k,
                }
            )
        if not is_gemini_3:
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
        if is_gemini_3 or is_gemma:
            reasoning_effort = config.reasoning_effort or "high"
            if (
                is_gemini_37_flash_model(model_name)
                or is_gemini_38_flash_model(model_name)
            ) and reasoning_effort == "minimal":
                reasoning_effort = "high"
            generation_config["thinkingConfig"] = {"thinkingLevel": reasoning_effort}
            log_message(
                f"Using reasoning effort '{reasoning_effort}' for {model_name}",
                verbose=debug,
            )
        elif _is_reasoning_model_google(model_name) and not is_gemini_3:
            reasoning_effort = config.reasoning_effort or "auto"
            is_flash = is_gemini_25_flash_model(model_name)
            is_pro = is_gemini_25_pro_model(model_name)
            if reasoning_effort == "none":
                if is_flash:
                    generation_config["thinkingConfig"] = {"thinkingBudget": 0}
                    log_message(f"Disabled reasoning for {model_name}", verbose=debug)
                elif is_pro:
                    generation_config["thinkingConfig"] = {"thinkingBudget": 128}
                    log_message(
                        f"Using 'none' reasoning effort (thinkingBudget: 128) for {model_name}",
                        verbose=debug,
                    )
                else:
                    log_message(
                        f"Warning: 'none' not supported for {model_name}, using 'auto'",
                        verbose=debug,
                    )
            elif reasoning_effort == "auto":
                log_message(
                    f"Using auto reasoning allocation for {model_name}", verbose=debug
                )
            else:
                thinking_budget = calculate_reasoning_budget(
                    max_tokens_value, reasoning_effort
                )
                generation_config["thinkingConfig"] = {
                    "thinkingBudget": thinking_budget
                }
                log_message(
                    f"Using reasoning effort '{reasoning_effort}' (budget: {thinking_budget} tokens) for {model_name}",
                    verbose=debug,
                )
        return generation_config

    elif provider == "OpenAI":
        generation_config = {
            "max_output_tokens": max_tokens_value,
        }
        if use_sampling:
            generation_config.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )  # top_k not supported by OpenAI
        generation_config["image_detail"] = normalize_image_detail()
        is_chat = is_gpt5_chat_variant(model_name)
        if _is_reasoning_model_openai(model_name) and not is_chat:
            gen = get_gpt5_generation(model_name)
            reasoning_effort = config.reasoning_effort or "high"
            effort = reasoning_effort
            if is_gpt6_astra(model_name):
                if effort in ("none", "minimal"):
                    effort = "low"
                generation_config["reasoning_effort"] = effort
            else:
                if effort == "max" and not supports_gpt5_max_effort(model_name):
                    effort = (
                        "xhigh" if supports_gpt5_xhigh_effort(model_name) else "high"
                    )
                if effort == "xhigh" and not supports_gpt5_xhigh_effort(model_name):
                    effort = "high"
                none_capable = gen is not None and gen != "5"
                if none_capable or effort != "none":
                    generation_config["reasoning_effort"] = effort
            if is_openai_virtual_pro(model_name):
                generation_config["reasoning_mode"] = "pro"
        if supports_openai_verbosity(model_name):
            generation_config["verbosity"] = config.verbosity or "low"
        return generation_config

    elif provider == "Anthropic":
        is_reasoning = _is_reasoning_model_anthropic(model_name)
        anthropic_flags = anthropic_model_flags(model_name)
        generation_config = {
            "max_tokens": max_tokens_value,
            "_metadata": dict(anthropic_flags),
        }
        if use_sampling:
            generation_config.update(
                {
                    "temperature": min(temperature, 1.0),
                    "top_k": top_k,
                }
            )
        if is_reasoning:
            omit_thinking = anthropic_flags.get("is_claude_omit_thinking", False)
            adaptive_default = anthropic_flags.get("is_claude_adaptive_default", False)
            reasoning_effort = config.reasoning_effort or (
                "auto" if adaptive_default else "none"
            )
            generation_config["reasoning_effort"] = reasoning_effort
            if adaptive_default and not omit_thinking:
                generation_config["thinking_type"] = (
                    "disabled" if reasoning_effort == "none" else "adaptive"
                )
            elif anthropic_flags.get("is_claude_effort_max") and not omit_thinking:
                if reasoning_effort == "auto":
                    generation_config["thinking_type"] = "adaptive"
            elif not omit_thinking and reasoning_effort != "none":
                generation_config["thinking_type"] = "enabled"
        if anthropic_flags and config.effort:
            generation_config["effort"] = config.effort
        return generation_config

    elif provider == "SpaceXAI":
        generation_config = {
            "max_tokens": max_tokens_value,
            "media_resolution": config.media_resolution,
        }
        if prompt_cache_key:
            generation_config["prompt_cache_key"] = prompt_cache_key
        if use_sampling:
            generation_config.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        if supports_xai_reasoning_parameter(model_name):
            reasoning_effort = config.reasoning_effort or "high"
            generation_config["reasoning_effort"] = reasoning_effort
        return generation_config

    elif provider == "DeepSeek":
        is_reasoning = is_deepseek_reasoning_model(model_name)
        generation_config = {
            "max_tokens": max_tokens_value,
        }
        if use_sampling:
            generation_config.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        if is_reasoning:
            reasoning_effort = config.reasoning_effort or "high"
            thinking_type = "enabled" if reasoning_effort != "none" else "disabled"
            generation_config["thinking"] = {"type": thinking_type}
            if thinking_type == "enabled":
                generation_config["reasoning_effort"] = reasoning_effort
        return generation_config

    elif provider == "Z.ai":
        is_reasoning = is_zai_reasoning_model(model_name)
        generation_config = {
            "max_tokens": max_tokens_value,
        }
        if use_sampling:
            generation_config.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        if is_reasoning:
            lm = (model_name or "").lower()
            if "glm-5.3" in lm:
                reasoning_effort = config.reasoning_effort or "high"
                generation_config["thinking"] = {"type": "enabled"}
                generation_config["reasoning_effort"] = reasoning_effort
            else:
                reasoning_effort = config.reasoning_effort or (
                    "high" if supports_zai_reasoning_effort(model_name) else "auto"
                )
                thinking_type = "enabled" if reasoning_effort != "none" else "disabled"
                generation_config["thinking"] = {"type": thinking_type}
                if thinking_type == "enabled" and supports_zai_reasoning_effort(
                    model_name
                ):
                    generation_config["reasoning_effort"] = reasoning_effort
        return generation_config

    elif provider == "Moonshot AI":
        is_reasoning = is_moonshot_reasoning_model(model_name)

        generation_config = {
            "max_tokens": max_tokens_value,
        }
        if use_sampling:
            generation_config.update(
                {
                    "temperature": min(temperature, 1.0),
                    "top_p": top_p,
                }
            )

        if is_reasoning:
            if supports_moonshot_reasoning_effort(model_name):
                reasoning_effort = config.reasoning_effort or "high"
                if reasoning_effort not in ("low", "high", "max"):
                    reasoning_effort = "high"
                generation_config["reasoning_effort"] = reasoning_effort
                log_message(
                    f"Using reasoning effort '{reasoning_effort}' for {model_name}",
                    verbose=debug,
                )
            else:
                reasoning_effort = config.reasoning_effort or "auto"
                thinking_type = "enabled" if reasoning_effort != "none" else "disabled"
                generation_config["thinking"] = {"type": thinking_type}
        return generation_config

    elif provider == "Xiaomi MiMo":
        is_reasoning = is_mimo_reasoning_model(model_name)

        generation_config = {
            "max_tokens": max_tokens_value,
        }
        if use_sampling:
            generation_config.update(
                {
                    "temperature": min(temperature, 1.0),
                    "top_p": top_p,
                }
            )

        if is_reasoning:
            reasoning_effort = config.reasoning_effort or "auto"
            thinking_type = "enabled" if reasoning_effort != "none" else "disabled"
            generation_config["thinking"] = {"type": thinking_type}
        return generation_config

    elif provider == "QwenCloud":
        is_reasoning = is_qwencloud_reasoning_model(model_name)

        generation_config = {
            "max_tokens": max_tokens_value,
        }
        if use_sampling:
            generation_config.update(
                {
                    "temperature": min(temperature, 1.0),
                    "top_p": top_p,
                }
            )

        if is_reasoning:
            reasoning_effort = config.reasoning_effort or (
                "xhigh" if supports_qwencloud_reasoning_effort(model_name) else "auto"
            )
            thinking_type = "enabled" if reasoning_effort != "none" else "disabled"
            generation_config["thinking"] = {"type": thinking_type}
            if (
                thinking_type == "enabled"
                and reasoning_effort not in ("auto", "none")
                and supports_qwencloud_reasoning_effort(model_name)
            ):
                generation_config["reasoning_effort"] = reasoning_effort
        return generation_config

    elif provider == "Meta Model":
        is_reasoning = is_meta_reasoning_model(model_name)

        generation_config = {
            "max_tokens": max_tokens_value,
        }
        if use_sampling:
            generation_config.update(
                {
                    "temperature": min(temperature, 1.0),
                    "top_p": top_p,
                }
            )

        if is_reasoning:
            reasoning_effort = config.reasoning_effort or "auto"
            if reasoning_effort not in (
                "auto",
                "none",
            ) and supports_meta_reasoning_effort(model_name):
                generation_config["reasoning_effort"] = reasoning_effort
        return generation_config

    elif provider == "OpenCode":
        is_reasoning = (
            is_deepseek_reasoning_model(model_name)
            or is_qwencloud_reasoning_model(model_name)
            or is_zai_reasoning_model(model_name)
            or is_openai_reasoning_model(model_name)
            or is_moonshot_reasoning_model(model_name)
            or is_mimo_reasoning_model(model_name)
            or is_xai_reasoning_model(model_name)
            or is_meta_reasoning_model(model_name)
            or is_anthropic_reasoning_model(model_name)
        )

        generation_config = {
            "max_tokens": max_tokens_value,
        }
        if use_sampling:
            generation_config.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

        if (
            is_reasoning
            and config.reasoning_effort
            and config.reasoning_effort != "none"
        ):
            generation_config["reasoning_effort"] = config.reasoning_effort
        return generation_config

    elif provider == "OpenRouter":
        model_lower = (model_name or "").lower()
        is_openai_model = is_openai_model_family(model_name)
        is_anthropic_model = is_anthropic_model_family(model_name)
        is_grok_model = "grok-4" in model_lower
        is_gemini_3 = is_gemini_3_model(model_name)
        is_google_model = is_google_model_family(model_name)

        generation_config = {"max_tokens": max_tokens_value}
        if use_sampling and not is_gemini_no_sampling_model(model_name):
            generation_config.update(
                {
                    "temperature": temperature,
                    "top_p": top_p if not is_anthropic_model else None,
                    "top_k": top_k,
                }
            )
        if is_openai_model:
            generation_config["image_detail"] = normalize_image_detail()

        is_openai_reasoning = is_openai_model and is_openai_reasoning_model(model_name)
        is_gpt5_model = is_openai_model and is_gpt5_series(model_name)
        is_gpt6_model = is_openai_model and is_gpt6_astra(model_name)
        is_gpt5_1 = is_openai_model and "gpt-5.1" in model_lower
        is_gpt5 = is_openai_model and "gpt-5" in model_lower and not is_gpt5_1
        is_anthropic_reasoning = is_anthropic_reasoning_model(model_name)
        # OpenRouter Grok metadata omit explicit reasoning tags in name
        is_grok_reasoning = is_grok_model and "non-reasoning" not in model_lower

        anthropic_flags = anthropic_model_flags(model_name)
        generation_config["_metadata"] = {
            "is_openai_model": is_openai_model,
            "is_anthropic_model": is_anthropic_model,
            "is_grok_model": is_grok_model,
            "is_gemini_3": is_gemini_3,
            "is_google_model": is_google_model,
            "is_gemini_no_sampling": is_gemini_no_sampling_model(model_name),
            "is_openai_reasoning": is_openai_reasoning,
            "is_anthropic_reasoning": is_anthropic_reasoning,
            "is_grok_reasoning": is_grok_reasoning,
            "is_gpt5_1": is_gpt5_1,
            "is_gpt5": is_gpt5,
            "is_gpt5_model": is_gpt5_model,
            "is_gpt6_astra": is_gpt6_model,
            "supports_verbosity": is_openai_model
            and supports_openai_verbosity(model_name),
            **anthropic_flags,
        }

        if is_openai_reasoning or is_anthropic_reasoning or is_grok_reasoning:
            if is_anthropic_reasoning:
                is_claude_adaptive_default = anthropic_flags.get(
                    "is_claude_adaptive_default", False
                )
                is_claude_46 = (
                    anthropic_flags.get("is_claude_effort_max")
                    and not anthropic_flags.get("is_claude_effort_xhigh")
                    and not is_claude_adaptive_default
                )
                reasoning_effort = config.reasoning_effort or (
                    "auto" if (is_claude_46 or is_claude_adaptive_default) else "none"
                )
                generation_config["reasoning_effort"] = reasoning_effort
            elif is_gpt6_model:
                effort = config.reasoning_effort or "high"
                if effort in ("none", "minimal"):
                    effort = "low"
                generation_config["reasoning_effort"] = effort
            elif (
                is_gpt5_1
                or config.reasoning_effort
                and config.reasoning_effort != "none"
            ):
                generation_config["reasoning_effort"] = config.reasoning_effort
        elif is_google_model and config.reasoning_effort:
            effort = config.reasoning_effort
            if (
                is_gemini_37_flash_model(model_name)
                or is_gemini_38_flash_model(model_name)
            ) and effort == "minimal":
                effort = "high"
            generation_config["reasoning_effort"] = effort

        if anthropic_flags and config.effort:
            generation_config["effort"] = config.effort

        if is_openai_model and supports_openai_verbosity(model_name):
            generation_config["verbosity"] = config.verbosity or "low"

        return generation_config

    elif provider == "OpenAI-Compatible":
        is_openai_model = is_openai_model_family(model_name)
        is_anthropic_model = is_anthropic_model_family(model_name)
        is_azure = is_azure_url(config.openai_compatible_url)

        is_openai_reasoning = is_openai_model and _is_openai_reasoning_meta(model_name)
        is_gpt6_model = is_openai_model and is_gpt6_astra(model_name)
        is_anthropic_reasoning = is_anthropic_reasoning_model(model_name)
        anthropic_flags = anthropic_model_flags(model_name)

        generation_config = {"max_tokens": max_tokens_value}
        if (
            use_sampling
            and not is_gemini_no_sampling_model(model_name)
            and not is_openai_reasoning
        ):
            generation_config.update(
                {
                    "temperature": min(temperature, 1.0)
                    if is_anthropic_model
                    else temperature,
                    "top_p": top_p if not is_anthropic_model else None,
                }
            )
            if top_k is not None and not is_openai_model and not is_anthropic_model:
                generation_config["top_k"] = top_k

        if is_openai_model:
            generation_config["image_detail"] = normalize_image_detail()

        generation_config["_metadata"] = {
            "is_openai_model": is_openai_model,
            "is_anthropic_model": is_anthropic_model,
            "is_azure": is_azure,
            "is_openai_reasoning": is_openai_reasoning,
            "is_anthropic_reasoning": is_anthropic_reasoning,
            "is_gemini_no_sampling": is_gemini_no_sampling_model(model_name),
            "is_gpt5_model": is_openai_model and is_gpt5_series(model_name),
            "is_gpt6_astra": is_gpt6_model,
            "supports_verbosity": is_openai_model
            and supports_openai_verbosity(model_name),
            **anthropic_flags,
        }

        if (
            is_openai_reasoning
            or is_anthropic_reasoning
            or is_openai_compatible_reasoning_model(model_name)
        ) and config.reasoning_effort:
            effort = config.reasoning_effort
            if is_gpt6_model and effort in ("none", "minimal"):
                effort = "low"
            generation_config["reasoning_effort"] = effort

        if anthropic_flags and config.effort:
            generation_config["effort"] = config.effort

        if is_openai_model and supports_openai_verbosity(model_name):
            generation_config["verbosity"] = config.verbosity or "low"

        return generation_config

    else:
        raise TranslationError(f"Unknown provider for generation config: {provider}")


def _call_llm_endpoint(
    config: TranslationConfig,
    parts: list[dict[str, Any]],
    prompt_text: str,
    debug: bool = False,
    system_prompt: str | None = None,
    prompt_cache_key: str | None = None,
) -> str | None:
    """Internal helper to dispatch API calls based on provider."""
    provider = config.provider
    model_name = config.model_name
    api_parts = parts + [{"text": prompt_text}]
    coordinator = getattr(config, "request_coordinator", None)
    if coordinator is not None and not coordinator.in_slot():
        return coordinator.run(
            _call_llm_endpoint,
            config,
            parts,
            prompt_text,
            debug,
            system_prompt,
            prompt_cache_key,
        )

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
                enable_web_search=config.enable_web_search,
                enable_code_execution=config.enable_code_execution,
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
                enable_web_search=config.enable_web_search,
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
                enable_web_search=config.enable_web_search,
            )
        elif provider == "SpaceXAI":
            api_key = config.xai_api_key
            if not api_key:
                raise TranslationError("SpaceXAI API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug, prompt_cache_key=prompt_cache_key
            )
            return call_xai_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "DeepSeek":
            api_key = config.deepseek_api_key
            if not api_key:
                raise TranslationError("DeepSeek API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_deepseek_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "Z.ai":
            api_key = config.zai_api_key
            if not api_key:
                raise TranslationError("Z.ai API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_zai_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "Moonshot AI":
            api_key = config.moonshot_api_key
            if not api_key:
                raise TranslationError("Moonshot API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_moonshot_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "Xiaomi MiMo":
            api_key = config.mimo_api_key
            if not api_key:
                raise TranslationError("MiMo API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_mimo_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "QwenCloud":
            api_key = config.qwencloud_api_key
            if not api_key:
                raise TranslationError("QwenCloud API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_qwencloud_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "Meta Model":
            api_key = config.meta_api_key
            if not api_key:
                raise TranslationError("Meta Model API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_meta_model_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "OpenCode":
            api_key = config.opencode_api_key
            if not api_key:
                raise TranslationError("OpenCode API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            tier = getattr(config, "opencode_tier", "zen") or "zen"
            return call_opencode_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                tier=tier,
                debug=debug,
                enable_web_search=config.enable_web_search,
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
                enable_web_search=config.enable_web_search,
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

    except (ValueError, RuntimeError):  # noqa: TRY203
        raise


def _parse_llm_response_unified(
    response_text: str | None,
    total_elements: int,
    provider: str,
    debug: bool = False,
) -> list[str]:
    """Parse LLM response with a single numbered list."""
    if response_text is None:
        log_message(f"API call failed: {provider} returned None", always_print=True)
        raise TranslationError(f"{provider}: API failed (returned None)")
    elif response_text == "":
        log_message(f"API call returned empty response: {provider}", always_print=True)
        raise TranslationError(f"{provider}: Empty response")

    try:
        log_message(
            f"Parsing {provider} unified response: {len(response_text)} chars",
            verbose=debug,
        )
        log_message(f"Raw response:\n---\n{response_text}\n---", always_print=True)

        # Pattern matches "1: text" or "1. text" or "1 text" etc.
        pattern = re.compile(
            r'^\s*(\d+)\s*[:.]\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*[:.]|\s*$)',
            re.MULTILINE | re.DOTALL,
        )

        matches = pattern.findall(response_text)
        result_dict = {}

        for num_str, text in matches:
            try:
                num = int(num_str)
                if 1 <= num <= total_elements:
                    result_dict[num] = text.strip()
            except ValueError:
                continue

        final_list = []
        for i in range(1, total_elements + 1):
            if i in result_dict:
                final_list.append(result_dict[i])
            else:
                final_list.append(f"[{provider}: Missing item {i}]")

        log_message(
            f"Parsed {len(result_dict)} items from unified response (expected {total_elements})",
            verbose=debug,
        )
        return final_list

    except Exception as e:
        log_message(
            f"Failed to parse {provider} unified response: {e!s}",
            always_print=True,
        )
        return [f"[{provider}: Parse error]"] * total_elements


def _prepare_images_for_ocr(
    images_b64: list[str], verbose: bool = False
) -> list[Image.Image | None]:
    """Prepare base64-encoded images for OCR by decoding and converting to RGB.

    Args:
        images_b64: List of base64-encoded image strings
        verbose: Whether to print verbose logging

    Returns:
        List of PIL Images (or None for decode failures), all in RGB mode
    """
    pil_images = []
    for img_b64 in images_b64:
        try:
            image_data = base64.b64decode(img_b64)
            pil_img = Image.open(BytesIO(image_data))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
        except Exception as e:
            log_message(
                f"Failed to decode image for manga-ocr: {e}",
                always_print=True,
            )
            pil_images.append(None)
    return pil_images


def _format_ocr_results(
    extracted_texts: list[str],
    bubble_metadata: list[dict[str, Any]],
) -> None:
    """Format and log OCR results.

    Args:
        extracted_texts: List of extracted text strings
        bubble_metadata: List of metadata dicts for text elements
        verbose: Whether to print verbose logging
    """
    log_lines = []

    for i, text in enumerate(extracted_texts):
        metadata = bubble_metadata[i] if i < len(bubble_metadata) else {}
        is_osb = metadata.get("is_outside_text", False)
        prefix = f"{i + 1}"
        type_label = "[OSB]" if is_osb else "[Bubble]"

        log_lines.append(f"{prefix}: {type_label} {text}")

    if log_lines:
        log_message(
            f"Raw OCR output:\n---\n{chr(10).join(log_lines)}\n---",
            always_print=True,
        )


def _check_ocr_failure(texts: list[str], provider: str | None = None) -> bool:
    """Check if all OCR results indicate failure.

    Args:
        texts: List of extracted text strings
        provider: Optional provider name for LLM OCR failure detection

    Returns:
        True if all texts indicate failure, False otherwise
    """
    if not texts:
        return True

    if provider:
        for text in texts:
            if f"[{provider}-OCR:" not in text:
                return False
        return True
    else:
        return all(text == "[OCR FAILED]" for text in texts)


def _format_previous_context_texts(
    previous_context_texts: list[list[str]] | None,
) -> str:
    """Format previous-page OCR transcripts as a labeled context block.

    Pages are listed oldest-to-newest (matching the previous-image convention).
    Empty/failed entries are omitted, and the section is suppressed entirely
    when no usable transcripts remain.
    """
    if not previous_context_texts:
        return ""

    page_blocks = []
    for page_index, page_texts in enumerate(previous_context_texts, start=1):
        if not page_texts:
            continue
        lines = []
        for idx, text in enumerate(page_texts, start=1):
            cleaned = (text or "").strip()
            if not cleaned or cleaned == "[OCR FAILED]":
                continue
            lines.append(f"{idx}: {cleaned}")
        if not lines:
            continue
        page_blocks.append(f"### Previous Page {page_index}\n" + "\n".join(lines))

    if not page_blocks:
        return ""

    return (
        "\n## PREVIOUS PAGE TRANSCRIPTS (REFERENCE ONLY)\n"
        "Listed oldest-to-newest.\n" + "\n\n".join(page_blocks) + "\n"
    )


def _format_special_instructions(config: TranslationConfig) -> str:
    """Format user's special instructions section for prompts.

    Args:
        config: TranslationConfig with special_instructions

    Returns:
        Formatted special instructions string (empty if none)
    """
    if config.special_instructions and config.special_instructions.strip():
        return f"""

## CUSTOM USER INSTRUCTIONS
{config.special_instructions.strip()}
"""
    return ""


def _build_rosetta_instruction(
    output_language: str,
    special_instructions: str | None = None,
) -> str:
    """Build instruction prompt for YanoljaNEXT Rosetta translation models.

    Follows the Rosetta chat template format: concise instruction with target
    language, context, tone, optional glossary, and output format.
    Special instructions are mapped to the Glossary field.
    """
    instruction = (
        f"Translate the user's text to {output_language}. "
        f"Keep the JSON structure and keys.\n"
        f"Context: Manga dialogue, sound effects, and narration.\n"
        f"Tone: Natural-sounding manga localization"
    )

    if special_instructions and special_instructions.strip():
        glossary_lines = []
        for line in special_instructions.strip().splitlines():
            line = line.strip()
            if line:
                entry = line if line.startswith("- ") else f"- {line}"
                glossary_lines.append(entry)
        if glossary_lines:
            instruction += "\nGlossary:\n" + "\n".join(glossary_lines)

    instruction += (
        "\nOutput format: JSON\n"
        "Provide the final translation immediately without any other text."
    )
    return instruction


def _build_rosetta_source_prompt(extracted_texts: list[str]) -> str:
    """Format OCR texts as JSON for Rosetta models.

    Returns a JSON object with string keys "1", "2", etc.
    """
    data = {str(i + 1): text for i, text in enumerate(extracted_texts)}
    return json.dumps(data, ensure_ascii=False)


def _parse_rosetta_response(
    response_text: str | None,
    total_elements: int,
    provider: str,
    debug: bool = False,
) -> list[str]:
    """Parse JSON response from a Rosetta (or Hy-MT2) model.

    Falls back to _parse_llm_response_unified if JSON parsing fails.
    """
    if response_text is None:
        raise TranslationError(f"{provider}: API failed (returned None)")
    if response_text == "":
        raise TranslationError(f"{provider}: Empty response")

    log_message(f"Raw response:\n---\n{response_text}\n---", always_print=True)

    # Strip markdown code fences if present
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            result = []
            for i in range(1, total_elements + 1):
                val = parsed.get(str(i))
                if val is not None:
                    result.append(str(val).strip())
                else:
                    result.append(f"[{provider}: Missing item {i}]")
            log_message(
                f"Parsed {len(parsed)} items from JSON (expected {total_elements})",
                verbose=debug,
            )
            return result
    except (json.JSONDecodeError, TypeError):
        log_message(
            "JSON response invalid, falling back to numbered-list parser",
            verbose=debug,
        )

    return _parse_llm_response_unified(response_text, total_elements, provider, debug)


def _build_hy_mt2_prompt(
    output_language: str,
    extracted_texts: list[str],
    special_instructions: str | None = None,
) -> str:
    """Build user-only prompt for Hy-MT2 (no system prompt per model card).

    Follows the card's default/terminology templates; source is JSON for multi-bubble.
    """
    source_data = json.dumps(
        {str(i + 1): text for i, text in enumerate(extracted_texts)},
        ensure_ascii=False,
    )

    prompt = ""
    if special_instructions and special_instructions.strip():
        term_lines = []
        for line in special_instructions.strip().splitlines():
            line = line.strip().removeprefix("- ").strip()
            if not line:
                continue
            if "->" in line:
                src, _, tgt = line.partition("->")
                term_lines.append(f"{src.strip()} translates to {tgt.strip()}")
            elif "=>" in line:
                src, _, tgt = line.partition("=>")
                term_lines.append(f"{src.strip()} translates to {tgt.strip()}")
            else:
                term_lines.append(line)
        if term_lines:
            prompt += (
                "Reference the following translations:\n"
                + "\n".join(term_lines)
                + "\n\n"
            )

    prompt += (
        f"Translate the following text into {output_language}. "
        f"Keep the JSON structure and keys. "
        f"Note that you should **only output the translated result without any "
        f"additional explanation**:\n\n"
        f"{source_data}"
    )
    return prompt


def _perform_manga_ocr(
    images_b64: list[str],
    bubble_metadata: list[dict[str, Any]],
    debug: bool = False,
) -> list[str]:
    """Perform OCR using manga-ocr model.

    Args:
        images_b64: List of base64-encoded images
        bubble_metadata: List of metadata dicts for text elements
        debug: Whether to print verbose logging

    Returns:
        List of extracted text strings, or early return with failure list
    """
    total_elements = len(images_b64)
    log_message("Using manga-ocr for text extraction", verbose=debug)

    cache = get_cache()
    cache_key = cache.get_manga_ocr_cache_key(images_b64, total_elements)
    cached_ocr = cache.get_manga_ocr_result(cache_key)
    if cached_ocr is not None:
        if len(cached_ocr) == total_elements:
            log_message("Using cached manga-ocr results", verbose=debug)
            return cached_ocr
        log_message("Discarding manga-ocr cache due to length mismatch", verbose=debug)

    pil_images = _prepare_images_for_ocr(images_b64, verbose=debug)
    extracted_texts = extract_text_with_manga_ocr(pil_images, verbose=debug)

    formatted_texts = []
    for i, text in enumerate(extracted_texts):
        if text == "[OCR FAILED]" or not text:
            formatted_texts.append(text if text else "[OCR FAILED]")
        else:
            formatted_texts.append(text)

    extracted_texts = formatted_texts

    _format_ocr_results(extracted_texts, bubble_metadata)

    if len(extracted_texts) != total_elements:
        msg = (
            f"Warning: extracted_texts length ({len(extracted_texts)}) "
            f"doesn't match total_elements ({total_elements})"
        )
        log_message(msg, always_print=True)
        while len(extracted_texts) < total_elements:
            extracted_texts.append("[OCR FAILED]")
        extracted_texts = extracted_texts[:total_elements]

    if not extracted_texts:
        log_message("manga-ocr returned empty results", verbose=debug)
        failure_results = ["[OCR FAILED]"] * total_elements
        cache.set_manga_ocr_result(cache_key, failure_results, debug)
        return failure_results

    if _check_ocr_failure(extracted_texts):
        log_message("manga-ocr returned only failures", verbose=debug)
        cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
        return extracted_texts

    cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
    return extracted_texts


def _perform_paddle_ocr_vl(
    images_b64: list[str],
    bubble_metadata: list[dict[str, Any]],
    debug: bool = False,
) -> list[str]:
    """Perform OCR using PaddleOCR-VL-1.6 model.

    Args:
        images_b64: List of base64-encoded images
        bubble_metadata: List of metadata dicts for text elements
        debug: Whether to print verbose logging

    Returns:
        List of extracted text strings, or early return with failure list
    """
    total_elements = len(images_b64)
    log_message("Using PaddleOCR-VL-1.6 for text extraction", verbose=debug)

    cache = get_cache()
    cache_key = cache.get_manga_ocr_cache_key(
        images_b64, total_elements, prefix="pocr16_"
    )
    cached_ocr = cache.get_manga_ocr_result(cache_key)
    if cached_ocr is not None:
        if len(cached_ocr) == total_elements:
            log_message("Using cached PaddleOCR-VL-1.6 results", verbose=debug)
            return cached_ocr
        log_message(
            "Discarding PaddleOCR-VL-1.6 cache due to length mismatch", verbose=debug
        )

    pil_images = _prepare_images_for_ocr(images_b64, verbose=debug)
    extracted_texts = extract_text_with_paddle_ocr_vl(pil_images, verbose=debug)

    formatted_texts = []
    for i, text in enumerate(extracted_texts):
        if text == "[OCR FAILED]" or not text:
            formatted_texts.append(text if text else "[OCR FAILED]")
        else:
            # Collapse newlines hallucinated by PaddleOCR-VL
            formatted_texts.append(" ".join(text.split()))

    extracted_texts = formatted_texts

    _format_ocr_results(extracted_texts, bubble_metadata)

    if len(extracted_texts) != total_elements:
        msg = (
            f"Warning: extracted_texts length ({len(extracted_texts)}) "
            f"doesn't match total_elements ({total_elements})"
        )
        log_message(msg, always_print=True)
        while len(extracted_texts) < total_elements:
            extracted_texts.append("[OCR FAILED]")
        extracted_texts = extracted_texts[:total_elements]

    if not extracted_texts:
        log_message("PaddleOCR-VL-1.6 returned empty results", verbose=debug)
        failure_results = ["[OCR FAILED]"] * total_elements
        cache.set_manga_ocr_result(cache_key, failure_results, debug)
        return failure_results

    if _check_ocr_failure(extracted_texts):
        log_message("PaddleOCR-VL-1.6 returned only failures", verbose=debug)
        cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
        return extracted_texts

    cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
    return extracted_texts


def _perform_llm_ocr(
    config: TranslationConfig,
    images_b64: list[str],
    mime_types: list[str],
    ocr_prompt: str,
    provider: str,
    input_language: str | None,
    debug: bool = False,
    prompt_cache_key: str | None = None,
) -> list[str]:
    """Perform OCR using vision LLM.

    Args:
        config: TranslationConfig
        images_b64: List of base64-encoded images
        mime_types: List of MIME types for each image
        ocr_prompt: OCR prompt text
        provider: Provider name
        input_language: Input language
        debug: Whether to print verbose logging

    Returns:
        List of extracted text strings, or early return with failure list
    """
    total_elements = len(images_b64)
    ocr_parts = []
    for i, img_b64 in enumerate(images_b64):
        mime_type = mime_types[i] if i < len(mime_types) else "image/jpeg"
        bubble_part = {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        supports_per_part_res = (
            provider == "Google" and is_gemini_3_model(config.model_name)
        ) or provider == "SpaceXAI"
        if supports_per_part_res:
            bubble_part = _add_media_resolution_to_part(
                bubble_part, config.media_resolution_bubbles
            )
        ocr_parts.append(bubble_part)

    ocr_system = _build_system_prompt_ocr(input_language)
    ocr_response_text = _call_llm_endpoint(
        config,
        ocr_parts,
        ocr_prompt,
        debug,
        system_prompt=ocr_system,
        prompt_cache_key=prompt_cache_key,
    )
    extracted_texts = _parse_llm_response_unified(
        ocr_response_text,
        total_elements,
        provider + "-OCR",
        debug,
    )

    if extracted_texts is None:
        log_message("OCR API call failed", always_print=True)
        return [f"[{provider}: OCR failed]"] * total_elements

    if _check_ocr_failure(extracted_texts, provider):
        log_message("OCR returned only placeholders", verbose=debug)
        return extracted_texts

    return extracted_texts


def call_translation_api_batch(
    config: TranslationConfig,
    images_b64: list[str],
    full_image_b64: str,
    mime_types: list[str],
    full_image_mime_type: str,
    bubble_metadata: list[dict[str, Any]],
    previous_context_images: list[dict[str, str]] | None = None,
    previous_context_texts: list[list[str]] | None = None,
    ocr_texts_output: list[str] | None = None,
    debug: bool = False,
) -> list[str]:
    """
    Generates prompts and calls the appropriate LLM API endpoint based on the provider and mode
    specified in the configuration, translating text from speech bubbles and outside-bubble text.

    Supports "one-step" (OCR+Translate+Style) and "two-step" (OCR then Translate+Style) modes.

    Args:
        config (TranslationConfig): Configuration object.
        images_b64 (list): List of base64 encoded images of all text elements, in reading order.
        full_image_b64 (str): Base64 encoded image of the full manga page.
        mime_types (List[str]): List of MIME types for each text element image.
        full_image_mime_type (str): MIME type of the full page image.
        bubble_metadata (List[Dict]): List of metadata dicts with 'is_outside_text' flags for each image.
        previous_context_images: Previous source page images, oldest-to-newest, as reference only.
        previous_context_texts: Per-previous-page OCR transcripts (oldest-to-newest) included as
            narrative reference only. Each inner list contains the OCR strings for that page in reading order.
        ocr_texts_output: Optional mutable list. When provided, OCR transcripts (source-language text) for
            the current page's bubbles are appended in reading order so callers can propagate them as
            previous-page text context for subsequent calls.
        debug (bool): Whether to print debugging information.

    Returns:
        list: List of translated strings (potentially with style markers), one for each input text element.
              Returns placeholder messages on errors or empty responses.

    Raises:
        ValueError: If required config (API key, provider, URL) is missing or invalid.
        RuntimeError: If an API call fails irrecoverably after retries (raised by endpoint functions).
    """
    provider = config.provider
    import uuid

    session_prompt_cache_key = f"manga-translation-{uuid.uuid4()}"
    input_language = config.input_language
    output_language = config.output_language
    translation_mode = config.translation_mode
    previous_context_images = previous_context_images or []
    if not config.send_full_page_context or config.ocr_method != "LLM":
        previous_context_images = []
    previous_context_image_count = len(previous_context_images)
    # Filter out empty pages (no usable OCR) and trim to configured cap so the
    # request order matches the prompt order regardless of upstream history gaps.
    cleaned_previous_texts: list[list[str]] = []
    configured_text_count = int(getattr(config, "previous_context_text_count", 0) or 0)
    if previous_context_texts and configured_text_count > 0:
        for page_texts in previous_context_texts:
            usable = [
                (t or "").strip()
                for t in (page_texts or [])
                if (t or "").strip() and (t or "").strip() != "[OCR FAILED]"
            ]
            if usable:
                cleaned_previous_texts.append(usable)
        cleaned_previous_texts = cleaned_previous_texts[-configured_text_count:]
    previous_context_text_count = len(cleaned_previous_texts)
    previous_text_section = _format_previous_context_texts(cleaned_previous_texts)

    # Include conditional bubble hints
    total_elements = len(images_b64)
    dialogue_indices = [
        i + 1
        for i, meta in enumerate(bubble_metadata)
        if not meta.get("is_outside_text", False)
    ]
    osb_indices = [
        i + 1
        for i, meta in enumerate(bubble_metadata)
        if meta.get("is_outside_text", False)
    ]

    hints = []
    if dialogue_indices:
        dialogue_list_str = ", ".join(map(str, dialogue_indices))
        hints.append(f"Items [{dialogue_list_str}] contain spoken dialogue.")
    if osb_indices:
        osb_list_str = ", ".join(map(str, osb_indices))
        hints.append(
            f"Items [{osb_list_str}] contain sound effects, mimetic effects, narration, or internal monologues."
        )

    context_hints = ""
    if hints:
        context_hints = "\nNote: " + " ".join(hints) + " Translate them accordingly."

    cache = get_cache()
    cache_key = cache.get_translation_cache_key(
        images_b64,
        full_image_b64,
        config,
        previous_context_images=previous_context_images,
        previous_context_texts=cleaned_previous_texts,
    )
    cached_translation, cached_ocr_texts = cache.get_translation(cache_key)
    if cached_translation is not None:
        log_message("  - Using cached translation", verbose=debug)
        if ocr_texts_output is not None and cached_ocr_texts is not None:
            ocr_texts_output.extend(cached_ocr_texts)
        return cached_translation

    model_name = config.model_name
    is_gemini_3 = provider == "Google" and is_gemini_3_model(model_name)
    supports_per_part_res = is_gemini_3 or provider == "SpaceXAI"

    base_parts = []
    for i, img_b64 in enumerate(images_b64):
        mime_type = mime_types[i] if i < len(mime_types) else "image/jpeg"
        bubble_part = {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        if supports_per_part_res:
            bubble_part = _add_media_resolution_to_part(
                bubble_part, config.media_resolution_bubbles
            )
        base_parts.append(bubble_part)

    if config.send_full_page_context and full_image_b64:
        context_part = {
            "inline_data": {
                "mime_type": full_image_mime_type,
                "data": full_image_b64,
            }
        }
        if supports_per_part_res:
            context_part = _add_media_resolution_to_part(
                context_part, config.media_resolution_context
            )
        base_parts.append(context_part)

    for image in previous_context_images:
        previous_part = {
            "inline_data": {
                "mime_type": image.get("mime_type", "image/jpeg"),
                "data": image.get("data", ""),
            }
        }
        if supports_per_part_res:
            previous_part = _add_media_resolution_to_part(
                previous_part, config.media_resolution_context
            )
        base_parts.append(previous_part)

    try:
        if translation_mode == "two-step":
            ocr_prompt = f"""
## CONTEXT
You have been provided with {total_elements} individual text images from a manga page.

## TASK
Transcribe the text in each image according to your transcription rules.
"""

            log_message("Starting OCR step", verbose=debug)

            if config.ocr_method == "manga-ocr":
                extracted_texts = _perform_manga_ocr(
                    images_b64,
                    bubble_metadata,
                    debug,
                )
            elif config.ocr_method == "paddleocr-vl-1.6":
                extracted_texts = _perform_paddle_ocr_vl(
                    images_b64,
                    bubble_metadata,
                    debug,
                )
            else:
                extracted_texts = _perform_llm_ocr(
                    config,
                    images_b64,
                    mime_types,
                    ocr_prompt,
                    provider,
                    input_language,
                    debug,
                    prompt_cache_key=session_prompt_cache_key,
                )

            log_message("Starting translation step", verbose=debug)

            formatted_texts = []
            ocr_failed_indices = set()
            for i, text in enumerate(extracted_texts):
                if f"[{provider}-OCR:" in text or text == "[OCR FAILED]":
                    formatted_texts.append("[OCR FAILED]")
                    ocr_failed_indices.add(i)
                else:
                    formatted_texts.append(text)

            ocr_input_section = """
## INPUT DATA
"""
            for i, text in enumerate(formatted_texts):
                ocr_input_section += f"{i + 1}: {text}\n"

            full_page_context = (
                "A full-page image is also provided for visual and narrative context."
                if (
                    config.ocr_method not in ("manga-ocr", "paddleocr-vl-1.6")
                    and config.send_full_page_context
                    and full_image_b64
                )
                else ""
            )
            previous_page_context = _format_previous_context_prompt_note(
                previous_context_image_count,
                previous_context_text_count,
                (
                    "current full page first (when present), then previous "
                    "source pages oldest-to-newest"
                ),
            )

            special_instructions_section = _format_special_instructions(config)

            translation_prompt = f"""
## CONTEXT
You have been provided with a list of {total_elements} transcribed text segments from a manga page. {full_page_context}{previous_page_context}
{context_hints}
{previous_text_section}
{ocr_input_section}

## TASK
Apply your translation and styling rules to the text in the `## INPUT DATA` section. 
The target language is {output_language}. Use the appropriate translation approach for each text type.{special_instructions_section}
"""

            translation_parts = []
            if (
                config.ocr_method not in ("manga-ocr", "paddleocr-vl-1.6")
                and config.send_full_page_context
                and full_image_b64
            ):
                context_part = {
                    "inline_data": {
                        "mime_type": full_image_mime_type,
                        "data": full_image_b64,
                    }
                }
                if supports_per_part_res:
                    context_part = _add_media_resolution_to_part(
                        context_part, config.media_resolution_context
                    )
                translation_parts.append(context_part)

            for image in previous_context_images:
                previous_part = {
                    "inline_data": {
                        "mime_type": image.get("mime_type", "image/jpeg"),
                        "data": image.get("data", ""),
                    }
                }
                if supports_per_part_res:
                    previous_part = _add_media_resolution_to_part(
                        previous_part, config.media_resolution_context
                    )
                translation_parts.append(previous_part)

            use_rosetta = is_rosetta_model(model_name)
            use_hy_mt2 = is_hy_mt2_model(model_name)
            if use_rosetta:
                log_message(
                    "YanoljaNEXT Rosetta model detected — using Rosetta prompt format",
                    always_print=True,
                )
                translation_system = _build_rosetta_instruction(
                    output_language,
                    config.special_instructions,
                )
                translation_prompt = _build_rosetta_source_prompt(formatted_texts)
                translation_parts = []  # text-only model, no image parts
            elif use_hy_mt2:
                log_message(
                    "Hy-MT2 model detected — using Hy-MT2 prompt format",
                    always_print=True,
                )
                translation_system = None
                translation_prompt = _build_hy_mt2_prompt(
                    output_language,
                    formatted_texts,
                    config.special_instructions,
                )
                translation_parts = []
            else:
                translation_system = _build_system_prompt_translation(
                    output_language,
                    mode="two-step",
                    full_page_context=(
                        config.send_full_page_context and bool(full_image_b64)
                    ),
                )
            translation_response_text = _call_llm_endpoint(
                config,
                translation_parts,
                translation_prompt,
                debug,
                system_prompt=translation_system,
                prompt_cache_key=session_prompt_cache_key,
            )
            if use_rosetta or use_hy_mt2:
                final_translations = _parse_rosetta_response(
                    translation_response_text,
                    total_elements,
                    provider + "-Translate",
                    debug,
                )
            else:
                final_translations = _parse_llm_response_unified(
                    translation_response_text,
                    total_elements,
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
                if ocr_texts_output is not None:
                    ocr_texts_output.extend(extracted_texts)
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

            cache.set_translation(
                cache_key, combined_results, ocr_texts=extracted_texts
            )
            if ocr_texts_output is not None:
                ocr_texts_output.extend(extracted_texts)
            return combined_results

        elif translation_mode == "one-step":
            log_message("Starting one-step translation", verbose=debug)

            full_page_context = (
                "A full-page image is also provided for visual and narrative context."
                if config.send_full_page_context
                else ""
            )
            previous_page_context = _format_previous_context_prompt_note(
                previous_context_image_count,
                previous_context_text_count,
                (
                    "text crops first, optional current full page, then previous "
                    "source pages oldest-to-newest"
                ),
            )

            special_instructions_section = _format_special_instructions(config)

            one_step_prompt = f"""
## CONTEXT
You have been provided with {total_elements} individual text images from a manga page. {full_page_context}{previous_page_context}
{context_hints}
{previous_text_section}
## TASK
For each image, perform two steps:
1.  **Transcribe:** Extract the original text exactly as it appears.
2.  **Translate:** Translate the text you just transcribed into {output_language}, applying your translation and styling rules.
Provide both the transcription and translation separated by ` || ` in the required output schema.{special_instructions_section}
"""

            one_step_system = _build_system_prompt_translation(
                output_language,
                mode="one-step",
                full_page_context=(
                    config.send_full_page_context and bool(full_image_b64)
                ),
            )
            response_text = _call_llm_endpoint(
                config,
                base_parts,
                one_step_prompt,
                debug,
                system_prompt=one_step_system,
                prompt_cache_key=session_prompt_cache_key,
            )

            # Parse one-step format ("Original || Translated")
            raw_lines = _parse_llm_response_unified(
                response_text, total_elements, provider, debug
            )

            translations = []
            ocr_texts = []
            for line in raw_lines:
                if "||" in line:
                    parts = line.split("||", 1)
                    ocr_texts.append(parts[0].strip())
                    translations.append(parts[1].strip())
                else:
                    # Model violated the format — keep the line as the translation
                    # and mark OCR as failed so it isn't reused as prior context.
                    ocr_texts.append("[OCR FAILED]")
                    translations.append(line)

            cache.set_translation(cache_key, translations, ocr_texts=ocr_texts)
            if ocr_texts_output is not None:
                ocr_texts_output.extend(ocr_texts)
            return translations
        else:
            raise TranslationError(
                f"Unknown translation_mode specified in config: {translation_mode}"
            )
    except TranslationError:
        raise
    except (ValueError, RuntimeError) as e:
        log_message(f"Translation error: {e}", always_print=True)
        return [f"[Translation Error: {e}]"] * total_elements


def prepare_bubble_images_for_translation(
    bubble_data: list[dict[str, Any]],
    original_cv_image: np.ndarray,
    upscale_model: Any,
    device: Any,
    mime_type: str,
    bubble_min_side_pixels: int,
    upscale_method: str = "model_lite",
    whiteout_conjoined_bubbles: bool = True,
    verbose: bool = False,
) -> list[dict[str, Any]]:
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
    cv2_ext = ".png" if mime_type == "image/png" else ".jpg"

    prepared_bubbles = []

    mask_lookup = {}
    for b in bubble_data:
        b_bbox = tuple(round(v) for v in b["bbox"])
        mask_lookup[b_bbox] = b.get("sam_mask")

    if upscale_method == "model":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with 2x-AnimeSharpV4_RCAN",
            always_print=True,
        )
    elif upscale_method == "model_lite":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with 2x-AnimeSharpV4_Fast_RCAN_PU (Lite)",
            always_print=True,
        )
    elif upscale_method == "lanczos":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with LANCZOS",
            always_print=True,
        )
    else:
        log_message(
            f"Processing {len(bubble_data)} bubble images without upscaling",
            always_print=True,
        )

    for bubble in bubble_data:
        prepared_bubble = bubble.copy()
        x1, y1, x2, y2 = bubble["bbox"]

        # Use the tight bbox of the mask
        _mask = bubble.get("sam_mask")
        _ma = None
        if _mask is not None:
            _ma = np.asarray(_mask)
            if _ma.ndim == 3:
                _ma = _ma[..., 0]
            if _ma.ndim == 2:
                _rows, _cols = np.where(_ma > 0)
                if _rows.size and _cols.size:
                    mx1, my1 = int(_cols.min()), int(_rows.min())
                    mx2, my2 = int(_cols.max()) + 1, int(_rows.max()) + 1
                    x1 = min(x1, mx1)
                    y1 = min(y1, my1)
                    x2 = max(x2, mx2)
                    y2 = max(y2, my2)

        bubble_image_cv = original_cv_image[y1:y2, x1:x2].copy()

        # White-out conjoined neighbor text regions visible in this crop
        neighbor_bboxes = bubble.get("conjoined_neighbor_bboxes")
        if whiteout_conjoined_bubbles and neighbor_bboxes:
            own_mask_crop = (
                _ma[y1:y2, x1:x2] > 0 if (_ma is not None and _ma.ndim == 2) else None
            )

            for nb in neighbor_bboxes:
                nb_tuple = tuple(round(v) for v in nb)
                neighbor_mask = mask_lookup.get(nb_tuple)

                if neighbor_mask is not None:
                    _nm = np.asarray(neighbor_mask)
                    if _nm.ndim == 3:
                        _nm = _nm[..., 0]
                    if _nm.ndim == 2:
                        nm_crop = _nm[y1:y2, x1:x2] > 0

                        if own_mask_crop is not None:
                            region_mask = nm_crop & ~own_mask_crop
                        else:
                            region_mask = nm_crop

                        # Apply whiteout precisely on neighbor's mask pixels
                        bubble_image_cv[region_mask] = 255

        bubble_image_pil = cv2_to_pil(bubble_image_cv)

        if upscale_method == "model" or upscale_method == "model_lite":
            final_bubble_pil = process_bubble_image_cached(
                bubble_image_pil,
                upscale_model,
                device,
                bubble_min_side_pixels,
                "min",
                upscale_method,
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
        else:
            final_bubble_pil = bubble_image_pil

        final_bubble_cv = pil_to_cv2(final_bubble_pil)

        try:
            is_success, buffer = cv2.imencode(cv2_ext, final_bubble_cv)
            if is_success:
                image_b64 = base64.b64encode(buffer).decode("utf-8")
                prepared_bubble["image_b64"] = image_b64
                prepared_bubble["mime_type"] = mime_type
                log_message(
                    f"Bubble {x1},{y1} ({final_bubble_pil.size[0]}x{final_bubble_pil.size[1]})",
                    verbose=verbose,
                )
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
