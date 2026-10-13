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
from core.image.image_utils import (
    cv2_to_pil,
    pil_to_cv2,
    process_bubble_image_cached,
    resize_lanczos_antiring,
)
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
    NO_WEB_SEARCH_PROVIDERS,
    anthropic_model_flags,
    get_max_tokens_cap,
    is_anthropic_model_family,
    is_anthropic_reasoning_model,
    is_azure_url,
    is_deepseek_reasoning_model,
    is_gemini_3_model,
    is_gemini_25_flash_model,
    is_gemini_25_pro_model,
    is_gemini_38_flash_model,
    is_gemini_no_sampling_model,
    is_gemma_model,
    is_google_model_family,
    is_google_reasoning_model,
    is_gpt5_chat_variant,
    is_gpt5_series,
    is_gpt6_astra,
    is_gpt6_series,
    is_gpt61_sol,
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

_FULL_PAGE_LABEL = "Full page (context only):"
_PREVIOUS_PAGE_LABEL = "Previous page, {} back (context only):"
_FULL_PAGE_NOTE = " The image labeled `Full page` shows the whole page for context."


def _lang_label(input_language: str | None) -> str:
    return (
        f"{input_language.strip()} "
        if input_language and input_language.strip()
        else ""
    )


def _transcription_rules(input_language: str | None) -> str:
    """Transcription rules shared by the OCR prompt and the one-step prompt."""
    lang_clean = (input_language or "").strip().lower()
    primary_code = lang_clean.split("-")[0].split("_")[0]
    no_space_keywords = ("japanese", "chinese", "mandarin", "cantonese")
    no_space_codes = {"ja", "zh", "jpn", "chi", "zho"}
    is_no_space_lang = bool(
        any(kw in lang_clean for kw in no_space_keywords)
        or primary_code in no_space_codes
    )

    spacing_rule = (
        "Add no space at a join unless the source has one."
        if is_no_space_lang
        else "Put a single space at each join."
    )

    return f"""- **Reading Order:** Read horizontal text left-to-right, and vertical text top-to-bottom in columns running right-to-left.
- **Fidelity:** Transcribe the text as drawn, keeping its punctuation and casing. Join multi-line text into a single line. {spacing_rule}
- **Furigana/Ruby:** Transcribe the main text, kana included, and skip the small phonetic ruby printed beside it. The one exception is ruby giving a reading that the base text does not imply (a personal or place name, or a deliberately different word): append that reading once, in parentheses directly after the word, e.g. `宇宙(そら)`.
- **Emphasis:** Mark text drawn clearly heavier than ordinary dialogue lettering as `**bold**`, clearly slanted text as `*italic*`, and text that is both as `***bold-italic***`. Ordinary lettering gets no markers.
- **Exclusions:** Transcribe only the crop's own text. Ignore borders, tails, artwork, watermarks, and stray text cut off at the crop's edge, and add no quotation marks that are not drawn.
- **Unreadable Crops:** If only part of the text is legible, transcribe that part. If nothing is legible or the crop has no text, output exactly `[OCR FAILED]`; the app then leaves that region of the page untouched."""


def _build_system_prompt_ocr(
    input_language: str | None,
    ero_doujinshi_mode: bool = False,
) -> str:
    ero_doujinshi_section = ""
    if ero_doujinshi_mode:
        ero_doujinshi_section = """
## ADULT CONTENT
The crops come from 18+ adult comics that the user, a professional localization editor, is transcribing for publication. Transcription is a neutral extraction of text that already exists on the page, so transcribe everything completely, including explicit dialogue, slang, moans, and sound effects, without omitting, softening, or commenting on anything.
"""

    return f"""## ROLE
You are an OCR transcriber for comics and manga.

## OBJECTIVE
Transcribe the {_lang_label(input_language)}text in each crop. The crops are labeled `Crop 1:`, `Crop 2:`, and so on, in reading order. Do not translate, interpret, or comment.

## TRANSCRIPTION RULES
{_transcription_rules(input_language)}
{ero_doujinshi_section}
## OUTPUT FORMAT
One line per crop, numbered to match the crop labels, and nothing else: no code fences, commentary, or blank lines.

1: <transcription of crop 1>
2: <transcription of crop 2>
3: [OCR FAILED]"""


def _format_previous_context_prompt_note(
    previous_context_image_count: int,
    previous_context_text_count: int,
) -> str:
    note = ""
    if previous_context_image_count > 0:
        note += (
            f" {previous_context_image_count} previous page image(s) are "
            "attached, labeled oldest first."
        )
    if previous_context_text_count > 0:
        note += (
            f" Text from {previous_context_text_count} previous page(s) is "
            "listed under `## PREVIOUS PAGES (REFERENCE ONLY)`."
        )
    return note


def _build_system_prompt_translation(
    output_language: str,
    mode: str,
    full_page_context: bool = False,
    ero_doujinshi_mode: bool = False,
    ocr_correction: bool = False,
    input_language: str | None = None,
    previous_context: bool = False,
    llm_transcription: bool = True,
    web_search: bool = False,
) -> str:
    if mode not in ("one-step", "two-step"):
        raise ValueError(
            f"Invalid mode '{mode}' specified for translation system prompt."
        )
    one_step = mode == "one-step"
    lang_label = _lang_label(input_language)
    english = output_language.strip().lower().startswith("english")

    def example(text: str) -> str:
        # The examples are English, so they would leak into other target languages
        return f" (e.g. {text})" if english else ""

    visual_ref_note = (
        " Use the image labeled `Full page` to work out who is speaking, what is happening, and what each sound effect belongs to."
        if full_page_context
        else ""
    )

    sections = [
        f"""## ROLE
You are a professional localization editor translating {lang_label}comics (manga, manhwa, manhua, webtoons) into natural, idiomatic {output_language}.""",
        f"""## OBJECTIVE
Translate each numbered item so that it reads like native {output_language} comic lettering while staying faithful to the source's meaning, tone, and subtext.{visual_ref_note}""",
    ]

    if one_step:
        sections.append(
            f"""## TRANSCRIPTION RULES
The items are image crops labeled `Crop 1:`, `Crop 2:`, and so on. Transcribe each crop before translating it:
{_transcription_rules(input_language)}"""
        )

    failed_rule = (
        "- **Failed Crops:** A crop transcribed as `[OCR FAILED]` gets `[OCR FAILED]` as its translation too."
        if one_step
        else "- **Failed Items:** Return `[OCR FAILED]` for an item that is marked `[OCR FAILED]` or that is plainly OCR noise rather than text; the app then leaves the original artwork in place."
    )
    furigana_rule = (
        "\n- **Furigana Readings:** A reading in parentheses directly after a word in a transcription, e.g. `宇宙(そら)`, is that word's furigana. Use it to romanize the name or to convey the intended word, and do not reproduce the parentheses."
        if llm_transcription
        else ""
    )
    sections.append(
        f"""## TRANSLATION RULES
- **Continuous Narrative:** The items are in reading order and form one continuous scene, so translate each with its neighbors in mind. Translate meaning and intent rather than word for word.
- **Register:** Match each speaker's voice and the register of the source: formal stays formal, crude stays crude, childish stays childish. Do not soften or intensify it.
- **Length:** Each translation is typeset back into the space the original text occupied, so keep it about as compact as the source, and prefer the shorter of two equally good phrasings.
- **Split Bubbles:** A sentence often runs across consecutive bubbles. Keep each fragment in its own numbered slot (never merge slots or leave one empty), and do not add ellipses at the break unless the source has them.
- **Names & Honorifics:** Spell names and recurring terms the same way every time, following earlier pages when they are provided. Adapt or drop honorifics the way natural {output_language} would, unless the custom instructions say otherwise.{furigana_rule}
- **Vocalizations:** Render laughs, gasps, grunts, and stutters phonetically in {output_language} spelling, keeping their length and rhythm.
- **Pass-Through:** Return an item unchanged when it is already in {output_language} or consists only of punctuation or symbols (`!?`, `♥`, `...`).
- **Punctuation:** Write ellipses as periods (`...`, or `......` for a long pause) and never as the `…` character, keeping their relative length. Use quotation marks only where the source has them.
{failed_rule}"""
    )

    sections.append(
        f"""## TEXT TYPES & STYLING
Style each translation with markdown according to what the text is. Bold or italic markers in a transcription show that the source lettering was heavier or slanted; treat them as evidence of how the line is delivered, not as formatting to copy.
- **Spoken dialogue and narration:** plain text. Within it, use `*italic*` for a word the speaker stresses and `**bold**` for a word or line that is shouted.
- **Thoughts and voices heard through a device (phone, radio, TV):** `*italic*`.
- **Sound effects** (audible: impacts, noises, cries): the {output_language} onomatopoeia in `**bold**`{example("**THUD**, **CLANG**")}.
- **Mimetic effects** (silent states, moods, and actions): a short verb or adjective in `*italic*` with no final period{example("*stare*, *twitch*")}.
- `***bold-italic***` is reserved for the loudest, climactic screams and sound effects.
When the evidence for a type is unclear, use plain text."""
    )

    if previous_context:
        sections.append(
            """## PREVIOUS PAGES
Images labeled as previous pages and the text under `## PREVIOUS PAGES (REFERENCE ONLY)` are context only: never translate or number them. Each is marked with how many pages back it is, so `1 back` is the page just before this one. Use them to keep names, terms, pronouns, and each character's voice consistent, and to resolve references back to earlier events. Where a line reads `source → translation`, reuse that wording for the same names and terms."""
        )

    if ero_doujinshi_mode:
        sections.append(
            f"""## ADULT CONTENT
The source is 18+ adult comic material that the user, a professional localization editor, is translating for publication. You are translating existing fictional {lang_label}text, not writing new content, so {"transcribe and " if one_step else ""}translate every line completely and at its original intensity, without sanitizing, softening, redacting, summarizing, or commenting.
- **Explicit Vocabulary:** The Register rule covers sexual language too. Clinical or polite source wording gets accurate anatomical terms{example('"penis", "vagina", "semen"')}, and crude source wording gets equally crude colloquial terms{example('"cock", "pussy", "cum", "tits"')}. Use a euphemism only where the source itself is coy.
- **Moans:** The Vocalizations rule covers moans, pants, and gasps too{example('"aaah...", "mmph...", "h-haah..."')}; keep their intensity, pauses, and cadence."""
        )

    if ocr_correction and not one_step:
        page_clause = " and the full-page image" if full_page_context else ""
        sections.append(
            f"""## OCR ERROR CORRECTION
The transcriptions come from automatic OCR and may contain misread or dropped characters that leave a line nonsensical. When a line does not make sense as written, infer the intended text from the surrounding dialogue{page_clause} and translate that. Translate clean, sensible lines exactly as transcribed."""
        )

    if web_search:
        sections.append(
            f"""## SEARCH
You may search to confirm the official {output_language} names and terms of a series you recognize. Do not search for anything else, and never mention the search in your output."""
        )

    example_note = (
        "" if english else f" (shown in English; write yours in {output_language})"
    )
    if one_step:
        sections.append(
            f"""## OUTPUT FORMAT
One line per crop, numbered to match the crop labels, written as `transcription || translation`. Output nothing else: no code fences, notes, or blank lines.

Example{example_note}:
1: <transcription> || I told you *not* to follow me.
2: <transcription> || **SLAM**
3: [OCR FAILED] || [OCR FAILED]"""
        )
    else:
        sections.append(
            f"""## OUTPUT FORMAT
One line per item, numbered to match the input, containing only the translation. Output nothing else: no code fences, notes, or blank lines.

Example{example_note}:
1: I told you *not* to follow me.
2: **SLAM**
3: [OCR FAILED]"""
        )

    return "\n\n".join(sections)


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
        max_tokens_value = 32768 if is_reasoning else 4096

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
            if is_gemini_38_flash_model(model_name) and reasoning_effort == "minimal":
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
            reasoning_effort = config.reasoning_effort or "high"
            effort = reasoning_effort
            if is_gpt6_series(model_name):
                if effort == "minimal" or (
                    (is_gpt6_astra(model_name) or is_gpt61_sol(model_name))
                    and effort == "none"
                ):
                    effort = "low"
                generation_config["reasoning_effort"] = effort
            else:
                if effort == "max" and not supports_gpt5_max_effort(model_name):
                    effort = (
                        "xhigh" if supports_gpt5_xhigh_effort(model_name) else "high"
                    )
                if effort == "xhigh" and not supports_gpt5_xhigh_effort(model_name):
                    effort = "high"
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
                if reasoning_effort == "none":
                    generation_config["thinking_type"] = (
                        "between_tools"
                        if anthropic_flags.get("is_claude_between_tools")
                        else "disabled"
                    )
                else:
                    generation_config["thinking_type"] = "adaptive"
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
                    "temperature": min(temperature, 1.0),
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
        is_gpt6_model = is_openai_model and is_gpt6_series(model_name)
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
            "is_gpt5_model": is_gpt5_model,
            "is_gpt6_astra": is_openai_model and is_gpt6_astra(model_name),
            "is_gpt6_model": is_gpt6_model,
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
                if effort == "minimal" or (
                    (is_gpt6_astra(model_name) or is_gpt61_sol(model_name))
                    and effort == "none"
                ):
                    effort = "low"
                generation_config["reasoning_effort"] = effort
            elif is_gpt5_model:
                generation_config["reasoning_effort"] = config.reasoning_effort
        elif is_google_model and config.reasoning_effort:
            effort = config.reasoning_effort
            if is_gemini_38_flash_model(model_name) and effort == "minimal":
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
        is_gpt6_model = is_openai_model and is_gpt6_series(model_name)
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
            "is_gpt6_astra": is_openai_model and is_gpt6_astra(model_name),
            "is_gpt6_model": is_gpt6_model,
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
            if is_gpt6_model and (
                effort == "minimal"
                or (
                    (is_gpt6_astra(model_name) or is_gpt61_sol(model_name))
                    and effort == "none"
                )
            ):
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


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


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

        # Pattern matches "1: text" or "1. text". An item runs until the next
        # numbered line, a blank line, or end of text (\Z, since $ would stop
        # at the first line break under MULTILINE).
        pattern = re.compile(
            r'^\s*(\d+)\s*[:.]\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*[:.]|\n\s*\n|\s*\Z)',
            re.MULTILINE | re.DOTALL,
        )

        matches = pattern.findall(_strip_code_fences(response_text))
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


# Kana, CJK ideographs and punctuation, full-width forms (not Hangul, which uses spaces)
_CJK_CHARS = "\u3000-\u9fff\uff00-\uffef"


def _join_ocr_lines(text: str) -> str:
    """Collapse line breaks: nothing between CJK characters, a space otherwise."""
    text = re.sub(rf"(?<=[{_CJK_CHARS}])\s*\n\s*(?=[{_CJK_CHARS}])", "", text)
    return " ".join(text.split())


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


def _pair_context_lines(ocr_texts: list[str], translations: list[str]) -> list[str]:
    """Join each source line to its translation so later pages reuse names and terms."""
    if len(ocr_texts) != len(translations):
        return ocr_texts
    return [
        # A leading "[" marks a failure placeholder, not a translation
        ocr
        if ocr == "[OCR FAILED]" or not tr or tr.startswith("[")
        else f"{ocr} → {tr}"
        for ocr, tr in zip(ocr_texts, translations)
    ]


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
        pages_back = len(previous_context_texts) - page_index + 1
        page_blocks.append(f"### Previous page, {pages_back} back\n" + "\n".join(lines))

    if not page_blocks:
        return ""

    return (
        "\n## PREVIOUS PAGES (REFERENCE ONLY)\n"
        "Oldest first. A line with `→` gives the source text and the "
        "translation already used for it.\n" + "\n\n".join(page_blocks) + "\n"
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
These come from the user and override the translation and styling rules where they conflict; the output format still applies.
{config.special_instructions.strip()}
"""
    return ""


def _split_special_instructions(
    special_instructions: str | None,
) -> tuple[list[str], list[str]]:
    """Split custom instructions into glossary pairs (`a -> b`) and free-form notes."""
    glossary_lines, note_lines = [], []
    for line in (special_instructions or "").splitlines():
        line = line.strip().removeprefix("- ").strip()
        if line:
            is_pair = "->" in line or "=>" in line
            (glossary_lines if is_pair else note_lines).append(line)
    return glossary_lines, note_lines


def _build_rosetta_instruction(
    output_language: str,
    special_instructions: str | None = None,
) -> str:
    """Build instruction prompt for YanoljaNEXT Rosetta translation models.

    Follows the Rosetta chat template format: concise instruction with target
    language, context, tone, optional glossary, and output format.
    Special instructions of the form `a -> b` are mapped to the Glossary field;
    free-form ones are appended to the Context field.
    """
    glossary_lines, note_lines = _split_special_instructions(special_instructions)
    context = " ".join(["Comic dialogue, sound effects, and narration.", *note_lines])
    instruction = (
        f"Translate the user's text to {output_language}. "
        f"Keep the JSON structure and keys.\n"
        f"Context: {context}\n"
        f"Tone: Natural-sounding comic localization"
    )

    if glossary_lines:
        instruction += "\nGlossary:\n" + "\n".join(
            f"- {line}" for line in glossary_lines
        )

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

    cleaned = _strip_code_fences(response_text)

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

    glossary_lines, note_lines = _split_special_instructions(special_instructions)
    prompt = ""
    if glossary_lines:
        term_lines = []
        for line in glossary_lines:
            src, _, tgt = line.partition("->" if "->" in line else "=>")
            term_lines.append(f"{src.strip()} translates to {tgt.strip()}")
        prompt += (
            "Reference the following translations:\n" + "\n".join(term_lines) + "\n\n"
        )

    notes = "".join(f"{line.rstrip('.')}. " for line in note_lines)
    prompt += (
        f"Translate the following text into {output_language}. "
        f"Keep the JSON structure and keys. {notes}"
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
            log_message("  - Using cached manga-ocr results", always_print=True)
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
            log_message("  - Using cached PaddleOCR-VL-1.6 results", always_print=True)
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
            formatted_texts.append(_join_ocr_lines(text))

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
        ocr_parts += [{"text": f"Crop {i + 1}:"}, bubble_part]

    ocr_system = _build_system_prompt_ocr(
        input_language,
        ero_doujinshi_mode=getattr(config, "ero_doujinshi_mode", False),
    )
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
        hints.append(
            f"Items [{dialogue_list_str}] are inside speech bubbles or caption "
            "boxes (dialogue, thoughts, or narration)."
        )
    if osb_indices:
        osb_list_str = ", ".join(map(str, osb_indices))
        hints.append(
            f"Items [{osb_list_str}] are free-standing text outside bubbles "
            "(usually sound effects or mimetic effects, sometimes narration or asides)."
        )

    context_hints = ""
    if hints:
        context_hints = "\nLayout: " + " ".join(hints)

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
        log_message("  - Using cached translation", always_print=True)
        if ocr_texts_output is not None and cached_ocr_texts is not None:
            ocr_texts_output.extend(cached_ocr_texts)
        return cached_translation

    model_name = config.model_name
    is_gemini_3 = provider == "Google" and is_gemini_3_model(model_name)
    supports_per_part_res = is_gemini_3 or provider == "SpaceXAI"
    has_full_page = bool(config.send_full_page_context and full_image_b64)
    has_previous_context = bool(
        previous_context_image_count or previous_context_text_count
    )
    web_search = (
        bool(getattr(config, "enable_web_search", False))
        and provider not in NO_WEB_SEARCH_PROVIDERS
    )

    base_parts = []
    for i, img_b64 in enumerate(images_b64):
        mime_type = mime_types[i] if i < len(mime_types) else "image/jpeg"
        bubble_part = {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        if supports_per_part_res:
            bubble_part = _add_media_resolution_to_part(
                bubble_part, config.media_resolution_bubbles
            )
        base_parts += [{"text": f"Crop {i + 1}:"}, bubble_part]

    if has_full_page:
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
        base_parts += [{"text": _FULL_PAGE_LABEL}, context_part]

    for pages_back, image in zip(
        range(previous_context_image_count, 0, -1), previous_context_images
    ):
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
        base_parts += [
            {"text": _PREVIOUS_PAGE_LABEL.format(pages_back)},
            previous_part,
        ]

    try:
        if translation_mode == "two-step":
            ocr_prompt = f"Transcribe the {total_elements} crops above."

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

            llm_ocr = config.ocr_method not in ("manga-ocr", "paddleocr-vl-1.6")
            # Local OCR implies a text-only translation request
            page_attached = has_full_page and llm_ocr
            full_page_context = _FULL_PAGE_NOTE if page_attached else ""
            previous_page_context = _format_previous_context_prompt_note(
                previous_context_image_count,
                previous_context_text_count,
            )

            special_instructions_section = _format_special_instructions(config)

            translation_prompt = f"""## CONTEXT
{total_elements} transcribed text items from one comic page, in reading order.{full_page_context}{previous_page_context}
{context_hints}
{previous_text_section}
{ocr_input_section}
## TASK
Translate each item under `## INPUT DATA` into {output_language}.{special_instructions_section}
"""

            translation_parts = []
            if page_attached:
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
                translation_parts += [{"text": _FULL_PAGE_LABEL}, context_part]

            for pages_back, image in zip(
                range(previous_context_image_count, 0, -1), previous_context_images
            ):
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
                translation_parts += [
                    {"text": _PREVIOUS_PAGE_LABEL.format(pages_back)},
                    previous_part,
                ]

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
                    full_page_context=page_attached,
                    ero_doujinshi_mode=getattr(config, "ero_doujinshi_mode", False),
                    ocr_correction=getattr(config, "ocr_correction", False),
                    input_language=input_language,
                    previous_context=has_previous_context,
                    llm_transcription=llm_ocr,
                    web_search=web_search,
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

            full_page_context = _FULL_PAGE_NOTE if has_full_page else ""
            previous_page_context = _format_previous_context_prompt_note(
                previous_context_image_count,
                previous_context_text_count,
            )

            special_instructions_section = _format_special_instructions(config)

            one_step_prompt = f"""## CONTEXT
{total_elements} text crops from one comic page, in reading order.{full_page_context}{previous_page_context}
{context_hints}
{previous_text_section}
## TASK
Transcribe each crop, then translate that transcription into {output_language}.{special_instructions_section}
"""

            one_step_system = _build_system_prompt_translation(
                output_language,
                mode="one-step",
                full_page_context=has_full_page,
                ero_doujinshi_mode=getattr(config, "ero_doujinshi_mode", False),
                input_language=input_language,
                previous_context=has_previous_context,
                web_search=web_search,
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
        bx1, by1, bx2, by2 = [round(c) for c in bubble["bbox"]]
        bx1 = max(0, min(original_cv_image.shape[1], bx1))
        by1 = max(0, min(original_cv_image.shape[0], by1))
        bx2 = max(0, min(original_cv_image.shape[1], bx2))
        by2 = max(0, min(original_cv_image.shape[0], by2))
        if bx2 > bx1 and by2 > by1:
            prepared_bubble["original_crop_pil"] = cv2_to_pil(
                original_cv_image[by1:by2, bx1:bx2].copy()
            )

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
                resized_bubble = resize_lanczos_antiring(
                    bubble_image_pil, (new_w, new_h)
                )
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
