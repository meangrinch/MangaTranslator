import base64
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from core.caching import get_cache
from core.config import TranslationConfig, calculate_reasoning_budget
from core.image.image_utils import cv2_to_pil, pil_to_cv2, process_bubble_image_cached
from core.image.ocr_detection import extract_text_with_manga_ocr
from utils.endpoints import (
    call_anthropic_endpoint,
    call_deepseek_endpoint,
    call_gemini_endpoint,
    call_moonshot_endpoint,
    call_openai_compatible_endpoint,
    call_openai_endpoint,
    call_openrouter_endpoint,
    call_xai_endpoint,
    call_zai_endpoint,
    openrouter_is_reasoning_model,
)
from utils.exceptions import TranslationError
from utils.logging import log_message
from utils.model_metadata import (
    get_max_tokens_cap,
    is_deepseek_reasoning_model,
    is_openai_compatible_reasoning_model,
    is_opus_45_model,
    is_xai_reasoning_model,
    is_zai_reasoning_model,
)

TRANSLATION_PATTERN = re.compile(
    r'^\s*(\d+)\s*:\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*:|\s*$)',
    re.MULTILINE | re.DOTALL,
)


def _build_system_prompt_ocr(
    input_language: Optional[str],
    reading_direction: str,
) -> str:
    lang_label = f"{input_language} " if input_language else ""
    direction = (
        "right-to-left"
        if (reading_direction or "rtl").lower() == "rtl"
        else "left-to-right"
    )

    return f"""
## ROLE
You are an expert manga OCR transcriber.

## OBJECTIVE
Your sole purpose is to accurately transcribe the original text from a series of provided images. You must not translate, interpret, or add commentary.

## CORE RULES
- **Reading Context:** The image crops are presented in a {direction} reading order. Do not reorder them.
- **Transcription Policy:** Preserve all original punctuation, ellipses, and casing. Collapse multi-line text into a single line, separated by a single space.
- **Ignore Policy:** You must ignore image borders, speech bubble tails, watermarks, page numbers, and any decorative elements outside the text itself.
- **Language Focus:** Transcribe only the original {lang_label}text.
- **Ruby/Furigana Policy:** If small phonetic characters (ruby/furigana) are present, you must ignore them and transcribe only the main, larger base text.
- **Visual Emphasis Policy:** If the source text is visually emphasized (bold, slanted, etc.), you must mirror that emphasis in your transcription using markdown-style markers: `*italic*` for slanted text, `**bold**` for bold text, `***bold-italic***` for both.
- **Edge Cases:**
  - If an image contains standalone periods/ellipses, you must return it exactly as it appears.
  - If text is indecipherable, you must return the exact token: `[OCR FAILED]`.

## OUTPUT SCHEMA
- You must return your response as a single numbered list with exactly one line per input image.
- The numbering must correspond to the input image order (1, 2, 3...).
- The format must be `i: <transcribed {lang_label}text>` where `i` is the input image number.
- Do not include section headers, explanations, or formatting outside of this list.
"""  # noqa


def _build_system_prompt_translation(
    output_language: str,
    mode: str,
    reading_direction: str,
    full_page_context: bool = False,
) -> str:
    direction = (
        "right-to-left"
        if (reading_direction or "rtl").lower() == "rtl"
        else "left-to-right"
    )
    input_type = "transcriptions" if mode == "two-step" else "image crops"

    cohesion_visual = (
        " Refer to the full-page image to resolve ambiguous context."
        if full_page_context
        else ""
    )

    if mode == "two-step":
        edge_cases = """- **Edge Cases:**
  - If an input line contains standalone periods/ellipses, you must return it exactly as it appears.
  - If an input line is the exact token `[OCR FAILED]`, you must output it unchanged."""
    else:
        edge_cases = """- **Edge Cases:**
  - If an image contains standalone periods/ellipses, you must return it exactly as it appears.
  - If text is indecipherable, you must return the exact token: `[OCR FAILED]`."""

    core_rules = f"""
## CORE RULES
- **Reading Context:** The {input_type} are presented in a {direction} reading order. Do not reorder them.
- **Cohesion:** Treat the input lines as a continuous narrative. Ensure the translation flows logically and naturally as a cohesive whole.{cohesion_visual}
- **Fidelity:** Focus on intent; translate functionally rather than literally.
- **Conciseness:** Keep translations idiomatic and concise.
- **Emphasis:** If the source text is visually emphasized (bold, slanted, etc.), mirror that emphasis using the STYLING GUIDE.
- **Punctuation:** Replace ellipses (e.g., "…") with consecutive periods (e.g., "...").
- **Text Types:**
  - **Spoken Dialogue/Internal Monologue:** Translate naturally, matching the character's personality.
  - **Narration:** Translate neutrally without special styling.
  - **Audible SFX:** Translate physical sounds (Giongo) as standard onomatopoeia.
  - **Mimetic FX:** Translate atmospheric text (Gitaigo) or silent actions as descriptive verbs or adjectives. Do not add a period at the end of the word.
{edge_cases}
"""  # noqa

    shared_components = f"""
## ROLE
You are a professional manga localization translator and editor.

## OBJECTIVE
Your goal is to produce natural-sounding, high-quality translations in {output_language} that are faithful to the original source's meaning, tone, and visual emphasis.

## STYLING GUIDE
You must use the following markdown-style markers to convey emphasis:
- `*italic*`: Used for onomatopoeias, thoughts, flashbacks, distant sounds, or dialogue mediated by a device (e.g., phone, radio).
- `**bold**`: Used for sound effects (SFX), shouting, timestamps, or individual emphatic words.
- `***bold-italic***`: Used for extremely loud sounds or dialogue that also meets the criteria for italics (e.g., shouting over a radio).

{core_rules}
"""  # noqa

    if mode == "one-step":
        output_schema = f"""
## OUTPUT SCHEMA
- You must return your response as a single numbered list with exactly one line per input image.
- The numbering must correspond to the input image order (1, 2, 3...).
- For each item, provide both transcription and translation in the format:
  `i: <transcribed text> || <translated {output_language} text>` where `i` is the input image number.
- Do not include section headers, explanations, or formatting outside of this list.
"""
    elif mode == "two-step":
        output_schema = f"""
## OUTPUT SCHEMA
- You must return your response as a single numbered list with exactly one line per input text.
- The numbering must correspond to the input order (1, 2, 3...).
- The format must be `i: <translated {output_language} text>` where `i` is the input text number.
- Do not include section headers, explanations, or formatting outside of this list.
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
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "claude-3-7-sonnet",
    ]
    return any(lm.startswith(p) for p in reasoning_prefixes)


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
        elif provider == "xAI":
            is_reasoning = is_xai_reasoning_model(model_name)
        elif provider == "OpenRouter":
            is_reasoning = openrouter_is_reasoning_model(model_name, debug)
        elif provider == "OpenAI-Compatible":
            is_reasoning = is_openai_compatible_reasoning_model(model_name)
        elif provider == "DeepSeek":
            is_reasoning = is_deepseek_reasoning_model(model_name)
        elif provider == "Z.ai":
            is_reasoning = is_zai_reasoning_model(model_name)
        elif provider == "Moonshot AI":
            lm = (model_name or "").lower()
            is_reasoning = "thinking" in lm or "kimi-k2.5" in lm
        max_tokens_value = 16384 if is_reasoning else 4096

    max_tokens_cap = get_max_tokens_cap(provider, model_name)
    if max_tokens_cap is not None and max_tokens_value > max_tokens_cap:
        max_tokens_value = max_tokens_cap

    if provider == "Google":
        is_gemini_3 = "gemini-3" in model_name.lower()
        generation_config = {
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "maxOutputTokens": max_tokens_value,
        }
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
        if is_gemini_3:
            reasoning_effort = config.reasoning_effort or "high"
            generation_config["thinkingConfig"] = {"thinkingLevel": reasoning_effort}
            log_message(
                f"Using reasoning effort '{reasoning_effort}' for {model_name}",
                verbose=debug,
            )
        elif _is_reasoning_model_google(model_name) and not is_gemini_3:
            reasoning_effort = config.reasoning_effort or "auto"
            is_flash = "gemini-2.5-flash" in model_name.lower()
            is_pro = "gemini-2.5-pro" in model_name.lower()
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
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens_value,
        }  # top_k not supported by OpenAI
        if config.reasoning_effort:
            lm = (model_name or "").lower()
            is_chat_variant = "chat" in lm
            is_gpt5_1 = lm.startswith("gpt-5.1")
            is_gpt5_2 = lm.startswith("gpt-5.2")
            effort = config.reasoning_effort
            if effort == "xhigh" and not is_gpt5_2:
                effort = "high"
            if not is_chat_variant and (is_gpt5_1 or is_gpt5_2 or effort != "none"):
                generation_config["reasoning_effort"] = effort
        return generation_config

    elif provider == "Anthropic":
        is_reasoning = _is_reasoning_model_anthropic(model_name)
        is_opus_45 = is_opus_45_model(model_name)
        clamped_temp = min(temperature, 1.0)  # Anthropic caps at 1.0
        generation_config = {
            "temperature": clamped_temp,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens_value,
        }
        if is_reasoning:
            generation_config["reasoning_effort"] = config.reasoning_effort or "none"
        if is_opus_45 and config.effort:
            generation_config["effort"] = config.effort
        return generation_config

    elif provider == "xAI":
        is_reasoning = is_xai_reasoning_model(model_name)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens_value,
        }
        if is_reasoning:
            generation_config["reasoning_effort"] = config.reasoning_effort or "high"
        return generation_config

    elif provider == "DeepSeek":
        is_reasoning = is_deepseek_reasoning_model(model_name)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens_value,
        }
        return generation_config

    elif provider == "Z.ai":
        is_reasoning = is_zai_reasoning_model(model_name)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens_value,
        }
        if is_reasoning:
            # Z.ai uses thinking parameter with {"type": "enabled"} or {"type": "disabled"}
            # Map reasoning_effort: "high" -> enabled, "none" -> disabled
            reasoning_effort = config.reasoning_effort or "high"
            thinking_type = "enabled" if reasoning_effort == "high" else "disabled"
            generation_config["thinking"] = {"type": thinking_type}
        return generation_config

    elif provider == "Moonshot AI":
        lm = (model_name or "").lower()
        is_kimi_25 = "kimi-k2.5" in lm
        is_reasoning = is_kimi_25 or "thinking" in lm

        generation_config = {
            "temperature": min(temperature, 1.0),  # Moonshot caps at 1.0
            "top_p": top_p,
            "max_tokens": max_tokens_value,
        }

        if is_reasoning:
            # Map reasoning_effort: "high" -> enabled, "none" -> disabled
            reasoning_effort = config.reasoning_effort or "high"
            thinking_type = "enabled" if reasoning_effort == "high" else "disabled"
            generation_config["thinking"] = {"type": thinking_type}
        return generation_config

    elif provider == "OpenRouter":
        model_lower = (model_name or "").lower()
        is_openai_model = "openai/" in model_lower or model_lower.startswith("gpt-")
        is_anthropic_model = "anthropic/" in model_lower or model_lower.startswith(
            "claude-"
        )
        is_grok_model = "grok-4" in model_lower
        is_gemini_3 = "gemini-3" in model_lower

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens_value,
        }

        is_openai_reasoning = is_openai_model and (
            "gpt-5" in model_lower
            or "o1" in model_lower
            or "o3" in model_lower
            or "o4-mini" in model_lower
        )
        is_gpt5_1 = is_openai_model and "gpt-5.1" in model_lower
        is_gpt5 = is_openai_model and "gpt-5" in model_lower and not is_gpt5_1
        # For OpenRouter, Anthropic models use dots (4.5) not hyphens (4-5)
        # Claude 3.7 Sonnet :thinking variant is reasoning-capable, non-thinking is not
        is_claude_37_sonnet_thinking = (
            is_anthropic_model
            and "claude-3.7-sonnet" in model_lower
            and ":thinking" in model_lower
        )
        is_anthropic_reasoning = is_anthropic_model and (
            "claude-opus-4" in model_lower
            or "claude-sonnet-4" in model_lower
            or "claude-haiku-4.5" in model_lower
            or is_claude_37_sonnet_thinking
        )
        # For OpenRouter, Grok models don't have "reasoning" in the name (e.g., "grok-4.1-fast")
        is_grok_reasoning = is_grok_model and "non-reasoning" not in model_lower

        # Add metadata flags for OpenRouter endpoint to avoid re-parsing model names
        generation_config["_metadata"] = {
            "is_openai_model": is_openai_model,
            "is_anthropic_model": is_anthropic_model,
            "is_grok_model": is_grok_model,
            "is_gemini_3": is_gemini_3,
            "is_google_model": "google/" in model_lower or "gemini" in model_lower,
            "is_openai_reasoning": is_openai_reasoning,
            "is_anthropic_reasoning": is_anthropic_reasoning,
            "is_grok_reasoning": is_grok_reasoning,
            "is_claude_37_sonnet_thinking": is_claude_37_sonnet_thinking,
            "is_gpt5_1": is_gpt5_1,
            "is_gpt5": is_gpt5,
        }

        if is_openai_reasoning or is_anthropic_reasoning or is_grok_reasoning:
            if is_anthropic_reasoning:
                reasoning_effort = config.reasoning_effort or "none"
                generation_config["reasoning_effort"] = reasoning_effort
            elif is_gpt5_1:
                generation_config["reasoning_effort"] = config.reasoning_effort
            elif config.reasoning_effort and config.reasoning_effort != "none":
                generation_config["reasoning_effort"] = config.reasoning_effort
        elif "gemini" in model_lower or "google/" in model_lower:
            if config.reasoning_effort:
                generation_config["reasoning_effort"] = config.reasoning_effort

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
                enable_web_search=config.enable_web_search,
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

    except (ValueError, RuntimeError):
        raise


def _parse_llm_response_unified(
    response_text: Optional[str],
    total_elements: int,
    provider: str,
    debug: bool = False,
) -> List[str]:
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
            f"Failed to parse {provider} unified response: {str(e)}",
            always_print=True,
        )
        return [f"[{provider}: Parse error]"] * total_elements


def _prepare_images_for_ocr(
    images_b64: List[str], verbose: bool = False
) -> List[Optional[Image.Image]]:
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
    extracted_texts: List[str],
    bubble_metadata: List[Dict[str, Any]],
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


def _check_ocr_failure(texts: List[str], provider: Optional[str] = None) -> bool:
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


def _format_special_instructions(config: TranslationConfig) -> str:
    """Format user's special instructions section for prompts.

    Args:
        config: TranslationConfig with special_instructions

    Returns:
        Formatted special instructions string (empty if none)
    """
    if config.special_instructions and config.special_instructions.strip():
        return f"""

## SPECIAL INSTRUCTIONS
{config.special_instructions.strip()}
"""
    return ""


def _perform_manga_ocr(
    images_b64: List[str],
    bubble_metadata: List[Dict[str, Any]],
    debug: bool = False,
) -> List[str]:
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


def _perform_llm_ocr(
    config: TranslationConfig,
    images_b64: List[str],
    mime_types: List[str],
    ocr_prompt: str,
    is_gemini_3: bool,
    provider: str,
    input_language: Optional[str],
    reading_direction: str,
    debug: bool = False,
) -> List[str]:
    """Perform OCR using vision LLM.

    Args:
        config: TranslationConfig
        images_b64: List of base64-encoded images
        mime_types: List of MIME types for each image
        ocr_prompt: OCR prompt text
        is_gemini_3: Whether model is Gemini 3
        provider: Provider name
        input_language: Input language
        reading_direction: Reading direction
        debug: Whether to print verbose logging

    Returns:
        List of extracted text strings, or early return with failure list
    """
    total_elements = len(images_b64)
    ocr_parts = []
    for i, img_b64 in enumerate(images_b64):
        mime_type = mime_types[i] if i < len(mime_types) else "image/jpeg"
        bubble_part = {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        if is_gemini_3:
            bubble_part = _add_media_resolution_to_part(
                bubble_part, config.media_resolution_bubbles, is_gemini_3
            )
        ocr_parts.append(bubble_part)

    ocr_system = _build_system_prompt_ocr(input_language, reading_direction)
    ocr_response_text = _call_llm_endpoint(
        config,
        ocr_parts,
        ocr_prompt,
        debug,
        system_prompt=ocr_system,
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
    images_b64: List[str],
    full_image_b64: str,
    mime_types: List[str],
    full_image_mime_type: str,
    bubble_metadata: List[Dict[str, Any]],
    debug: bool = False,
) -> List[str]:
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

    reading_order_desc = (
        "right-to-left, top-to-bottom"
        if reading_direction == "rtl"
        else "left-to-right, top-to-bottom"
    )

    cache = get_cache()
    cache_key = cache.get_translation_cache_key(images_b64, full_image_b64, config)
    cached_translation = cache.get_translation(cache_key)
    if cached_translation is not None:
        log_message("  - Using cached translation", verbose=debug)
        return cached_translation

    model_name = config.model_name
    is_gemini_3 = provider == "Google" and "gemini-3" in model_name.lower()

    base_parts = []
    for i, img_b64 in enumerate(images_b64):
        mime_type = mime_types[i] if i < len(mime_types) else "image/jpeg"
        bubble_part = {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        if is_gemini_3:
            bubble_part = _add_media_resolution_to_part(
                bubble_part, config.media_resolution_bubbles, is_gemini_3
            )
        base_parts.append(bubble_part)

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

    try:
        if translation_mode == "two-step":
            special_instructions_section = _format_special_instructions(config)

            ocr_prompt = f"""
## CONTEXT
You have been provided with {total_elements} individual text images from a manga page. They are presented in their natural reading order ({reading_order_desc}).

## TASK
Apply your OCR transcription rules to each image provided.{special_instructions_section}
"""  # noqa

            log_message("Starting OCR step", verbose=debug)

            if config.ocr_method == "manga-ocr":
                extracted_texts = _perform_manga_ocr(
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
                    is_gemini_3,
                    provider,
                    input_language,
                    reading_direction,
                    debug,
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
                    config.ocr_method != "manga-ocr"
                    and config.send_full_page_context
                    and full_image_b64
                )
                else ""
            )

            special_instructions_section = _format_special_instructions(config)

            translation_prompt = f"""
## CONTEXT
You have been provided with a list of {total_elements} transcribed text segments from a manga page. {full_page_context}
{context_hints}

{ocr_input_section}

## TASK
Apply your translation and styling rules to the text in the `## INPUT DATA` section. 
The target language is {output_language}. Use the appropriate translation approach for each text type.{special_instructions_section}
"""  # noqa

            translation_parts = []
            if (
                config.ocr_method != "manga-ocr"
                and config.send_full_page_context
                and full_image_b64
            ):
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
                reading_direction=reading_direction,
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
            )
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

            cache.set_translation(cache_key, combined_results)
            return combined_results

        elif translation_mode == "one-step":
            log_message("Starting one-step translation", verbose=debug)

            full_page_context = (
                "A full-page image is also provided for visual and narrative context."
                if config.send_full_page_context
                else ""
            )

            special_instructions_section = _format_special_instructions(config)

            one_step_prompt = f"""
## CONTEXT
You have been provided with {total_elements} individual text images from a manga page. {full_page_context}
{context_hints}

## TASK
For each image, you must perform two steps:
1.  **Transcribe:** Extract the original text exactly as it appears.
2.  **Translate:** Translate the text you just transcribed into {output_language}, applying your translation and styling rules.{special_instructions_section}

## OUTPUT FORMAT
You must return your response as a single numbered list with exactly one line per input image.
The numbering must correspond to the input image order (1, 2, 3...).
Format: `i: <transcribed text> || <translated {output_language} text>`
"""  # noqa

            one_step_system = _build_system_prompt_translation(
                output_language,
                mode="one-step",
                reading_direction=reading_direction,
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
            )

            # Parse one-step format ("Original || Translated")
            raw_lines = _parse_llm_response_unified(
                response_text, total_elements, provider, debug
            )

            translations = []
            for line in raw_lines:
                if "||" in line:
                    parts = line.split("||", 1)
                    translations.append(parts[1].strip())
                else:
                    translations.append(line)

            cache.set_translation(cache_key, translations)
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
    bubble_data: List[Dict[str, Any]],
    original_cv_image: np.ndarray,
    upscale_model: Any,
    device: Any,
    mime_type: str,
    bubble_min_side_pixels: int,
    upscale_method: str = "model_lite",
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
    cv2_ext = ".png" if mime_type == "image/png" else ".jpg"

    prepared_bubbles = []

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
    else:  # upscale_method == "none"
        log_message(
            f"Processing {len(bubble_data)} bubble images without upscaling",
            always_print=True,
        )

    for bubble in bubble_data:
        prepared_bubble = bubble.copy()
        x1, y1, x2, y2 = bubble["bbox"]

        bubble_image_cv = original_cv_image[y1:y2, x1:x2].copy()
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
        else:  # upscale_method == "none"
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
