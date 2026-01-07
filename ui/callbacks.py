import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
from PIL import Image

from core.config import MangaTranslatorConfig
from core.validation import (
    normalize_zip_file_input,
    validate_mutually_exclusive_modes,
    validate_zip_file,
)
from utils.exceptions import CancellationError, ValidationError

from . import logic, settings_manager, utils
from .cancellation import CancellationManager
from .ui_models import (
    UICleaningSettings,
    UIConfigState,
    UIDetectionSettings,
    UIGeneralSettings,
    UIOutputSettings,
    UIOutsideTextSettings,
    UIRenderingSettings,
    UITranslationLLMSettings,
    UITranslationProviderSettings,
    map_ui_to_backend_config,
)

ERROR_PREFIX = "❌ Error: "
SUCCESS_PREFIX = "✅ "

CANCELLATION_MANAGER: Optional[CancellationManager] = None


def _clean_error_message(message: Any) -> str:
    """Normalize error text for display and avoid duplicate prefixes/quotes."""
    try:
        text = str(message).strip()
    except Exception:
        text = ""

    if (len(text) >= 2) and (
        (text[0] == text[-1] == "'") or (text[0] == text[-1] == '"')
    ):
        text = text[1:-1].strip()

    if text.startswith(ERROR_PREFIX):
        return text
    return f"{ERROR_PREFIX}{text}"


def _build_ui_state_from_args(args: tuple, is_batch: bool) -> UIConfigState:
    """Build UIConfigState from UI component arguments, handling single vs batch mode differences."""
    (
        confidence,
        conjoined_confidence,
        panel_confidence,
        use_sam2_checkbox_val,
        conjoined_detection_checkbox_val,
        osb_text_verification_checkbox_val,
        use_panel_sorting_checkbox_val,
        thresholding_value,
        use_otsu_threshold,
        inpaint_colored_bubbles,
        roi_shrink_px,
        provider_selector,
        google_api_key,
        openai_api_key,
        anthropic_api_key,
        xai_api_key,
        deepseek_api_key,
        zai_api_key,
        moonshot_api_key,
        openrouter_api_key,
        openai_compatible_url_input,
        openai_compatible_api_key_input,
        config_model_name,
        temperature,
        top_p,
        top_k,
        max_tokens,
        config_reading_direction,
        config_translation_mode,
        ocr_method_val,
        input_language,
        output_language,
        font_dropdown,
        max_font_size,
        min_font_size,
        line_spacing_mult,
        use_subpixel_rendering,
        font_hinting,
        use_ligatures,
        output_format,
        jpeg_quality,
        png_compression,
        verbose,
        cleaning_only_toggle,
        upscaling_only_val,
        test_mode_toggle,
        enable_web_search_val,
        media_resolution_val,
        media_resolution_bubbles_val,
        media_resolution_context_val,
        reasoning_effort_val,
        effort_val,
        send_full_page_context_val,
        upscale_method_val,
        bubble_min_side_pixels_val,
        context_image_max_side_pixels_val,
        osb_min_side_pixels_val,
        hyphenate_before_scaling_val,
        hyphen_penalty_val,
        hyphenation_min_word_length_val,
        badness_exponent_val,
        padding_pixels_val,
        supersampling_factor_val,
        outside_text_enabled_val,
        outside_text_seed_val,
        outside_text_force_cv2_inpainting_val,
        outside_text_flux_num_inference_steps_val,
        outside_text_flux_residual_diff_threshold_val,
        outside_text_osb_confidence_val,
        outside_text_enable_page_number_filtering_val,
        outside_text_page_filter_margin_threshold_val,
        outside_text_page_filter_min_area_ratio_val,
        outside_text_huggingface_token_val,
        outside_text_osb_font_pack_val,
        outside_text_osb_max_font_size_val,
        outside_text_osb_min_font_size_val,
        outside_text_osb_use_ligatures_val,
        outside_text_osb_outline_width_val,
        outside_text_osb_line_spacing_val,
        outside_text_osb_use_subpixel_rendering_val,
        outside_text_osb_font_hinting_val,
        outside_text_bbox_expansion_percent_val,
        outside_text_text_box_proximity_ratio_val,
        image_upscale_mode_val,
        image_upscale_factor_val,
        image_upscale_model_val,
        auto_scale_val,
        batch_input_language,
        batch_output_language,
        batch_font_dropdown,
        special_instructions_val,
        batch_special_instructions_val,
    ) = args

    final_input_language = batch_input_language if is_batch else input_language
    final_output_language = batch_output_language if is_batch else output_language
    final_font_pack = batch_font_dropdown if is_batch else font_dropdown
    final_special_instructions = (
        batch_special_instructions_val if is_batch else special_instructions_val
    )

    return UIConfigState(
        detection=UIDetectionSettings(
            confidence=confidence,
            conjoined_confidence=conjoined_confidence,
            panel_confidence=panel_confidence,
            use_sam2=use_sam2_checkbox_val,
            conjoined_detection=conjoined_detection_checkbox_val,
            use_osb_text_verification=osb_text_verification_checkbox_val,
            use_panel_sorting=use_panel_sorting_checkbox_val,
        ),
        cleaning=UICleaningSettings(
            thresholding_value=thresholding_value,
            use_otsu_threshold=use_otsu_threshold,
            roi_shrink_px=int(max(0, min(8, roi_shrink_px))),
            inpaint_colored_bubbles=inpaint_colored_bubbles,
        ),
        outside_text=UIOutsideTextSettings(
            enabled=outside_text_enabled_val,
            seed=int(outside_text_seed_val),
            huggingface_token=outside_text_huggingface_token_val,
            force_cv2_inpainting=outside_text_force_cv2_inpainting_val,
            flux_num_inference_steps=int(outside_text_flux_num_inference_steps_val),
            flux_residual_diff_threshold=float(
                outside_text_flux_residual_diff_threshold_val
            ),
            osb_confidence=float(outside_text_osb_confidence_val),
            enable_page_number_filtering=outside_text_enable_page_number_filtering_val,
            page_filter_margin_threshold=float(
                outside_text_page_filter_margin_threshold_val
            ),
            page_filter_min_area_ratio=float(
                outside_text_page_filter_min_area_ratio_val
            ),
            osb_font_name=outside_text_osb_font_pack_val,
            osb_max_font_size=int(outside_text_osb_max_font_size_val),
            osb_min_font_size=int(outside_text_osb_min_font_size_val),
            osb_use_ligatures=outside_text_osb_use_ligatures_val,
            osb_outline_width=float(outside_text_osb_outline_width_val),
            osb_line_spacing=float(outside_text_osb_line_spacing_val),
            osb_use_subpixel_rendering=outside_text_osb_use_subpixel_rendering_val,
            osb_font_hinting=outside_text_osb_font_hinting_val,
            bbox_expansion_percent=float(outside_text_bbox_expansion_percent_val),
            text_box_proximity_ratio=float(outside_text_text_box_proximity_ratio_val),
        ),
        provider_settings=UITranslationProviderSettings(
            provider=provider_selector,
            google_api_key=google_api_key,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            xai_api_key=xai_api_key,
            deepseek_api_key=deepseek_api_key,
            zai_api_key=zai_api_key,
            moonshot_api_key=moonshot_api_key,
            openrouter_api_key=openrouter_api_key,
            openai_compatible_url=openai_compatible_url_input,
            openai_compatible_api_key=openai_compatible_api_key_input,
        ),
        llm_settings=UITranslationLLMSettings(
            model_name=config_model_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reading_direction=config_reading_direction,
            translation_mode=config_translation_mode,
            ocr_method=ocr_method_val,
            send_full_page_context=send_full_page_context_val,
            upscale_method=upscale_method_val,
            bubble_min_side_pixels=bubble_min_side_pixels_val,
            context_image_max_side_pixels=context_image_max_side_pixels_val,
            osb_min_side_pixels=osb_min_side_pixels_val,
            special_instructions=final_special_instructions,
        ),
        rendering=UIRenderingSettings(
            max_font_size=max_font_size,
            min_font_size=min_font_size,
            line_spacing_mult=line_spacing_mult,
            use_subpixel_rendering=use_subpixel_rendering,
            font_hinting=font_hinting,
            use_ligatures=use_ligatures,
            hyphenate_before_scaling=hyphenate_before_scaling_val,
            hyphen_penalty=hyphen_penalty_val,
            hyphenation_min_word_length=hyphenation_min_word_length_val,
            badness_exponent=badness_exponent_val,
            padding_pixels=padding_pixels_val,
            supersampling_factor=int(supersampling_factor_val),
        ),
        output=UIOutputSettings(
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
            image_upscale_mode=image_upscale_mode_val,
            image_upscale_factor=image_upscale_factor_val,
            image_upscale_model=image_upscale_model_val,
        ),
        general=UIGeneralSettings(
            verbose=verbose,
            cleaning_only=cleaning_only_toggle,
            upscaling_only=upscaling_only_val,
            test_mode=test_mode_toggle,
            enable_web_search=enable_web_search_val,
            media_resolution=media_resolution_val,
            media_resolution_bubbles=media_resolution_bubbles_val,
            media_resolution_context=media_resolution_context_val,
            reasoning_effort=reasoning_effort_val,
            effort=effort_val,
            auto_scale=auto_scale_val,
        ),
        input_language=final_input_language,
        output_language=final_output_language,
        font_pack=final_font_pack,
        batch_input_language=batch_input_language,
        batch_output_language=batch_output_language,
        batch_font_pack=batch_font_dropdown,
    )


def _validate_ui_state(ui_state: UIConfigState) -> None:
    """Validate UI state including modes and API keys. Raises gr.Error on validation failure."""
    try:
        validate_mutually_exclusive_modes(
            ui_state.general.cleaning_only,
            ui_state.general.upscaling_only,
            ui_state.general.test_mode,
        )
    except ValidationError as e:
        raise gr.Error(f"{ERROR_PREFIX}{str(e)}")

    # Skip API key validation if in cleaning-only, upscaling-only, or test mode
    if (
        not ui_state.general.cleaning_only
        and not ui_state.general.upscaling_only
        and not ui_state.general.test_mode
    ):
        api_key_to_validate = ""
        provider_selector = ui_state.provider_settings.provider
        if provider_selector == "Google":
            api_key_to_validate = ui_state.provider_settings.google_api_key
        elif provider_selector == "OpenAI":
            api_key_to_validate = ui_state.provider_settings.openai_api_key
        elif provider_selector == "Anthropic":
            api_key_to_validate = ui_state.provider_settings.anthropic_api_key
        elif provider_selector == "xAI":
            api_key_to_validate = ui_state.provider_settings.xai_api_key
        elif provider_selector == "DeepSeek":
            api_key_to_validate = ui_state.provider_settings.deepseek_api_key
        elif provider_selector == "Z.ai":
            api_key_to_validate = ui_state.provider_settings.zai_api_key
        elif provider_selector == "Moonshot AI":
            api_key_to_validate = ui_state.provider_settings.moonshot_api_key
        elif provider_selector == "OpenRouter":
            api_key_to_validate = ui_state.provider_settings.openrouter_api_key
        elif provider_selector == "OpenAI-Compatible":
            api_key_to_validate = ui_state.provider_settings.openai_compatible_api_key

        api_valid, api_msg = utils.validate_api_key(
            api_key_to_validate, provider_selector
        )
        if not api_valid and not (
            provider_selector == "OpenAI-Compatible" and not api_key_to_validate
        ):
            raise gr.Error(f"{ERROR_PREFIX}{api_msg}")

    if (
        ui_state.provider_settings.provider == "OpenAI-Compatible"
        and not ui_state.general.cleaning_only
        and not ui_state.general.upscaling_only
        and not ui_state.general.test_mode
    ):
        if not ui_state.provider_settings.openai_compatible_url:
            raise gr.Error(f"{ERROR_PREFIX}OpenAI-Compatible URL is required.")
        if not ui_state.provider_settings.openai_compatible_url.startswith(
            ("http://", "https://")
        ):
            raise gr.Error(
                f"{ERROR_PREFIX}Invalid OpenAI-Compatible URL format. Must start with http:// or https://",
            )

    if ui_state.outside_text.enabled and ui_state.outside_text.huggingface_token:
        hf_valid, hf_msg = utils.validate_huggingface_token(
            ui_state.outside_text.huggingface_token
        )
        if not hf_valid:
            raise gr.Error(f"{ERROR_PREFIX}{hf_msg}")


def _format_single_success_message(
    result_image: Image.Image,
    backend_config: MangaTranslatorConfig,
    font_dir_path: Path,
    save_path: Path,
    processing_time: float,
) -> str:
    """Formats the success message for single image translation."""
    width, height = result_image.size
    provider = backend_config.translation.provider
    model_name = backend_config.translation.model_name
    thinking_status_str = utils.format_thinking_status(
        provider,
        model_name,
        backend_config.translation.reasoning_effort,
    )
    temp_val = backend_config.translation.temperature
    top_p_val = backend_config.translation.top_p
    top_k_val = backend_config.translation.top_k

    llm_params_str = f"• LLM Params: Temp={temp_val:.2f}, Top-P={top_p_val:.2f}"
    param_notes = ""
    if provider == "Google":
        llm_params_str += f", Top-K={top_k_val}"
    elif provider == "Anthropic":
        llm_params_str += f", Top-K={top_k_val}"
        param_notes = " (Temp clamped <= 1.0)"
    elif provider == "OpenAI":
        param_notes = " (Top-K N/A)"
    elif provider == "OpenRouter":
        is_openai_model = "openai/" in model_name
        is_anthropic_model = "anthropic/" in model_name
        if is_openai_model or is_anthropic_model:
            param_notes = " (Temp clamped <= 1.0, Top-K N/A)"
        else:
            llm_params_str += f", Top-K={top_k_val}"
    elif provider == "OpenAI-Compatible":
        llm_params_str += f", Top-K={top_k_val}"
    llm_params_str += param_notes

    if backend_config.cleaning_only:
        processing_mode_str = "Cleaning Only"
    elif backend_config.upscaling_only:
        processing_mode_str = "Upscaling Only"
    else:
        processing_mode_str = "Translation"

    if backend_config.test_mode:
        model_display = "<Test Mode>"
    elif backend_config.cleaning_only:
        model_display = "<Cleaning-only Mode>"
    elif backend_config.upscaling_only:
        model_display = "<Upscaling-only Mode>"
    else:
        model_display = f"{model_name}{thinking_status_str}"

    msg_parts = [
        f"{SUCCESS_PREFIX}{processing_mode_str} completed!\n",
        f"• Image Size: {width}x{height} pixels\n",
        f"• Outside Text Detection: {'Enabled' if backend_config.outside_text.enabled else 'Disabled'}\n",
    ]

    if (
        backend_config.outside_text.enabled
        and backend_config.outside_text.osb_font_name
    ):
        osb_font_name = Path(backend_config.outside_text.osb_font_name).name
        msg_parts.append(f"• OSB Font Pack: {osb_font_name}\n")

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(
            f"• Full-Page Context: {'On' if backend_config.translation.send_full_page_context else 'Off'}\n"
        )

    msg_parts.append(
        f"• Upscale Method: {backend_config.translation.upscale_method.title()}\n"
    )

    msg_parts.extend(
        [
            f"• Provider: {provider}\n",
            f"• Model: {model_display}\n",
            f"• Source Language: {backend_config.translation.input_language}\n",
            f"• Target Language: {backend_config.translation.output_language}\n",
            f"• Reading Direction: {backend_config.translation.reading_direction.upper()}\n",
            f"• Translation Mode: {backend_config.translation.translation_mode}\n",
        ]
    )

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(f"{llm_params_str}\n")

    msg_parts.append(f"• Font Pack: {font_dir_path.name}\n")

    if (
        getattr(backend_config, "preprocessing", None)
        and backend_config.preprocessing.enabled
    ):
        msg_parts.append(
            f"• Initial Image Upscaling: {backend_config.preprocessing.factor}x\n"
        )

    if backend_config.output.upscale_final_image:
        msg_parts.append(
            f"• Final Image Upscaling: {backend_config.output.image_upscale_factor}x\n"
        )

    if backend_config.output.output_format == "png":
        msg_parts.append("• Output Format: png\n")
        msg_parts.append(
            f"• PNG Compression: {backend_config.output.png_compression}\n"
        )
    elif backend_config.output.output_format == "jpeg":
        msg_parts.append("• Output Format: jpeg\n")
        msg_parts.append(f"• JPEG Quality: {backend_config.output.jpeg_quality}\n")
    else:
        try:
            ext = save_path.suffix.lower()
        except Exception:
            ext = ""
        if ext in {".jpg", ".jpeg"}:
            msg_parts.append("• Output Format: auto → jpeg\n")
            msg_parts.append(f"• JPEG Quality: {backend_config.output.jpeg_quality}\n")
        elif ext == ".png":
            msg_parts.append("• Output Format: auto → png\n")
            msg_parts.append(
                f"• PNG Compression: {backend_config.output.png_compression}\n"
            )
        elif ext == ".webp":
            msg_parts.append("• Output Format: auto → webp\n")
        else:
            msg_parts.append("• Output Format: auto\n")

    msg_parts.extend(
        [
            f"• Processing Time: {processing_time:.2f} seconds\n",
            f"• Saved To: {save_path}",
        ]
    )
    return "".join(msg_parts)


def _format_batch_success_message(
    results: Dict[str, Any],
    backend_config: MangaTranslatorConfig,
    font_dir_path: Path,
) -> str:
    """Formats the success message for batch processing."""
    success_count = results["success_count"]
    error_count = results["error_count"]
    total_images = success_count + error_count
    processing_time = results["processing_time"]
    output_path = results["output_path"]
    seconds_per_image = processing_time / success_count if success_count > 0 else 0

    provider = backend_config.translation.provider
    model_name = backend_config.translation.model_name
    thinking_status_str = utils.format_thinking_status(
        provider,
        model_name,
        backend_config.translation.reasoning_effort,
    )
    temp_val = backend_config.translation.temperature
    top_p_val = backend_config.translation.top_p
    top_k_val = backend_config.translation.top_k

    llm_params_str = f"• LLM Params: Temp={temp_val:.2f}, Top-P={top_p_val:.2f}"
    param_notes = ""
    if provider == "Google":
        llm_params_str += f", Top-K={top_k_val}"
    elif provider == "Anthropic":
        llm_params_str += f", Top-K={top_k_val}"
        param_notes = " (Temp clamped <= 1.0)"
    elif provider == "OpenAI":
        param_notes = " (Top-K N/A)"
    elif provider == "OpenRouter":
        is_openai_model = "openai/" in model_name or model_name.startswith("gpt-")
        is_anthropic_model = "anthropic/" in model_name or model_name.startswith(
            "claude-"
        )
        if is_openai_model or is_anthropic_model:
            param_notes = " (Temp clamped <= 1.0, Top-K N/A)"
        else:
            llm_params_str += f", Top-K={top_k_val}"
    elif provider == "OpenAI-Compatible":
        param_notes = " (Top-K N/A)"
    llm_params_str += param_notes

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        llm_params_str = (
            f"• Full-Page Context: {'On' if backend_config.translation.send_full_page_context else 'Off'}\n"
            + llm_params_str
        )

    error_summary = ""
    if error_count > 0:
        error_summary = f"\n• Warning: {error_count} image(s) failed to process."

    if backend_config.cleaning_only:
        processing_mode_str = "Cleaning Only"
    elif backend_config.upscaling_only:
        processing_mode_str = "Upscaling Only"
    else:
        processing_mode_str = "Translation"

    # Determine model display based on mode
    if backend_config.test_mode:
        model_display = "<Test Mode>"
    elif backend_config.cleaning_only:
        model_display = "<Cleaning-only Mode>"
    elif backend_config.upscaling_only:
        model_display = "<Upscaling-only Mode>"
    else:
        model_display = f"{model_name}{thinking_status_str}"

    msg_parts = [
        f"{SUCCESS_PREFIX}Batch {processing_mode_str.lower()} completed!\n",
        f"• Outside Text Detection: {'Enabled' if backend_config.outside_text.enabled else 'Disabled'}\n",
    ]

    if (
        backend_config.outside_text.enabled
        and backend_config.outside_text.osb_font_name
    ):
        osb_font_name = Path(backend_config.outside_text.osb_font_name).name
        msg_parts.append(f"• OSB Font Pack: {osb_font_name}\n")

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(
            f"• Full-Page Context: {'On' if backend_config.translation.send_full_page_context else 'Off'}\n"
        )

    msg_parts.append(
        f"• Upscale Method: {backend_config.translation.upscale_method.title()}\n"
    )

    msg_parts.extend(
        [
            f"• Provider: {provider}\n",
            f"• Model: {model_display}\n",
            f"• Source Language: {backend_config.translation.input_language}\n",
            f"• Target Language: {backend_config.translation.output_language}\n",
            f"• Reading Direction: {backend_config.translation.reading_direction.upper()}\n",
            f"• Translation Mode: {backend_config.translation.translation_mode}\n",
        ]
    )

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(f"{llm_params_str}\n")

    msg_parts.append(f"• Font Pack: {font_dir_path.name}\n")

    if (
        getattr(backend_config, "preprocessing", None)
        and backend_config.preprocessing.enabled
    ):
        msg_parts.append(
            f"• Initial Image Upscaling: {backend_config.preprocessing.factor}x\n"
        )

    if backend_config.output.upscale_final_image:
        msg_parts.append(
            f"• Final Image Upscaling: {backend_config.output.image_upscale_factor}x\n"
        )

    if backend_config.output.output_format == "png":
        msg_parts.append("• Output Format: png\n")
        msg_parts.append(
            f"• PNG Compression: {backend_config.output.png_compression}\n"
        )
    elif backend_config.output.output_format == "jpeg":
        msg_parts.append("• Output Format: jpeg\n")
        msg_parts.append(f"• JPEG Quality: {backend_config.output.jpeg_quality}\n")
    else:
        msg_parts.append("• Output Format: auto\n")
        try:
            out_dir = Path(output_path)
            if out_dir.exists() and out_dir.is_dir():
                exts = {p.suffix.lower() for p in out_dir.glob("*.*") if p.is_file()}
                only_jpeg = exts and all(e in {".jpg", ".jpeg"} for e in exts)
                only_png = exts == {".png"}
                if only_png:
                    msg_parts.append(
                        f"• PNG Compression: {backend_config.output.png_compression}\n"
                    )
                elif only_jpeg:
                    msg_parts.append(
                        f"• JPEG Quality: {backend_config.output.jpeg_quality}\n"
                    )
                # If mixed or unknown, omit specific settings to avoid showing both
        except Exception:
            pass

    msg_parts.extend(
        [
            f"• Successful Translations: {success_count}/{total_images}{error_summary}\n",
            f"• Total Processing Time: {processing_time:.2f} seconds ({seconds_per_image:.2f} seconds/image)\n",
            f"• Saved To: {output_path}",
        ]
    )
    return "".join(msg_parts)


def handle_translate_click(
    *args: Any,
    models_dir: Path,
    fonts_base_dir: Path,
    target_device: Optional[torch.device],
) -> Tuple[Optional[Image.Image], str]:
    """Callback for the 'Translate' button click. Uses dataclasses for config."""
    input_image_path = args[0]
    start_time = time.time()
    global CANCELLATION_MANAGER
    CANCELLATION_MANAGER = CancellationManager()
    try:
        img_valid, img_msg = utils.validate_image(input_image_path)
        if not img_valid:
            raise gr.Error(f"{ERROR_PREFIX}{img_msg}")

        ui_state = _build_ui_state_from_args(args[1:], is_batch=False)

        _validate_ui_state(ui_state)

        backend_config = map_ui_to_backend_config(
            ui_state=ui_state,
            fonts_base_dir=fonts_base_dir,
            target_device=target_device,
            is_batch=False,
        )
        selected_font_pack_name = ui_state.font_pack

        result_image, save_path = logic.translate_manga_logic(
            image=input_image_path,
            config=backend_config,
            selected_font_pack_name=selected_font_pack_name,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            cancellation_manager=CANCELLATION_MANAGER,
        )

        # Re-get font path for success message (logic already validated it)
        font_dir_path = (
            fonts_base_dir / selected_font_pack_name
            if selected_font_pack_name
            else Path(".")
        )

        processing_time = time.time() - start_time
        status_msg = _format_single_success_message(
            result_image,
            backend_config,
            font_dir_path,
            save_path,
            processing_time,
        )

        return result_image.copy(), status_msg

    except gr.Error as e:
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return gr.update(), cleaned
    except CancellationError:
        return gr.update(), "Translation cancelled by user."
    except (ValidationError, FileNotFoundError, ValueError, logic.LogicError) as e:
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return gr.update(), cleaned
    except Exception as e:
        import traceback

        traceback.print_exc()
        cleaned = _clean_error_message(f"An unexpected error occurred: {str(e)}")
        gr.Error(cleaned)
        return gr.update(), cleaned


def handle_batch_click(
    *args: Any,
    models_dir: Path,
    fonts_base_dir: Path,
    target_device: Optional[torch.device],
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[List[str], str]:
    """Callback for the 'Start Batch Translating' button click. Uses dataclasses."""
    input_files = args[0]
    input_zip = args[1] if len(args) > 1 else None
    progress(0, desc="Starting batch process...")
    global CANCELLATION_MANAGER
    CANCELLATION_MANAGER = CancellationManager()
    try:
        zip_file_path = None
        if input_zip:
            try:
                zip_path_str = normalize_zip_file_input(input_zip)
                zip_file_path = validate_zip_file(zip_path_str)
            except (ValidationError, FileNotFoundError) as e:
                raise gr.Error(f"{ERROR_PREFIX}{str(e)}")

        if input_files:
            if not isinstance(input_files, list):
                raise gr.Error(
                    f"{ERROR_PREFIX}Invalid input format. Expected a list of files."
                )

        if not zip_file_path and not input_files:
            raise gr.Error(
                f"{ERROR_PREFIX}Please upload images, a folder, or a ZIP archive."
            )

        if zip_file_path and input_files:
            input_to_process = {
                "zip": str(zip_file_path),
                "files": input_files,
            }
        elif zip_file_path:
            input_to_process = str(zip_file_path)
        else:
            input_to_process = input_files

        ui_state = _build_ui_state_from_args(args[2:], is_batch=True)

        _validate_ui_state(ui_state)

        backend_config = map_ui_to_backend_config(
            ui_state=ui_state,
            fonts_base_dir=fonts_base_dir,
            target_device=target_device,
            is_batch=True,
        )
        selected_font_pack_name = ui_state.batch_font_pack

        results = logic.process_batch_logic(
            input_dir_or_files=input_to_process,
            config=backend_config,
            selected_font_pack_name=selected_font_pack_name,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            gradio_progress=progress,
            cancellation_manager=CANCELLATION_MANAGER,
        )

        output_path = results["output_path"]
        gallery_images = []
        if output_path.exists():
            # Use rglob to recursively find all images when structure is preserved
            processed_files = list(output_path.rglob("*.*"))
            image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
            processed_files = [
                f
                for f in processed_files
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            processed_files.sort(  # Sort naturally (e.g., page_1, page_2, page_10)
                key=lambda x: tuple(
                    int(part) if part.isdigit() else part
                    for part in re.split(r"(\d+)", x.stem)
                )
            )
            gallery_images = [str(file_path) for file_path in processed_files]

        # Re-get font path for success message (logic already validated it)
        font_dir_path = (
            fonts_base_dir / selected_font_pack_name
            if selected_font_pack_name
            else Path(".")
        )

        status_msg = _format_batch_success_message(
            results, backend_config, font_dir_path
        )
        progress(1.0, desc="Batch complete!")
        return gallery_images, status_msg

    except gr.Error as e:
        progress(1.0, desc="Error occurred")
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return gr.update(), cleaned
    except CancellationError:
        progress(1.0, desc="Cancelled")
        return gr.update(), "Batch process cancelled by user."
    except (ValidationError, FileNotFoundError, ValueError, logic.LogicError) as e:
        progress(1.0, desc="Error occurred")
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return gr.update(), cleaned
    except Exception as e:
        progress(1.0, desc="Error occurred")
        import traceback

        traceback.print_exc()
        cleaned = _clean_error_message(
            f"An unexpected error occurred during batch processing: {str(e)}"
        )
        gr.Error(cleaned)
        return gr.update(), cleaned


def handle_save_config_click(*args: Any) -> str:
    """Callback for the 'Save Config' button. Uses dataclasses."""
    (
        conf,
        conjoined_conf,
        panel_conf,
        use_sam2,
        conjoined_detection,
        osb_text_verification,
        use_panel_sorting,
        rd,
        thresholding_val,
        otsu,
        inpaint_colored_bubbles,
        roi_shrink_px,
        prov,
        gem_key,
        oai_key,
        ant_key,
        xai_key,
        deepseek_key,
        zai_key,
        moonshot_key,
        or_key,
        comp_url,
        comp_key,
        model,
        temp,
        tp,
        tk,
        max_tokens,
        trans_mode,
        ocr_method_val,
        max_fs,
        min_fs,
        ls,
        subpix,
        hint,
        liga,
        out_fmt,
        jq,
        pngc,
        verb,
        cleaning_only_val,
        upscaling_only_val,
        test_mode_val,
        s_in_lang,
        s_out_lang,
        s_font,
        b_in_lang,
        b_out_lang,
        b_font,
        enable_web_search_val,
        media_resolution_val,
        media_resolution_bubbles_val,
        media_resolution_context_val,
        reasoning_effort_val,
        effort_val,
        send_full_page_context_val,
        upscale_method_val,
        bubble_min_side_pixels_val,
        context_image_max_side_pixels_val,
        osb_min_side_pixels_val,
        hyphenate_before_scaling_val,
        special_instructions_val,
        batch_special_instructions_val,
        hyphen_penalty_val,
        hyphenation_min_word_length_val,
        badness_exponent_val,
        padding_pixels_val,
        supersampling_factor_val,
        outside_text_enabled_val,
        outside_text_seed_val,
        outside_text_force_cv2_inpainting_val,
        outside_text_flux_num_inference_steps_val,
        outside_text_flux_residual_diff_threshold_val,
        outside_text_osb_confidence_val,
        outside_text_enable_page_number_filtering_val,
        outside_text_page_filter_margin_threshold_val,
        outside_text_page_filter_min_area_ratio_val,
        outside_text_huggingface_token_val,
        outside_text_osb_font_pack_val,
        outside_text_osb_max_font_size_val,
        outside_text_osb_min_font_size_val,
        outside_text_osb_use_ligatures_val,
        outside_text_osb_outline_width_val,
        outside_text_osb_line_spacing_val,
        outside_text_osb_use_subpixel_rendering_val,
        outside_text_osb_font_hinting_val,
        outside_text_bbox_expansion_percent_val,
        outside_text_text_box_proximity_ratio_val,
        image_upscale_mode_val,
        image_upscale_factor_val,
        image_upscale_model_val,
        auto_scale_val,
    ) = args
    ui_state = UIConfigState(
        detection=UIDetectionSettings(
            confidence=conf,
            conjoined_confidence=conjoined_conf,
            panel_confidence=panel_conf,
            use_sam2=use_sam2,
            conjoined_detection=conjoined_detection,
            use_osb_text_verification=osb_text_verification,
            use_panel_sorting=use_panel_sorting,
        ),
        cleaning=UICleaningSettings(
            thresholding_value=thresholding_val,
            use_otsu_threshold=otsu,
            inpaint_colored_bubbles=inpaint_colored_bubbles,
            roi_shrink_px=int(max(0, min(8, roi_shrink_px))),
        ),
        outside_text=UIOutsideTextSettings(
            enabled=outside_text_enabled_val,
            seed=int(outside_text_seed_val),
            huggingface_token=outside_text_huggingface_token_val,
            force_cv2_inpainting=outside_text_force_cv2_inpainting_val,
            flux_num_inference_steps=int(outside_text_flux_num_inference_steps_val),
            flux_residual_diff_threshold=float(
                outside_text_flux_residual_diff_threshold_val
            ),
            osb_confidence=float(outside_text_osb_confidence_val),
            enable_page_number_filtering=outside_text_enable_page_number_filtering_val,
            page_filter_margin_threshold=float(
                outside_text_page_filter_margin_threshold_val
            ),
            page_filter_min_area_ratio=float(
                outside_text_page_filter_min_area_ratio_val
            ),
            osb_font_name=outside_text_osb_font_pack_val,
            osb_max_font_size=int(outside_text_osb_max_font_size_val),
            osb_min_font_size=int(outside_text_osb_min_font_size_val),
            osb_use_ligatures=outside_text_osb_use_ligatures_val,
            osb_outline_width=float(outside_text_osb_outline_width_val),
            osb_line_spacing=float(outside_text_osb_line_spacing_val),
            osb_use_subpixel_rendering=outside_text_osb_use_subpixel_rendering_val,
            osb_font_hinting=outside_text_osb_font_hinting_val,
            bbox_expansion_percent=float(outside_text_bbox_expansion_percent_val),
            text_box_proximity_ratio=float(outside_text_text_box_proximity_ratio_val),
        ),
        provider_settings=UITranslationProviderSettings(
            provider=prov,
            google_api_key=gem_key,
            openai_api_key=oai_key,
            anthropic_api_key=ant_key,
            xai_api_key=xai_key,
            deepseek_api_key=deepseek_key,
            zai_api_key=zai_key,
            moonshot_api_key=moonshot_key,
            openrouter_api_key=or_key,
            openai_compatible_url=comp_url,
            openai_compatible_api_key=comp_key,
        ),
        llm_settings=UITranslationLLMSettings(
            model_name=model,
            temperature=temp,
            top_p=tp,
            top_k=tk,
            max_tokens=max_tokens,
            reading_direction=rd,
            translation_mode=trans_mode,
            ocr_method=ocr_method_val,
            send_full_page_context=send_full_page_context_val,
            upscale_method=upscale_method_val,
            bubble_min_side_pixels=bubble_min_side_pixels_val,
            context_image_max_side_pixels=context_image_max_side_pixels_val,
            osb_min_side_pixels=osb_min_side_pixels_val,
            special_instructions=special_instructions_val,
        ),
        rendering=UIRenderingSettings(
            max_font_size=max_fs,
            min_font_size=min_fs,
            line_spacing_mult=ls,
            use_subpixel_rendering=subpix,
            font_hinting=hint,
            use_ligatures=liga,
            hyphenate_before_scaling=hyphenate_before_scaling_val,
            hyphen_penalty=hyphen_penalty_val,
            hyphenation_min_word_length=hyphenation_min_word_length_val,
            badness_exponent=badness_exponent_val,
            padding_pixels=padding_pixels_val,
            supersampling_factor=int(supersampling_factor_val),
        ),
        output=UIOutputSettings(
            output_format=out_fmt,
            jpeg_quality=jq,
            png_compression=pngc,
            image_upscale_mode=image_upscale_mode_val,
            image_upscale_factor=image_upscale_factor_val,
            image_upscale_model=image_upscale_model_val,
        ),
        general=UIGeneralSettings(
            verbose=verb,
            cleaning_only=cleaning_only_val,
            upscaling_only=upscaling_only_val,
            test_mode=test_mode_val,
            enable_web_search=enable_web_search_val,
            media_resolution=media_resolution_val,
            media_resolution_bubbles=media_resolution_bubbles_val,
            media_resolution_context=media_resolution_context_val,
            reasoning_effort=reasoning_effort_val,
            effort=effort_val,
            auto_scale=auto_scale_val,
        ),
        input_language=s_in_lang,
        output_language=s_out_lang,
        font_pack=s_font,
        batch_input_language=b_in_lang,
        batch_output_language=b_out_lang,
        batch_font_pack=b_font,
        batch_special_instructions=batch_special_instructions_val,
    )

    # Convert UI state to dictionary for saving
    settings_dict = ui_state.to_save_dict()
    # Persist send_full_page_context from the single UI value (used for both single and batch flows).
    settings_dict["send_full_page_context"] = bool(send_full_page_context_val)
    message = settings_manager.save_config(settings_dict)
    return message


def handle_reset_defaults_click(fonts_base_dir: Path) -> List[gr.update]:
    """Callback for the 'Reset Defaults' button. Uses dataclasses."""

    default_settings_dict = settings_manager.reset_to_defaults()
    default_ui_state = UIConfigState.from_dict(default_settings_dict)

    available_fonts, _ = utils.get_available_font_packs(fonts_base_dir)
    reset_single_font = default_ui_state.font_pack
    if reset_single_font not in available_fonts:
        reset_single_font = available_fonts[0] if available_fonts else None
    default_ui_state.font_pack = reset_single_font

    batch_reset_font = default_ui_state.batch_font_pack
    if batch_reset_font not in available_fonts:
        batch_reset_font = available_fonts[0] if available_fonts else None
    default_ui_state.batch_font_pack = batch_reset_font

    default_provider = default_ui_state.provider_settings.provider
    default_model_name = default_ui_state.llm_settings.model_name
    if not default_model_name:
        default_model_name = settings_manager.DEFAULT_SETTINGS["provider_models"].get(
            default_provider
        )
        default_ui_state.llm_settings.model_name = default_model_name

    if default_provider == "OpenRouter" or default_provider == "OpenAI-Compatible":
        default_models_choices = [default_model_name] if default_model_name else []
    else:
        default_models_choices = settings_manager.PROVIDER_MODELS.get(
            default_provider, []
        )

    gemini_visible = default_provider == "Google"
    openai_visible = default_provider == "OpenAI"
    anthropic_visible = default_provider == "Anthropic"
    xai_visible = default_provider == "xAI"
    deepseek_visible = default_provider == "DeepSeek"
    zai_visible = default_provider == "Z.ai"
    moonshot_visible = default_provider == "Moonshot AI"
    openrouter_visible = default_provider == "OpenRouter"
    compatible_visible = default_provider == "OpenAI-Compatible"
    (
        temp_update,
        top_k_update,
        _,  # max_tokens_update - unused (using saved default instead)
        enable_web_search_update,
        media_resolution_update,
        media_resolution_bubbles_update,
        media_resolution_context_update,
        reasoning_effort_update,
        effort_update,
    ) = utils.update_params_for_model(
        default_provider, default_model_name, default_ui_state.llm_settings.temperature
    )
    temp_val = temp_update.get("value", default_ui_state.llm_settings.temperature)
    temp_max = temp_update.get("maximum", 2.0)
    top_k_interactive = top_k_update.get("interactive", True)
    top_k_val = top_k_update.get("value", default_ui_state.llm_settings.top_k)
    is_reasoning = utils.is_reasoning_model(default_provider, default_model_name)
    max_tokens_val = 16384 if is_reasoning else 4096
    enable_web_search_visible = enable_web_search_update.get("visible", False)
    media_resolution_visible = media_resolution_update.get("visible", False)
    media_resolution_bubbles_visible = media_resolution_bubbles_update.get(
        "visible", False
    )
    media_resolution_context_visible = media_resolution_context_update.get(
        "visible", False
    )
    reasoning_visible = reasoning_effort_update.get("visible", False)
    reasoning_effort_val = reasoning_effort_update.get(
        "value", default_ui_state.general.reasoning_effort
    )
    effort_visible = effort_update.get("visible", False)
    effort_val = effort_update.get("value", default_ui_state.general.effort)

    return [
        default_ui_state.detection.confidence,
        default_ui_state.detection.conjoined_confidence,
        gr.update(
            value=default_ui_state.detection.panel_confidence,
            interactive=default_ui_state.detection.use_panel_sorting,
        ),
        default_ui_state.detection.use_sam2,
        default_ui_state.detection.conjoined_detection,
        default_ui_state.detection.use_osb_text_verification,
        default_ui_state.detection.use_panel_sorting,
        default_ui_state.llm_settings.reading_direction,
        default_ui_state.cleaning.thresholding_value,
        default_ui_state.cleaning.use_otsu_threshold,
        default_ui_state.cleaning.inpaint_colored_bubbles,
        default_ui_state.cleaning.roi_shrink_px,
        gr.update(value=default_provider),
        gr.update(
            value=default_ui_state.provider_settings.google_api_key,
            visible=gemini_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openai_api_key,
            visible=openai_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.anthropic_api_key,
            visible=anthropic_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.xai_api_key, visible=xai_visible
        ),
        gr.update(
            value=default_ui_state.provider_settings.deepseek_api_key,
            visible=deepseek_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.zai_api_key,
            visible=zai_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.moonshot_api_key,
            visible=moonshot_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openrouter_api_key,
            visible=openrouter_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openai_compatible_url,
            visible=compatible_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openai_compatible_api_key,
            visible=compatible_visible,
        ),
        gr.update(choices=default_models_choices, value=default_model_name),
        gr.update(value=temp_val, maximum=temp_max),
        default_ui_state.llm_settings.top_p,
        gr.update(value=top_k_val, interactive=top_k_interactive),
        gr.update(value=max_tokens_val),
        gr.update(value=default_ui_state.llm_settings.translation_mode),
        gr.update(value=default_ui_state.llm_settings.ocr_method),
        default_ui_state.rendering.max_font_size,
        default_ui_state.rendering.min_font_size,
        default_ui_state.rendering.line_spacing_mult,
        default_ui_state.rendering.use_subpixel_rendering,
        default_ui_state.rendering.font_hinting,
        default_ui_state.rendering.use_ligatures,
        default_ui_state.output.output_format,
        default_ui_state.output.jpeg_quality,
        default_ui_state.output.png_compression,
        default_ui_state.general.verbose,
        default_ui_state.general.cleaning_only,
        default_ui_state.general.upscaling_only,
        default_ui_state.general.test_mode,
        default_ui_state.input_language,
        default_ui_state.output_language,
        gr.update(value=default_ui_state.font_pack),
        default_ui_state.batch_input_language,
        default_ui_state.batch_output_language,
        gr.update(value=default_ui_state.batch_font_pack),
        gr.update(
            value=default_ui_state.general.enable_web_search,
            visible=enable_web_search_visible,
        ),
        gr.update(
            value=default_ui_state.general.media_resolution,
            visible=media_resolution_visible,
        ),
        gr.update(
            value=default_ui_state.general.media_resolution_bubbles,
            visible=media_resolution_bubbles_visible,
        ),
        gr.update(
            value=default_ui_state.general.media_resolution_context,
            visible=media_resolution_context_visible,
        ),
        gr.update(value=reasoning_effort_val, visible=reasoning_visible),
        gr.update(value=effort_val, visible=effort_visible),
        "Settings reset to defaults (API keys preserved).",
        gr.update(value=default_ui_state.llm_settings.send_full_page_context),
        gr.update(value=default_ui_state.llm_settings.upscale_method),
        default_ui_state.llm_settings.bubble_min_side_pixels,
        default_ui_state.llm_settings.context_image_max_side_pixels,
        default_ui_state.llm_settings.osb_min_side_pixels,
        gr.update(value=default_ui_state.rendering.hyphenate_before_scaling),
        default_ui_state.llm_settings.special_instructions or "",
        default_ui_state.batch_special_instructions or "",
        default_ui_state.outside_text.enabled,
        default_ui_state.outside_text.seed,
        gr.update(value=default_ui_state.outside_text.force_cv2_inpainting),
        default_ui_state.outside_text.flux_num_inference_steps,
        default_ui_state.outside_text.flux_residual_diff_threshold,
        default_ui_state.outside_text.osb_confidence,
        gr.update(value=default_ui_state.outside_text.enable_page_number_filtering),
        gr.update(
            value=default_ui_state.outside_text.page_filter_margin_threshold,
            interactive=default_ui_state.outside_text.enable_page_number_filtering,
        ),
        gr.update(
            value=default_ui_state.outside_text.page_filter_min_area_ratio,
            interactive=default_ui_state.outside_text.enable_page_number_filtering,
        ),
        default_ui_state.outside_text.huggingface_token,
        gr.update(value=default_ui_state.outside_text.osb_font_name),
        default_ui_state.outside_text.osb_max_font_size,
        default_ui_state.outside_text.osb_min_font_size,
        default_ui_state.outside_text.osb_use_ligatures,
        default_ui_state.outside_text.osb_outline_width,
        default_ui_state.outside_text.osb_line_spacing,
        default_ui_state.outside_text.osb_use_subpixel_rendering,
        default_ui_state.outside_text.osb_font_hinting,
        default_ui_state.outside_text.bbox_expansion_percent,
        default_ui_state.outside_text.text_box_proximity_ratio,
        gr.update(value=default_ui_state.output.image_upscale_mode),
        gr.update(
            value=default_ui_state.output.image_upscale_factor,
            interactive=default_ui_state.output.image_upscale_mode != "off",
        ),
        gr.update(
            value=default_ui_state.output.image_upscale_model,
            interactive=default_ui_state.output.image_upscale_mode != "off",
        ),
        default_ui_state.general.auto_scale,
    ]


def handle_provider_change(provider: str, current_temp: float, ocr_method: str = "LLM"):
    """Handles changes in the provider selector."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_translation_cache()
    cache.clear_manga_ocr_cache()
    return utils.update_translation_ui(provider, current_temp, ocr_method)


def handle_output_format_change(output_format_value: str):
    """Handles changes in the output format radio button."""
    is_jpeg = output_format_value == "jpeg"
    is_png = output_format_value == "png"

    jpeg_interactive = not is_png
    png_interactive = not is_jpeg

    return gr.update(interactive=jpeg_interactive), gr.update(
        interactive=png_interactive
    )


def handle_refresh_resources_click(fonts_base_dir: Path):
    """Callback for the 'Refresh Models / Fonts' button."""
    return utils.refresh_models_and_fonts(fonts_base_dir)


def handle_unload_models_click():
    """Callback for the 'Unload Models' button."""
    try:
        from core.ml.model_manager import get_model_manager

        model_manager = get_model_manager()
        model_manager.unload_all()
        gr.Info("All models unloaded from memory successfully.")
    except Exception as e:
        gr.Error(f"Error unloading models: {str(e)}")


def handle_model_change(provider: str, model_name: Optional[str], current_temp: float):
    """Handles changes in the model name dropdown."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_translation_cache()
    cache.clear_manga_ocr_cache()
    return utils.update_params_for_model(provider, model_name, current_temp)


def handle_app_load(provider: str, url: str, key: Optional[str]):
    """Callback for the app.load event to fetch dynamic models."""
    return utils.initial_dynamic_fetch(provider, url, key)


def update_process_buttons(
    processing: bool, button_text_processing: str, button_text_idle: str
):
    """Toggle visibility and interactivity of processing buttons."""
    global CANCELLATION_MANAGER
    if not processing:
        CANCELLATION_MANAGER = None
    return (
        gr.update(
            interactive=not processing,
            value=button_text_processing if processing else button_text_idle,
        ),
        gr.update(visible=not processing),  # Clear button
        gr.update(visible=processing),  # Cancel button
        # Also update the other tab's buttons to prevent concurrent runs
        gr.update(interactive=not processing),
        gr.update(visible=not processing),
        gr.update(visible=processing),
    )


def cancel_process():
    """Signal that the current process should be cancelled."""
    global CANCELLATION_MANAGER
    if CANCELLATION_MANAGER:
        CANCELLATION_MANAGER.cancel()


def handle_thresholding_change(use_otsu_threshold: bool):
    """Handles changes in the Otsu thresholding checkbox to enable/disable thresholding_value slider."""
    return gr.update(interactive=not use_otsu_threshold)


def handle_hyphenation_change(hyphenate_before_scaling: bool):
    """Handles changes in the hyphenation checkbox to enable/disable hyphenation-related sliders."""
    return gr.update(interactive=hyphenate_before_scaling), gr.update(
        interactive=hyphenate_before_scaling
    )


def handle_cleaning_only_change(cleaning_only: bool):
    """Handles changes in the cleaning-only checkbox to disable upscaling-only and test mode if enabled."""
    return (
        gr.update(
            interactive=not cleaning_only, value=False if cleaning_only else None
        ),
        gr.update(
            interactive=not cleaning_only, value=False if cleaning_only else None
        ),
    )


def handle_upscaling_only_change(upscaling_only: bool):
    """Handles changes in the upscaling-only checkbox to disable cleaning-only and test mode if enabled."""
    return (
        gr.update(
            interactive=not upscaling_only, value=False if upscaling_only else None
        ),
        gr.update(
            interactive=not upscaling_only, value=False if upscaling_only else None
        ),
    )


def handle_test_mode_change(test_mode: bool):
    """Handles changes in the test mode checkbox to disable cleaning-only and upscaling-only if enabled."""
    return (
        gr.update(interactive=not test_mode, value=False if test_mode else None),
        gr.update(interactive=not test_mode, value=False if test_mode else None),
    )


def handle_conjoined_detection_change(_conjoined_detection: bool):
    """Handles changes in the conjoined detection checkbox to clear SAM cache."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_sam_cache()
    return gr.update(interactive=_conjoined_detection)


def handle_confidence_threshold_change(_confidence: float):
    """Handles changes in confidence threshold settings to clear YOLO cache."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_yolo_cache()
    return None


def handle_ocr_method_change(
    ocr_method: str,
    input_language: str,
    original_language_state: str,
    batch_input_language: str,
    batch_original_language_state: str,
    provider: str,
    current_model: Optional[str],
    openai_compatible_url: str,
    openai_compatible_api_key: Optional[str],
):
    """Handles changes in OCR method selection."""
    import gradio as gr

    from core.caching import get_cache

    from . import layout, utils

    cache = get_cache()
    cache.clear_translation_cache()
    cache.clear_manga_ocr_cache()

    updates = []

    # Filter provider selector based on OCR method
    available_providers = utils.get_available_providers(ocr_method)
    current_provider = (
        provider
        if provider in available_providers
        else available_providers[0] if available_providers else "Google"
    )
    provider_selector_update = gr.update(
        choices=available_providers, value=current_provider
    )
    updates.append(provider_selector_update)

    if ocr_method == "manga-ocr":
        if input_language != "Japanese":
            saved_language = input_language
        else:
            saved_language = original_language_state

        updates.append(
            gr.update(value="Japanese", choices=["Japanese"], interactive=False)
        )
        updates.append(saved_language)

        if batch_input_language != "Japanese":
            batch_saved_language = batch_input_language
        else:
            batch_saved_language = batch_original_language_state

        updates.append(
            gr.update(value="Japanese", choices=["Japanese"], interactive=False)
        )
        updates.append(batch_saved_language)

        updates.append(gr.update(value=False, interactive=False))

        # Trigger model list refresh for providers with dynamic model lists
        if provider == "OpenRouter":
            model_update = utils.fetch_and_update_openrouter_models(
                ocr_method="manga-ocr", current_model=current_model
            )
            updates.append(model_update)
        elif provider == "OpenAI-Compatible":
            model_update = utils.fetch_and_update_compatible_models(
                openai_compatible_url, openai_compatible_api_key, current_model
            )
            updates.append(model_update)
        elif provider == "Z.ai":
            # For manga-ocr mode, show all Z.ai models (text-only models work)
            models = settings_manager.PROVIDER_MODELS.get("Z.ai", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Z.ai")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        else:
            updates.append(gr.update())
    else:
        restored_language = (
            original_language_state if original_language_state else input_language
        )
        updates.append(
            gr.update(
                value=restored_language,
                choices=layout.SOURCE_LANGUAGES,
                interactive=True,
            )
        )
        updates.append(original_language_state)

        batch_restored_language = (
            batch_original_language_state
            if batch_original_language_state
            else batch_input_language
        )
        updates.append(
            gr.update(
                value=batch_restored_language,
                choices=layout.SOURCE_LANGUAGES,
                interactive=True,
            )
        )
        updates.append(batch_original_language_state)

        saved_settings = settings_manager.get_saved_settings()
        restored_send_full_page_context = saved_settings.get(
            "send_full_page_context", True
        )
        updates.append(
            gr.update(value=restored_send_full_page_context, interactive=True)
        )

        # Trigger model list refresh for providers with dynamic or filtered model lists
        if provider == "OpenRouter":
            model_update = utils.fetch_and_update_openrouter_models(
                ocr_method="LLM", current_model=current_model
            )
            updates.append(model_update)
        elif provider == "OpenAI-Compatible":
            model_update = utils.fetch_and_update_compatible_models(
                openai_compatible_url, openai_compatible_api_key, current_model
            )
            updates.append(model_update)
        elif provider == "Z.ai":
            # For LLM OCR mode, only show Z.ai vision models
            models = ["glm-4.5v", "glm-4.6v"]
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Z.ai")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        else:
            updates.append(gr.update())

    return updates


def handle_translation_mode_change(translation_mode: str, current_ocr_method: str):
    """Handles changes in translation mode to enable/disable OCR method selection."""
    import gradio as gr

    if translation_mode == "one-step":
        if current_ocr_method == "manga-ocr":
            return gr.update(value="LLM", interactive=False)
        else:
            return gr.update(interactive=False)
    else:
        return gr.update(interactive=True)
