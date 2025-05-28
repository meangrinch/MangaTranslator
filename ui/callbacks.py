import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch

import gradio as gr
from PIL import Image

from . import logic
from . import utils
from . import settings_manager
from .ui_models import (
    UIConfigState,
    UIDetectionSettings,
    UICleaningSettings,
    UITranslationProviderSettings,
    UITranslationLLMSettings,
    UIRenderingSettings,
    UIOutputSettings,
    UIGeneralSettings,
    map_ui_to_backend_config,
)
from core.models import MangaTranslatorConfig
from utils.exceptions import ValidationError

ERROR_PREFIX = "❌ Error: "
SUCCESS_PREFIX = "✅ "


def _format_single_success_message(
    result_image: Image.Image,
    backend_config: MangaTranslatorConfig,
    selected_yolo_model_name: str,
    font_dir_path: Path,
    save_path: Path,
    processing_time: float,
) -> str:
    """Formats the success message for single image translation."""
    width, height = result_image.size
    provider = backend_config.translation.provider
    model_name = backend_config.translation.model_name
    thinking_status_str = ""
    if provider == "Gemini" and "gemini-2.5-flash" in model_name:
        if backend_config.translation.enable_thinking:
            thinking_status_str = " (thinking)"
        else:
            thinking_status_str = " (no thinking)"
    temp_val = backend_config.translation.temperature
    top_p_val = backend_config.translation.top_p
    top_k_val = backend_config.translation.top_k

    llm_params_str = f"• LLM Params: Temp={temp_val:.2f}, Top-P={top_p_val:.2f}"
    param_notes = ""
    if provider == "Gemini":
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
    elif provider == "OpenAI-compatible":
        llm_params_str += f", Top-K={top_k_val}"
    llm_params_str += param_notes

    processing_mode_str = "Cleaning Only" if backend_config.cleaning_only else "Translation"
    msg_parts = [
        f"{SUCCESS_PREFIX}{processing_mode_str} completed!\n",
        f"• Image size: {width}x{height} pixels\n",
        f"• Provider: {provider}\n",
        f"• Model: {model_name}{thinking_status_str}\n",
        f"• Source language: {backend_config.translation.input_language}\n",
        f"• Target language: {backend_config.translation.output_language}\n",
        f"• Reading Direction: {backend_config.translation.reading_direction.upper()}\n",
        f"• Font pack: {font_dir_path.name}\n",
        f"• Translation Mode: {backend_config.translation.translation_mode}\n",
        f"• YOLO Model: {selected_yolo_model_name}\n",
        (
            f"• Cleaning Params: DKS:{backend_config.cleaning.dilation_kernel_size}, "
            f"DI:{backend_config.cleaning.dilation_iterations}, "
            f"Otsu:{backend_config.cleaning.use_otsu_threshold}, "
            f"MCA:{backend_config.cleaning.min_contour_area}, "
            f"CKS:{backend_config.cleaning.closing_kernel_size}, "
            f"CI:{backend_config.cleaning.closing_iterations}, "
            f"CEKS:{backend_config.cleaning.constraint_erosion_kernel_size}, "
            f"CEI:{backend_config.cleaning.constraint_erosion_iterations}\n"
        ),
    ]
    if not backend_config.cleaning_only:
        msg_parts.append(f"{llm_params_str}\n")

    if backend_config.output.output_format == "png":
        msg_parts.append("• Output format: png\n")
        msg_parts.append(f"• PNG compression: {backend_config.output.png_compression}\n")
    elif backend_config.output.output_format == "jpeg":
        msg_parts.append("• Output format: jpeg\n")
        msg_parts.append(f"• JPEG quality: {backend_config.output.jpeg_quality}\n")
    else:  # Auto
        msg_parts.append("• Output format: auto\n")
        msg_parts.append(f"• PNG compression: {backend_config.output.png_compression}\n")
        msg_parts.append(f"• JPEG quality: {backend_config.output.jpeg_quality}\n")

    msg_parts.extend([f"• Processing time: {processing_time:.2f} seconds\n", f"• Saved to: {save_path}"])
    return "".join(msg_parts)


def _format_batch_success_message(
    results: Dict[str, Any], backend_config: MangaTranslatorConfig, selected_yolo_model_name: str, font_dir_path: Path
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
    thinking_status_str = ""
    if provider == "Gemini" and "gemini-2.5-flash" in model_name:
        if backend_config.translation.enable_thinking:
            thinking_status_str = " (thinking)"
        else:
            thinking_status_str = " (no thinking)"
    temp_val = backend_config.translation.temperature
    top_p_val = backend_config.translation.top_p
    top_k_val = backend_config.translation.top_k

    llm_params_str = f"• LLM Params: Temp={temp_val:.2f}, Top-P={top_p_val:.2f}"
    param_notes = ""
    if provider == "Gemini":
        llm_params_str += f", Top-K={top_k_val}"
    elif provider == "Anthropic":
        llm_params_str += f", Top-K={top_k_val}"
        param_notes = " (Temp clamped <= 1.0)"
    elif provider == "OpenAI":
        param_notes = " (Top-K N/A)"
    elif provider == "OpenRouter":
        is_openai_model = "openai/" in model_name or model_name.startswith("gpt-")
        is_anthropic_model = "anthropic/" in model_name or model_name.startswith("claude-")
        if is_openai_model or is_anthropic_model:
            param_notes = " (Temp clamped <= 1.0, Top-K N/A)"
        else:
            llm_params_str += f", Top-K={top_k_val}"
    elif provider == "OpenAI-compatible":
        param_notes = " (Top-K N/A)"
    llm_params_str += param_notes

    error_summary = ""
    if error_count > 0:
        error_summary = f"\n• Warning: {error_count} image(s) failed to process."

    processing_mode_str = "Cleaning Only" if backend_config.cleaning_only else "Translation"
    msg_parts = [
        f"{SUCCESS_PREFIX}Batch {processing_mode_str.lower()} completed!\n",
        f"• Provider: {provider}\n",
        f"• Model: {model_name}{thinking_status_str}\n",
        f"• Source language: {backend_config.translation.input_language}\n",
        f"• Target language: {backend_config.translation.output_language}\n",
        f"• Reading Direction: {backend_config.translation.reading_direction.upper()}\n",
        f"• Font pack: {font_dir_path.name}\n",
        f"• YOLO Model: {selected_yolo_model_name}\n",
        f"• Translation Mode: {backend_config.translation.translation_mode}\n",
        (
            f"• Cleaning Params: DKS:{backend_config.cleaning.dilation_kernel_size}, "
            f"DI:{backend_config.cleaning.dilation_iterations}, "
            f"Otsu:{backend_config.cleaning.use_otsu_threshold}, "
            f"MCA:{backend_config.cleaning.min_contour_area}, "
            f"CKS:{backend_config.cleaning.closing_kernel_size}, "
            f"CI:{backend_config.cleaning.closing_iterations}, "
            f"CEKS:{backend_config.cleaning.constraint_erosion_kernel_size}, "
            f"CEI:{backend_config.cleaning.constraint_erosion_iterations}\n"
        ),
    ]
    if not backend_config.cleaning_only:
        msg_parts.append(f"{llm_params_str}\n")

    if backend_config.output.output_format == "png":
        msg_parts.append("• Output format: png\n")
        msg_parts.append(f"• PNG compression: {backend_config.output.png_compression}\n")
    elif backend_config.output.output_format == "jpeg":
        msg_parts.append("• Output format: jpeg\n")
        msg_parts.append(f"• JPEG quality: {backend_config.output.jpeg_quality}\n")
    else:  # Auto
        msg_parts.append("• Output format: auto\n")
        msg_parts.append(f"• PNG compression: {backend_config.output.png_compression}\n")
        msg_parts.append(f"• JPEG quality: {backend_config.output.jpeg_quality}\n")

    msg_parts.extend(
        [
            f"• Successful translations: {success_count}/{total_images}{error_summary}\n",
            f"• Total processing time: {processing_time:.2f} seconds ({seconds_per_image:.2f} seconds/image)\n",
            f"• Saved to: {output_path}",
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
    (
        input_image_path,
        config_yolo_model_dropdown,
        confidence,
        dilation_kernel_size,
        dilation_iterations,
        use_otsu_threshold,
        min_contour_area,
        closing_kernel_size,
        closing_iterations,
        constraint_erosion_kernel_size,
        constraint_erosion_iterations,
        provider_selector,
        gemini_api_key,
        openai_api_key,
        anthropic_api_key,
        openrouter_api_key,
        openai_compatible_url_input,
        openai_compatible_api_key_input,
        config_model_name,
        temperature,
        top_p,
        top_k,
        config_reading_direction,
        config_translation_mode,
        input_language,
        output_language,
        font_dropdown,
        max_font_size,
        min_font_size,
        line_spacing,
        use_subpixel_rendering,
        font_hinting,
        use_ligatures,
        output_format,
        jpeg_quality,
        png_compression,
        verbose,
        cleaning_only_toggle,
        enable_thinking_checkbox_val,
    ) = args
    """Callback for the 'Translate' button click."""
    start_time = time.time()
    try:
        # --- UI Validation ---
        img_valid, img_msg = utils.validate_image(input_image_path)
        if not img_valid:
            raise gr.Error(f"{ERROR_PREFIX}{img_msg}")

        api_key_to_validate = ""
        if provider_selector == "Gemini":
            api_key_to_validate = gemini_api_key
        elif provider_selector == "OpenAI":
            api_key_to_validate = openai_api_key
        elif provider_selector == "Anthropic":
            api_key_to_validate = anthropic_api_key
        elif provider_selector == "OpenRouter":
            api_key_to_validate = openrouter_api_key
        elif provider_selector == "OpenAI-compatible":
            api_key_to_validate = openai_compatible_api_key_input

        api_valid, api_msg = utils.validate_api_key(api_key_to_validate, provider_selector)
        if not api_valid and not (provider_selector == "OpenAI-compatible" and not api_key_to_validate):
            raise gr.Error(f"{ERROR_PREFIX}{api_msg}")

        if provider_selector == "OpenAI-compatible":
            if not openai_compatible_url_input:
                raise gr.Error(f"{ERROR_PREFIX}OpenAI-Compatible URL is required.")
            if not openai_compatible_url_input.startswith(("http://", "https://")):
                raise gr.Error(
                    f"{ERROR_PREFIX}Invalid OpenAI-Compatible URL format. Must start with http:// or https://",
                )

        # --- Build UI State Dataclass ---
        ui_state = UIConfigState(
            yolo_model=config_yolo_model_dropdown,
            detection=UIDetectionSettings(confidence=confidence),
            cleaning=UICleaningSettings(
                dilation_kernel_size=dilation_kernel_size,
                dilation_iterations=dilation_iterations,
                use_otsu_threshold=use_otsu_threshold,
                min_contour_area=min_contour_area,
                closing_kernel_size=closing_kernel_size,
                closing_iterations=closing_iterations,
                constraint_erosion_kernel_size=constraint_erosion_kernel_size,
                constraint_erosion_iterations=constraint_erosion_iterations,
            ),
            provider_settings=UITranslationProviderSettings(
                provider=provider_selector,
                gemini_api_key=gemini_api_key,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                openrouter_api_key=openrouter_api_key,
                openai_compatible_url=openai_compatible_url_input,
                openai_compatible_api_key=openai_compatible_api_key_input,
            ),
            llm_settings=UITranslationLLMSettings(
                model_name=config_model_name,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                reading_direction=config_reading_direction,
                translation_mode=config_translation_mode,
            ),
            rendering=UIRenderingSettings(
                max_font_size=max_font_size,
                min_font_size=min_font_size,
                line_spacing=line_spacing,
                use_subpixel_rendering=use_subpixel_rendering,
                font_hinting=font_hinting,
                use_ligatures=use_ligatures,
            ),
            output=UIOutputSettings(
                output_format=output_format,
                jpeg_quality=jpeg_quality,
                png_compression=png_compression,
            ),
            general=UIGeneralSettings(
                verbose=verbose, cleaning_only=cleaning_only_toggle, enable_thinking=enable_thinking_checkbox_val
            ),
            input_language=input_language,
            output_language=output_language,
            font_pack=font_dropdown,
        )

        # --- Map UI State to Backend Config ---
        backend_config = map_ui_to_backend_config(
            ui_state=ui_state,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            target_device=target_device,
            is_batch=False,
        )
        selected_yolo_model_name = ui_state.yolo_model
        selected_font_pack_name = ui_state.font_pack

        # --- Call Logic ---
        result_image, save_path = logic.translate_manga_logic(
            image=input_image_path,
            config=backend_config,
            selected_yolo_model_name=selected_yolo_model_name,
            selected_font_pack_name=selected_font_pack_name,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
        )

        # --- Format Success Output ---
        # Re-get font path for success message (logic already validated it)
        font_dir_path = fonts_base_dir / selected_font_pack_name if selected_font_pack_name else Path(".")

        processing_time = time.time() - start_time
        status_msg = _format_single_success_message(
            result_image, backend_config, selected_yolo_model_name, font_dir_path, save_path, processing_time
        )

        return result_image.copy(), status_msg

    except (ValidationError, FileNotFoundError, ValueError, logic.LogicError) as e:
        # Catch specific errors from validation or logic
        return gr.update(), f"{ERROR_PREFIX}Processing Error: {str(e)}"
    except gr.Error as e:
        # Catch UI validation errors
        return gr.update(), str(e)
    except Exception as e:
        # Catch unexpected errors
        import traceback

        traceback.print_exc()
        return gr.update(), f"{ERROR_PREFIX}An unexpected error occurred: {str(e)}"


def handle_batch_click(
    *args: Any,
    models_dir: Path,
    fonts_base_dir: Path,
    target_device: Optional[torch.device],
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[List[str], str]:
    """Callback for the 'Start Batch Translating' button click. Uses dataclasses."""
    (
        input_files,
        config_yolo_model_dropdown,
        confidence,
        dilation_kernel_size,
        dilation_iterations,
        use_otsu_threshold,
        min_contour_area,
        closing_kernel_size,
        closing_iterations,
        constraint_erosion_kernel_size,
        constraint_erosion_iterations,
        provider_selector,
        gemini_api_key,
        openai_api_key,
        anthropic_api_key,
        openrouter_api_key,
        openai_compatible_url_input,
        openai_compatible_api_key_input,
        config_model_name,
        temperature,
        top_p,
        top_k,
        config_reading_direction,
        config_translation_mode,
        batch_input_language,
        batch_output_language,
        batch_font_dropdown,
        max_font_size,
        min_font_size,
        line_spacing,
        use_subpixel_rendering,
        font_hinting,
        use_ligatures,
        output_format,
        jpeg_quality,
        png_compression,
        verbose,
        cleaning_only_toggle,
        enable_thinking_checkbox_val,
    ) = args
    """Callback for the 'Start Batch Translating' button click."""
    progress(0, desc="Starting batch process...")
    try:
        # --- UI Validation ---
        if not input_files:
            raise gr.Error(f"{ERROR_PREFIX}Please upload images or a folder.")

        if not isinstance(input_files, list):
            raise gr.Error(f"{ERROR_PREFIX}Invalid input format. Expected a list of files.")

        api_key_to_validate = ""
        if provider_selector == "Gemini":
            api_key_to_validate = gemini_api_key
        elif provider_selector == "OpenAI":
            api_key_to_validate = openai_api_key
        elif provider_selector == "Anthropic":
            api_key_to_validate = anthropic_api_key
        elif provider_selector == "OpenRouter":
            api_key_to_validate = openrouter_api_key
        elif provider_selector == "OpenAI-compatible":
            api_key_to_validate = openai_compatible_api_key_input

        api_valid, api_msg = utils.validate_api_key(api_key_to_validate, provider_selector)
        if not api_valid and not (provider_selector == "OpenAI-compatible" and not api_key_to_validate):
            raise gr.Error(f"{ERROR_PREFIX}{api_msg}")

        if provider_selector == "OpenAI-compatible":
            if not openai_compatible_url_input:
                raise gr.Error(f"{ERROR_PREFIX}OpenAI-Compatible URL is required.")
            if not openai_compatible_url_input.startswith(("http://", "https://")):
                raise gr.Error(
                    f"{ERROR_PREFIX}Invalid OpenAI-Compatible URL format. Must start with http:// or https://",
                )

        # --- Build UI State Dataclass ---
        ui_state = UIConfigState(
            yolo_model=config_yolo_model_dropdown,
            detection=UIDetectionSettings(confidence=confidence),
            cleaning=UICleaningSettings(
                dilation_kernel_size=dilation_kernel_size,
                dilation_iterations=dilation_iterations,
                use_otsu_threshold=use_otsu_threshold,
                min_contour_area=min_contour_area,
                closing_kernel_size=closing_kernel_size,
                closing_iterations=closing_iterations,
                constraint_erosion_kernel_size=constraint_erosion_kernel_size,
                constraint_erosion_iterations=constraint_erosion_iterations,
            ),
            provider_settings=UITranslationProviderSettings(
                provider=provider_selector,
                gemini_api_key=gemini_api_key,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                openrouter_api_key=openrouter_api_key,
                openai_compatible_url=openai_compatible_url_input,
                openai_compatible_api_key=openai_compatible_api_key_input,
            ),
            llm_settings=UITranslationLLMSettings(
                model_name=config_model_name,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                reading_direction=config_reading_direction,
                translation_mode=config_translation_mode,
            ),
            rendering=UIRenderingSettings(
                max_font_size=max_font_size,
                min_font_size=min_font_size,
                line_spacing=line_spacing,
                use_subpixel_rendering=use_subpixel_rendering,
                font_hinting=font_hinting,
                use_ligatures=use_ligatures,
            ),
            output=UIOutputSettings(
                output_format=output_format,
                jpeg_quality=jpeg_quality,
                png_compression=png_compression,
            ),
            general=UIGeneralSettings(
                verbose=verbose, cleaning_only=cleaning_only_toggle, enable_thinking=enable_thinking_checkbox_val
            ),
            batch_input_language=batch_input_language,
            batch_output_language=batch_output_language,
            batch_font_pack=batch_font_dropdown,
        )

        # --- Map UI State to Backend Config ---
        backend_config = map_ui_to_backend_config(
            ui_state=ui_state,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            target_device=target_device,
            is_batch=True,
        )
        selected_yolo_model_name = ui_state.yolo_model
        selected_font_pack_name = ui_state.batch_font_pack

        # --- Call Logic ---
        results = logic.process_batch_logic(
            input_dir_or_files=input_files,
            config=backend_config,
            selected_yolo_model_name=selected_yolo_model_name,
            selected_font_pack_name=selected_font_pack_name,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            gradio_progress=progress,
        )

        # --- Format Success Output ---
        output_path = results["output_path"]
        gallery_images = []
        if output_path.exists():
            processed_files = list(output_path.glob("*.*"))
            processed_files.sort(  # Sort naturally (e.g., page_1, page_2, page_10)
                key=lambda x: tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", x.stem))
            )
            gallery_images = [str(file_path) for file_path in processed_files]

        # Re-get font path for success message (logic already validated it)
        font_dir_path = fonts_base_dir / selected_font_pack_name if selected_font_pack_name else Path(".")

        status_msg = _format_batch_success_message(results, backend_config, selected_yolo_model_name, font_dir_path)
        progress(1.0, desc="Batch complete!")
        return gallery_images, status_msg

    except (ValidationError, FileNotFoundError, ValueError, logic.LogicError) as e:
        progress(1.0, desc="Error occurred")
        return gr.update(), f"{ERROR_PREFIX}Batch Processing Error: {str(e)}"
    except gr.Error as e:
        progress(1.0, desc="Error occurred")
        return gr.update(), str(e)
    except Exception as e:
        progress(1.0, desc="Error occurred")
        import traceback

        traceback.print_exc()
        return gr.update(), f"{ERROR_PREFIX}An unexpected error occurred during batch processing: {str(e)}"


def handle_save_config_click(*args: Any) -> str:
    """Callback for the 'Save Config' button. Uses dataclasses."""
    (
        yolo,
        conf,
        rd,
        dks,
        di,
        otsu,
        mca,
        cks,
        ci,
        ceks,
        cei,
        prov,
        gem_key,
        oai_key,
        ant_key,
        or_key,
        comp_url,
        comp_key,
        model,
        temp,
        tp,
        tk,
        trans_mode,
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
        s_in_lang,
        s_out_lang,
        s_font,
        b_in_lang,
        b_out_lang,
        b_font,
        enable_thinking_val,
    ) = args
    """Callback for the 'Save Config' button."""
    # Build UI State Dataclass from inputs
    ui_state = UIConfigState(
        yolo_model=yolo,
        detection=UIDetectionSettings(confidence=conf),
        cleaning=UICleaningSettings(
            dilation_kernel_size=dks,
            dilation_iterations=di,
            use_otsu_threshold=otsu,
            min_contour_area=mca,
            closing_kernel_size=cks,
            closing_iterations=ci,
            constraint_erosion_kernel_size=ceks,
            constraint_erosion_iterations=cei,
        ),
        provider_settings=UITranslationProviderSettings(
            provider=prov,
            gemini_api_key=gem_key,
            openai_api_key=oai_key,
            anthropic_api_key=ant_key,
            openrouter_api_key=or_key,
            openai_compatible_url=comp_url,
            openai_compatible_api_key=comp_key,
        ),
        llm_settings=UITranslationLLMSettings(
            model_name=model,
            temperature=temp,
            top_p=tp,
            top_k=tk,
            reading_direction=rd,
            translation_mode=trans_mode,
        ),
        rendering=UIRenderingSettings(
            max_font_size=max_fs,
            min_font_size=min_fs,
            line_spacing=ls,
            use_subpixel_rendering=subpix,
            font_hinting=hint,
            use_ligatures=liga,
        ),
        output=UIOutputSettings(
            output_format=out_fmt,
            jpeg_quality=jq,
            png_compression=pngc,
        ),
        general=UIGeneralSettings(verbose=verb, cleaning_only=cleaning_only_val, enable_thinking=enable_thinking_val),
        input_language=s_in_lang,
        output_language=s_out_lang,
        font_pack=s_font,
        batch_input_language=b_in_lang,
        batch_output_language=b_out_lang,
        batch_font_pack=b_font,
    )

    # Convert UI state to dictionary for saving
    settings_dict = ui_state.to_save_dict()
    message = settings_manager.save_config(settings_dict)
    return message


def handle_reset_defaults_click(models_dir: Path, fonts_base_dir: Path) -> List[gr.update]:
    """Callback for the 'Reset Defaults' button. Uses dataclasses."""

    default_settings_dict = settings_manager.reset_to_defaults()
    default_ui_state = UIConfigState.from_dict(default_settings_dict)

    # --- Handle dynamic choices based on defaults ---
    available_yolo_models = utils.get_available_models(models_dir)
    reset_yolo_model = default_ui_state.yolo_model
    if reset_yolo_model not in available_yolo_models:
        reset_yolo_model = available_yolo_models[0] if available_yolo_models else None
    default_ui_state.yolo_model = reset_yolo_model

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
    default_model_name = settings_manager.DEFAULT_SETTINGS["provider_models"].get(default_provider)
    default_ui_state.llm_settings.model_name = default_model_name

    if default_provider == "OpenRouter" or default_provider == "OpenAI-compatible":
        default_models_choices = [default_model_name] if default_model_name else []
    else:
        default_models_choices = settings_manager.PROVIDER_MODELS.get(default_provider, [])

    gemini_visible = default_provider == "Gemini"
    openai_visible = default_provider == "OpenAI"
    anthropic_visible = default_provider == "Anthropic"
    openrouter_visible = default_provider == "OpenRouter"
    compatible_visible = default_provider == "OpenAI-compatible"
    temp_update, top_k_update = utils.update_params_for_model(
        default_provider, default_model_name, default_ui_state.llm_settings.temperature
    )
    temp_val = temp_update.get("value", default_ui_state.llm_settings.temperature)
    temp_max = temp_update.get("maximum", 2.0)
    top_k_interactive = top_k_update.get("interactive", True)
    top_k_val = top_k_update.get("value", default_ui_state.llm_settings.top_k)
    enable_thinking_visible = (
        default_provider == "Gemini" and "gemini-2.5-flash" in default_model_name
    )

    return [
        gr.update(value=default_ui_state.yolo_model),
        default_ui_state.detection.confidence,
        default_ui_state.llm_settings.reading_direction,
        default_ui_state.cleaning.dilation_kernel_size,
        default_ui_state.cleaning.dilation_iterations,
        default_ui_state.cleaning.use_otsu_threshold,
        default_ui_state.cleaning.min_contour_area,
        default_ui_state.cleaning.closing_kernel_size,
        default_ui_state.cleaning.closing_iterations,
        default_ui_state.cleaning.constraint_erosion_kernel_size,
        default_ui_state.cleaning.constraint_erosion_iterations,
        gr.update(value=default_provider),
        gr.update(value=default_ui_state.provider_settings.gemini_api_key, visible=gemini_visible),
        gr.update(value=default_ui_state.provider_settings.openai_api_key, visible=openai_visible),
        gr.update(value=default_ui_state.provider_settings.anthropic_api_key, visible=anthropic_visible),
        gr.update(value=default_ui_state.provider_settings.openrouter_api_key, visible=openrouter_visible),
        gr.update(value=default_ui_state.provider_settings.openai_compatible_url, visible=compatible_visible),
        gr.update(value=default_ui_state.provider_settings.openai_compatible_api_key, visible=compatible_visible),
        gr.update(choices=default_models_choices, value=default_model_name),
        gr.update(value=temp_val, maximum=temp_max),
        default_ui_state.llm_settings.top_p,
        gr.update(value=top_k_val, interactive=top_k_interactive),
        gr.update(value=default_ui_state.llm_settings.translation_mode),
        default_ui_state.rendering.max_font_size,
        default_ui_state.rendering.min_font_size,
        default_ui_state.rendering.line_spacing,
        default_ui_state.rendering.use_subpixel_rendering,
        default_ui_state.rendering.font_hinting,
        default_ui_state.rendering.use_ligatures,
        default_ui_state.output.output_format,
        default_ui_state.output.jpeg_quality,
        default_ui_state.output.png_compression,
        default_ui_state.general.verbose,
        default_ui_state.general.cleaning_only,
        default_ui_state.input_language,
        default_ui_state.output_language,
        gr.update(value=default_ui_state.font_pack),
        default_ui_state.batch_input_language,
        default_ui_state.batch_output_language,
        gr.update(value=default_ui_state.batch_font_pack),
        gr.update(value=default_ui_state.general.enable_thinking, visible=enable_thinking_visible),
        "Settings reset to defaults (API keys preserved).",
    ]


# --- UI Element Update Callbacks ---
def handle_provider_change(provider: str, current_temp: float):
    """Handles changes in the provider selector."""
    return utils.update_translation_ui(provider, current_temp)


def handle_output_format_change(output_format_value: str):
    """Handles changes in the output format radio button."""
    is_jpeg = output_format_value == "jpeg"
    is_png = output_format_value == "png"

    jpeg_interactive = not is_png
    png_interactive = not is_jpeg

    return gr.update(interactive=jpeg_interactive), gr.update(interactive=png_interactive)


def handle_refresh_resources_click(models_dir: Path, fonts_base_dir: Path):
    """Callback for the 'Refresh Models / Fonts' button."""
    return utils.refresh_models_and_fonts(models_dir, fonts_base_dir)


def handle_model_change(provider: str, model_name: Optional[str], current_temp: float):
    """Handles changes in the model name dropdown."""
    return utils.update_params_for_model(provider, model_name, current_temp)


def handle_app_load(provider: str, url: str, key: Optional[str]):
    """Callback for the app.load event to fetch dynamic models."""
    return utils.initial_dynamic_fetch(provider, url, key)


# --- Button/Status Update Callbacks ---
def update_button_state(processing: bool, button_text_processing: str, button_text_idle: str):
    """Generic function to update button text and interactivity."""
    return gr.update(interactive=not processing, value=button_text_processing if processing else button_text_idle)


def trigger_status_fade(status_element_id: str):
    """Returns JS code to fade out a status message element after a delay."""
    return None
