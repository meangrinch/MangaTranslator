import argparse
import os
import tempfile
import time
import zipfile
from pathlib import Path

import torch

from core.config import (CleaningConfig, DetectionConfig,
                         MangaTranslatorConfig, OutputConfig,
                         OutsideTextConfig, PreprocessingConfig,
                         RenderingConfig, TranslationConfig)
from core.pipeline import batch_translate_images, translate_and_render
from core.validation import (autodetect_yolo_model_path,
                             validate_mutually_exclusive_modes)
from utils.logging import log_message


def main():
    parser = argparse.ArgumentParser(
        description="Translate manga/comic speech bubbles using a configuration approach"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image, directory, or ZIP archive (if using --batch)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save the translated image or directory (if using --batch)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all images in the input directory or ZIP archive (preserves folder structure for ZIP files)",
    )
    # --- Provider and API Key Arguments ---
    parser.add_argument(
        "--provider",
        type=str,
        default="Google",
        choices=[
            "Google",
            "OpenAI",
            "Anthropic",
            "xAI",
            "OpenRouter",
            "OpenAI-Compatible",
        ],
        help="LLM provider to use for translation",
    )
    parser.add_argument(
        "--google-api-key",
        dest="google_api_key",
        type=str,
        default=None,
        help="Google API key (overrides GOOGLE_API_KEY env var if --provider is Google)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var if --provider is OpenAI)",
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var if --provider is Anthropic)",
    )
    parser.add_argument(
        "--xai-api-key",
        type=str,
        default=None,
        help="xAI API key (overrides XAI_API_KEY env var if --provider is xAI)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=None,
        help="OpenRouter API key (overrides OPENROUTER_API_KEY env var if --provider is OpenRouter)",
    )
    parser.add_argument(
        "--openai-compatible-url",
        type=str,
        default="http://localhost:11434/v1",
        help="Base URL for the OpenAI-Compatible endpoint (e.g., http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--openai-compatible-api-key",
        type=str,
        default=None,
        help="Optional API key for the OpenAI-Compatible endpoint (overrides OPENAI_COMPATIBLE_API_KEY env var)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for the selected provider (e.g., 'gemini-2.5-flash'). "
        "If not provided, a default will be attempted based on the provider.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="./models",
        help="Directory containing YOLO model files",
    )
    parser.add_argument(
        "--font-dir",
        type=str,
        default="./fonts",
        help="Directory containing font files",
    )
    parser.add_argument(
        "--input-language",
        type=str,
        default="Japanese",
        help="Source language",
    )
    parser.add_argument(
        "--output-language", type=str, default="English", help="Target language"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35, help="Confidence threshold for detection"
    )
    parser.add_argument(
        "--no-sam2",
        dest="use_sam2",
        action="store_false",
        help="Disable SAM 2.1 segmentation",
    )
    parser.set_defaults(use_sam2=True)
    parser.add_argument(
        "--no-conjoined-detection",
        dest="enable_conjoined_detection",
        action="store_false",
        help="Disable conjoined bubble detection using secondary YOLO model",
    )
    parser.set_defaults(enable_conjoined_detection=True)
    parser.add_argument(
        "--reading-direction",
        type=str,
        default="rtl",
        choices=["rtl", "ltr"],
        help="Reading direction for sorting bubbles (rtl or ltr)",
    )
    # Cleaning args
    parser.add_argument(
        "--use-otsu-threshold",
        action="store_true",
        help="Use Otsu's method for thresholding instead of the fixed value",
    )
    parser.add_argument(
        "--thresholding-value",
        type=int,
        default=190,
        help=(
            "Fixed threshold value for text detection (0-255). "
            "Lower values help clean edge-hugging text."
        ),
    )
    parser.add_argument(
        "--roi-shrink-px",
        type=int,
        default=4,
        help=(
            "Shrink the threshold ROI inward by N pixels (0-8) before fill. "
            "Lower helps clean edge-hugging text; higher preserves outlines."
        ),
    )
    # Translation args
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Controls creativity. Lower is more deterministic, higher is more random (0.0-2.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Controls diversity. Lower is more focused, higher is more random (0.0-1.0)",
    )
    parser.add_argument(
        "--top-k", type=int, default=1, help="Limits sampling pool to top K tokens"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of tokens in the response (2048-32768). "
            "Default: 4096 for non-reasoning models, 16384 for reasoning models"
        ),
    )
    parser.add_argument(
        "--translation-mode",
        type=str,
        default="one-step",
        choices=["one-step", "two-step"],
        help=(
            "Method for translation ('one-step' combines OCR/Translate, 'two-step' separates them). "
            "'two-step' might improve translation quality for less-capable LLMs"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["high", "medium", "low", "minimal"],
        help=(
            "Controls internal reasoning effort for OpenAI/OpenRouter OpenAI reasoning models "
            "(o1/o3/o4-mini/gpt-5*). Note: 'minimal' is only supported by gpt-5 series."
        ),
    )
    # Rendering args
    parser.add_argument(
        "--max-font-size",
        type=int,
        default=15,
        help="Max font size for rendering text (px)",
    )
    parser.add_argument(
        "--min-font-size",
        type=int,
        default=8,
        help="Min font size for rendering text (px)",
    )
    parser.add_argument(
        "--line-spacing",
        type=float,
        default=1.0,
        help="Line spacing for rendering text (1.0 = standard)",
    )
    parser.add_argument(
        "--no-subpixel-rendering",
        dest="use_subpixel_rendering",
        action="store_false",
        help="Disable subpixel rendering for speech bubble text (disable for OLED displays)",
    )
    parser.set_defaults(use_subpixel_rendering=True)
    parser.add_argument(
        "--font-hinting",
        type=str,
        choices=["none", "slight", "normal", "full"],
        default="none",
        help="Font hinting mode for speech bubble text",
    )
    parser.add_argument(
        "--use-ligatures",
        action="store_true",
        help="Enable standard ligatures for speech bubble text (e.g., fi, fl)",
    )
    parser.add_argument(
        "--no-hyphenate-before-scaling",
        dest="hyphenate_before_scaling",
        action="store_false",
        help="Disable hyphenation of long words before reducing font size.",
    )
    parser.set_defaults(hyphenate_before_scaling=True)
    parser.add_argument(
        "--hyphen-penalty",
        type=float,
        default=1000.0,
        help="Penalty for hyphenated line breaks in text layout (100-2000). Increase to discourage hyphenation.",
    )
    parser.add_argument(
        "--hyphenation-min-word-length",
        type=int,
        default=8,
        help="Minimum word length required for hyphenation (6-10)",
    )
    parser.add_argument(
        "--badness-exponent",
        type=float,
        default=3.0,
        help="Exponent for line badness calculation in text layout (2-4). Increase to avoid loose lines.",
    )
    parser.add_argument(
        "--padding-pixels",
        type=float,
        default=5.0,
        help="Padding between text and the edge of the speech bubble (2-12). "
        "Increase for more space between text and bubble boundaries.",
    )
    # Output args
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG compression quality (1-100)",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=6,
        help="PNG compression level (0-9)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["auto", "png", "jpeg"],
        default="auto",
        help="Output image format (auto uses input format)",
    )
    parser.add_argument(
        "--image-upscale-mode",
        choices=["off", "initial", "final"],
        default="off",
        help="Image upscaling mode: 'off' (none), 'initial' (before processing), or 'final' (after processing).",
    )
    parser.add_argument(
        "--image-upscale-factor",
        type=float,
        default=2.0,
        help="Factor for the selected upscaling mode (1.0-8.0).",
    )
    parser.add_argument(
        "--enable-auto-scale",
        action="store_true",
        help=(
            "Enable automatic scaling of pipeline parameters (fonts, kernels, etc.) "
            "based on image size relative to 1MP. Ensures consistent behavior across different image resolutions."
        ),
    )
    # General args
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--cleaning-only",
        action="store_true",
        help="Skip translation and text rendering, output only the cleaned speech bubbles",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Skip translation and render placeholder text (lorem ipsum)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enables 'thinking' capabilities for Gemini 2.5 Flash (Lite), Claude reasoning models, "
            "and OpenRouter Gemini/Anthropic/Grok models"
        ),
    )
    parser.add_argument(
        "--thinking-level",
        type=str,
        default="high",
        choices=["high", "low"],
        help=("Reasoning effort for Gemini 3 models"),
    )
    parser.add_argument(
        "--enable-grounding",
        action="store_true",
        help=(
            "Enable Google Search grounding for Gemini models (Google and OpenRouter)"
        ),
    )
    parser.add_argument(
        "--media-resolution",
        type=str,
        choices=["auto", "high", "medium", "low"],
        default="auto",
        help="Media resolution for Gemini models (Google provider only, not used for Gemini 3)",
    )
    parser.add_argument(
        "--media-resolution-bubbles",
        type=str,
        choices=["auto", "high", "medium", "low"],
        default="auto",
        help="Media resolution for bubble images (Gemini 3 only)",
    )
    parser.add_argument(
        "--media-resolution-context",
        type=str,
        choices=["auto", "high", "medium", "low"],
        default="auto",
        help="Media resolution for context (full page) images (Gemini 3 only)",
    )
    parser.add_argument(
        "--special-instructions",
        type=str,
        default=None,
        help="Optional special instructions for the LLM (formatting, context, character names, etc.)",
    )
    # Full page context toggle
    parser.add_argument(
        "--no-full-page-context",
        dest="send_full_page_context",
        action="store_false",
        help=(
            "Disable including the full page image as context for translation. Enable if "
            "encountering refusals or using less-capable LLMs"
        ),
    )
    # Translation upscaling method
    parser.add_argument(
        "--upscale-method",
        type=str,
        choices=["model", "lanczos", "none"],
        default="model",
        help=(
            "Method for upscaling images before translation API. "
            "model: Use 2x-AnimeSharpV4 upscaling model (best quality, slower), "
            "lanczos: Use LANCZOS resampling (fast, good quality), "
            "none: No upscaling (may affect OCR quality for small text)"
        ),
    )

    # --- Outside Speech Bubble (OSB) Text Settings ---
    parser.add_argument(
        "--osb-enable",
        action="store_true",
        help="Enable outside speech bubble text detection and removal",
    )
    parser.add_argument(
        "--osb-huggingface-token",
        type=str,
        default=None,
        help="HuggingFace token for Flux Kontext model downloads (overrides HUGGINGFACE_TOKEN env var)",
    )
    parser.add_argument(
        "--osb-flux-steps",
        type=int,
        default=8,
        help=(
            "Number of denoising steps for Flux Kontext (1-30). "
            "15 is best for quality (diminishing returns beyond); "
            "below 6 shows noticeable degradation."
        ),
    )
    parser.add_argument(
        "--osb-flux-residual-threshold",
        type=float,
        default=0.12,
        help="Residual diff threshold for Flux inference (0.0-1.0)",
    )
    parser.add_argument(
        "--osb-seed",
        type=int,
        default=1,
        help="Seed for reproducible inpainting (-1 = random)",
    )
    parser.add_argument(
        "--osb-font-name",
        type=str,
        default=None,
        help="Font name for OSB text rendering (default: use main font)",
    )
    parser.add_argument(
        "--osb-max-font-size",
        type=int,
        default=64,
        help="Maximum font size for OSB text (5-96px)",
    )
    parser.add_argument(
        "--osb-min-font-size",
        type=int,
        default=12,
        help="Minimum font size for OSB text (5-50px)",
    )
    parser.add_argument(
        "--osb-use-ligatures",
        action="store_true",
        help="Enable standard ligatures for OSB text (e.g., fi, fl)",
    )
    parser.add_argument(
        "--osb-outline-width",
        type=float,
        default=3.0,
        help="Outline width for OSB text (0-10px)",
    )
    parser.add_argument(
        "--osb-line-spacing",
        type=float,
        default=1.0,
        help="Line spacing multiplier for OSB text (0.5-2.0)",
    )
    parser.add_argument(
        "--osb-use-subpixel",
        action="store_true",
        default=True,
        help="Enable subpixel rendering for OSB text (disable for OLED displays)",
    )
    parser.add_argument(
        "--osb-font-hinting",
        type=str,
        choices=["none", "slight", "normal", "full"],
        default="none",
        help="Font hinting mode for OSB text",
    )
    parser.add_argument(
        "--osb-bbox-expansion",
        type=float,
        default=0.1,
        help="Bounding box expansion percent for OSB detection",
    )
    parser.add_argument(
        "--osb-easyocr-min-size",
        type=int,
        default=200,
        help="Minimum text region size in pixels for OSB text detection",
    )
    parser.add_argument(
        "--bubble-min-side-pixels",
        type=int,
        default=128,
        help="Target minimum side length for speech bubble upscaling",
    )
    parser.add_argument(
        "--context-image-max-side-pixels",
        type=int,
        default=1536,
        help="Target maximum side length for full page image",
    )
    parser.add_argument(
        "--osb-min-side-pixels",
        type=int,
        default=128,
        help="Target minimum side length for outside speech bubble upscaling",
    )

    parser.set_defaults(send_full_page_context=True)
    parser.set_defaults(
        verbose=False,
        cpu=False,
        cleaning_only=False,
        enable_thinking=False,
        enable_grounding=False,
    )

    args = parser.parse_args()

    # --- Validate mutually exclusive flags ---
    try:
        validate_mutually_exclusive_modes(args.cleaning_only, args.test_mode)
    except Exception as e:
        parser.error(str(e))

    # --- Create Config Object ---
    provider = args.provider
    api_key = None
    api_key_arg_name = ""
    api_key_env_var = ""
    compatible_url = None

    if provider == "Google":
        api_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")
        api_key_arg_name = "--google-api-key"
        api_key_env_var = "GOOGLE_API_KEY"
        default_model = "gemini-2.0-flash"
    elif provider == "OpenAI":
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        api_key_arg_name = "--openai-api-key"
        api_key_env_var = "OPENAI_API_KEY"
        default_model = "gpt-4o"
    elif provider == "Anthropic":
        api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        api_key_arg_name = "--anthropic-api-key"
        api_key_env_var = "ANTHROPIC_API_KEY"
        default_model = "claude-3.7-sonnet-latest"
    elif provider == "xAI":
        api_key = args.xai_api_key or os.environ.get("XAI_API_KEY")
        api_key_arg_name = "--xai-api-key"
        api_key_env_var = "XAI_API_KEY"
        default_model = "grok-4-fast-reasoning"
    elif provider == "OpenRouter":
        api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        api_key_arg_name = "--openrouter-api-key"
        api_key_env_var = "OPENROUTER_API_KEY"
        default_model = "openrouter/auto"
    elif provider == "OpenAI-Compatible":
        compatible_url = args.openai_compatible_url
        api_key = args.openai_compatible_api_key or os.environ.get(
            "OPENAI_COMPATIBLE_API_KEY"
        )
        api_key_arg_name = "--openai-compatible-api-key"
        api_key_env_var = "OPENAI_COMPATIBLE_API_KEY"
        default_model = "default"

    if provider != "OpenAI-Compatible" and not api_key:
        log_message(
            f"Warning: {provider} API key not provided via {api_key_arg_name} or {api_key_env_var} "
            f"environment variable. Translation will likely fail.",
            always_print=True,
        )

    model_name = args.model_name or default_model
    if not args.model_name:
        log_message(f"Using default model for {provider}: {model_name}", verbose=True)

    target_device = (
        torch.device("cpu")
        if args.cpu or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    log_message(
        f"Using {'CPU' if target_device.type == 'cpu' else 'CUDA'} device.",
        always_print=True,
    )

    use_otsu_config_val = args.use_otsu_threshold

    # Determine YOLO model path
    models_dir = Path(args.models).resolve()
    # Create models directory if it doesn't exist (model manager will handle model downloads)
    models_dir.mkdir(parents=True, exist_ok=True)
    yolo_model_path = autodetect_yolo_model_path(models_dir)

    config = MangaTranslatorConfig(
        yolo_model_path=str(yolo_model_path),
        verbose=args.verbose,
        device=target_device,
        cleaning_only=args.cleaning_only,
        detection=DetectionConfig(
            confidence=args.conf,
            use_sam2=args.use_sam2,
            enable_conjoined_detection=args.enable_conjoined_detection,
        ),
        cleaning=CleaningConfig(
            thresholding_value=args.thresholding_value,
            use_otsu_threshold=use_otsu_config_val,
            roi_shrink_px=max(0, min(8, int(args.roi_shrink_px))),
        ),
        translation=TranslationConfig(
            provider=provider,
            google_api_key=(
                api_key
                if provider == "Google"
                else os.environ.get("GOOGLE_API_KEY", "")
            ),
            openai_api_key=(
                api_key
                if provider == "OpenAI"
                else os.environ.get("OPENAI_API_KEY", "")
            ),
            anthropic_api_key=(
                api_key
                if provider == "Anthropic"
                else os.environ.get("ANTHROPIC_API_KEY", "")
            ),
            xai_api_key=(
                api_key if provider == "xAI" else os.environ.get("XAI_API_KEY", "")
            ),
            openrouter_api_key=(
                api_key
                if provider == "OpenRouter"
                else os.environ.get("OPENROUTER_API_KEY", "")
            ),
            openai_compatible_url=compatible_url,
            openai_compatible_api_key=(
                api_key
                if provider == "OpenAI-Compatible"
                else os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")
            ),
            model_name=model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            input_language=args.input_language,
            output_language=args.output_language,
            reading_direction=args.reading_direction,
            translation_mode=args.translation_mode,
            enable_thinking=args.enable_thinking,
            thinking_level=args.thinking_level,
            enable_grounding=args.enable_grounding,
            media_resolution=args.media_resolution,
            media_resolution_bubbles=args.media_resolution_bubbles,
            media_resolution_context=args.media_resolution_context,
            reasoning_effort=args.reasoning_effort,
            send_full_page_context=args.send_full_page_context,
            upscale_method=args.upscale_method,
            bubble_min_side_pixels=args.bubble_min_side_pixels,
            context_image_max_side_pixels=args.context_image_max_side_pixels,
            osb_min_side_pixels=args.osb_min_side_pixels,
            special_instructions=args.special_instructions,
        ),
        rendering=RenderingConfig(
            font_dir=args.font_dir,
            max_font_size=args.max_font_size,
            min_font_size=args.min_font_size,
            line_spacing=args.line_spacing,
            use_subpixel_rendering=args.use_subpixel_rendering,
            font_hinting=args.font_hinting,
            use_ligatures=args.use_ligatures,
            hyphenate_before_scaling=args.hyphenate_before_scaling,
            hyphen_penalty=args.hyphen_penalty,
            hyphenation_min_word_length=args.hyphenation_min_word_length,
            badness_exponent=args.badness_exponent,
            padding_pixels=args.padding_pixels,
        ),
        output=OutputConfig(
            output_format=args.output_format,
            jpeg_quality=args.jpeg_quality,
            png_compression=args.png_compression,
            upscale_final_image=args.image_upscale_mode == "final",
            image_upscale_factor=args.image_upscale_factor,
        ),
        outside_text=OutsideTextConfig(
            enabled=args.osb_enable,
            huggingface_token=args.osb_huggingface_token
            or os.environ.get("HUGGINGFACE_TOKEN", ""),
            flux_num_inference_steps=args.osb_flux_steps,
            flux_residual_diff_threshold=args.osb_flux_residual_threshold,
            seed=args.osb_seed,
            osb_font_name=args.osb_font_name,
            osb_max_font_size=args.osb_max_font_size,
            osb_min_font_size=args.osb_min_font_size,
            osb_use_ligatures=args.osb_use_ligatures,
            osb_outline_width=args.osb_outline_width,
            osb_line_spacing=args.osb_line_spacing,
            osb_use_subpixel_rendering=args.osb_use_subpixel,
            osb_font_hinting=args.osb_font_hinting,
            bbox_expansion_percent=args.osb_bbox_expansion,
            easyocr_min_size=args.osb_easyocr_min_size,
        ),
        preprocessing=PreprocessingConfig(
            enabled=args.image_upscale_mode == "initial",
            factor=args.image_upscale_factor,
            enable_auto_scale=args.enable_auto_scale,
        ),
        test_mode=args.test_mode,
    )

    # --- Execute ---
    if args.batch:
        input_path = Path(args.input)
        zip_temp_dir_obj = None
        preserve_structure = False

        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            log_message(f"Detected ZIP archive: {input_path.name}", always_print=True)
            try:
                temp_dir_obj = tempfile.TemporaryDirectory()
                temp_dir_path = Path(temp_dir_obj.name)
                zip_temp_dir_obj = temp_dir_obj

                with zipfile.ZipFile(input_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir_path)

                log_message(
                    "Extracted ZIP archive to temporary directory",
                    always_print=True,
                )

                input_path = temp_dir_path
                preserve_structure = True
            except zipfile.BadZipFile:
                log_message(
                    f"Error: '{args.input}' is not a valid ZIP archive.",
                    always_print=True,
                )
                exit(1)
            except Exception as e:
                log_message(
                    f"Error extracting ZIP archive: {str(e)}",
                    always_print=True,
                )
                exit(1)
        elif not input_path.is_dir():
            log_message(
                f"Error: --batch requires --input '{args.input}' to be a directory or ZIP archive.",
                always_print=True,
            )
            exit(1)

        output_dir = Path(args.output) if args.output else None

        if args.output:
            output_dir = Path(args.output)
            if not output_dir.exists():
                log_message(
                    f"Creating output directory: {output_dir}", always_print=True
                )
                output_dir.mkdir(parents=True, exist_ok=True)
            elif not output_dir.is_dir():
                log_message(
                    f"Error: Specified --output '{output_dir}' is not a directory.",
                    always_print=True,
                )
                exit(1)

        try:
            batch_translate_images(
                input_path, config, output_dir, preserve_structure=preserve_structure
            )
        finally:
            if zip_temp_dir_obj:
                try:
                    zip_temp_dir_obj.cleanup()
                    log_message(
                        "Cleaned up ZIP extraction temporary directory",
                        always_print=True,
                    )
                except Exception as e_clean:
                    log_message(
                        f"Warning: Failed to clean up temporary directory: {e_clean}",
                        always_print=True,
                    )
    else:
        input_path = Path(args.input)
        if not input_path.is_file():
            log_message(
                f"Error: Input '{args.input}' is not a valid file.", always_print=True
            )
            exit(1)

        output_path_arg = args.output
        if not output_path_arg:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            original_ext = input_path.suffix.lower()
            output_ext = original_ext
            output_dir = Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir / f"{input_path.stem}_translated_{timestamp}{output_ext}"
            )
            log_message(
                f"--output not specified, using default: {output_path}",
                always_print=True,
            )
        else:
            output_path = Path(output_path_arg)
            if not output_path.parent.exists():
                log_message(
                    f"Creating directory for output file: {output_path.parent}",
                    always_print=True,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            log_message(f"Processing {input_path}...", always_print=True)
            translate_and_render(input_path, config, output_path)
            log_message(
                f"Translation complete. Result saved to {output_path}",
                always_print=True,
            )
        except Exception as e:
            log_message(f"Error processing {input_path}: {e}", always_print=True)


if __name__ == "__main__":
    main()
