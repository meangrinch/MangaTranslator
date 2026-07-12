import argparse
import os

# Suppress OpenBLAS NUM_THREADS warnings on machines with >24 logical cores.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import tempfile
import time
import zipfile
from pathlib import Path

from utils.archive import safe_extract_zip
from utils.model_metadata import (
    FLUX_SDCPP_DIFFUSION_QUANTS,
    FLUX_SDCPP_TEXT_ENCODER_QUANTS,
    flux_sdcpp_text_encoder_quants,
)


def main():
    parser = argparse.ArgumentParser(
        description="Translate manga/comic speech bubbles using a configuration approach",
        usage="%(prog)s --input INPUT [options]",
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
    parser.add_argument(
        "--parallel-requests",
        type=int,
        default=1,
        help=(
            "Batch only: number of parallel workers, one per page (1-20, default: 1)"
        ),
    )
    parser.add_argument(
        "--batch-parallel-within-pages",
        action="store_true",
        help=(
            "Batch only: allows parallel workers to be shared with "
            "independent LLM/Flux work within each page"
        ),
    )
    parser.add_argument(
        "--batch-previous-context-images",
        type=int,
        default=0,
        help="Batch only: include this many previous source pages as visual context (0-10, default: 0)",
    )
    parser.add_argument(
        "--batch-previous-context-texts",
        type=int,
        default=3,
        help=(
            "Batch only: include this many previous pages' OCR text transcripts "
            "as narrative context (0-50, default: 3)"
        ),
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
            "DeepSeek",
            "Z.ai",
            "Moonshot AI",
            "Xiaomi MiMo",
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
        "--deepseek-api-key",
        type=str,
        default=None,
        help="DeepSeek API key (overrides DEEPSEEK_API_KEY env var if --provider is DeepSeek)",
    )
    parser.add_argument(
        "--zai-api-key",
        type=str,
        default=None,
        help="Z.ai API key (overrides ZAI_API_KEY env var if --provider is Z.ai)",
    )
    parser.add_argument(
        "--moonshot-api-key",
        type=str,
        default=None,
        help="Moonshot API key (overrides MOONSHOT_API_KEY env var if --provider is Moonshot)",
    )
    parser.add_argument(
        "--mimo-api-key",
        type=str,
        default=None,
        help=(
            "Xiaomi MiMo API key, sk-... (pay-as-you-go) or tp-... (Token Plan); "
            "overrides MIMO_API_KEY env var if --provider is Xiaomi MiMo"
        ),
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
        default="http://localhost:8080/v1",
        help="Base URL for the OpenAI-Compatible endpoint (default is llama.cpp)",
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
        help="Model name for the selected provider (e.g., 'gemini-3.1-flash-lite'). "
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
        "--confidence",
        type=float,
        default=0.6,
        help="Confidence threshold for speech bubble detection (0.0-1.0)",
    )
    parser.add_argument(
        "--conjoined-confidence",
        type=float,
        default=0.35,
        help="Confidence threshold for conjoined bubble detection (0.0-1.0)",
    )
    parser.add_argument(
        "--panel-confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for panel detection YOLO (0.0-1.0)",
    )
    parser.add_argument(
        "--seg-model",
        dest="seg_model",
        type=str,
        choices=["sam3", "sam2", "yolo"],
        default="yolo",
        help="Segmentation method",
    )
    parser.add_argument(
        "--no-conjoined-detection",
        dest="conjoined_detection",
        action="store_false",
        help="Disable conjoined bubble detection using secondary YOLO model",
    )
    parser.set_defaults(conjoined_detection=True)
    parser.add_argument(
        "--bubble-detector-model",
        dest="bubble_detector_model",
        type=str,
        choices=["yolo_1", "yolo_2"],
        default="yolo_1",
        help="Primary bubble detector model",
    )
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
        help="Force Otsu's method for thresholding instead of the fixed value (on all bubbles)",
    )
    parser.add_argument(
        "--thresholding-value",
        type=int,
        default=200,
        help=(
            "Fixed threshold value for text detection (0-255). "
            "Lower values help clean edge-hugging text."
        ),
    )
    parser.add_argument(
        "--roi-shrink-px",
        type=int,
        default=5,
        help=(
            "Shrink the threshold ROI inward by N pixels (0-10) before fill. "
            "Lower helps clean edge-hugging text; higher preserves outlines."
        ),
    )
    parser.add_argument(
        "--inpaint-colored-bubbles",
        action="store_true",
        help="Use Flux model to inpaint colored/grayscale bubbles",
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
        choices=["max", "xhigh", "high", "medium", "low", "minimal", "none"],
        help=(
            "OpenAI/Gemini 3: Controls internal reasoning effort. "
            "`max` is available for GPT-5.6, `xhigh` for GPT-5.2+ (OpenAI), "
            "and `minimal` for GPT-5. "
            "Other providers: Controls reasoning token budget allocation relative to "
            "`max_tokens` (high=80%%, medium=50%%, low=20%%). "
            "Use 'none' to disable thinking for certain models."
        ),
    )
    parser.add_argument(
        "--effort",
        type=str,
        default="medium",
        choices=["high", "medium", "low"],
        help=(
            "Opus 4.5/4.6, Sonnet 4.6 only: Controls token spending eagerness when responding. "
            "Separate from 'max_tokens' and 'reasoning_effort'."
        ),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="low",
        choices=["high", "medium", "low"],
        help=(
            "GPT-5 series only: Controls response verbosity. "
            "Separate from 'max_tokens' and 'reasoning_effort'."
        ),
    )
    parser.add_argument(
        "--ocr-method",
        type=str,
        default="LLM",
        choices=["LLM", "manga-ocr", "paddleocr-vl"],
        help=(
            "Determines whether to use a vision-capable LLM or a local OCR model for OCR. "
            "Local OCR options enable text-only LLMs for translation "
            "and must be used in 'two-step' translation mode."
        ),
    )
    # Rendering args
    parser.add_argument(
        "--max-font-size",
        type=int,
        default=16,
        help="Max font size for rendering text (px)",
    )
    parser.add_argument(
        "--min-font-size",
        type=int,
        default=8,
        help="Min font size for rendering text (px)",
    )
    parser.add_argument(
        "--line-spacing-mult",
        type=float,
        default=1.0,
        help="Line spacing multiplier for rendering text (1.0 = standard)",
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
        help="Minimum word length required for hyphenation (4-10)",
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
    parser.add_argument(
        "--supersampling-factor",
        type=int,
        default=4,
        help="Render text at Nx resolution then downscale for smoother edges (1-4). "
        "Higher values improve quality but use slightly more memory. 1 = disabled.",
    )
    parser.add_argument(
        "--no-detach-trailing-punctuation",
        dest="detach_trailing_punctuation",
        action="store_false",
        help=(
            "Disable detaching trailing punctuation clusters onto a new line "
            "for better wrapping."
        ),
    )
    parser.set_defaults(detach_trailing_punctuation=True)
    parser.add_argument(
        "--auto-vertical-text",
        action="store_true",
        help=(
            "Automatically stack short translated text vertically in tall speech bubbles "
            "when it improves the layout."
        ),
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
        default=2,
        help="PNG compression level (0-6)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["auto", "png", "jpeg"],
        default="png",
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
        "--no-auto-scale",
        action="store_false",
        dest="auto_scale",
        help=(
            "Disable automatic scaling of pipeline parameters (fonts, kernels, etc.) "
            "based on image size relative to 1MP. Prevents consistent behavior across different image resolutions."
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
        "--upscaling-only",
        action="store_true",
        help="Skip detection and translation, only upscale the image",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Skip translation and render placeholder text (lorem ipsum)",
    )
    parser.add_argument(
        "--enable-web-search",
        action="store_true",
        help=(
            "Enable model's built-in web search for up-to-date information. OpenRouter uses its own web search tool."
        ),
    )
    parser.add_argument(
        "--enable-code-execution",
        action="store_true",
        help="Enable Gemini's code execution tool for image zoom/inspection (Gemini 3 Flash only)",
    )
    parser.add_argument(
        "--no-custom-sampling",
        dest="use_custom_sampling",
        action="store_false",
        help="Omit temperature, top-p, and top-k from API requests",
    )
    parser.set_defaults(use_custom_sampling=True)
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
        help="Media resolution for bubble images (Gemini 3 and xAI models)",
    )
    parser.add_argument(
        "--media-resolution-context",
        type=str,
        choices=["auto", "high", "medium", "low"],
        default="auto",
        help="Media resolution for context (full page) images (Gemini 3 and xAI only)",
    )
    parser.add_argument(
        "--image-detail",
        type=str,
        choices=["auto", "original", "high", "low"],
        default="auto",
        help=(
            "Image detail for OpenAI to process bubble/context images. "
            "'original' is supported on GPT-5.4+ base/pro models."
        ),
    )
    parser.add_argument(
        "--special-instructions",
        type=str,
        default=None,
        help="Optional special instructions for the LLM (formatting, context, character names, etc.)",
    )
    parser.add_argument(
        "--no-full-page-context",
        dest="send_full_page_context",
        action="store_false",
        help=(
            "Disable including the full page image as context for translation. Enable if "
            "encountering refusals or using less-capable LLMs"
        ),
    )
    parser.add_argument(
        "--no-whiteout-conjoined-bubbles",
        dest="whiteout_conjoined_bubbles",
        action="store_false",
        help="Disable whiting out text from neighboring conjoined bubbles during preparation.",
    )
    parser.add_argument(
        "--upscale-method",
        type=str,
        choices=["model", "model_lite", "lanczos", "none"],
        default="model_lite",
        help=(
            "Method for upscaling images before translation API. "
            "model: Use 2x-AnimeSharpV4 upscaling model (best quality, slower), "
            "model_lite: Use 2x-AnimeSharpV4 Fast RCAN PU model (worse quality, faster/less memory), "
            "lanczos: Use LANCZOS resampling (worst quality, fastest/least memory), "
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
        "--osb-hf-token",
        type=str,
        default=None,
        help="HuggingFace token for downloading OSB pipeline models (overrides HF_TOKEN env var)",
    )
    parser.add_argument(
        "--osb-inpainting-method",
        type=str,
        choices=["flux_klein_9b", "flux_klein_4b", "flux_kontext", "opencv", "none"],
        default="flux_klein_4b",
        help="Inpainting method for outside text removal.",
    )
    parser.add_argument(
        "--osb-flux-backend",
        type=str,
        choices=["sdcpp", "sdnq", "nunchaku"],
        default="sdnq",
        help=(
            "Backend for Flux inpainting. "
            "'sdcpp' and 'sdnq' support Flux Klein/Kontext; "
            "'nunchaku' is CUDA-only and Kontext-only."
        ),
    )
    parser.add_argument(
        "--osb-flux-low-vram",
        action="store_true",
        help="Enable CPU offload for Flux SDNQ models (reduces VRAM usage)",
    )
    parser.add_argument(
        "--osb-flux-sdcpp-cache-mode",
        type=str,
        choices=["spectrum", "cache-dit", "taylorseer", "dbcache", "none"],
        default="none",
        help=(
            "Caching method for Flux via sd.cpp. Ordered from fastest/worst quality to slowest/best quality."
            "Warmup is 25%% of the selected step count."
        ),
    )
    parser.add_argument(
        "--osb-flux-sdcpp-diffusion-quant",
        type=str,
        choices=FLUX_SDCPP_DIFFUSION_QUANTS,
        default="Q4_K_M",
        help=(
            "Quantization level for the Flux model via sd.cpp. "
            "Ordered from largest/best quality to smallest/lowest quality."
        ),
    )
    parser.add_argument(
        "--osb-flux-sdcpp-text-encoder-quant",
        type=str,
        choices=FLUX_SDCPP_TEXT_ENCODER_QUANTS,
        default=None,
        help=(
            "Quantization level for the text encoder model via sd.cpp. "
            "Ordered from largest/best quality to smallest/lowest quality. "
            "(defaults: Q4_K_XL for Klein and Q4_K_M for Kontext)."
        ),
    )
    parser.add_argument(
        "--osb-no-flux-upscale-small-crops",
        dest="osb_flux_upscale_small_crops",
        action="store_false",
        help="Disable scaling small Flux Klein OSB crops to ~1MP before inference",
    )
    parser.set_defaults(osb_flux_upscale_small_crops=True)
    parser.add_argument(
        "--osb-flux-group-regions",
        action="store_true",
        help="Group non-solid OSB regions into one expanded Flux mask per page",
    )

    parser.add_argument(
        "--osb-flux-steps",
        type=int,
        default=4,
        help=(
            "Number of denoising steps for Flux models. "
            "Klein: 4 is recommended (1-12). "
            "Kontext: 6-15 is recommended (1-30). "
        ),
    )
    parser.add_argument(
        "--osb-no-luminance-correction",
        dest="osb_flux_luminance_correction",
        action="store_false",
        help="Disable luminance correction that matches generated patch brightness to surrounding context (Klein only)",
    )
    parser.set_defaults(osb_flux_luminance_correction=True)
    parser.add_argument(
        "--osb-flux-residual-threshold",
        type=float,
        default=0.15,
        help="Residual diff threshold for Flux.1 Kontext via Nunchaku (0.0-1.0)",
    )
    parser.add_argument(
        "--osb-seed",
        type=int,
        default=1,
        help="Seed for reproducible inpainting (-1 = random)",
    )
    parser.add_argument(
        "--osb-font-dir",
        type=str,
        default=None,
        help="Font pack directory for OSB text rendering (default: use main font)",
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
        default=10,
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
        "--osb-render-expansion-narrow",
        type=float,
        default=1.0,
        help="Multiplier for narrow/tall OSB render boxes (1.0-3.0)",
    )
    parser.add_argument(
        "--osb-render-expansion-tiny",
        type=float,
        default=1.0,
        help="Multiplier for tiny OSB render boxes (1.0-3.0)",
    )
    parser.add_argument(
        "--osb-render-expansion-aspect-threshold",
        type=float,
        default=0.4,
        help="Classify OSB boxes as narrow/tall at or below this width/height ratio",
    )
    parser.add_argument(
        "--osb-render-expansion-area-threshold",
        type=float,
        default=0.005,
        help="Classify OSB boxes as tiny below this image area ratio",
    )
    parser.add_argument(
        "--osb-text-box-proximity-ratio",
        type=float,
        default=0.02,
        help="Proximity ratio for grouping nearby text boxes (as fraction of image dimension)",
    )
    parser.add_argument(
        "--osb-confidence",
        type=float,
        default=0.6,
        help="Confidence threshold for OSB text detection (0.0-1.0)",
    )
    parser.add_argument(
        "--osb-filter-page-numbers",
        action="store_true",
        help=(
            "Filter probable page numbers near margins using manga-ocr (slightly slower and may detect false positives)"
        ),
    )
    parser.add_argument(
        "--osb-page-filter-margin",
        type=float,
        default=0.1,
        help=(
            "Margin ratio (0-0.3) for page-number filtering; only used when page-number filtering is enabled"
        ),
    )
    parser.add_argument(
        "--osb-page-filter-min-area",
        type=float,
        default=0.05,
        help=(
            "Minimum area ratio (0-0.2) to treat detection as a potential page number; "
            "only used when page-number filtering is enabled"
        ),
    )
    parser.add_argument(
        "--osb-min-area-ignore-ratio",
        type=float,
        default=0.0,
        help=(
            "Skip OSB regions whose bounding-box area is below this ratio of the image "
            "(0.0-0.05)"
        ),
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
        default=1024,
        help="Target maximum side length for full page image",
    )
    parser.add_argument(
        "--osb-min-side-pixels",
        type=int,
        default=128,
        help="Target minimum side length for outside speech bubble upscaling",
    )

    parser.set_defaults(send_full_page_context=True)
    parser.set_defaults(whiteout_conjoined_bubbles=True)
    parser.set_defaults(auto_scale=True)
    parser.set_defaults(
        verbose=False,
        cpu=False,
        cleaning_only=False,
        enable_web_search=False,
    )

    args = parser.parse_args()

    if args.osb_flux_backend == "nunchaku" and args.osb_inpainting_method in (
        "flux_klein_9b",
        "flux_klein_4b",
    ):
        parser.error(
            "--osb-flux-backend nunchaku is only supported with "
            "--osb-inpainting-method flux_kontext."
        )
    if args.osb_flux_backend == "sdcpp" and args.osb_flux_sdcpp_text_encoder_quant:
        valid_text_encoder_quants = flux_sdcpp_text_encoder_quants(
            args.osb_inpainting_method
        )
        if (
            not valid_text_encoder_quants
            or args.osb_flux_sdcpp_text_encoder_quant not in valid_text_encoder_quants
        ):
            parser.error(
                "--osb-flux-sdcpp-text-encoder-quant "
                f"{args.osb_flux_sdcpp_text_encoder_quant} is not available for "
                f"{args.osb_inpainting_method}. "
                "Valid text encoder quants: "
                f"{', '.join(valid_text_encoder_quants) or 'none'}."
            )

    import torch

    from core.config import (
        CleaningConfig,
        DetectionConfig,
        MangaTranslatorConfig,
        OutputConfig,
        OutsideTextConfig,
        PreprocessingConfig,
        RenderingConfig,
        TranslationConfig,
    )
    from core.pipeline import batch_translate_images, translate_and_render
    from core.validation import (
        autodetect_yolo_model_path,
        clamp_settings,
        validate_mutually_exclusive_modes,
    )
    from utils.logging import log_message

    # --- Validate mutually exclusive flags ---
    try:
        validate_mutually_exclusive_modes(
            args.cleaning_only, args.upscaling_only, args.test_mode
        )
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
        default_model = "gemini-3.1-flash-lite"
    elif provider == "OpenAI":
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        api_key_arg_name = "--openai-api-key"
        api_key_env_var = "OPENAI_API_KEY"
        default_model = "gpt-5.4-nano-2026-03-17"
    elif provider == "Anthropic":
        api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        api_key_arg_name = "--anthropic-api-key"
        api_key_env_var = "ANTHROPIC_API_KEY"
        default_model = "claude-sonnet-5"
    elif provider == "xAI":
        api_key = args.xai_api_key or os.environ.get("XAI_API_KEY")
        api_key_arg_name = "--xai-api-key"
        api_key_env_var = "XAI_API_KEY"
        default_model = "grok-4.5"
    elif provider == "DeepSeek":
        api_key = args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
        api_key_arg_name = "--deepseek-api-key"
        api_key_env_var = "DEEPSEEK_API_KEY"
        default_model = "deepseek-v4-flash"
    elif provider == "Z.ai":
        api_key = args.zai_api_key or os.environ.get("ZAI_API_KEY")
        api_key_arg_name = "--zai-api-key"
        api_key_env_var = "ZAI_API_KEY"
        default_model = "glm-5v-turbo"
    elif provider == "Moonshot AI":
        api_key = args.moonshot_api_key or os.environ.get("MOONSHOT_API_KEY")
        api_key_arg_name = "--moonshot-api-key"
        api_key_env_var = "MOONSHOT_API_KEY"
        default_model = "kimi-k2.6"
    elif provider == "Xiaomi MiMo":
        api_key = args.mimo_api_key or os.environ.get("MIMO_API_KEY")
        api_key_arg_name = "--mimo-api-key"
        api_key_env_var = "MIMO_API_KEY"
        default_model = "mimo-v2.5"
    elif provider == "OpenRouter":
        api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        api_key_arg_name = "--openrouter-api-key"
        api_key_env_var = "OPENROUTER_API_KEY"
        default_model = "google/gemini-3.1-flash-lite"
    elif provider == "OpenAI-Compatible":
        compatible_url = args.openai_compatible_url
        api_key = args.openai_compatible_api_key or os.environ.get(
            "OPENAI_COMPATIBLE_API_KEY"
        )
        api_key_arg_name = "--openai-compatible-api-key"
        api_key_env_var = "OPENAI_COMPATIBLE_API_KEY"
        default_model = "default"

    if (
        provider != "OpenAI-Compatible"
        and not api_key
        and not args.cleaning_only
        and not args.test_mode
    ):
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
        if args.cpu
        else torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    log_message(
        f"Using {target_device.type.upper()} device.",
        always_print=True,
    )

    use_otsu_config_val = args.use_otsu_threshold

    # Determine YOLO model path
    models_dir = Path(args.models).resolve()
    # Create models directory if it doesn't exist (model manager will handle model downloads)
    models_dir.mkdir(parents=True, exist_ok=True)
    yolo_model_path = autodetect_yolo_model_path(models_dir, args.bubble_detector_model)

    config = MangaTranslatorConfig(
        yolo_model_path=str(yolo_model_path),
        verbose=args.verbose,
        device=target_device,
        cleaning_only=args.cleaning_only,
        upscaling_only=args.upscaling_only,
        parallel_requests=args.parallel_requests,
        batch_parallel_within_pages=bool(
            args.batch and args.batch_parallel_within_pages
        ),
        detection=DetectionConfig(
            confidence=args.confidence,
            conjoined_confidence=args.conjoined_confidence,
            panel_confidence=args.panel_confidence,
            seg_model=args.seg_model,
            bubble_detector_model=args.bubble_detector_model,
            conjoined_detection=args.conjoined_detection,
        ),
        cleaning=CleaningConfig(
            thresholding_value=args.thresholding_value,
            use_otsu_threshold=use_otsu_config_val,
            roi_shrink_px=max(0, min(10, int(args.roi_shrink_px))),
            inpaint_colored_bubbles=args.inpaint_colored_bubbles,
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
            deepseek_api_key=(
                api_key
                if provider == "DeepSeek"
                else os.environ.get("DEEPSEEK_API_KEY", "")
            ),
            zai_api_key=(
                api_key if provider == "Z.ai" else os.environ.get("ZAI_API_KEY", "")
            ),
            moonshot_api_key=(
                api_key
                if provider == "Moonshot AI"
                else os.environ.get("MOONSHOT_API_KEY", "")
            ),
            mimo_api_key=(
                api_key
                if provider == "Xiaomi MiMo"
                else os.environ.get("MIMO_API_KEY", "")
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
            enable_web_search=args.enable_web_search,
            enable_code_execution=args.enable_code_execution,
            use_custom_sampling=args.use_custom_sampling,
            image_detail=args.image_detail,
            media_resolution=args.media_resolution,
            media_resolution_bubbles=args.media_resolution_bubbles,
            media_resolution_context=args.media_resolution_context,
            reasoning_effort=args.reasoning_effort,
            effort=args.effort,
            verbosity=args.verbosity,
            send_full_page_context=args.send_full_page_context,
            whiteout_conjoined_bubbles=args.whiteout_conjoined_bubbles,
            upscale_method=args.upscale_method,
            bubble_min_side_pixels=args.bubble_min_side_pixels,
            context_image_max_side_pixels=args.context_image_max_side_pixels,
            previous_context_image_count=(
                args.batch_previous_context_images
                if (
                    args.batch
                    and args.send_full_page_context
                    and args.ocr_method == "LLM"
                )
                else 0
            ),
            previous_context_text_count=(
                args.batch_previous_context_texts if args.batch else 0
            ),
            osb_min_side_pixels=args.osb_min_side_pixels,
            special_instructions=args.special_instructions,
            ocr_method=args.ocr_method,
        ),
        rendering=RenderingConfig(
            font_dir=args.font_dir,
            max_font_size=args.max_font_size,
            min_font_size=args.min_font_size,
            line_spacing_mult=args.line_spacing_mult,
            use_subpixel_rendering=args.use_subpixel_rendering,
            font_hinting=args.font_hinting,
            use_ligatures=args.use_ligatures,
            hyphenate_before_scaling=args.hyphenate_before_scaling,
            hyphen_penalty=args.hyphen_penalty,
            hyphenation_min_word_length=args.hyphenation_min_word_length,
            badness_exponent=args.badness_exponent,
            padding_pixels=args.padding_pixels,
            supersampling_factor=args.supersampling_factor,
            detach_trailing_punctuation=args.detach_trailing_punctuation,
            auto_vertical_text=args.auto_vertical_text,
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
            enable_page_number_filtering=args.osb_filter_page_numbers,
            page_filter_margin_threshold=args.osb_page_filter_margin,
            page_filter_min_area_ratio=args.osb_page_filter_min_area,
            min_area_ignore_ratio=args.osb_min_area_ignore_ratio,
            huggingface_token=args.osb_hf_token or os.environ.get("HF_TOKEN", ""),
            inpainting_method=args.osb_inpainting_method,
            flux_backend=args.osb_flux_backend,
            flux_low_vram=args.osb_flux_low_vram,
            flux_sdcpp_cache_mode=args.osb_flux_sdcpp_cache_mode,
            flux_sdcpp_diffusion_quant=args.osb_flux_sdcpp_diffusion_quant,
            flux_sdcpp_text_encoder_quant=args.osb_flux_sdcpp_text_encoder_quant or "",
            flux_num_inference_steps=args.osb_flux_steps,
            flux_luminance_correction=args.osb_flux_luminance_correction,
            flux_upscale_small_crops=args.osb_flux_upscale_small_crops,
            flux_group_regions=args.osb_flux_group_regions,
            flux_residual_diff_threshold=args.osb_flux_residual_threshold,
            osb_confidence=args.osb_confidence,
            seed=args.osb_seed,
            osb_font_dir=args.osb_font_dir,
            osb_max_font_size=args.osb_max_font_size,
            osb_min_font_size=args.osb_min_font_size,
            osb_use_ligatures=args.osb_use_ligatures,
            osb_outline_width=args.osb_outline_width,
            osb_line_spacing=args.osb_line_spacing,
            osb_use_subpixel_rendering=args.osb_use_subpixel,
            osb_font_hinting=args.osb_font_hinting,
            bbox_expansion_percent=args.osb_bbox_expansion,
            osb_render_expansion_narrow_multiplier=args.osb_render_expansion_narrow,
            osb_render_expansion_tiny_multiplier=args.osb_render_expansion_tiny,
            osb_render_expansion_aspect_ratio_threshold=(
                args.osb_render_expansion_aspect_threshold
            ),
            osb_render_expansion_area_ratio_threshold=(
                args.osb_render_expansion_area_threshold
            ),
            text_box_proximity_ratio=args.osb_text_box_proximity_ratio,
        ),
        preprocessing=PreprocessingConfig(
            enabled=args.image_upscale_mode == "initial",
            factor=args.image_upscale_factor,
            auto_scale=args.auto_scale,
        ),
        test_mode=args.test_mode,
    )

    clamp_settings(config)
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
                    safe_extract_zip(zip_ref, temp_dir_path)

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
            results = batch_translate_images(
                input_path, config, output_dir, preserve_structure=preserve_structure
            )
            if results.get("error_count", 0) > 0:
                raise SystemExit(1)
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
        original_ext = input_path.suffix.lower()
        output_format = args.output_format
        if output_format == "auto":
            output_ext = (
                original_ext
                if original_ext in [".jpg", ".jpeg", ".png", ".webp"]
                else ".png"
            )
        elif output_format == "png":
            output_ext = ".png"
        elif output_format == "jpeg":
            output_ext = ".jpg"
        else:
            output_ext = ".png"

        if not output_path_arg:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
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
            if args.output_format != "auto" or not output_path.suffix:
                output_path = output_path.with_suffix(output_ext)
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
            raise SystemExit(1) from e


if __name__ == "__main__":
    main()
