import argparse
import os

# Suppress OpenBLAS NUM_THREADS warnings on machines with >24 logical cores.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")

import sys
import tempfile
import time
import zipfile
from pathlib import Path

from utils.archive import ARCHIVE_SUFFIXES, safe_extract_zip
from utils.model_metadata import (
    FLUX_SDCPP_DIFFUSION_QUANTS,
    FLUX_SDCPP_TEXT_ENCODER_QUANTS,
    NO_WEB_SEARCH_PROVIDERS,
    flux_method_backends,
    flux_sdcpp_diffusion_quants,
    flux_sdcpp_text_encoder_quants,
    flux_steps_range,
    flux_valid_backend,
)

# Model used when --model-name is not given; the keys are the --provider choices
CLI_DEFAULT_MODELS = {
    "Google": "gemini-3.8-flash",
    "OpenAI": "gpt-6-luna",
    "Anthropic": "claude-haiku-5-5",
    "SpaceXAI": "grok-4.7",
    "Meta Model": "muse-spark-1.3-contributor",
    "DeepSeek": "deepseek-flash",
    "Z.ai": "glm-5.3-flash",
    "Moonshot AI": "kimi-k3.1-pro",
    "Xiaomi MiMo": "mimo-v2.6-flash",
    "QwenCloud": "qwen3.8-flash",
    "OpenCode": "deepseek-v4.1-flash",
    "OpenRouter": "google/gemini-3.8-flash",
    "OpenAI-Compatible": "default",
}


def main():
    parser = argparse.ArgumentParser(
        description="Translate manga/comic speech bubbles using a configuration approach",
        usage="%(prog)s --input INPUT [options]",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to input image; with --batch: directory, ZIP/CBZ archive, "
            "or failed_paths.txt"
        ),
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
        help=(
            "Process all images in the input directory, ZIP/CBZ archive "
            "(preserves folder structure), or failed_paths.txt"
        ),
    )
    parser.add_argument(
        "--parallel-requests",
        type=int,
        default=1,
        help=(
            "Batch only: number of parallel workers, one per page (1-20, default: 1). "
            "LLM calls only run in parallel when --batch-previous-context-texts is 0"
        ),
    )
    parser.add_argument(
        "--batch-parallel-within-pages",
        action="store_true",
        help=(
            "Batch only: allows parallel workers to be shared with "
            "independent Flux work within each page"
        ),
    )
    parser.add_argument(
        "--no-retry-failed-once",
        dest="retry_failed_once",
        action="store_false",
        help="Batch only: do not retry each failed image once at the very end",
    )
    parser.add_argument(
        "--no-overlap-llm-with-inpaint",
        dest="overlap_llm_with_inpaint",
        action="store_false",
        help="Do not run LLM translation concurrently with inpainting",
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
    parser.add_argument(
        "--batch-terminology-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Batch only: have the LLM save names and terms to a glossary file and "
            "reuse them for consistency (default: on)"
        ),
    )
    parser.add_argument(
        "--batch-glossary-file",
        type=str,
        default=None,
        help=(
            "Batch only: glossary file to read and update, e.g. the previous chapter's. "
            "Default: a new output/.glossary/<input name>_<date>.txt"
        ),
    )
    # --- Provider and API Key Arguments ---
    parser.add_argument(
        "--provider",
        type=str,
        default="Google",
        choices=list(CLI_DEFAULT_MODELS),
        help="LLM provider to use for translation",
    )
    parser.add_argument(
        "--google-api-key",
        dest="google_api_key",
        type=str,
        default=None,
        help=(
            "Google API key (overrides GOOGLE_API_KEY or GEMINI_API_KEY env var "
            "if --provider is Google)"
        ),
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
        "--spacexai-api-key",
        type=str,
        default=None,
        help=(
            "SpaceXAI API key (overrides SPACEXAI_API_KEY or XAI_API_KEY env var "
            "if --provider is SpaceXAI)"
        ),
    )
    parser.add_argument(
        "--meta-api-key",
        dest="meta_api_key",
        type=str,
        default=None,
        help=(
            "Meta Model API key; overrides META_MODEL_API_KEY or META_API_KEY env var "
            "if --provider is Meta Model"
        ),
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
        "--qwencloud-api-key",
        dest="qwencloud_api_key",
        type=str,
        default=None,
        help=(
            "QwenCloud API key (starts with sk-...); "
            "overrides QWENCLOUD_API_KEY or QWEN_API_KEY env var if --provider is QwenCloud"
        ),
    )
    parser.add_argument(
        "--opencode-api-key",
        type=str,
        default=None,
        help="OpenCode API key (overrides OPENCODE_API_KEY env var if --provider is OpenCode)",
    )
    parser.add_argument(
        "--opencode-tier",
        type=str,
        default="zen",
        choices=["zen", "go", "OpenCode Zen", "OpenCode Go"],
        help="OpenCode endpoint tier ('zen' or 'go', default: 'zen')",
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
        help="Base URL for the OpenAI-Compatible endpoint. Also supports Azure OpenAI endpoint URLs (e.g. https://<resource>.openai.azure.com)",
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
        help="Model name for the selected provider (e.g., 'gemini-3.8-flash'). "
        "If not provided, a default will be attempted based on the provider.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="./models",
        help="Directory where downloaded models are stored",
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
        help="Disable conjoined bubble detection using secondary RT-DETR model",
    )
    parser.set_defaults(conjoined_detection=True)
    parser.add_argument(
        "--no-panel-sorting",
        dest="use_panel_sorting",
        action="store_false",
        help="Sort bubbles across the whole page instead of within detected panels",
    )
    parser.add_argument(
        "--no-osb-text-verification",
        dest="use_osb_text_verification",
        action="store_false",
        help=(
            "Skip checking with the AnimeText YOLO model that bubble boxes "
            "fully cover their text"
        ),
    )
    parser.add_argument(
        "--bubble-detector-model",
        dest="bubble_detector_model",
        type=str,
        choices=["yolo_1", "yolo_2"],
        default="yolo_2",
        help="Primary bubble detector model",
    )
    parser.add_argument(
        "--reading-direction",
        type=str,
        default="rtl",
        choices=["rtl", "ltr"],
        help="Reading direction for sorting bubbles (rtl or ltr)",
    )
    parser.add_argument(
        "--no-split-long-strips",
        action="store_false",
        dest="split_long_strips",
        help=(
            "Disable splitting of long manhwa/manhua strips (taller than 8x their width) "
            "into sections that are translated separately and stitched back together."
        ),
    )
    # Cleaning args
    parser.add_argument(
        "--use-otsu-threshold",
        action="store_true",
        help="Force Otsu's method for thresholding instead of the fixed value (on all bubbles)",
    )
    parser.add_argument(
        "--no-auto-threshold",
        dest="auto_threshold",
        action="store_false",
        help=(
            "Disable scaling --thresholding-value to each bubble's background brightness "
            "(scaling cleans off-white or gray bubbles like white ones; ignored with "
            "--use-otsu-threshold)"
        ),
    )
    parser.set_defaults(auto_threshold=True)
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
        "--top-k",
        type=int,
        default=None,
        help="Limits sampling pool to top K tokens (default: provider default)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of tokens in the response (4096-63488). "
            "Default: 4096 for non-reasoning models, 32768 for reasoning models"
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
        default=None,
        choices=["max", "xhigh", "high", "medium", "low", "minimal", "none"],
        help=(
            "Controls reasoning effort level or thinking token budget. "
            "Defaults to the model's own default when omitted. "
            "For models supporting native reasoning effort: "
            "controls the internal reasoning effort level. "
            "For budget-based models: allocates reasoning tokens relative to "
            "`max_tokens` (high=80%%, medium=50%%, low=20%%, minimal=10%%). "
            "Use 'none' to disable thinking where supported."
        ),
    )
    parser.add_argument(
        "--effort",
        type=str,
        default="high",
        choices=["max", "xhigh", "high", "medium", "low"],
        help=(
            "Claude Opus 4.5+, Sonnet 4.6+, Haiku 5.5+, Fable 5+ only: Controls token spending eagerness when responding. "
            "'max' and 'xhigh' apply only to models that support them. "
            "Separate from 'max_tokens' and 'reasoning_effort'."
        ),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="low",
        choices=["high", "medium", "low"],
        help=(
            "GPT-5 and GPT-6 series: Controls response verbosity. "
            "Separate from 'max_tokens' and 'reasoning_effort'."
        ),
    )
    parser.add_argument(
        "--ocr-method",
        type=str,
        default="LLM",
        choices=["LLM", "manga-ocr", "paddleocr-vl-1.6"],
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
        default=32,
        help="Max font size for rendering text (px)",
    )
    parser.add_argument(
        "--min-font-size",
        type=int,
        default=8,
        help="Min font size for rendering text (px)",
    )
    parser.add_argument(
        "--no-auto-font-size",
        dest="auto_font_size",
        action="store_false",
        help=(
            "Disable matching each page's speech bubble font size to its original lettering "
            "and use the largest size that fits within the min/max font size range instead"
        ),
    )
    parser.set_defaults(auto_font_size=True)
    parser.add_argument(
        "--line-spacing-mult",
        type=float,
        default=1.0,
        help="Line spacing multiplier for rendering text (1.0 = standard)",
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
        default=4.0,
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
        "--no-auto-vertical-text",
        dest="auto_vertical_text",
        action="store_false",
        help=(
            "Disable automatically stacking short translated text vertically in tall speech bubbles"
        ),
    )
    parser.set_defaults(auto_vertical_text=True)
    parser.add_argument(
        "--vertical-line-spacing-mult",
        type=float,
        default=1.0,
        help="Line spacing multiplier for vertical bubble text (0.5-2.0)",
    )
    parser.add_argument(
        "--vertical-font-size-mult",
        type=float,
        default=1.0,
        help="Font size multiplier for vertical bubble text (0.1-3.0)",
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
        "--image-upscale-model",
        choices=["model", "model_lite"],
        default="model_lite",
        help="Model for image upscaling: 'model' (best quality) or 'model_lite' (faster).",
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
        "--hf-cache",
        action="store_true",
        help="Store Hugging Face models in the default HF cache instead of ./models",
    )
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
            "Enable model's built-in web search for up-to-date information. OpenRouter uses its own web search tool. "
            f"Not available for {', '.join(NO_WEB_SEARCH_PROVIDERS)}."
        ),
    )
    parser.add_argument(
        "--enable-code-execution",
        action="store_true",
        help="Enable Gemini's code execution tool for image zoom/inspection (Gemini 3+ only)",
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
        help="Media resolution for Gemini models (Google provider only, not used for Gemini 3+)",
    )
    parser.add_argument(
        "--media-resolution-bubbles",
        type=str,
        choices=["auto", "high", "medium", "low"],
        default="auto",
        help="Media resolution for bubble images (Gemini 3+ and Grok only)",
    )
    parser.add_argument(
        "--media-resolution-context",
        type=str,
        choices=["auto", "high", "medium", "low"],
        default="auto",
        help="Media resolution for context (full page) images (Gemini 3+ and Grok only)",
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
        "--ero-doujinshi-mode",
        action="store_true",
        help="Optimize translation for 18+ content. Uses more tokens, but may reduce refusals.",
    )
    parser.add_argument(
        "--ocr-correction",
        action="store_true",
        help="When using two-step translation mode, instruct the LLM to contextually correct poor OCR during translation.",
    )
    parser.add_argument(
        "--force-cache-translations",
        action="store_true",
        help="Explicitly save and reuse translations from cache, even for non-deterministic sampling.",
    )
    parser.add_argument(
        "--no-full-page-context",
        dest="send_full_page_context",
        action="store_false",
        help=(
            "Disable including the full page image as context for translation. Enable if "
            "encountering refusals or to reduce token usage"
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
            "lanczos: Use anti-ringing LANCZOS resampling (softer than the models, fastest/least memory), "
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
        choices=[
            "fibo2_edit_turbo",
            "flux3_klein",
            "qwen_image_2_1_turbo",
            "flux_klein_9b",
            "flux_klein_4b",
            "flux_kontext",
            "lama_manga",
            "opencv",
            "none",
        ],
        default="flux_klein_4b",
        help="Inpainting method for outside text removal.",
    )
    parser.add_argument(
        "--osb-flux-backend",
        type=str,
        choices=["sdcpp", "sdnq", "nunchaku"],
        default=None,
        help=(
            "Backend for diffusion inpainting (default: sdnq, or sdcpp for Fibo). "
            "'sdcpp' supports every diffusion model; 'sdnq' all but Fibo 2.0; "
            "'nunchaku' is CUDA-only and Kontext-only."
        ),
    )
    parser.add_argument(
        "--osb-no-flux-low-vram",
        dest="osb_flux_low_vram",
        action="store_false",
        help="Disable CPU offload for SDNQ models. Faster but uses more VRAM.)",
    )
    parser.set_defaults(osb_flux_low_vram=True)
    parser.add_argument(
        "--osb-flux-sdcpp-cache-mode",
        type=str,
        choices=["spectrum", "step-skip", "none"],
        default="step-skip",
        help=(
            "Step caching for sd.cpp. 'spectrum' predicts skipped steps from earlier ones; "
            "'step-skip' reuses the last step's change when the image barely moved. "
            "Warmup is 25%% of the selected step count."
        ),
    )
    parser.add_argument(
        "--osb-flux-sdcpp-diffusion-quant",
        type=str,
        choices=FLUX_SDCPP_DIFFUSION_QUANTS,
        default=None,
        help=(
            "Quantization level for the diffusion model via sd.cpp. "
            "Ordered from largest/best quality to smallest/lowest quality "
            "(defaults: Q4_K_M, or INT8-ConvRot for Fibo 2.0)."
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
            "(defaults: Q4_K_XL, or Q4_K_M for Kontext)."
        ),
    )
    parser.add_argument(
        "--osb-no-flux-upscale-small-crops",
        dest="osb_flux_upscale_small_crops",
        action="store_false",
        help="Disable scaling small OSB crops to ~1MP before inference (not Kontext or LaMa)",
    )
    parser.set_defaults(osb_flux_upscale_small_crops=True)
    parser.add_argument(
        "--osb-flux-group-regions",
        action="store_true",
        help="Group non-solid OSB regions into one expanded inpainting mask per page",
    )

    parser.add_argument(
        "--osb-flux-steps",
        type=int,
        default=None,
        help=(
            "Number of denoising steps for diffusion models (default: the model's). "
            "Fibo: 4 is recommended (1-12). "
            "Flux.3: 16 is recommended (1-16). "
            "Qwen-Image Turbo: fixed at 8. "
            "Klein: 4 is recommended (1-12). "
            "Kontext: 6-15 is recommended (1-30)."
        ),
    )
    parser.add_argument(
        "--osb-no-luminance-correction",
        dest="osb_flux_luminance_correction",
        action="store_false",
        help="Disable luminance correction that matches generated patch brightness to surrounding context",
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
        "--osb-outline-ratio",
        type=float,
        default=0.15,
        help="Outline width for OSB text as fraction of font size (0-0.3)",
    )
    parser.add_argument(
        "--osb-line-spacing",
        type=float,
        default=1.0,
        help="Line spacing multiplier for OSB text (0.5-2.0)",
    )
    parser.add_argument(
        "--osb-padding-pixels",
        type=float,
        default=4.0,
        help="Padding between text and the edge of the OSB region (2-12). "
        "Increase for more space between text and region boundaries.",
    )
    parser.add_argument(
        "--osb-no-auto-vertical-text",
        dest="osb_auto_vertical_text",
        action="store_false",
        help="Disable automatically stacking short translated OSB text vertically",
    )
    parser.set_defaults(osb_auto_vertical_text=True)
    parser.add_argument(
        "--osb-vertical-line-spacing-mult",
        type=float,
        default=1.0,
        help="Line spacing multiplier for vertical OSB text (0.5-2.0)",
    )
    parser.add_argument(
        "--osb-vertical-font-size-mult",
        type=float,
        default=1.0,
        help="Font size multiplier for vertical OSB text (0.1-3.0)",
    )
    parser.add_argument(
        "--osb-bbox-expansion-width",
        type=float,
        default=0.1,
        help="Bounding box width expansion percent for OSB detection",
    )
    parser.add_argument(
        "--osb-bbox-expansion-height",
        type=float,
        default=0.1,
        help="Bounding box height expansion percent for OSB detection",
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
        default=0.5,
        help="Confidence threshold for OSB text detection (0.0-1.0)",
    )
    parser.add_argument(
        "--osb-text-free-only",
        action="store_true",
        help=(
            "Only target 'text_free' detections from the RT-DETR model "
            "(narration-style text blocks), ignoring stray SFX and other detections"
        ),
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
        "--osb-no-follow-tilt",
        dest="osb_follow_tilt",
        action="store_false",
        help="Disable rotation-matching text rendering for outside speech bubble text",
    )
    parser.set_defaults(osb_follow_tilt=True)
    parser.add_argument(
        "--osb-max-tilt-deg",
        type=float,
        default=20.0,
        help="Maximum tilt angle clamp for OSB text rendering (0.0-45.0 degrees)",
    )
    parser.add_argument(
        "--osb-min-tilt-deg",
        type=float,
        default=3.0,
        help="Deadband threshold below which OSB tilt snaps to 0.0 degrees",
    )
    parser.add_argument(
        "--osb-tilt-confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence required to apply OSB tilt rotation (0.0-1.0)",
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
    parser.set_defaults(split_long_strips=True)
    parser.set_defaults(
        verbose=False,
        cpu=False,
        cleaning_only=False,
        enable_web_search=False,
    )

    args = parser.parse_args()
    from utils.logging import init_file_logging

    init_file_logging()

    osb_method = args.osb_inpainting_method
    osb_backends = flux_method_backends(osb_method)
    if args.osb_flux_backend is None:
        args.osb_flux_backend = flux_valid_backend(osb_method, "sdnq")
    elif osb_backends and args.osb_flux_backend not in osb_backends:
        parser.error(
            f"--osb-flux-backend {args.osb_flux_backend} is not supported with "
            f"--osb-inpainting-method {osb_method}. "
            f"Valid backends: {', '.join(osb_backends)}."
        )
    if args.osb_flux_steps is None:
        args.osb_flux_steps = flux_steps_range(osb_method)[0]
    if args.osb_flux_backend == "sdcpp" and args.osb_flux_sdcpp_diffusion_quant:
        valid_diffusion_quants = flux_sdcpp_diffusion_quants(osb_method)
        if args.osb_flux_sdcpp_diffusion_quant not in valid_diffusion_quants:
            parser.error(
                "--osb-flux-sdcpp-diffusion-quant "
                f"{args.osb_flux_sdcpp_diffusion_quant} is not available for "
                f"{osb_method}. "
                "Valid diffusion quants: "
                f"{', '.join(valid_diffusion_quants) or 'none'}."
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
        PROVIDER_API_KEYS,
        CleaningConfig,
        DetectionConfig,
        MangaTranslatorConfig,
        OutputConfig,
        OutsideTextConfig,
        PreprocessingConfig,
        RenderingConfig,
        TranslationConfig,
        env_api_key,
    )
    from core.llm_defaults import get_provider_sampling_defaults
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
    key_field, key_env_vars = PROVIDER_API_KEYS[provider]
    # The CLI flag for SpaceXAI is --spacexai-api-key; others match the config field
    key_arg = "spacexai_api_key" if provider == "SpaceXAI" else key_field
    api_key = getattr(args, key_arg, None) or env_api_key(provider)
    api_key_arg_name = "--" + key_arg.replace("_", "-")
    api_key_env_var = " or ".join(key_env_vars)
    default_model = CLI_DEFAULT_MODELS[provider]
    compatible_url = (
        args.openai_compatible_url if provider == "OpenAI-Compatible" else None
    )

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

    if args.enable_web_search and provider in NO_WEB_SEARCH_PROVIDERS:
        log_message(
            f"Warning: {provider} has no built-in web search; "
            "--enable-web-search is ignored.",
            always_print=True,
        )

    model_name = args.model_name or default_model
    if not args.model_name:
        log_message(f"Using default model for {provider}: {model_name}", verbose=True)

    from core.ml.model_manager import get_model_manager, migrate_model_dirs

    models_dir = Path(args.models).resolve()
    # Create models directory if it doesn't exist (model manager will handle model downloads)
    models_dir.mkdir(parents=True, exist_ok=True)
    migrate_model_dirs(models_dir)
    get_model_manager().set_models_dir(models_dir)

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
    if args.cpu:
        get_model_manager().set_device(target_device)
    if args.hf_cache:
        get_model_manager().use_hf_cache = True
    log_message(
        f"Using {target_device.type.upper()} device.",
        always_print=True,
    )

    use_otsu_config_val = args.use_otsu_threshold

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
        retry_failed_once=bool(args.batch and args.retry_failed_once),
        overlap_llm_with_inpaint=bool(args.overlap_llm_with_inpaint),
        detection=DetectionConfig(
            confidence=args.confidence,
            conjoined_confidence=args.conjoined_confidence,
            panel_confidence=args.panel_confidence,
            seg_model=args.seg_model,
            bubble_detector_model=args.bubble_detector_model,
            conjoined_detection=args.conjoined_detection,
            use_panel_sorting=args.use_panel_sorting,
            use_osb_text_verification=args.use_osb_text_verification,
        ),
        cleaning=CleaningConfig(
            thresholding_value=args.thresholding_value,
            use_otsu_threshold=use_otsu_config_val,
            auto_threshold=args.auto_threshold,
            roi_shrink_px=max(0, min(10, int(args.roi_shrink_px))),
            inpaint_colored_bubbles=args.inpaint_colored_bubbles,
        ),
        translation=TranslationConfig(
            provider=provider,
            # Keys for other providers come from the environment in __post_init__
            **{key_field: api_key},
            openai_compatible_url=compatible_url,
            opencode_tier=getattr(args, "opencode_tier", "zen"),
            model_name=model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=(
                args.top_k
                if args.top_k is not None
                else get_provider_sampling_defaults(provider)["top_k"]
            ),
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
            terminology_memory=args.batch and args.batch_terminology_memory,
            terminology_memory_path=(args.batch_glossary_file if args.batch else None),
            osb_min_side_pixels=args.osb_min_side_pixels,
            special_instructions=args.special_instructions,
            ero_doujinshi_mode=args.ero_doujinshi_mode,
            ocr_correction=args.ocr_correction,
            force_cache_translations=args.force_cache_translations,
            ocr_method=args.ocr_method,
        ),
        rendering=RenderingConfig(
            font_dir=args.font_dir,
            max_font_size=args.max_font_size,
            min_font_size=args.min_font_size,
            auto_font_size=args.auto_font_size,
            line_spacing_mult=args.line_spacing_mult,
            use_ligatures=args.use_ligatures,
            hyphenate_before_scaling=args.hyphenate_before_scaling,
            hyphenation_min_word_length=args.hyphenation_min_word_length,
            badness_exponent=args.badness_exponent,
            padding_pixels=args.padding_pixels,
            supersampling_factor=args.supersampling_factor,
            detach_trailing_punctuation=args.detach_trailing_punctuation,
            auto_vertical_text=args.auto_vertical_text,
            vertical_line_spacing_mult=args.vertical_line_spacing_mult,
            vertical_font_size_mult=args.vertical_font_size_mult,
        ),
        output=OutputConfig(
            output_format=args.output_format,
            jpeg_quality=args.jpeg_quality,
            png_compression=args.png_compression,
            upscale_final_image=args.image_upscale_mode == "final",
            image_upscale_factor=args.image_upscale_factor,
            image_upscale_model=args.image_upscale_model,
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
            flux_sdcpp_diffusion_quant=args.osb_flux_sdcpp_diffusion_quant or "",
            flux_sdcpp_text_encoder_quant=args.osb_flux_sdcpp_text_encoder_quant or "",
            flux_num_inference_steps=args.osb_flux_steps,
            flux_luminance_correction=args.osb_flux_luminance_correction,
            flux_upscale_small_crops=args.osb_flux_upscale_small_crops,
            flux_group_regions=args.osb_flux_group_regions,
            flux_residual_diff_threshold=args.osb_flux_residual_threshold,
            osb_confidence=args.osb_confidence,
            osb_text_free_only=args.osb_text_free_only,
            seed=args.osb_seed,
            osb_font_dir=args.osb_font_dir,
            osb_max_font_size=args.osb_max_font_size,
            osb_min_font_size=args.osb_min_font_size,
            osb_use_ligatures=args.osb_use_ligatures,
            osb_outline_ratio=args.osb_outline_ratio,
            osb_line_spacing=args.osb_line_spacing,
            osb_padding_pixels=args.osb_padding_pixels,
            osb_auto_vertical_text=args.osb_auto_vertical_text,
            osb_vertical_line_spacing_mult=args.osb_vertical_line_spacing_mult,
            osb_vertical_font_size_mult=args.osb_vertical_font_size_mult,
            bbox_expansion_percent_width=args.osb_bbox_expansion_width,
            bbox_expansion_percent_height=args.osb_bbox_expansion_height,
            osb_render_expansion_narrow_multiplier=args.osb_render_expansion_narrow,
            osb_render_expansion_tiny_multiplier=args.osb_render_expansion_tiny,
            osb_render_expansion_aspect_ratio_threshold=(
                args.osb_render_expansion_aspect_threshold
            ),
            osb_render_expansion_area_ratio_threshold=(
                args.osb_render_expansion_area_threshold
            ),
            text_box_proximity_ratio=args.osb_text_box_proximity_ratio,
            osb_follow_tilt=args.osb_follow_tilt,
            osb_max_tilt_deg=args.osb_max_tilt_deg,
            osb_min_tilt_deg=args.osb_min_tilt_deg,
            osb_tilt_confidence_threshold=args.osb_tilt_confidence_threshold,
        ),
        preprocessing=PreprocessingConfig(
            enabled=args.image_upscale_mode == "initial",
            factor=args.image_upscale_factor,
            auto_scale=args.auto_scale,
            split_long_strips=args.split_long_strips,
        ),
        test_mode=args.test_mode,
    )

    clamp_settings(config)
    # --- Execute ---
    if args.batch:
        input_path = Path(args.input)
        suffix = input_path.suffix.lower()
        glossary_name = (
            None
            if suffix == ".txt"
            else input_path.stem
            if suffix in ARCHIVE_SUFFIXES
            else input_path.resolve().name
        )
        zip_temp_dir_obj = None
        path_list_temp_dir_obj = None
        path_list_run_dir = None
        glossary_path = None
        preserve_structure = False
        source_path_map = None

        if input_path.is_file() and suffix in ARCHIVE_SUFFIXES:
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
                sys.exit(1)
            except Exception as e:
                log_message(
                    f"Error extracting ZIP archive: {e!s}",
                    always_print=True,
                )
                sys.exit(1)
        elif input_path.is_file() and input_path.suffix.lower() == ".txt":
            from utils.path_list import stage_path_list

            log_message(
                f"Detected failed-paths list: {input_path.name}",
                always_print=True,
            )
            path_list_temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(path_list_temp_dir_obj.name)
            source_path_map, path_list_run_dir, glossary_path = stage_path_list(
                input_path, temp_dir_path
            )
            if not source_path_map:
                log_message(
                    f"Error: no existing image paths found in '{args.input}'.",
                    always_print=True,
                )
                sys.exit(1)

            log_message(
                f"Prepared {len(source_path_map)} image(s) from path list",
                always_print=True,
            )
            input_path = temp_dir_path
            preserve_structure = path_list_run_dir is not None
        elif not input_path.is_dir():
            log_message(
                f"Error: --batch requires --input '{args.input}' to be a directory, "
                "ZIP/CBZ archive, or failed-paths .txt file.",
                always_print=True,
            )
            sys.exit(1)

        # A failed-paths list writes back into its original run unless --output is set
        output_dir = Path(args.output) if args.output else path_list_run_dir

        if args.output:
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
                sys.exit(1)

        try:
            results = batch_translate_images(
                input_path,
                config,
                output_dir,
                preserve_structure=preserve_structure,
                source_path_map=source_path_map,
                glossary_name=glossary_name,
                glossary_path=glossary_path,
            )
            failed_paths_file = results.get("failed_paths_file")
            if failed_paths_file:
                log_message(
                    f"Failed image paths saved to: {failed_paths_file}",
                    always_print=True,
                )
            if results.get("error_count", 0) > 0:
                raise SystemExit(1)
        finally:
            for temp_obj, label in (
                (zip_temp_dir_obj, "ZIP extraction"),
                (path_list_temp_dir_obj, "path list staging"),
            ):
                if not temp_obj:
                    continue
                try:
                    temp_obj.cleanup()
                    log_message(
                        f"Cleaned up {label} temporary directory",
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
            sys.exit(1)

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
