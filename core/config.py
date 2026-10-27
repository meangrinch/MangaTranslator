import os
from dataclasses import dataclass, field
from typing import Any

import torch

from core.device import get_best_device
from core.llm_defaults import DEFAULT_LLM_PROVIDER, get_provider_sampling_defaults

# Each provider's TranslationConfig key field and the env vars it falls back to, in order
PROVIDER_API_KEYS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Google": ("google_api_key", ("GOOGLE_API_KEY", "GEMINI_API_KEY")),
    "OpenAI": ("openai_api_key", ("OPENAI_API_KEY",)),
    "Anthropic": ("anthropic_api_key", ("ANTHROPIC_API_KEY",)),
    "SpaceXAI": ("xai_api_key", ("SPACEXAI_API_KEY", "XAI_API_KEY")),
    "Meta Model": ("meta_api_key", ("META_MODEL_API_KEY", "META_API_KEY")),
    "DeepSeek": ("deepseek_api_key", ("DEEPSEEK_API_KEY",)),
    "Z.ai": ("zai_api_key", ("ZAI_API_KEY",)),
    "Moonshot AI": ("moonshot_api_key", ("MOONSHOT_API_KEY",)),
    "Xiaomi MiMo": ("mimo_api_key", ("MIMO_API_KEY",)),
    "QwenCloud": ("qwencloud_api_key", ("QWENCLOUD_API_KEY", "QWEN_API_KEY")),
    "OpenCode": (
        "opencode_api_key",
        ("OPENCODE_API_KEY", "OPENCODE_ZEN_API_KEY", "OPENCODE_GO_API_KEY"),
    ),
    "OpenRouter": ("openrouter_api_key", ("OPENROUTER_API_KEY",)),
    "OpenAI-Compatible": (
        "openai_compatible_api_key",
        ("OPENAI_COMPATIBLE_API_KEY",),
    ),
}


def env_api_key(provider: str) -> str:
    """First non-empty environment variable holding `provider`'s API key, or ""."""
    env_names = PROVIDER_API_KEYS.get(provider, ("", ()))[1]
    return next((value for name in env_names if (value := os.environ.get(name))), "")


@dataclass
class DetectionConfig:
    """Configuration for speech bubble detection."""

    confidence: float = 0.6
    conjoined_confidence: float = 0.35
    panel_confidence: float = 0.25
    seg_model: str = "yolo"  # "sam3", "sam2", or "yolo"
    bubble_detector_model: str = "yolo_2"  # "yolo_1" or "yolo_2"
    conjoined_detection: bool = True
    use_panel_sorting: bool = True
    use_osb_text_verification: bool = True


@dataclass
class CleaningConfig:
    """Configuration for speech bubble cleaning."""

    thresholding_value: int = 200
    use_otsu_threshold: bool = False
    auto_threshold: bool = True
    roi_shrink_px: int = 5
    inpaint_colored_bubbles: bool = False


_DEFAULT_TRANSLATION_PROVIDER = DEFAULT_LLM_PROVIDER
_DEFAULT_SAMPLING = get_provider_sampling_defaults(_DEFAULT_TRANSLATION_PROVIDER)


@dataclass
class TranslationConfig:
    """Configuration for text translation."""

    provider: str = _DEFAULT_TRANSLATION_PROVIDER
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""
    meta_api_key: str = ""
    deepseek_api_key: str = ""
    zai_api_key: str = ""
    moonshot_api_key: str = ""
    mimo_api_key: str = ""
    qwencloud_api_key: str = ""
    opencode_api_key: str = ""
    opencode_tier: str = "zen"
    openrouter_api_key: str = ""
    openai_compatible_url: str = "http://localhost:8080/v1"
    openai_compatible_api_key: str | None = ""
    model_name: str = "gemini-3.8-flash"
    provider_models: dict[str, str | None] = field(default_factory=dict)
    temperature: float = float(_DEFAULT_SAMPLING["temperature"])
    top_p: float = float(_DEFAULT_SAMPLING["top_p"])
    top_k: int = int(_DEFAULT_SAMPLING["top_k"])
    max_tokens: int | None = (
        None  # None = use default logic (32768 for reasoning, 4096 otherwise)
    )
    input_language: str = "Japanese"
    output_language: str = "English"
    reading_direction: str = "rtl"
    translation_mode: str = "one-step"
    reasoning_effort: str | None = (
        None  # None = per-provider fallback in translation._build_generation_config
    )
    effort: str | None = (
        "high"  # Claude Opus 4.5+, Sonnet 4.6+, Haiku 5.5+, Fable 5+: token spending eagerness (max/xhigh/high/medium/low)
    )
    verbosity: str | None = (
        None  # GPT-5 and GPT-6 series: controls response verbosity (high/medium/low)
    )
    send_full_page_context: bool = True
    whiteout_conjoined_bubbles: bool = True
    upscale_method: str = "model_lite"  # "model", "model_lite", "lanczos", or "none"
    enable_web_search: bool = False  # Enable model's built-in web search for up-to-date information. OpenRouter uses its own web search tool.
    enable_code_execution: bool = False  # Enable Gemini's code execution tool for image zoom/inspection (Gemini 3+ only)
    use_custom_sampling: bool = True
    image_detail: str = "auto"  # OpenAI image detail (auto/high/low/original)
    media_resolution: str = (
        "auto"  # Only available via Google provider (auto/high/medium/low)
    )
    media_resolution_bubbles: str = "auto"  # Gemini 3 and SpaceXAI models
    media_resolution_context: str = "auto"  # Gemini 3 and SpaceXAI models
    bubble_min_side_pixels: int = 128
    context_image_max_side_pixels: int = 1024
    previous_context_image_count: int = 0
    previous_context_text_count: int = 0
    terminology_memory: bool = False
    terminology_memory_path: str | None = None  # Batch resolves a default when unset
    osb_min_side_pixels: int = 128
    special_instructions: str | None = None
    ero_doujinshi_mode: bool = False
    ocr_correction: bool = False
    force_cache_translations: bool = False
    ocr_method: str = "LLM"  # "LLM", "manga-ocr", or "paddleocr-vl-1.6"
    request_coordinator: Any | None = None


@dataclass
class RenderingConfig:
    """Configuration for rendering translated text."""

    font_dir: str = "./fonts"
    max_font_size: int = 32
    min_font_size: int = 8
    auto_font_size: bool = True
    line_spacing_mult: float = 1.0
    use_ligatures: bool = False
    hyphenate_before_scaling: bool = True
    hyphenation_min_word_length: int = 8
    # Set per render from the output language; not a user setting.
    hyphenation_language: str = ""
    badness_exponent: float = 3.0
    padding_pixels: float = 4.0
    outline_ratio: float = 0.0
    supersampling_factor: int = 4
    detach_trailing_punctuation: bool = True
    auto_vertical_text: bool = True
    vertical_line_spacing_mult: float = 1.0
    vertical_font_size_mult: float = 1.0


@dataclass
class OutsideTextConfig:
    """Configuration for outside speech bubble text detection and removal."""

    enabled: bool = False
    enable_page_number_filtering: bool = False
    page_filter_margin_threshold: float = 0.1
    page_filter_min_area_ratio: float = 0.05
    min_area_ignore_ratio: float = 0.0
    seed: int = 1  # -1 = random
    huggingface_token: str = ""  # Required for Flux Kontext model downloads
    # flux_klein_9b, flux_klein_4b, flux_kontext, flux3_klein, fibo2_edit_turbo,
    # qwen_image_2_1_turbo, lama_manga, opencv, none
    inpainting_method: str = "flux_klein_4b"
    flux_backend: str = "sdnq"  # "sdcpp", "sdnq", "nunchaku"; see FLUX_METHOD_BACKENDS
    flux_low_vram: bool = True  # Use CPU offload for SDNQ
    flux_sdcpp_cache_mode: str = "step-skip"
    flux_sdcpp_diffusion_quant: str = "Q4_K_M"
    flux_sdcpp_text_encoder_quant: str = ""
    flux_num_inference_steps: int = 4
    flux_luminance_correction: bool = (
        True  # Match patch luminance to surrounding context
    )
    flux_upscale_small_crops: bool = True
    flux_group_regions: bool = False
    # Paste edit-model redraws only over the letters, keeping the art around them
    flux_preserve_art: bool = True
    flux_residual_diff_threshold: float = 0.15
    osb_confidence: float = 0.5
    osb_text_free_only: bool = False  # Use the RT-DETR text_free class as OSB source
    # Add text the OSB model missed: letters the text segmentation model finds outside
    # bubbles and OSB boxes, kept where the OSB model confirms them on crops
    osb_recover_missed_text: bool = True
    osb_font_dir: str | None = None  # None = use main font as fallback
    osb_max_font_size: int = 64
    osb_min_font_size: int = 10
    osb_use_ligatures: bool = False
    osb_outline_ratio: float = 0.15
    osb_line_spacing: float = 1.0
    osb_padding_pixels: float = 4.0
    osb_auto_vertical_text: bool = True
    osb_vertical_line_spacing_mult: float = 1.0
    osb_vertical_font_size_mult: float = 1.0
    bbox_expansion_percent_width: float = 0.1
    bbox_expansion_percent_height: float = 0.1
    osb_render_expansion_narrow_multiplier: float = 1.0
    osb_render_expansion_tiny_multiplier: float = 1.0
    osb_render_expansion_aspect_ratio_threshold: float = 0.4
    osb_render_expansion_area_ratio_threshold: float = 0.005
    text_box_proximity_ratio: float = 0.02  # 2% of image dimension
    flux_guidance_scale: float = 2.5
    flux_prompt: str = "Remove all text."
    osb_follow_tilt: bool = True  # Enable rotation-matching text rendering for OSB
    osb_max_tilt_deg: float = 20.0
    osb_min_tilt_deg: float = 3.0
    osb_tilt_confidence_threshold: float = 0.5


@dataclass
class OutputConfig:
    """Configuration for saving output images."""

    jpeg_quality: int = 95
    png_compression: int = 2
    output_format: str = "png"
    upscale_final_image: bool = False
    image_upscale_factor: float = 2.0
    image_upscale_model: str = "model_lite"  # "model" or "model_lite"


@dataclass
class MangaTranslatorConfig:
    """Main configuration for the MangaTranslator pipeline."""

    yolo_model_path: str
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    outside_text: OutsideTextConfig = field(default_factory=OutsideTextConfig)
    preprocessing: "PreprocessingConfig" = field(
        default_factory=lambda: PreprocessingConfig()
    )
    verbose: bool = False
    device: torch.device | None = None
    cleaning_only: bool = False
    upscaling_only: bool = False
    test_mode: bool = False
    processing_scale: float = 1.0
    parallel_requests: int = 1
    batch_parallel_within_pages: bool = False
    overlap_llm_with_inpaint: bool = True
    retry_failed_once: bool = True
    request_coordinator: Any | None = None

    def __post_init__(self):
        # Load API keys from environment variables if not already set
        for provider, (key_field, _) in PROVIDER_API_KEYS.items():
            if not getattr(self.translation, key_field):
                setattr(self.translation, key_field, env_api_key(provider))
        if not self.outside_text.huggingface_token:
            self.outside_text.huggingface_token = os.environ.get("HF_TOKEN", "")

        # Autodetect device if not specified
        if self.device is None:
            self.device = get_best_device()


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing before detection/cleaning."""

    enabled: bool = False
    factor: float = 2.0
    auto_scale: bool = True
    split_long_strips: bool = True


def calculate_reasoning_budget(total_tokens: int, effort_level: str) -> int:
    """
    Calculate reasoning token budget based on effort level.

    Args:
        total_tokens: Total available tokens (typically max_tokens)
        effort_level: Reasoning effort level ("high", "medium", "low", "minimal", "auto", or "none")

    Returns:
        int: Calculated budget in tokens
        - "high": 80% of total_tokens
        - "medium": 50% of total_tokens
        - "low": 20% of total_tokens
        - "minimal": 10% of total_tokens
        - "auto" or "none": Returns 0 (caller should handle these cases separately)
    """
    if effort_level == "high":
        return int(total_tokens * 0.8)
    elif effort_level == "medium":
        return int(total_tokens * 0.5)
    elif effort_level == "low":
        return int(total_tokens * 0.2)
    elif effort_level == "minimal":
        return int(total_tokens * 0.1)
    else:
        # "auto" or "none" - return 0, caller should handle these cases
        return 0
