from dataclasses import dataclass, field
from typing import Optional

import torch

from core.llm_defaults import DEFAULT_LLM_PROVIDER, get_provider_sampling_defaults


@dataclass
class DetectionConfig:
    """Configuration for speech bubble detection."""

    confidence: float = 0.6
    conjoined_confidence: float = 0.35
    panel_confidence: float = 0.25
    use_sam2: bool = True
    conjoined_detection: bool = True
    use_panel_sorting: bool = True
    use_osb_text_verification: bool = True


@dataclass
class CleaningConfig:
    """Configuration for speech bubble cleaning."""

    thresholding_value: int = 190
    use_otsu_threshold: bool = False
    roi_shrink_px: int = 4
    inpaint_colored_bubbles: bool = True


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
    deepseek_api_key: str = ""
    zai_api_key: str = ""
    moonshot_api_key: str = ""
    openrouter_api_key: str = ""
    openai_compatible_url: str = "http://localhost:1234/v1"
    openai_compatible_api_key: Optional[str] = ""
    model_name: str = "gemini-2.5-flash"
    provider_models: dict[str, Optional[str]] = field(default_factory=dict)
    temperature: float = float(_DEFAULT_SAMPLING["temperature"])
    top_p: float = float(_DEFAULT_SAMPLING["top_p"])
    top_k: int = int(_DEFAULT_SAMPLING["top_k"])
    max_tokens: Optional[int] = (
        None  # None = use default logic (16384 for reasoning, 4096 otherwise)
    )
    input_language: str = "Japanese"
    output_language: str = "English"
    reading_direction: str = "rtl"
    translation_mode: str = "one-step"
    reasoning_effort: Optional[str] = (
        None  # Default: Google uses "auto", Anthropic uses "none", others use "medium"
    )
    effort: Optional[str] = (
        None  # Claude Opus 4.5 only: Controls token spending eagerness (high/medium/low)
    )
    send_full_page_context: bool = True
    upscale_method: str = "model_lite"  # "model", "model_lite", "lanczos", or "none"
    enable_web_search: bool = (
        False  # Enable model's built-in web search for up-to-date information. OpenRouter uses its own web search tool.
    )
    media_resolution: str = (
        "auto"  # Only available via Google provider (auto/high/medium/low)
    )
    media_resolution_bubbles: str = "auto"  # Gemini 3 models
    media_resolution_context: str = "auto"  # Gemini 3 models
    bubble_min_side_pixels: int = 128
    context_image_max_side_pixels: int = 1024
    osb_min_side_pixels: int = 128
    special_instructions: Optional[str] = None
    ocr_method: str = "LLM"  # "LLM" or "manga-ocr"


@dataclass
class RenderingConfig:
    """Configuration for rendering translated text."""

    font_dir: str = "./fonts"
    max_font_size: int = 16
    min_font_size: int = 8
    line_spacing_mult: float = 1.0
    use_subpixel_rendering: bool = False
    font_hinting: str = "none"
    use_ligatures: bool = False
    hyphenate_before_scaling: bool = True
    hyphen_penalty: float = 1000.0
    hyphenation_min_word_length: int = 8
    badness_exponent: float = 3.0
    padding_pixels: float = 5.0
    outline_width: float = 0.0
    supersampling_factor: int = 4


@dataclass
class OutsideTextConfig:
    """Configuration for outside speech bubble text detection and removal."""

    enabled: bool = False
    enable_page_number_filtering: bool = False
    page_filter_margin_threshold: float = 0.1
    page_filter_min_area_ratio: float = 0.05
    seed: int = 1  # -1 = random
    huggingface_token: str = ""  # Required for Flux Kontext model downloads
    force_cv2_inpainting: bool = False
    flux_num_inference_steps: int = 8
    flux_residual_diff_threshold: float = 0.15
    osb_confidence: float = 0.6
    osb_font_name: Optional[str] = None  # None = use main font as fallback
    osb_max_font_size: int = 64
    osb_min_font_size: int = 12
    osb_use_ligatures: bool = False
    osb_outline_width: float = 3.0
    osb_line_spacing: float = 1.0
    osb_use_subpixel_rendering: bool = False
    osb_font_hinting: str = "none"
    bbox_expansion_percent: float = 0.1
    text_box_proximity_ratio: float = 0.02  # 2% of image dimension
    flux_guidance_scale: float = 2.5
    flux_prompt: str = "Remove all text."


@dataclass
class OutputConfig:
    """Configuration for saving output images."""

    jpeg_quality: int = 95
    png_compression: int = 2
    output_format: str = "auto"
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
    device: Optional[torch.device] = None
    cleaning_only: bool = False
    upscaling_only: bool = False
    test_mode: bool = False
    processing_scale: float = 1.0

    def __post_init__(self):
        # Load API keys from environment variables if not already set
        import os

        if not self.translation.google_api_key:
            self.translation.google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not self.translation.openai_api_key:
            self.translation.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.translation.anthropic_api_key:
            self.translation.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.translation.xai_api_key:
            self.translation.xai_api_key = os.environ.get("XAI_API_KEY", "")
        if not self.translation.deepseek_api_key:
            self.translation.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.translation.moonshot_api_key:
            self.translation.moonshot_api_key = os.environ.get("MOONSHOT_API_KEY", "")
        if not self.translation.openrouter_api_key:
            self.translation.openrouter_api_key = os.environ.get(
                "OPENROUTER_API_KEY", ""
            )
        if (
            not self.translation.openai_compatible_api_key
        ):  # Check if it's None or empty string
            self.translation.openai_compatible_api_key = os.environ.get(
                "OPENAI_COMPATIBLE_API_KEY", ""
            )

        # Autodetect device if not specified
        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        pass


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing before detection/cleaning."""

    enabled: bool = False
    factor: float = 2.0
    auto_scale: bool = True


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
