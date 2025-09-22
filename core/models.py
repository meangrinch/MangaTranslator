from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class DetectionConfig:
    """Configuration for speech bubble detection."""

    confidence: float = 0.35
    use_sam2: bool = True
    sam2_model_id: str = "facebook/sam2.1-hiera-large"


@dataclass
class CleaningConfig:
    """Configuration for speech bubble cleaning."""

    thresholding_value: int = 190
    use_otsu_threshold: bool = False
    roi_shrink_px: int = 4


@dataclass
class TranslationConfig:
    """Configuration for text translation."""

    provider: str = "Gemini"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    openai_compatible_url: str = "http://localhost:11434/v1"
    openai_compatible_api_key: Optional[str] = ""
    model_name: str = "gemini-2.5-flash"
    provider_models: dict[str, Optional[str]] = field(default_factory=dict)
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 64
    input_language: str = "Japanese"
    output_language: str = "English"
    reading_direction: str = "rtl"
    translation_mode: str = "one-step"
    enable_thinking: bool = True  # Gemini 2.5 Flash & Claude reasoning models
    reasoning_effort: Optional[str] = "medium"  # OpenAI reasoning models (o1/o3/o4-mini/gpt-5*)
    send_full_page_context: bool = True
    openrouter_reasoning_override: bool = False  # Forces max output tokens to 8192


@dataclass
class RenderingConfig:
    """Configuration for rendering translated text."""

    font_dir: str = "./fonts"
    max_font_size: int = 14
    min_font_size: int = 8
    line_spacing: float = 1.0
    use_subpixel_rendering: bool = False
    font_hinting: str = "none"
    use_ligatures: bool = False
    hyphenate_before_scaling: bool = True
    hyphen_penalty: float = 1000.0
    hyphenation_min_word_length: int = 8
    badness_exponent: float = 3.0
    padding_pixels: float = 8.0


@dataclass
class OutputConfig:
    """Configuration for saving output images."""

    jpeg_quality: int = 95
    png_compression: int = 6
    image_mode: str = "RGBA"  # Default image mode
    output_format: str = "auto"


@dataclass
class MangaTranslatorConfig:
    """Main configuration for the MangaTranslator pipeline."""

    yolo_model_path: str
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    verbose: bool = False
    device: Optional[torch.device] = None
    cleaning_only: bool = False

    def __post_init__(self):
        # Load API keys from environment variables if not already set
        import os

        if not self.translation.gemini_api_key:
            self.translation.gemini_api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not self.translation.openai_api_key:
            self.translation.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.translation.anthropic_api_key:
            self.translation.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.translation.openrouter_api_key:
            self.translation.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.translation.openai_compatible_api_key:  # Check if it's None or empty string
            self.translation.openai_compatible_api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "")

        # Autodetect device if not specified
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pass
