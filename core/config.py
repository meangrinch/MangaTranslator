from dataclasses import dataclass, field
from typing import Optional
import torch
import cv2

@dataclass
class DetectionConfig:
    """Configuration for speech bubble detection."""
    confidence: float = 0.35

@dataclass
class CleaningConfig:
    """Configuration for speech bubble cleaning."""
    dilation_kernel_size: int = 7
    dilation_iterations: int = 1
    use_otsu_threshold: bool = False
    min_contour_area: int = 50
    closing_kernel_size: int = 7
    closing_iterations: int = 1
    closing_kernel_shape: int = cv2.MORPH_ELLIPSE
    constraint_erosion_kernel_size: int = 5
    constraint_erosion_iterations: int = 1

@dataclass
class TranslationConfig:
    """Configuration for text translation."""
    api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 1
    input_language: str = "Japanese"
    output_language: str = "English"
    reading_direction: str = "rtl"

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

@dataclass
class OutputConfig:
    """Configuration for saving output images."""
    jpeg_quality: int = 95
    png_compression: int = 6
    image_mode: str = "RGBA" # Default based on batch processing logic

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

    def __post_init__(self):
        if not self.translation.api_key:
            import os
            self.translation.api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not self.translation.api_key:
                 print("Warning: Gemini API key not provided in config or GOOGLE_API_KEY env var.")

        # Autodetect device if not specified
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pass