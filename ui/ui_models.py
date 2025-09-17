from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from core.models import (
    MangaTranslatorConfig,
    DetectionConfig,
    CleaningConfig,
    TranslationConfig,
    RenderingConfig,
    OutputConfig,
)


@dataclass
class UIDetectionSettings:
    """UI state for detection settings."""

    confidence: float = 0.35
    use_sam2: bool = True


@dataclass
class UICleaningSettings:
    """UI state for cleaning settings."""

    thresholding_value: int = 210
    use_otsu_threshold: bool = False


@dataclass
class UITranslationProviderSettings:
    """UI state for translation provider settings."""

    provider: str = "Gemini"
    gemini_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""
    anthropic_api_key: Optional[str] = ""
    openrouter_api_key: Optional[str] = ""
    openai_compatible_url: str = "http://localhost:11434/v1"
    openai_compatible_api_key: Optional[str] = ""


@dataclass
class UITranslationLLMSettings:
    """UI state for LLM-specific translation settings."""

    model_name: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 64
    translation_mode: str = "one-step"
    reading_direction: str = "rtl"
    send_full_page_context: bool = True
    openrouter_reasoning_override: bool = False


@dataclass
class UIRenderingSettings:
    """UI state for rendering settings."""

    max_font_size: int = 14
    min_font_size: int = 8
    line_spacing: float = 1.0
    use_subpixel_rendering: bool = True
    font_hinting: str = "none"
    use_ligatures: bool = False
    hyphenate_before_scaling: bool = True


@dataclass
class UIOutputSettings:
    """UI state for output settings."""

    output_format: str = "auto"
    jpeg_quality: int = 95
    png_compression: int = 6


@dataclass
class UIGeneralSettings:
    """UI state for general application settings."""

    verbose: bool = False
    cleaning_only: bool = False
    enable_thinking: bool = True
    reasoning_effort: str = "medium"
    openrouter_reasoning_override: bool = False


@dataclass
class UIConfigState:
    """Represents the complete configuration state managed by the UI."""

    yolo_model: Optional[str] = None
    detection: UIDetectionSettings = field(default_factory=UIDetectionSettings)
    cleaning: UICleaningSettings = field(default_factory=UICleaningSettings)
    provider_settings: UITranslationProviderSettings = field(default_factory=UITranslationProviderSettings)
    llm_settings: UITranslationLLMSettings = field(default_factory=UITranslationLLMSettings)
    rendering: UIRenderingSettings = field(default_factory=UIRenderingSettings)
    output: UIOutputSettings = field(default_factory=UIOutputSettings)
    general: UIGeneralSettings = field(default_factory=UIGeneralSettings)

    # Specific UI elements state (saved in config.json)
    input_language: str = "Japanese"
    output_language: str = "English"
    font_pack: Optional[str] = None
    batch_input_language: str = "Japanese"
    batch_output_language: str = "English"
    batch_font_pack: Optional[str] = None

    def to_save_dict(self) -> Dict[str, Any]:
        """Converts the UI state into a dictionary suitable for saving to config.json."""
        data = {
            "yolo_model": self.yolo_model,
            "confidence": self.detection.confidence,
            "use_sam2": self.detection.use_sam2,
            "reading_direction": self.llm_settings.reading_direction,
            "thresholding_value": self.cleaning.thresholding_value,
            "use_otsu_threshold": self.cleaning.use_otsu_threshold,
            "provider": self.provider_settings.provider,
            "gemini_api_key": self.provider_settings.gemini_api_key,
            "openai_api_key": self.provider_settings.openai_api_key,
            "anthropic_api_key": self.provider_settings.anthropic_api_key,
            "openrouter_api_key": self.provider_settings.openrouter_api_key,
            "openai_compatible_url": self.provider_settings.openai_compatible_url,
            "openai_compatible_api_key": self.provider_settings.openai_compatible_api_key,
            "model_name": self.llm_settings.model_name,
            "temperature": self.llm_settings.temperature,
            "top_p": self.llm_settings.top_p,
            "top_k": self.llm_settings.top_k,
            "translation_mode": self.llm_settings.translation_mode,
            "send_full_page_context": self.llm_settings.send_full_page_context,
            "openrouter_reasoning_override": self.general.openrouter_reasoning_override,
            "font_pack": self.font_pack,
            "max_font_size": self.rendering.max_font_size,
            "min_font_size": self.rendering.min_font_size,
            "line_spacing": self.rendering.line_spacing,
            "use_subpixel_rendering": self.rendering.use_subpixel_rendering,
            "font_hinting": self.rendering.font_hinting,
            "use_ligatures": self.rendering.use_ligatures,
            "hyphenate_before_scaling": self.rendering.hyphenate_before_scaling,
            "output_format": self.output.output_format,
            "jpeg_quality": self.output.jpeg_quality,
            "png_compression": self.output.png_compression,
            "verbose": self.general.verbose,
            "cleaning_only": self.general.cleaning_only,
            "enable_thinking": self.general.enable_thinking,
            "reasoning_effort": self.general.reasoning_effort,
            "input_language": self.input_language,
            "output_language": self.output_language,
            "batch_input_language": self.batch_input_language,
            "batch_output_language": self.batch_output_language,
            "batch_font_pack": self.batch_font_pack,
        }
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UIConfigState":
        """Creates a UIConfigState instance from a dictionary (e.g., loaded from config.json)."""

        from . import settings_manager  # Local import to avoid circular dependency issues

        defaults = settings_manager.DEFAULT_SETTINGS.copy()
        defaults.update(settings_manager.DEFAULT_BATCH_SETTINGS)

        return UIConfigState(
            yolo_model=data.get("yolo_model"),
            detection=UIDetectionSettings(
                confidence=data.get("confidence", defaults["confidence"]),
                use_sam2=data.get("use_sam2", defaults.get("use_sam2", True)),
            ),
            cleaning=UICleaningSettings(
                thresholding_value=data.get("thresholding_value", defaults["thresholding_value"]),
                use_otsu_threshold=data.get("use_otsu_threshold", defaults["use_otsu_threshold"]),
            ),
            provider_settings=UITranslationProviderSettings(
                provider=data.get("provider", defaults["provider"]),
                gemini_api_key=data.get("gemini_api_key", defaults["gemini_api_key"]),
                openai_api_key=data.get("openai_api_key", defaults["openai_api_key"]),
                anthropic_api_key=data.get("anthropic_api_key", defaults["anthropic_api_key"]),
                openrouter_api_key=data.get("openrouter_api_key", defaults["openrouter_api_key"]),
                openai_compatible_url=data.get("openai_compatible_url", defaults["openai_compatible_url"]),
                openai_compatible_api_key=data.get(
                    "openai_compatible_api_key", defaults["openai_compatible_api_key"]
                ),
            ),
            llm_settings=UITranslationLLMSettings(
                model_name=data.get("model_name"),
                temperature=data.get("temperature", defaults["temperature"]),
                top_p=data.get("top_p", defaults["top_p"]),
                top_k=data.get("top_k", defaults["top_k"]),
                translation_mode=data.get("translation_mode", defaults["translation_mode"]),
                reading_direction=data.get("reading_direction", defaults["reading_direction"]),
                send_full_page_context=data.get("send_full_page_context", True),
            ),
            rendering=UIRenderingSettings(
                max_font_size=data.get("max_font_size", defaults["max_font_size"]),
                min_font_size=data.get("min_font_size", defaults["min_font_size"]),
                line_spacing=data.get("line_spacing", defaults["line_spacing"]),
                use_subpixel_rendering=data.get("use_subpixel_rendering", defaults["use_subpixel_rendering"]),
                font_hinting=data.get("font_hinting", defaults["font_hinting"]),
                use_ligatures=data.get("use_ligatures", defaults["use_ligatures"]),
                hyphenate_before_scaling=data.get(
                    "hyphenate_before_scaling",
                    defaults.get("hyphenate_before_scaling", True),
                ),
            ),
            output=UIOutputSettings(
                output_format=data.get("output_format", defaults["output_format"]),
                jpeg_quality=data.get("jpeg_quality", defaults["jpeg_quality"]),
                png_compression=data.get("png_compression", defaults["png_compression"]),
            ),
            general=UIGeneralSettings(
                verbose=data.get("verbose", defaults["verbose"]),
                cleaning_only=data.get("cleaning_only", defaults["cleaning_only"]),
                enable_thinking=data.get("enable_thinking", defaults.get("enable_thinking", True)),
                reasoning_effort=data.get("reasoning_effort", defaults.get("reasoning_effort", "medium")),
                openrouter_reasoning_override=data.get(
                    "openrouter_reasoning_override", defaults.get("openrouter_reasoning_override", False)
                ),
            ),
            input_language=data.get("input_language", defaults["input_language"]),
            output_language=data.get("output_language", defaults["output_language"]),
            font_pack=data.get("font_pack"),
            batch_input_language=data.get("batch_input_language", defaults["batch_input_language"]),
            batch_output_language=data.get("batch_output_language", defaults["batch_output_language"]),
            batch_font_pack=data.get("batch_font_pack"),
        )


def map_ui_to_backend_config(
    ui_state: UIConfigState,
    models_dir: Path,
    fonts_base_dir: Path,
    target_device: Optional[torch.device],
    is_batch: bool = False,
) -> MangaTranslatorConfig:
    """Maps the UIConfigState to the backend MangaTranslatorConfig."""

    yolo_path = ""
    font_pack_name = ui_state.batch_font_pack if is_batch else ui_state.font_pack
    font_dir_path = fonts_base_dir / font_pack_name if font_pack_name else ""
    input_lang = ui_state.batch_input_language if is_batch else ui_state.input_language
    output_lang = ui_state.batch_output_language if is_batch else ui_state.output_language

    detection_cfg = DetectionConfig(confidence=ui_state.detection.confidence)
    detection_cfg.use_sam2 = ui_state.detection.use_sam2

    cleaning_cfg = CleaningConfig(
        thresholding_value=ui_state.cleaning.thresholding_value,
        use_otsu_threshold=ui_state.cleaning.use_otsu_threshold,
    )

    translation_cfg = TranslationConfig(
        provider=ui_state.provider_settings.provider,
        gemini_api_key=ui_state.provider_settings.gemini_api_key or "",
        openai_api_key=ui_state.provider_settings.openai_api_key or "",
        anthropic_api_key=ui_state.provider_settings.anthropic_api_key or "",
        openrouter_api_key=ui_state.provider_settings.openrouter_api_key or "",
        openai_compatible_url=ui_state.provider_settings.openai_compatible_url,
        openai_compatible_api_key=ui_state.provider_settings.openai_compatible_api_key,
        model_name=ui_state.llm_settings.model_name or "",
        temperature=ui_state.llm_settings.temperature,
        top_p=ui_state.llm_settings.top_p,
        top_k=ui_state.llm_settings.top_k,
        openrouter_reasoning_override=ui_state.general.openrouter_reasoning_override,
        input_language=input_lang,
        output_language=output_lang,
        reading_direction=ui_state.llm_settings.reading_direction,
        translation_mode=ui_state.llm_settings.translation_mode,
        enable_thinking=ui_state.general.enable_thinking,
        send_full_page_context=ui_state.llm_settings.send_full_page_context,
    )

    rendering_cfg = RenderingConfig(
        font_dir=str(font_dir_path),
        max_font_size=ui_state.rendering.max_font_size,
        min_font_size=ui_state.rendering.min_font_size,
        line_spacing=ui_state.rendering.line_spacing,
        use_subpixel_rendering=ui_state.rendering.use_subpixel_rendering,
        font_hinting=ui_state.rendering.font_hinting,
        use_ligatures=ui_state.rendering.use_ligatures,
        hyphenate_before_scaling=ui_state.rendering.hyphenate_before_scaling,
    )

    output_cfg = OutputConfig(
        output_format=ui_state.output.output_format,
        jpeg_quality=ui_state.output.jpeg_quality,
        png_compression=ui_state.output.png_compression,
    )

    backend_config = MangaTranslatorConfig(
        yolo_model_path=str(yolo_path),
        verbose=ui_state.general.verbose,
        device=target_device,
        detection=detection_cfg,
        cleaning=cleaning_cfg,
        translation=translation_cfg,
        rendering=rendering_cfg,
        output=output_cfg,
        cleaning_only=ui_state.general.cleaning_only,
    )

    return backend_config
