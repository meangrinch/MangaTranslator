from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from core.config import (CleaningConfig, DetectionConfig,
                         MangaTranslatorConfig, OutputConfig,
                         OutsideTextConfig, PreprocessingConfig,
                         RenderingConfig, TranslationConfig)


@dataclass
class UIDetectionSettings:
    """UI state for detection settings."""

    confidence: float = 0.35
    use_sam2: bool = True


@dataclass
class UICleaningSettings:
    """UI state for cleaning settings."""

    thresholding_value: int = 190
    use_otsu_threshold: bool = False
    roi_shrink_px: int = 4


@dataclass
class UITranslationProviderSettings:
    """UI state for translation provider settings."""

    provider: str = "Google"
    google_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""
    anthropic_api_key: Optional[str] = ""
    xai_api_key: Optional[str] = ""
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
    max_tokens: Optional[int] = None
    translation_mode: str = "one-step"
    reading_direction: str = "rtl"
    send_full_page_context: bool = True
    upscale_method: str = "model"
    bubble_min_side_pixels: int = 128
    context_image_max_side_pixels: int = 1536
    osb_min_side_pixels: int = 128
    special_instructions: Optional[str] = None


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
    hyphen_penalty: float = 1000.0
    hyphenation_min_word_length: int = 8
    badness_exponent: float = 3.0
    padding_pixels: float = 5.0


@dataclass
class UIOutputSettings:
    """UI state for output settings."""

    output_format: str = "auto"
    jpeg_quality: int = 95
    png_compression: int = 6
    image_upscale_mode: str = "off"  # "off", "initial", "final"
    image_upscale_factor: float = 2.0


@dataclass
class UIOutsideTextSettings:
    """UI state for outside speech bubble text removal settings."""

    enabled: bool = False
    seed: int = 1  # -1 = random
    huggingface_token: str = ""
    flux_num_inference_steps: int = 8
    flux_residual_diff_threshold: float = 0.12
    osb_font_name: str = ""  # Empty = use main font
    osb_max_font_size: int = 64
    osb_min_font_size: int = 12
    osb_use_ligatures: bool = False
    osb_outline_width: float = 3.0
    osb_line_spacing: float = 1.0
    osb_use_subpixel_rendering: bool = True
    osb_font_hinting: str = "none"
    easyocr_min_size: int = 200


@dataclass
class UIGeneralSettings:
    """UI state for general application settings."""

    verbose: bool = False
    cleaning_only: bool = False
    test_mode: bool = False
    enable_thinking: bool = True
    enable_grounding: bool = False
    reasoning_effort: str = "medium"


@dataclass
class UIConfigState:
    """Represents the complete configuration state managed by the UI."""

    detection: UIDetectionSettings = field(default_factory=UIDetectionSettings)
    cleaning: UICleaningSettings = field(default_factory=UICleaningSettings)
    provider_settings: UITranslationProviderSettings = field(
        default_factory=UITranslationProviderSettings
    )
    llm_settings: UITranslationLLMSettings = field(
        default_factory=UITranslationLLMSettings
    )
    rendering: UIRenderingSettings = field(default_factory=UIRenderingSettings)
    output: UIOutputSettings = field(default_factory=UIOutputSettings)
    outside_text: UIOutsideTextSettings = field(default_factory=UIOutsideTextSettings)
    general: UIGeneralSettings = field(default_factory=UIGeneralSettings)

    # Specific UI elements state (saved in config.json)
    input_language: str = "Japanese"
    output_language: str = "English"
    font_pack: Optional[str] = None
    batch_input_language: str = "Japanese"
    batch_output_language: str = "English"
    batch_font_pack: Optional[str] = None
    batch_special_instructions: Optional[str] = None

    def to_save_dict(self) -> Dict[str, Any]:
        """Converts the UI state into a dictionary suitable for saving to config.json."""
        data = {
            "confidence": self.detection.confidence,
            "use_sam2": self.detection.use_sam2,
            "reading_direction": self.llm_settings.reading_direction,
            "thresholding_value": self.cleaning.thresholding_value,
            "use_otsu_threshold": self.cleaning.use_otsu_threshold,
            "roi_shrink_px": self.cleaning.roi_shrink_px,
            "provider": self.provider_settings.provider,
            "google_api_key": self.provider_settings.google_api_key,
            "openai_api_key": self.provider_settings.openai_api_key,
            "anthropic_api_key": self.provider_settings.anthropic_api_key,
            "xai_api_key": self.provider_settings.xai_api_key,
            "openrouter_api_key": self.provider_settings.openrouter_api_key,
            "openai_compatible_url": self.provider_settings.openai_compatible_url,
            "openai_compatible_api_key": self.provider_settings.openai_compatible_api_key,
            "model_name": self.llm_settings.model_name,
            "temperature": self.llm_settings.temperature,
            "top_p": self.llm_settings.top_p,
            "top_k": self.llm_settings.top_k,
            "max_tokens": self.llm_settings.max_tokens,
            "translation_mode": self.llm_settings.translation_mode,
            "send_full_page_context": self.llm_settings.send_full_page_context,
            "upscale_method": self.llm_settings.upscale_method,
            "bubble_min_side_pixels": self.llm_settings.bubble_min_side_pixels,
            "context_image_max_side_pixels": self.llm_settings.context_image_max_side_pixels,
            "osb_min_side_pixels": self.llm_settings.osb_min_side_pixels,
            "special_instructions": self.llm_settings.special_instructions or "",
            "font_pack": self.font_pack,
            "max_font_size": self.rendering.max_font_size,
            "min_font_size": self.rendering.min_font_size,
            "line_spacing": self.rendering.line_spacing,
            "use_subpixel_rendering": self.rendering.use_subpixel_rendering,
            "font_hinting": self.rendering.font_hinting,
            "use_ligatures": self.rendering.use_ligatures,
            "hyphenate_before_scaling": self.rendering.hyphenate_before_scaling,
            "hyphen_penalty": self.rendering.hyphen_penalty,
            "hyphenation_min_word_length": self.rendering.hyphenation_min_word_length,
            "badness_exponent": self.rendering.badness_exponent,
            "padding_pixels": self.rendering.padding_pixels,
            "outside_text_enabled": self.outside_text.enabled,
            "outside_text_seed": self.outside_text.seed,
            "outside_text_huggingface_token": self.outside_text.huggingface_token,
            "outside_text_flux_num_inference_steps": self.outside_text.flux_num_inference_steps,
            "outside_text_flux_residual_diff_threshold": self.outside_text.flux_residual_diff_threshold,
            "outside_text_osb_font_pack": self.outside_text.osb_font_name,
            "outside_text_osb_max_font_size": self.outside_text.osb_max_font_size,
            "outside_text_osb_min_font_size": self.outside_text.osb_min_font_size,
            "outside_text_osb_use_ligatures": self.outside_text.osb_use_ligatures,
            "outside_text_osb_outline_width": self.outside_text.osb_outline_width,
            "outside_text_osb_line_spacing": self.outside_text.osb_line_spacing,
            "outside_text_osb_use_subpixel_rendering": self.outside_text.osb_use_subpixel_rendering,
            "outside_text_osb_font_hinting": self.outside_text.osb_font_hinting,
            "outside_text_easyocr_min_size": self.outside_text.easyocr_min_size,
            "output_format": self.output.output_format,
            "jpeg_quality": self.output.jpeg_quality,
            "png_compression": self.output.png_compression,
            "image_upscale_mode": self.output.image_upscale_mode,
            "image_upscale_factor": self.output.image_upscale_factor,
            "verbose": self.general.verbose,
            "cleaning_only": self.general.cleaning_only,
            "test_mode": self.general.test_mode,
            "enable_thinking": self.general.enable_thinking,
            "enable_grounding": self.general.enable_grounding,
            "reasoning_effort": self.general.reasoning_effort,
            "input_language": self.input_language,
            "output_language": self.output_language,
            "batch_input_language": self.batch_input_language,
            "batch_output_language": self.batch_output_language,
            "batch_font_pack": self.batch_font_pack,
            "batch_special_instructions": self.batch_special_instructions or "",
        }
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UIConfigState":
        """Creates a UIConfigState instance from a dictionary (e.g., loaded from config.json)."""

        from . import \
            settings_manager  # Local import to avoid circular dependency issues

        defaults = settings_manager.DEFAULT_SETTINGS.copy()
        defaults.update(settings_manager.DEFAULT_BATCH_SETTINGS)

        return UIConfigState(
            detection=UIDetectionSettings(
                confidence=data.get("confidence", defaults["confidence"]),
                use_sam2=data.get("use_sam2", defaults.get("use_sam2", True)),
            ),
            cleaning=UICleaningSettings(
                thresholding_value=data.get(
                    "thresholding_value", defaults["thresholding_value"]
                ),
                use_otsu_threshold=data.get(
                    "use_otsu_threshold", defaults["use_otsu_threshold"]
                ),
                roi_shrink_px=data.get(
                    "roi_shrink_px", defaults.get("roi_shrink_px", 4)
                ),
            ),
            outside_text=UIOutsideTextSettings(
                enabled=data.get("outside_text_enabled", False),
                seed=data.get("outside_text_seed", 1),
                huggingface_token=data.get("outside_text_huggingface_token", ""),
                flux_num_inference_steps=data.get(
                    "outside_text_flux_num_inference_steps", 8
                ),
                flux_residual_diff_threshold=data.get(
                    "outside_text_flux_residual_diff_threshold", 0.12
                ),
                osb_font_name=data.get(
                    "outside_text_osb_font_pack",
                    defaults.get("outside_text_osb_font_pack", ""),
                ),
                osb_max_font_size=data.get("outside_text_osb_max_font_size", 64),
                osb_min_font_size=data.get("outside_text_osb_min_font_size", 12),
                osb_use_ligatures=data.get("outside_text_osb_use_ligatures", False),
                osb_outline_width=data.get("outside_text_osb_outline_width", 3.0),
                osb_line_spacing=data.get("outside_text_osb_line_spacing", 1.0),
                osb_use_subpixel_rendering=data.get(
                    "outside_text_osb_use_subpixel_rendering", True
                ),
                osb_font_hinting=data.get("outside_text_osb_font_hinting", "none"),
                easyocr_min_size=data.get("outside_text_easyocr_min_size", 200),
            ),
            provider_settings=UITranslationProviderSettings(
                provider=data.get("provider", defaults["provider"]),
                google_api_key=data.get(
                    "google_api_key", defaults.get("google_api_key", "")
                ),
                openai_api_key=data.get("openai_api_key", defaults["openai_api_key"]),
                anthropic_api_key=data.get(
                    "anthropic_api_key", defaults["anthropic_api_key"]
                ),
                xai_api_key=data.get("xai_api_key", defaults["xai_api_key"]),
                openrouter_api_key=data.get(
                    "openrouter_api_key", defaults["openrouter_api_key"]
                ),
                openai_compatible_url=data.get(
                    "openai_compatible_url", defaults["openai_compatible_url"]
                ),
                openai_compatible_api_key=data.get(
                    "openai_compatible_api_key", defaults["openai_compatible_api_key"]
                ),
            ),
            llm_settings=UITranslationLLMSettings(
                model_name=data.get("model_name"),
                temperature=data.get("temperature", defaults["temperature"]),
                top_p=data.get("top_p", defaults["top_p"]),
                top_k=data.get("top_k", defaults["top_k"]),
                max_tokens=data.get("max_tokens", defaults.get("max_tokens")),
                translation_mode=data.get(
                    "translation_mode", defaults["translation_mode"]
                ),
                reading_direction=data.get(
                    "reading_direction", defaults["reading_direction"]
                ),
                send_full_page_context=data.get("send_full_page_context", True),
                upscale_method=data.get(
                    "upscale_method", defaults.get("upscale_method", "model")
                ),
                bubble_min_side_pixels=data.get("bubble_min_side_pixels", 128),
                context_image_max_side_pixels=data.get(
                    "context_image_max_side_pixels", 1536
                ),
                osb_min_side_pixels=data.get("osb_min_side_pixels", 128),
                special_instructions=data.get("special_instructions") or None,
            ),
            rendering=UIRenderingSettings(
                max_font_size=data.get("max_font_size", defaults["max_font_size"]),
                min_font_size=data.get("min_font_size", defaults["min_font_size"]),
                line_spacing=data.get("line_spacing", defaults["line_spacing"]),
                use_subpixel_rendering=data.get(
                    "use_subpixel_rendering", defaults["use_subpixel_rendering"]
                ),
                font_hinting=data.get("font_hinting", defaults["font_hinting"]),
                use_ligatures=data.get("use_ligatures", defaults["use_ligatures"]),
                hyphenate_before_scaling=data.get(
                    "hyphenate_before_scaling",
                    defaults.get("hyphenate_before_scaling", True),
                ),
                hyphen_penalty=data.get(
                    "hyphen_penalty", defaults.get("hyphen_penalty", 1000.0)
                ),
                hyphenation_min_word_length=data.get(
                    "hyphenation_min_word_length",
                    defaults.get("hyphenation_min_word_length", 8),
                ),
                badness_exponent=data.get(
                    "badness_exponent", defaults.get("badness_exponent", 3.0)
                ),
                padding_pixels=data.get(
                    "padding_pixels", defaults.get("padding_pixels", 5.0)
                ),
            ),
            output=UIOutputSettings(
                output_format=data.get("output_format", defaults["output_format"]),
                jpeg_quality=data.get("jpeg_quality", defaults["jpeg_quality"]),
                png_compression=data.get(
                    "png_compression", defaults["png_compression"]
                ),
                image_upscale_mode=data.get(
                    "image_upscale_mode", defaults.get("image_upscale_mode", "off")
                ),
                image_upscale_factor=data.get(
                    "image_upscale_factor",
                    defaults.get("image_upscale_factor", 2.0),
                ),
            ),
            general=UIGeneralSettings(
                verbose=data.get("verbose", defaults["verbose"]),
                cleaning_only=data.get("cleaning_only", defaults["cleaning_only"]),
                test_mode=data.get("test_mode", defaults.get("test_mode", False)),
                enable_thinking=data.get(
                    "enable_thinking", defaults.get("enable_thinking", True)
                ),
                enable_grounding=data.get(
                    "enable_grounding", defaults.get("enable_grounding", False)
                ),
                reasoning_effort=data.get(
                    "reasoning_effort", defaults.get("reasoning_effort", "medium")
                ),
            ),
            input_language=data.get("input_language", defaults["input_language"]),
            output_language=data.get("output_language", defaults["output_language"]),
            font_pack=data.get("font_pack"),
            batch_input_language=data.get(
                "batch_input_language", defaults["batch_input_language"]
            ),
            batch_output_language=data.get(
                "batch_output_language", defaults["batch_output_language"]
            ),
            batch_font_pack=data.get("batch_font_pack"),
            batch_special_instructions=data.get("batch_special_instructions") or None,
        )


def map_ui_to_backend_config(
    ui_state: UIConfigState,
    fonts_base_dir: Path,
    target_device: Optional[torch.device],
    is_batch: bool = False,
) -> MangaTranslatorConfig:
    """Maps the UIConfigState to the backend MangaTranslatorConfig."""

    yolo_path = ""
    font_pack_name = ui_state.batch_font_pack if is_batch else ui_state.font_pack
    font_dir_path = fonts_base_dir / font_pack_name if font_pack_name else ""
    input_lang = ui_state.batch_input_language if is_batch else ui_state.input_language
    output_lang = (
        ui_state.batch_output_language if is_batch else ui_state.output_language
    )

    detection_cfg = DetectionConfig(confidence=ui_state.detection.confidence)
    detection_cfg.use_sam2 = ui_state.detection.use_sam2

    cleaning_cfg = CleaningConfig(
        thresholding_value=ui_state.cleaning.thresholding_value,
        use_otsu_threshold=ui_state.cleaning.use_otsu_threshold,
        roi_shrink_px=ui_state.cleaning.roi_shrink_px,
    )

    translation_cfg = TranslationConfig(
        provider=ui_state.provider_settings.provider,
        google_api_key=ui_state.provider_settings.google_api_key or "",
        openai_api_key=ui_state.provider_settings.openai_api_key or "",
        anthropic_api_key=ui_state.provider_settings.anthropic_api_key or "",
        xai_api_key=ui_state.provider_settings.xai_api_key or "",
        openrouter_api_key=ui_state.provider_settings.openrouter_api_key or "",
        openai_compatible_url=ui_state.provider_settings.openai_compatible_url,
        openai_compatible_api_key=ui_state.provider_settings.openai_compatible_api_key,
        model_name=ui_state.llm_settings.model_name or "",
        temperature=ui_state.llm_settings.temperature,
        top_p=ui_state.llm_settings.top_p,
        top_k=ui_state.llm_settings.top_k,
        max_tokens=ui_state.llm_settings.max_tokens,
        input_language=input_lang,
        output_language=output_lang,
        reading_direction=ui_state.llm_settings.reading_direction,
        translation_mode=ui_state.llm_settings.translation_mode,
        enable_thinking=ui_state.general.enable_thinking,
        send_full_page_context=ui_state.llm_settings.send_full_page_context,
        upscale_method=ui_state.llm_settings.upscale_method,
        bubble_min_side_pixels=ui_state.llm_settings.bubble_min_side_pixels,
        context_image_max_side_pixels=ui_state.llm_settings.context_image_max_side_pixels,
        osb_min_side_pixels=ui_state.llm_settings.osb_min_side_pixels,
        special_instructions=ui_state.llm_settings.special_instructions,
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
        hyphen_penalty=ui_state.rendering.hyphen_penalty,
        hyphenation_min_word_length=(ui_state.rendering.hyphenation_min_word_length),
        badness_exponent=ui_state.rendering.badness_exponent,
        padding_pixels=ui_state.rendering.padding_pixels,
    )

    upscale_mode = ui_state.output.image_upscale_mode
    upscale_factor = ui_state.output.image_upscale_factor

    output_cfg = OutputConfig(
        output_format=ui_state.output.output_format,
        jpeg_quality=ui_state.output.jpeg_quality,
        png_compression=ui_state.output.png_compression,
        upscale_final_image=upscale_mode == "final",
        image_upscale_factor=upscale_factor,
    )

    # Determine OSB font (use main font if not specified)
    osb_font = (
        ui_state.outside_text.osb_font_name
        if ui_state.outside_text.osb_font_name
        else ui_state.font_pack
    )
    osb_font_path = fonts_base_dir / osb_font if osb_font else None

    outside_text_cfg = OutsideTextConfig(
        enabled=ui_state.outside_text.enabled,
        seed=ui_state.outside_text.seed,
        huggingface_token=ui_state.outside_text.huggingface_token,
        flux_num_inference_steps=ui_state.outside_text.flux_num_inference_steps,
        flux_residual_diff_threshold=ui_state.outside_text.flux_residual_diff_threshold,
        osb_font_name=str(osb_font_path) if osb_font_path else None,
        osb_max_font_size=ui_state.outside_text.osb_max_font_size,
        osb_min_font_size=ui_state.outside_text.osb_min_font_size,
        osb_use_ligatures=ui_state.outside_text.osb_use_ligatures,
        osb_outline_width=ui_state.outside_text.osb_outline_width,
        osb_line_spacing=ui_state.outside_text.osb_line_spacing,
        osb_use_subpixel_rendering=ui_state.outside_text.osb_use_subpixel_rendering,
        osb_font_hinting=ui_state.outside_text.osb_font_hinting,
        easyocr_min_size=ui_state.outside_text.easyocr_min_size,
    )

    preprocessing_cfg = PreprocessingConfig(
        enabled=upscale_mode == "initial",
        factor=upscale_factor,
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
        outside_text=outside_text_cfg,
        preprocessing=preprocessing_cfg,
        cleaning_only=ui_state.general.cleaning_only,
        test_mode=ui_state.general.test_mode,
    )

    return backend_config
