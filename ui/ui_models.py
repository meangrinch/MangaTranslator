from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

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


@dataclass
class UIDetectionSettings:
    """UI state for detection settings."""

    confidence: float = 0.6
    conjoined_confidence: float = 0.35
    panel_confidence: float = 0.25
    use_sam2: bool = True
    conjoined_detection: bool = True
    use_panel_sorting: bool = True
    use_osb_text_verification: bool = True


@dataclass
class UICleaningSettings:
    """UI state for cleaning settings."""

    thresholding_value: int = 190
    use_otsu_threshold: bool = False
    roi_shrink_px: int = 4
    inpaint_colored_bubbles: bool = True


@dataclass
class UITranslationProviderSettings:
    """UI state for translation provider settings."""

    provider: str = "Google"
    google_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""
    anthropic_api_key: Optional[str] = ""
    xai_api_key: Optional[str] = ""
    deepseek_api_key: Optional[str] = ""
    zai_api_key: Optional[str] = ""
    moonshot_api_key: Optional[str] = ""
    openrouter_api_key: Optional[str] = ""
    openai_compatible_url: str = "http://localhost:1234/v1"
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
    upscale_method: str = "model_lite"
    bubble_min_side_pixels: int = 128
    context_image_max_side_pixels: int = 1024
    osb_min_side_pixels: int = 128
    special_instructions: Optional[str] = None
    ocr_method: str = "LLM"  # "LLM" or "manga-ocr"


@dataclass
class UIRenderingSettings:
    """UI state for rendering settings."""

    max_font_size: int = 16
    min_font_size: int = 8
    line_spacing_mult: float = 1.0
    use_subpixel_rendering: bool = True
    font_hinting: str = "none"
    use_ligatures: bool = False
    hyphenate_before_scaling: bool = True
    hyphen_penalty: float = 1000.0
    hyphenation_min_word_length: int = 8
    badness_exponent: float = 3.0
    padding_pixels: float = 5.0
    supersampling_factor: int = 4


@dataclass
class UIOutputSettings:
    """UI state for output settings."""

    output_format: str = "auto"
    jpeg_quality: int = 95
    png_compression: int = 2
    image_upscale_mode: str = "off"  # "off", "initial", "final"
    image_upscale_factor: float = 2.0
    image_upscale_model: str = "model_lite"  # "model" or "model_lite"


@dataclass
class UIOutsideTextSettings:
    """UI state for outside speech bubble text removal settings."""

    enabled: bool = False
    enable_page_number_filtering: bool = False
    page_filter_margin_threshold: float = 0.1
    page_filter_min_area_ratio: float = 0.05
    seed: int = 1  # -1 = random
    huggingface_token: str = ""
    force_cv2_inpainting: bool = False
    flux_num_inference_steps: int = 10
    flux_residual_diff_threshold: float = 0.15
    osb_confidence: float = 0.6
    osb_font_name: str = ""  # Empty = use main font
    osb_max_font_size: int = 64
    osb_min_font_size: int = 12
    osb_use_ligatures: bool = False
    osb_outline_width: float = 3.0
    osb_line_spacing: float = 1.0
    osb_use_subpixel_rendering: bool = True
    osb_font_hinting: str = "none"
    bbox_expansion_percent: float = 0.1
    text_box_proximity_ratio: float = 0.02


@dataclass
class UIGeneralSettings:
    """UI state for general application settings."""

    verbose: bool = False
    cleaning_only: bool = False
    upscaling_only: bool = False
    test_mode: bool = False
    enable_web_search: bool = (
        False  # Enable model's built-in web search for up-to-date information.
    )
    media_resolution: str = (
        "auto"  # Only available via Google provider (auto/high/medium/low)
    )
    media_resolution_bubbles: str = "auto"  # Gemini 3 models
    media_resolution_context: str = "auto"  # Gemini 3 models
    reasoning_effort: Optional[str] = None
    effort: Optional[str] = None  # Claude Opus 4.5 only: token spending eagerness
    auto_scale: bool = True


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
            "conjoined_confidence": self.detection.conjoined_confidence,
            "panel_confidence": self.detection.panel_confidence,
            "use_sam2": self.detection.use_sam2,
            "conjoined_detection": self.detection.conjoined_detection,
            "use_panel_sorting": self.detection.use_panel_sorting,
            "use_osb_text_verification": self.detection.use_osb_text_verification,
            "reading_direction": self.llm_settings.reading_direction,
            "thresholding_value": self.cleaning.thresholding_value,
            "use_otsu_threshold": self.cleaning.use_otsu_threshold,
            "roi_shrink_px": self.cleaning.roi_shrink_px,
            "inpaint_colored_bubbles": self.cleaning.inpaint_colored_bubbles,
            "provider": self.provider_settings.provider,
            "google_api_key": self.provider_settings.google_api_key,
            "openai_api_key": self.provider_settings.openai_api_key,
            "anthropic_api_key": self.provider_settings.anthropic_api_key,
            "xai_api_key": self.provider_settings.xai_api_key,
            "deepseek_api_key": self.provider_settings.deepseek_api_key,
            "zai_api_key": self.provider_settings.zai_api_key,
            "moonshot_api_key": self.provider_settings.moonshot_api_key,
            "openrouter_api_key": self.provider_settings.openrouter_api_key,
            "openai_compatible_url": self.provider_settings.openai_compatible_url,
            "openai_compatible_api_key": self.provider_settings.openai_compatible_api_key,
            "model_name": self.llm_settings.model_name,
            "temperature": self.llm_settings.temperature,
            "top_p": self.llm_settings.top_p,
            "top_k": self.llm_settings.top_k,
            "max_tokens": self.llm_settings.max_tokens,
            "translation_mode": self.llm_settings.translation_mode,
            "ocr_method": self.llm_settings.ocr_method,
            "send_full_page_context": self.llm_settings.send_full_page_context,
            "upscale_method": self.llm_settings.upscale_method,
            "bubble_min_side_pixels": self.llm_settings.bubble_min_side_pixels,
            "context_image_max_side_pixels": self.llm_settings.context_image_max_side_pixels,
            "osb_min_side_pixels": self.llm_settings.osb_min_side_pixels,
            "special_instructions": self.llm_settings.special_instructions or "",
            "font_pack": self.font_pack,
            "max_font_size": self.rendering.max_font_size,
            "min_font_size": self.rendering.min_font_size,
            "line_spacing_mult": self.rendering.line_spacing_mult,
            "use_subpixel_rendering": self.rendering.use_subpixel_rendering,
            "font_hinting": self.rendering.font_hinting,
            "use_ligatures": self.rendering.use_ligatures,
            "hyphenate_before_scaling": self.rendering.hyphenate_before_scaling,
            "hyphen_penalty": self.rendering.hyphen_penalty,
            "hyphenation_min_word_length": self.rendering.hyphenation_min_word_length,
            "badness_exponent": self.rendering.badness_exponent,
            "padding_pixels": self.rendering.padding_pixels,
            "supersampling_factor": self.rendering.supersampling_factor,
            "outside_text_enabled": self.outside_text.enabled,
            "outside_text_seed": self.outside_text.seed,
            "outside_text_huggingface_token": self.outside_text.huggingface_token,
            "outside_text_force_cv2_inpainting": self.outside_text.force_cv2_inpainting,
            "outside_text_flux_num_inference_steps": self.outside_text.flux_num_inference_steps,
            "outside_text_flux_residual_diff_threshold": self.outside_text.flux_residual_diff_threshold,
            "outside_text_osb_confidence": self.outside_text.osb_confidence,
            "outside_text_enable_page_number_filtering": self.outside_text.enable_page_number_filtering,
            "outside_text_page_filter_margin_threshold": self.outside_text.page_filter_margin_threshold,
            "outside_text_page_filter_min_area_ratio": self.outside_text.page_filter_min_area_ratio,
            "outside_text_osb_font_pack": self.outside_text.osb_font_name,
            "outside_text_osb_max_font_size": self.outside_text.osb_max_font_size,
            "outside_text_osb_min_font_size": self.outside_text.osb_min_font_size,
            "outside_text_osb_use_ligatures": self.outside_text.osb_use_ligatures,
            "outside_text_osb_outline_width": self.outside_text.osb_outline_width,
            "outside_text_osb_line_spacing": self.outside_text.osb_line_spacing,
            "outside_text_osb_use_subpixel_rendering": self.outside_text.osb_use_subpixel_rendering,
            "outside_text_osb_font_hinting": self.outside_text.osb_font_hinting,
            "outside_text_bbox_expansion_percent": self.outside_text.bbox_expansion_percent,
            "outside_text_text_box_proximity_ratio": self.outside_text.text_box_proximity_ratio,
            "output_format": self.output.output_format,
            "jpeg_quality": self.output.jpeg_quality,
            "png_compression": self.output.png_compression,
            "image_upscale_mode": self.output.image_upscale_mode,
            "image_upscale_factor": self.output.image_upscale_factor,
            "image_upscale_model": self.output.image_upscale_model,
            "verbose": self.general.verbose,
            "cleaning_only": self.general.cleaning_only,
            "upscaling_only": self.general.upscaling_only,
            "test_mode": self.general.test_mode,
            "enable_web_search": self.general.enable_web_search,
            "media_resolution": self.general.media_resolution,
            "media_resolution_bubbles": self.general.media_resolution_bubbles,
            "media_resolution_context": self.general.media_resolution_context,
            "reasoning_effort": self.general.reasoning_effort,
            "effort": self.general.effort,
            "auto_scale": self.general.auto_scale,
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

        from . import (
            settings_manager,
        )  # Local import to avoid circular dependency issues

        defaults = settings_manager.DEFAULT_SETTINGS.copy()
        defaults.update(settings_manager.DEFAULT_BATCH_SETTINGS)

        return UIConfigState(
            detection=UIDetectionSettings(
                confidence=data.get("confidence", defaults["confidence"]),
                conjoined_confidence=data.get(
                    "conjoined_confidence", defaults.get("conjoined_confidence", 0.35)
                ),
                panel_confidence=data.get(
                    "panel_confidence", defaults.get("panel_confidence", 0.25)
                ),
                use_sam2=data.get("use_sam2", defaults.get("use_sam2", True)),
                conjoined_detection=data.get(
                    "conjoined_detection",
                    defaults.get("conjoined_detection", True),
                ),
                use_osb_text_verification=data.get(
                    "use_osb_text_verification",
                    defaults.get("use_osb_text_verification", True),
                ),
                use_panel_sorting=data.get(
                    "use_panel_sorting",
                    defaults.get("use_panel_sorting", True),
                ),
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
                inpaint_colored_bubbles=data.get(
                    "inpaint_colored_bubbles",
                    defaults.get("inpaint_colored_bubbles", True),
                ),
            ),
            outside_text=UIOutsideTextSettings(
                enabled=data.get("outside_text_enabled", False),
                enable_page_number_filtering=data.get(
                    "outside_text_enable_page_number_filtering",
                    defaults.get("outside_text_enable_page_number_filtering", False),
                ),
                page_filter_margin_threshold=data.get(
                    "outside_text_page_filter_margin_threshold",
                    defaults.get("outside_text_page_filter_margin_threshold", 0.1),
                ),
                page_filter_min_area_ratio=data.get(
                    "outside_text_page_filter_min_area_ratio",
                    defaults.get("outside_text_page_filter_min_area_ratio", 0.05),
                ),
                seed=data.get("outside_text_seed", 1),
                huggingface_token=data.get("outside_text_huggingface_token", ""),
                force_cv2_inpainting=data.get(
                    "outside_text_force_cv2_inpainting",
                    defaults.get("outside_text_force_cv2_inpainting", False),
                ),
                flux_num_inference_steps=data.get(
                    "outside_text_flux_num_inference_steps", 8
                ),
                flux_residual_diff_threshold=data.get(
                    "outside_text_flux_residual_diff_threshold", 0.15
                ),
                osb_confidence=data.get("outside_text_osb_confidence", 0.6),
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
                bbox_expansion_percent=data.get(
                    "outside_text_bbox_expansion_percent", 0.1
                ),
                text_box_proximity_ratio=data.get(
                    "outside_text_text_box_proximity_ratio", 0.02
                ),
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
                deepseek_api_key=data.get(
                    "deepseek_api_key", defaults.get("deepseek_api_key", "")
                ),
                zai_api_key=data.get("zai_api_key", defaults.get("zai_api_key", "")),
                moonshot_api_key=data.get(
                    "moonshot_api_key", defaults.get("moonshot_api_key", "")
                ),
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
                ocr_method=data.get("ocr_method", defaults.get("ocr_method", "LLM")),
                reading_direction=data.get(
                    "reading_direction", defaults["reading_direction"]
                ),
                send_full_page_context=data.get("send_full_page_context", True),
                upscale_method=data.get(
                    "upscale_method", defaults.get("upscale_method", "model_lite")
                ),
                bubble_min_side_pixels=data.get("bubble_min_side_pixels", 128),
                context_image_max_side_pixels=data.get(
                    "context_image_max_side_pixels", 1024
                ),
                osb_min_side_pixels=data.get("osb_min_side_pixels", 128),
                special_instructions=data.get("special_instructions") or None,
            ),
            rendering=UIRenderingSettings(
                max_font_size=data.get("max_font_size", defaults["max_font_size"]),
                min_font_size=data.get("min_font_size", defaults["min_font_size"]),
                line_spacing_mult=data.get(
                    "line_spacing_mult", defaults["line_spacing_mult"]
                ),
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
                supersampling_factor=data.get(
                    "supersampling_factor", defaults.get("supersampling_factor", 4)
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
                image_upscale_model=data.get(
                    "image_upscale_model",
                    defaults.get("image_upscale_model", "model_lite"),
                ),
            ),
            general=UIGeneralSettings(
                verbose=data.get("verbose", defaults["verbose"]),
                cleaning_only=data.get("cleaning_only", defaults["cleaning_only"]),
                upscaling_only=data.get(
                    "upscaling_only", defaults.get("upscaling_only", False)
                ),
                test_mode=data.get("test_mode", defaults.get("test_mode", False)),
                enable_web_search=data.get(
                    "enable_web_search", defaults.get("enable_web_search", False)
                ),
                media_resolution=data.get(
                    "media_resolution",
                    defaults.get("media_resolution", "auto"),
                ),
                media_resolution_bubbles=data.get(
                    "media_resolution_bubbles",
                    defaults.get("media_resolution_bubbles", "auto"),
                ),
                media_resolution_context=data.get(
                    "media_resolution_context",
                    defaults.get("media_resolution_context", "auto"),
                ),
                reasoning_effort=data.get(
                    "reasoning_effort", defaults.get("reasoning_effort")
                ),
                effort=data.get("effort", defaults.get("effort", "medium")),
                auto_scale=data.get("auto_scale", defaults.get("auto_scale", True)),
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

    detection_cfg = DetectionConfig(
        confidence=ui_state.detection.confidence,
        panel_confidence=ui_state.detection.panel_confidence,
    )
    detection_cfg.conjoined_confidence = ui_state.detection.conjoined_confidence
    detection_cfg.use_sam2 = ui_state.detection.use_sam2
    detection_cfg.conjoined_detection = ui_state.detection.conjoined_detection
    detection_cfg.use_panel_sorting = ui_state.detection.use_panel_sorting
    detection_cfg.use_osb_text_verification = (
        ui_state.detection.use_osb_text_verification
    )

    cleaning_cfg = CleaningConfig(
        thresholding_value=ui_state.cleaning.thresholding_value,
        use_otsu_threshold=ui_state.cleaning.use_otsu_threshold,
        roi_shrink_px=ui_state.cleaning.roi_shrink_px,
        inpaint_colored_bubbles=ui_state.cleaning.inpaint_colored_bubbles,
    )

    translation_cfg = TranslationConfig(
        provider=ui_state.provider_settings.provider,
        google_api_key=ui_state.provider_settings.google_api_key or "",
        openai_api_key=ui_state.provider_settings.openai_api_key or "",
        anthropic_api_key=ui_state.provider_settings.anthropic_api_key or "",
        xai_api_key=ui_state.provider_settings.xai_api_key or "",
        deepseek_api_key=ui_state.provider_settings.deepseek_api_key or "",
        zai_api_key=ui_state.provider_settings.zai_api_key or "",
        moonshot_api_key=ui_state.provider_settings.moonshot_api_key or "",
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
        ocr_method=ui_state.llm_settings.ocr_method,
        enable_web_search=ui_state.general.enable_web_search,
        media_resolution=ui_state.general.media_resolution,
        media_resolution_bubbles=ui_state.general.media_resolution_bubbles,
        media_resolution_context=ui_state.general.media_resolution_context,
        send_full_page_context=ui_state.llm_settings.send_full_page_context,
        upscale_method=ui_state.llm_settings.upscale_method,
        bubble_min_side_pixels=ui_state.llm_settings.bubble_min_side_pixels,
        context_image_max_side_pixels=ui_state.llm_settings.context_image_max_side_pixels,
        osb_min_side_pixels=ui_state.llm_settings.osb_min_side_pixels,
        special_instructions=ui_state.llm_settings.special_instructions,
        reasoning_effort=ui_state.general.reasoning_effort,
        effort=ui_state.general.effort,
    )

    rendering_cfg = RenderingConfig(
        font_dir=str(font_dir_path),
        max_font_size=ui_state.rendering.max_font_size,
        min_font_size=ui_state.rendering.min_font_size,
        line_spacing_mult=ui_state.rendering.line_spacing_mult,
        use_subpixel_rendering=ui_state.rendering.use_subpixel_rendering,
        font_hinting=ui_state.rendering.font_hinting,
        use_ligatures=ui_state.rendering.use_ligatures,
        hyphenate_before_scaling=ui_state.rendering.hyphenate_before_scaling,
        hyphen_penalty=ui_state.rendering.hyphen_penalty,
        hyphenation_min_word_length=(ui_state.rendering.hyphenation_min_word_length),
        badness_exponent=ui_state.rendering.badness_exponent,
        padding_pixels=ui_state.rendering.padding_pixels,
        supersampling_factor=ui_state.rendering.supersampling_factor,
    )

    upscale_mode = ui_state.output.image_upscale_mode
    upscale_factor = ui_state.output.image_upscale_factor

    output_cfg = OutputConfig(
        output_format=ui_state.output.output_format,
        jpeg_quality=ui_state.output.jpeg_quality,
        png_compression=ui_state.output.png_compression,
        upscale_final_image=upscale_mode == "final",
        image_upscale_factor=upscale_factor,
        image_upscale_model=ui_state.output.image_upscale_model,
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
        enable_page_number_filtering=ui_state.outside_text.enable_page_number_filtering,
        page_filter_margin_threshold=ui_state.outside_text.page_filter_margin_threshold,
        page_filter_min_area_ratio=ui_state.outside_text.page_filter_min_area_ratio,
        seed=ui_state.outside_text.seed,
        huggingface_token=ui_state.outside_text.huggingface_token,
        force_cv2_inpainting=ui_state.outside_text.force_cv2_inpainting,
        flux_num_inference_steps=ui_state.outside_text.flux_num_inference_steps,
        flux_residual_diff_threshold=ui_state.outside_text.flux_residual_diff_threshold,
        osb_confidence=ui_state.outside_text.osb_confidence,
        osb_font_name=str(osb_font_path) if osb_font_path else None,
        osb_max_font_size=ui_state.outside_text.osb_max_font_size,
        osb_min_font_size=ui_state.outside_text.osb_min_font_size,
        osb_use_ligatures=ui_state.outside_text.osb_use_ligatures,
        osb_outline_width=ui_state.outside_text.osb_outline_width,
        osb_line_spacing=ui_state.outside_text.osb_line_spacing,
        osb_use_subpixel_rendering=ui_state.outside_text.osb_use_subpixel_rendering,
        osb_font_hinting=ui_state.outside_text.osb_font_hinting,
        bbox_expansion_percent=ui_state.outside_text.bbox_expansion_percent,
        text_box_proximity_ratio=ui_state.outside_text.text_box_proximity_ratio,
    )

    preprocessing_cfg = PreprocessingConfig(
        enabled=upscale_mode == "initial",
        factor=upscale_factor,
        auto_scale=ui_state.general.auto_scale,
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
        upscaling_only=ui_state.general.upscaling_only,
        test_mode=ui_state.general.test_mode,
    )

    return backend_config
