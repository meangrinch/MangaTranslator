from pathlib import Path
from typing import Any, Dict, Tuple, Union

from core.config import MangaTranslatorConfig, RenderingConfig, TranslationConfig
from utils.exceptions import ValidationError

SETTING_CONSTRAINTS: Dict[str, Tuple[float, float]] = {
    "confidence": (0.1, 1.0),
    "conjoined_confidence": (0.1, 1.0),
    "panel_confidence": (0.05, 1.0),
    "thresholding_value": (0, 255),
    "roi_shrink_px": (0, 8),
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0, 64),
    "max_tokens": (2048, 63488),
    "bubble_min_side_pixels": (64, 512),
    "context_image_max_side_pixels": (512, 2560),
    "osb_min_side_pixels": (64, 512),
    "max_font_size": (5, 50),
    "min_font_size": (5, 50),
    "line_spacing_mult": (0.5, 2.0),
    "hyphen_penalty": (100, 2000),
    "hyphenation_min_word_length": (6, 10),
    "badness_exponent": (2.0, 4.0),
    "padding_pixels": (2, 12),
    "supersampling_factor": (1, 16),
    "outside_text_osb_confidence": (0.0, 1.0),
    "outside_text_bbox_expansion_percent": (0.0, 1.0),
    "outside_text_text_box_proximity_ratio": (0.01, 0.1),
    "outside_text_page_filter_margin_threshold": (0.0, 0.3),
    "outside_text_page_filter_min_area_ratio": (0.0, 0.2),
    "outside_text_flux_num_inference_steps": (1, 30),
    "outside_text_flux_residual_diff_threshold": (0.0, 1.0),
    "outside_text_osb_max_font_size": (5, 96),
    "outside_text_osb_min_font_size": (5, 96),
    "outside_text_osb_line_spacing": (0.5, 2.0),
    "outside_text_osb_outline_width": (0.0, 10.0),
    "jpeg_quality": (1, 100),
    "png_compression": (0, 6),
    "image_upscale_factor": (1.0, 8.0),
}

# Attribute paths for clamping config objects (MangaTranslatorConfig and children)
_CONFIG_ATTR_PATHS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "confidence": (("detection", "confidence"),),
    "conjoined_confidence": (("detection", "conjoined_confidence"),),
    "panel_confidence": (("detection", "panel_confidence"),),
    "thresholding_value": (("cleaning", "thresholding_value"),),
    "roi_shrink_px": (("cleaning", "roi_shrink_px"),),
    "temperature": (("translation", "temperature"),),
    "top_p": (("translation", "top_p"),),
    "top_k": (("translation", "top_k"),),
    "max_tokens": (("translation", "max_tokens"),),
    "bubble_min_side_pixels": (("translation", "bubble_min_side_pixels"),),
    "context_image_max_side_pixels": (
        ("translation", "context_image_max_side_pixels"),
    ),
    "osb_min_side_pixels": (("translation", "osb_min_side_pixels"),),
    "max_font_size": (("rendering", "max_font_size"),),
    "min_font_size": (("rendering", "min_font_size"),),
    "line_spacing_mult": (("rendering", "line_spacing_mult"),),
    "hyphen_penalty": (("rendering", "hyphen_penalty"),),
    "hyphenation_min_word_length": (("rendering", "hyphenation_min_word_length"),),
    "badness_exponent": (("rendering", "badness_exponent"),),
    "padding_pixels": (("rendering", "padding_pixels"),),
    "supersampling_factor": (("rendering", "supersampling_factor"),),
    "outside_text_osb_confidence": (("outside_text", "osb_confidence"),),
    "outside_text_bbox_expansion_percent": (
        ("outside_text", "bbox_expansion_percent"),
    ),
    "outside_text_text_box_proximity_ratio": (
        ("outside_text", "text_box_proximity_ratio"),
    ),
    "outside_text_page_filter_margin_threshold": (
        ("outside_text", "page_filter_margin_threshold"),
    ),
    "outside_text_page_filter_min_area_ratio": (
        ("outside_text", "page_filter_min_area_ratio"),
    ),
    "outside_text_flux_num_inference_steps": (
        ("outside_text", "flux_num_inference_steps"),
    ),
    "outside_text_flux_residual_diff_threshold": (
        ("outside_text", "flux_residual_diff_threshold"),
    ),
    "outside_text_osb_max_font_size": (("outside_text", "osb_max_font_size"),),
    "outside_text_osb_min_font_size": (("outside_text", "osb_min_font_size"),),
    "outside_text_osb_line_spacing": (("outside_text", "osb_line_spacing"),),
    "outside_text_osb_outline_width": (("outside_text", "osb_outline_width"),),
    "jpeg_quality": (("output", "jpeg_quality"),),
    "png_compression": (("output", "png_compression"),),
    "image_upscale_factor": (
        ("output", "image_upscale_factor"),
        ("preprocessing", "factor"),
    ),
}


def _clamp_numeric(value: Any, min_value: float, max_value: float) -> Any:
    """Clamp numeric values to a min/max range, preserving int type where reasonable."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        clamped = max(min_value, min(value, max_value))
        if isinstance(value, int):
            return int(clamped)
        return clamped
    return value


def clamp_settings(settings: Any) -> Any:
    """
    Clamp numeric settings to UI-defined min/max constraints.

    Supports both flat dictionaries (UI settings/config.json) and MangaTranslatorConfig
    objects (including nested dataclasses). Returns the same object/dict for chaining.
    """

    # Dictionary path: flat keys matching SETTING_CONSTRAINTS
    if isinstance(settings, dict):
        clamped = settings.copy()
        for key, (min_val, max_val) in SETTING_CONSTRAINTS.items():
            if key in clamped:
                clamped[key] = _clamp_numeric(clamped[key], min_val, max_val)
        return clamped

    # Object path: traverse attribute paths if present
    for key, paths in _CONFIG_ATTR_PATHS.items():
        if not hasattr(settings, "__dict__"):
            break
        if key not in SETTING_CONSTRAINTS:
            continue
        min_val, max_val = SETTING_CONSTRAINTS[key]
        for path in paths:
            target = settings
            for attr in path[:-1]:
                target = getattr(target, attr, None)
                if target is None:
                    break
            else:
                leaf = path[-1]
                if hasattr(target, leaf):
                    current = getattr(target, leaf)
                    clamped_val = _clamp_numeric(current, min_val, max_val)
                    setattr(target, leaf, clamped_val)
    return settings


def autodetect_yolo_model_path(models_dir: Path) -> Path:
    """Returns the path for the primary YOLO speech bubble model.

    This function provides a consistent path for the model, which will be
    auto-downloaded by the ModelManager if it doesn't exist. It does not
    validate file existence here. This function previously scanned for any .pt
    file, but now returns a deterministic path to align with auto-downloading.
    """
    yolo_dir = models_dir / "yolo"
    return yolo_dir / "yolov8m_seg-speech-bubble.pt"


def validate_core_inputs(
    translation_cfg: TranslationConfig,
    rendering_cfg: RenderingConfig,
    models_dir: Path,
    fonts_base_dir: Path,
) -> Tuple[Path, Path]:
    """
    Validates core inputs required for translation, raising standard exceptions.

    Args:
        translation_cfg (TranslationConfig): Translation configuration.
        rendering_cfg (RenderingConfig): Rendering configuration.
        models_dir (Path): Absolute path to the directory containing YOLO models.
        fonts_base_dir (Path): Absolute path to the base directory containing font packs.

    Returns:
        tuple[Path, Path]: Validated absolute path to the YOLO model and font directory.

    Raises:
        FileNotFoundError: If required directories or files (model, font) are not found.
        ValidationError: If configuration values are invalid (e.g., model not selected,
                         font pack empty, invalid numeric values).
        ValueError: For general invalid parameter values.
    """
    # --- YOLO Model Auto-detection ---
    if not models_dir.is_dir():
        raise FileNotFoundError(f"YOLO models directory not found: {models_dir}")

    yolo_model_path = autodetect_yolo_model_path(models_dir)

    # --- Font Validation ---
    if not fonts_base_dir.is_dir():
        raise FileNotFoundError(f"Fonts base directory not found: {fonts_base_dir}")

    if not rendering_cfg.font_dir:
        raise ValidationError("Font pack (font_dir in rendering config) not specified.")

    font_dir_path = fonts_base_dir / rendering_cfg.font_dir
    if not font_dir_path.is_dir():
        raise FileNotFoundError(
            f"Specified font pack directory '{rendering_cfg.font_dir}' not found within {fonts_base_dir}"
        )

    font_files = list(font_dir_path.glob("*.ttf")) + list(font_dir_path.glob("*.otf"))
    if not font_files:
        raise ValidationError(
            f"No font files (.ttf or .otf) found in the font pack directory: '{font_dir_path}'"
        )

    # --- Rendering Config Validation ---
    if not (
        isinstance(rendering_cfg.max_font_size, int) and rendering_cfg.max_font_size > 0
    ):
        raise ValidationError("Max Font Size must be a positive integer.")
    if not (
        isinstance(rendering_cfg.min_font_size, int) and rendering_cfg.min_font_size > 0
    ):
        raise ValidationError("Min Font Size must be a positive integer.")
    if not (
        isinstance(rendering_cfg.line_spacing_mult, (int, float))
        and float(rendering_cfg.line_spacing_mult) > 0
    ):
        raise ValidationError("Line Spacing Multiplier must be a positive number.")
    if rendering_cfg.min_font_size > rendering_cfg.max_font_size:
        raise ValidationError("Min Font Size cannot be larger than Max Font Size.")
    if rendering_cfg.font_hinting not in ["none", "slight", "normal", "full"]:
        raise ValidationError(
            "Invalid Font Hinting value. Must be one of: none, slight, normal, full."
        )

    # --- Translation Config Validation (Basic) ---
    if not translation_cfg.provider:
        raise ValidationError("Translation provider cannot be empty.")
    if not translation_cfg.model_name:
        raise ValidationError("Translation model name cannot be empty.")
    if not translation_cfg.input_language:
        raise ValidationError("Input language cannot be empty.")
    if not translation_cfg.output_language:
        raise ValidationError("Output language cannot be empty.")
    if translation_cfg.reading_direction not in ["rtl", "ltr"]:
        raise ValidationError("Reading direction must be 'rtl' or 'ltr'.")

    return yolo_model_path.resolve(), font_dir_path.resolve()


def validate_mutually_exclusive_modes(
    cleaning_only: bool, upscaling_only: bool, test_mode: bool
) -> None:
    """
    Validates that cleaning_only, upscaling_only, and test_mode are mutually exclusive.

    Args:
        cleaning_only (bool): Whether cleaning-only mode is enabled.
        upscaling_only (bool): Whether upscaling-only mode is enabled.
        test_mode (bool): Whether test mode is enabled.

    Raises:
        ValidationError: If more than one mode is enabled simultaneously.
    """
    enabled_modes = sum([cleaning_only, upscaling_only, test_mode])
    if enabled_modes > 1:
        raise ValidationError(
            "Cleaning-only mode, Upscaling-only mode, and Test mode are mutually exclusive. "
            "Only one mode can be active at a time."
        )


def validate_config(config: MangaTranslatorConfig) -> None:
    """
    Validates the MangaTranslatorConfig object for mutually exclusive settings.

    Args:
        config (MangaTranslatorConfig): The configuration to validate.

    Raises:
        ValidationError: If invalid configuration is detected.
    """
    validate_mutually_exclusive_modes(
        config.cleaning_only, config.upscaling_only, config.test_mode
    )


def normalize_zip_file_input(zip_input: Any) -> str:
    """
    Normalizes ZIP file input from Gradio (handles both string and file-like objects).

    Args:
        zip_input: ZIP file input from Gradio (can be str or file-like object).

    Returns:
        str: Normalized file path as string.

    Raises:
        ValidationError: If the input format is invalid.
    """
    if isinstance(zip_input, str):
        return zip_input
    elif hasattr(zip_input, "name"):
        zip_path = zip_input.name
        return zip_path if isinstance(zip_path, str) else str(zip_path)
    else:
        raise ValidationError("Invalid ZIP file format.")


def validate_zip_file(zip_path: Union[str, Path]) -> Path:
    """
    Validates that a ZIP file exists and has the correct extension.

    Args:
        zip_path: Path to the ZIP file to validate.

    Returns:
        Path: Validated Path object to the ZIP file.

    Raises:
        FileNotFoundError: If the ZIP file does not exist.
        ValidationError: If the file is not a ZIP archive.
    """
    zip_file_path = Path(zip_path)
    if not zip_file_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    if not zip_file_path.suffix.lower() == ".zip":
        raise ValidationError(f"File is not a ZIP archive: {zip_path}")

    return zip_file_path


def validate_batch_input_path(input_path: Union[str, Path]) -> Path:
    """
    Validates that a batch input path exists and is either a directory or ZIP file.

    Args:
        input_path: Path to validate (directory or ZIP file).

    Returns:
        Path: Validated Path object.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValidationError: If the path is neither a directory nor a ZIP file.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")

    if not (path.is_dir() or (path.is_file() and path.suffix.lower() == ".zip")):
        raise ValidationError(
            f"Input path '{input_path}' is neither a directory nor a ZIP file."
        )

    return path
