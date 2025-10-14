from pathlib import Path
from typing import Tuple

from core.config import (MangaTranslatorConfig, RenderingConfig,
                         TranslationConfig)
from utils.exceptions import ValidationError


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
        isinstance(rendering_cfg.line_spacing, (int, float))
        and float(rendering_cfg.line_spacing) > 0
    ):
        raise ValidationError("Line Spacing must be a positive number.")
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


def validate_mutually_exclusive_modes(cleaning_only: bool, test_mode: bool) -> None:
    """
    Validates that cleaning_only and test_mode are not both enabled.

    Args:
        cleaning_only (bool): Whether cleaning-only mode is enabled.
        test_mode (bool): Whether test mode is enabled.

    Raises:
        ValidationError: If both modes are enabled simultaneously.
    """
    if cleaning_only and test_mode:
        raise ValidationError(
            "Cleaning-only mode and Test mode cannot be enabled together. "
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
    validate_mutually_exclusive_modes(config.cleaning_only, config.test_mode)
