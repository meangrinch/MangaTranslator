import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Union

from PIL import Image

from core.models import MangaTranslatorConfig, RenderingConfig
from core.pipeline import batch_translate_images, translate_and_render
from core.validation import validate_core_inputs
from utils.exceptions import ValidationError
from utils.logging import log_message


class LogicError(Exception):
    pass


def translate_manga_logic(
    image: Union[str, Path, Image.Image],
    config: MangaTranslatorConfig,
    selected_font_pack_name: str,
    models_dir: Path,
    fonts_base_dir: Path,
    output_base_dir: Path = Path("./output"),
) -> tuple[Image.Image, Path]:
    """
    Processes a single manga image. Handles core validation, calls the core pipeline,
    and returns the processed image or raises standard exceptions.

    Args:
        image: Input image (path or PIL object).
        config: The main configuration object.
        selected_font_pack_name: Name of the font pack directory.
        models_dir: Path to the directory containing YOLO models.
        fonts_base_dir: Path to the base directory containing font pack subdirectories.
        output_base_dir: Base directory to save the output image.

    Returns:
        A tuple containing:
            - The translated PIL Image object.
            - The Path object where the image was saved.

    Raises:
        FileNotFoundError: If required models or fonts are not found.
        ValidationError: If core configuration validation fails.
        ValueError: For general configuration issues.
        LogicError: For processing failures within this function.
        Exception: For unexpected errors during processing.
    """
    try:
        # Create temporary config for validation since we only have the font pack name
        rendering_cfg_for_val = RenderingConfig(
            font_dir=selected_font_pack_name,
            max_font_size=config.rendering.max_font_size,
            min_font_size=config.rendering.min_font_size,
            line_spacing=config.rendering.line_spacing,
            font_hinting=config.rendering.font_hinting,
        )
        yolo_model_path, font_dir_path = validate_core_inputs(
            translation_cfg=config.translation,
            rendering_cfg=rendering_cfg_for_val,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
        )
        config.yolo_model_path = str(yolo_model_path)
        config.rendering.font_dir = str(font_dir_path)

    except (FileNotFoundError, ValidationError, ValueError) as e:
        raise e

    temp_image_path = None
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if isinstance(image, (str, Path)):
            input_path = Path(image)
            original_ext = input_path.suffix.lower()
            image_path_for_processing = str(input_path)
        elif isinstance(image, Image.Image):
            # PIL objects need to be saved to disk for processing
            original_ext = ".png"
            temp_image_path = Path(tempfile.mktemp(suffix=".png"))
            image.save(temp_image_path)
            image_path_for_processing = str(temp_image_path)
        else:
            raise ValueError(f"Unsupported input image type: {type(image)}")

        output_format = config.output.output_format
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

        os.makedirs(output_base_dir, exist_ok=True)
        save_path = output_base_dir / f"MangaTranslator_{timestamp}{output_ext}"

        translated_image = translate_and_render(
            image_path=image_path_for_processing, config=config, output_path=save_path
        )

        if not translated_image:
            raise LogicError("Translation process failed to return an image.")

        return translated_image, save_path

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise LogicError(
            f"An unexpected error occurred during translation: {str(e)}"
        ) from e
    finally:
        if temp_image_path and temp_image_path.exists():
            try:
                os.remove(temp_image_path)
            except Exception as e_clean:
                log_message(
                    f"Failed to clean up temporary file: {e_clean}", always_print=True
                )


def process_batch_logic(
    input_dir_or_files: Union[str, List[str]],
    config: MangaTranslatorConfig,
    selected_font_pack_name: str,
    models_dir: Path,
    fonts_base_dir: Path,
    output_base_dir: Path = Path("./output"),
    gradio_progress: Any = None,
) -> Dict[str, Any]:
    """
    Processes a batch of manga images. Handles core validation, calls the core batch pipeline,
    and returns results or raises standard exceptions.

    Args:
        input_dir_or_files: Path to input directory or list of file paths.
        config: The main configuration object.
        selected_font_pack_name: Name of the font pack directory.
        models_dir: Path to the directory containing YOLO models.
        fonts_base_dir: Path to the base directory containing font pack subdirectories.
        output_base_dir: Base directory to save the output images.
        gradio_progress: Optional Gradio Progress object for UI updates.

    Returns:
        A dictionary containing processing results:
        {
            "success_count": int,
            "error_count": int,
            "errors": Dict[str, str],
            "output_path": Path,
            "processing_time": float
        }

    Raises:
        FileNotFoundError: If required models or fonts are not found, or input dir is invalid.
        ValidationError: If core configuration validation fails.
        ValueError: For general configuration issues or invalid inputs.
        LogicError: For processing failures within this function.
        Exception: For unexpected errors during processing.
    """
    start_time = time.time()

    try:
        # Create temporary config for validation since we only have the font pack name
        rendering_cfg_for_val = RenderingConfig(
            font_dir=selected_font_pack_name,
            max_font_size=config.rendering.max_font_size,
            min_font_size=config.rendering.min_font_size,
            line_spacing=config.rendering.line_spacing,
            font_hinting=config.rendering.font_hinting,
        )
        yolo_model_path, font_dir_path = validate_core_inputs(
            translation_cfg=config.translation,
            rendering_cfg=rendering_cfg_for_val,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
        )
        config.yolo_model_path = str(yolo_model_path)
        config.rendering.font_dir = str(font_dir_path)
    except (FileNotFoundError, ValidationError, ValueError) as e:
        raise e

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_output_path = output_base_dir / timestamp

    def _batch_progress_callback(value, desc="Processing..."):
        if gradio_progress is not None:
            gradio_progress(value, desc=desc)
        elif config.verbose:
            log_message(f"Progress: {desc} [{value * 100:.1f}%]", verbose=True)

    temp_dir_path_obj = None
    try:
        process_dir = None
        if isinstance(input_dir_or_files, list):
            if not input_dir_or_files:
                raise ValueError("No files provided for batch processing.")

            temp_dir_path_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir_path_obj.name)

            image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
            image_files_to_copy = []
            skipped_files = []
            for f_path_str in input_dir_or_files:
                p = Path(f_path_str)
                if p.is_file() and p.suffix.lower() in image_extensions:
                    # Verify file is actually a valid image
                    try:
                        with Image.open(p) as img:
                            img.verify()
                        image_files_to_copy.append(p)
                    except Exception as img_err:
                        skipped_files.append(f"{p.name} (Invalid: {img_err})")
                else:
                    skipped_files.append(f"{p.name} (Not a supported image file)")

            if not image_files_to_copy:
                raise ValueError(
                    f"No valid image files found in the selection. "
                    f"Skipped: {', '.join(skipped_files) if skipped_files else 'None'}"
                )

            if skipped_files:
                log_message(
                    f"Preparing {len(image_files_to_copy)} files (skipped {len(skipped_files)} invalid files)",
                    verbose=config.verbose,
                )
            else:
                log_message(
                    f"Preparing {len(image_files_to_copy)} files for batch processing",
                    verbose=config.verbose,
                )

            for img_file in image_files_to_copy:
                try:
                    # Preserve metadata when copying
                    shutil.copy2(img_file, temp_dir_path / img_file.name)
                except Exception as copy_err:
                    log_message(
                        f"Failed to copy {img_file.name}: {copy_err}", always_print=True
                    )
            process_dir = temp_dir_path

        elif isinstance(input_dir_or_files, str):
            input_path = Path(input_dir_or_files)
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Input directory '{input_dir_or_files}' does not exist."
                )
            if not input_path.is_dir():
                raise ValueError(
                    f"Input path '{input_dir_or_files}' is not a directory."
                )
            process_dir = input_path

        if not process_dir:
            raise LogicError("Could not determine processing directory.")

        results = batch_translate_images(
            input_dir=process_dir,
            config=config,
            output_dir=batch_output_path,
            progress_callback=_batch_progress_callback,
        )

        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        results["output_path"] = batch_output_path

        log_message(
            f"Batch completed: {results['success_count']} success, "
            f"{results['error_count']} errors in {processing_time:.2f}s",
            verbose=config.verbose,
        )
        if results["errors"]:
            log_message("Processing errors:", verbose=config.verbose)
            for fname, err in results["errors"].items():
                log_message(f"  {fname}: {err}", verbose=config.verbose)

        return results

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise LogicError(
            f"An unexpected error occurred during batch processing: {str(e)}"
        ) from e

    finally:
        if temp_dir_path_obj:
            try:
                temp_dir_path_obj.cleanup()
                log_message("Cleaned up temp directory", verbose=config.verbose)
            except Exception as e_clean:
                log_message(
                    f"Failed to clean up temp directory: {e_clean}",
                    always_print=True,
                )
