import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Union

from PIL import Image

from core.config import MangaTranslatorConfig, RenderingConfig
from core.pipeline import batch_translate_images, translate_and_render
from core.validation import validate_batch_input_path, validate_core_inputs
from utils.exceptions import (CleaningError, FontError, ImageProcessingError,
                              RenderingError, TranslationError,
                              ValidationError)
from utils.logging import log_message


class LogicError(Exception):
    pass


def extract_zip_to_temp(
    zip_file_path: Union[str, Path],
) -> tuple[Path, tempfile.TemporaryDirectory]:
    """
    Extracts a ZIP archive to a temporary directory, preserving folder structure.

    Args:
        zip_file_path: Path to the ZIP file to extract.

    Returns:
        Tuple of (Path to extracted directory, TemporaryDirectory object).
        The caller must keep a reference to the TemporaryDirectory object
        to prevent premature cleanup, and call cleanup() when done.

    Raises:
        ValidationError: If the ZIP file is invalid or cannot be extracted.
        FileNotFoundError: If the ZIP file does not exist.
    """
    zip_path = Path(zip_file_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file '{zip_file_path}' does not exist.")

    if not zip_path.is_file():
        raise ValidationError(f"Path '{zip_file_path}' is not a file.")

    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir_obj.name)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir_path)

        log_message(
            f"Extracted ZIP archive '{zip_path.name}' to temporary directory",
            verbose=True,
        )

        return temp_dir_path, temp_dir_obj

    except zipfile.BadZipFile:
        temp_dir_obj.cleanup()
        raise ValidationError(f"File '{zip_file_path}' is not a valid ZIP archive.")
    except Exception as e:
        temp_dir_obj.cleanup()
        raise ValidationError(f"Failed to extract ZIP archive: {str(e)}")


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
        if isinstance(image, (str, Path)):
            original_name = input_path.stem
        else:
            # PIL.Image object - no original filename available
            original_name = "MangaTranslator"
        save_path = output_base_dir / f"{original_name}_translated_{timestamp}{output_ext}"

        translated_image = translate_and_render(
            image_path=image_path_for_processing, config=config, output_path=save_path
        )

        if not translated_image:
            raise LogicError("Translation process failed to return an image.")

        return translated_image, save_path

    except (FontError, RenderingError) as e:
        raise LogicError(f"Text rendering failed: {str(e)}") from e
    except CleaningError as e:
        raise LogicError(f"Bubble cleaning failed: {str(e)}") from e
    except TranslationError as e:
        raise LogicError(f"Translation failed: {str(e)}") from e
    except ImageProcessingError as e:
        raise LogicError(f"Image processing failed: {str(e)}") from e
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
    input_dir_or_files: Union[str, List[str], Dict[str, Any]],
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
        input_dir_or_files: Can be one of:
            - Path to input directory (str)
            - Path to ZIP archive (str ending in .zip)
            - List of file paths (List[str])
            - Dictionary with both "zip" and "files" keys to process both simultaneously (Dict[str, Any])
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
    zip_temp_dir_obj = None
    try:
        process_dir = None
        preserve_structure = False

        if (
            isinstance(input_dir_or_files, dict)
            and "zip" in input_dir_or_files
            and "files" in input_dir_or_files
        ):
            zip_path_str = input_dir_or_files["zip"]
            files_list = input_dir_or_files["files"]

            temp_dir_path_obj = tempfile.TemporaryDirectory()
            combined_temp_dir = Path(temp_dir_path_obj.name)

            zip_path = Path(zip_path_str)
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file '{zip_path_str}' does not exist.")

            extracted_path, zip_temp_dir_obj = extract_zip_to_temp(zip_path)

            for item in extracted_path.iterdir():
                dest = combined_temp_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

            if files_list:
                files_subdir = combined_temp_dir / "uploaded_files"
                files_subdir.mkdir(exist_ok=True)

                image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
                image_files_to_copy = []
                skipped_files = []
                for f_path_str in files_list:
                    p = Path(f_path_str)
                    if p.is_file() and p.suffix.lower() in image_extensions:
                        try:
                            with Image.open(p) as img:
                                img.verify()
                            image_files_to_copy.append(p)
                        except Exception as img_err:
                            skipped_files.append(f"{p.name} (Invalid: {img_err})")
                    else:
                        skipped_files.append(f"{p.name} (Not a supported image file)")

                if image_files_to_copy:
                    for img_file in image_files_to_copy:
                        try:
                            shutil.copy2(img_file, files_subdir / img_file.name)
                        except Exception as copy_err:
                            log_message(
                                f"Failed to copy {img_file.name}: {copy_err}",
                                always_print=True,
                            )

                if skipped_files:
                    log_message(
                        f"Skipped {len(skipped_files)} invalid files from directory upload",
                        verbose=config.verbose,
                    )

            process_dir = combined_temp_dir
            preserve_structure = True

        elif isinstance(input_dir_or_files, list):
            if not input_dir_or_files:
                raise ValidationError("No files provided for batch processing.")

            temp_dir_path_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir_path_obj.name)

            image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
            image_files_to_copy = []
            skipped_files = []
            for f_path_str in input_dir_or_files:
                p = Path(f_path_str)
                if p.is_file() and p.suffix.lower() in image_extensions:
                    try:
                        with Image.open(p) as img:
                            img.verify()
                        image_files_to_copy.append(p)
                    except Exception as img_err:
                        skipped_files.append(f"{p.name} (Invalid: {img_err})")
                else:
                    skipped_files.append(f"{p.name} (Not a supported image file)")

            if not image_files_to_copy:
                raise ValidationError(
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
            input_path = validate_batch_input_path(input_dir_or_files)

            if input_path.is_file() and input_path.suffix.lower() == ".zip":
                extracted_path, zip_temp_dir_obj = extract_zip_to_temp(input_path)
                process_dir = extracted_path
                preserve_structure = True
            elif input_path.is_dir():
                process_dir = input_path
                preserve_structure = False

        if not process_dir:
            raise LogicError("Could not determine processing directory.")

        results = batch_translate_images(
            input_dir=process_dir,
            config=config,
            output_dir=batch_output_path,
            progress_callback=_batch_progress_callback,
            preserve_structure=preserve_structure,
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

    except (FontError, RenderingError) as e:
        raise LogicError(
            f"Text rendering failed during batch processing: {str(e)}"
        ) from e
    except CleaningError as e:
        raise LogicError(
            f"Bubble cleaning failed during batch processing: {str(e)}"
        ) from e
    except TranslationError as e:
        raise LogicError(f"Translation failed during batch processing: {str(e)}") from e
    except ImageProcessingError as e:
        raise LogicError(
            f"Image processing failed during batch processing: {str(e)}"
        ) from e
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
        if zip_temp_dir_obj:
            try:
                zip_temp_dir_obj.cleanup()
                log_message(
                    "Cleaned up ZIP extraction temp directory", verbose=config.verbose
                )
            except Exception as e_clean:
                log_message(
                    f"Failed to clean up ZIP extraction temp directory: {e_clean}",
                    always_print=True,
                )
