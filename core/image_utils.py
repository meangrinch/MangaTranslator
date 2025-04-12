import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
from utils.logging import log_message


def pil_to_cv2(pil_image):
    """
    Convert PIL Image to OpenCV format (numpy array)

    Args:
        pil_image (PIL.Image): PIL Image object

    Returns:
        numpy.ndarray: OpenCV image in BGR format
    """
    rgb_image = np.array(pil_image)
    if len(rgb_image.shape) == 3:
        if rgb_image.shape[2] == 3:  # RGB
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR
        elif rgb_image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGRA)  # OpenCV expects BGRA
    return rgb_image  # No conversion for other modes (e.g., grayscale)


def cv2_to_pil(cv2_image):
    """
    Convert OpenCV image to PIL Image

    Args:
        cv2_image (numpy.ndarray): OpenCV image in BGR or BGRA format

    Returns:
        PIL.Image: PIL Image object
    """
    if len(cv2_image.shape) == 3:
        if cv2_image.shape[2] == 3:  # BGR
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_image)
        elif cv2_image.shape[2] == 4:  # BGRA
            rgba_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba_image)
    # Handle grayscale or other formats
    return Image.fromarray(cv2_image)


def save_image_with_compression(image, output_path, jpeg_quality=95, png_compression=6, verbose=False):
    """
    Save an image with specified compression settings.

    Args:
        image (PIL.Image): Image to save
        output_path (str or Path): Path to save the image
        jpeg_quality (int): JPEG quality (1-100, higher is better quality)
        png_compression (int): PNG compression level (0-9, higher is more compression)
        verbose (bool): Whether to print verbose logging

    Returns:
        bool: Whether the save was successful
    """
    output_path = Path(output_path) if not isinstance(output_path, Path) else output_path

    extension = output_path.suffix.lower()
    output_format = None
    save_options = {}

    if extension in [".jpg", ".jpeg"]:
        output_format = "JPEG"
        # Convert RGBA/P/LA to RGB for JPEG by compositing on white background
        if image.mode in ["RGBA", "LA"]:
            log_message(f"Converting {image.mode} to RGB for JPEG output", verbose=verbose)
            background = Image.new("RGB", image.size, (255, 255, 255))
            # Use alpha channel if available (RGBA, LA)
            alpha_channel = image.split()[-1] if image.mode in ["RGBA", "LA"] else None
            background.paste(image, mask=alpha_channel)
            image = background
        elif image.mode == "P":  # Handle Palette mode
            log_message("Converting P mode to RGB for JPEG output", verbose=verbose)
            image = image.convert("RGB")
        elif image.mode != "RGB":
            log_message(f"Converting {image.mode} mode to RGB for JPEG output", verbose=verbose)
            image = image.convert("RGB")
        save_options["quality"] = max(1, min(jpeg_quality, 100))
        log_message(f"Saving JPEG image with quality {save_options['quality']} to {output_path}", verbose=verbose)

    elif extension == ".png":
        output_format = "PNG"
        save_options["compress_level"] = max(0, min(png_compression, 9))
        log_message(
            f"Saving PNG image with compression level {save_options['compress_level']} to {output_path}",
            verbose=verbose,
        )

    elif extension == ".webp":
        output_format = "WEBP"
        save_options["lossless"] = True
        log_message(f"Saving WEBP image with quality {save_options['quality']} to {output_path}", verbose=verbose)

    else:
        log_message(
            f"Warning: Unknown output extension '{extension}'. Saving as PNG.", verbose=verbose, always_print=True
        )
        output_format = "PNG"
        output_path = output_path.with_suffix(".png")
        save_options["compress_level"] = max(0, min(png_compression, 9))
        log_message(
            f"Saving PNG image with compression level {save_options['compress_level']} to {output_path}",
            verbose=verbose,
        )

    try:
        os.makedirs(output_path.parent, exist_ok=True)
        image.save(str(output_path), format=output_format, **save_options)
        return True
    except Exception as e:
        log_message(f"Error saving image to {output_path}: {e}", always_print=True)
        return False
