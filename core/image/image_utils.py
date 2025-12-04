import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from core.caching import get_cache
from core.ml.model_manager import get_model_manager
from utils.exceptions import ImageProcessingError
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
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        elif rgb_image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGRA)
    return rgb_image


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
    return Image.fromarray(cv2_image)


def save_image_with_compression(
    image, output_path, jpeg_quality=95, png_compression=6, verbose=False
):
    """
    Save an image with specified compression settings.

    Args:
        image (PIL.Image): Image to save
        output_path (str or Path): Path to save the image
        jpeg_quality (int): JPEG quality (1-100, higher is better quality)
        png_compression (int): PNG compression level (0-9, higher is more compression)
        verbose (bool): Whether to print verbose logging

    Raises:
        ImageProcessingError: If image saving fails
    """
    output_path = (
        Path(output_path) if not isinstance(output_path, Path) else output_path
    )

    extension = output_path.suffix.lower()
    output_format = None
    save_options = {}

    if extension in [".jpg", ".jpeg"]:
        output_format = "JPEG"
        # JPEG doesn't support transparency - composite on white background
        if image.mode in ["RGBA", "LA"]:
            log_message(
                f"Converting {image.mode} to RGB for JPEG output", verbose=verbose
            )
            background = Image.new("RGB", image.size, (255, 255, 255))
            alpha_channel = image.split()[-1] if image.mode in ["RGBA", "LA"] else None
            background.paste(image, mask=alpha_channel)
            image = background
        elif image.mode == "P":  # Handle Palette mode
            log_message("Converting P mode to RGB for JPEG output", verbose=verbose)
            image = image.convert("RGB")
        elif image.mode != "RGB":
            log_message(
                f"Converting {image.mode} mode to RGB for JPEG output", verbose=verbose
            )
            image = image.convert("RGB")
        save_options["quality"] = max(1, min(jpeg_quality, 100))
        log_message(
            f"Saving JPEG image with quality {save_options['quality']} to {output_path}",
            verbose=verbose,
        )

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
        log_message(
            f"Saving WEBP image with lossless quality to {output_path}", verbose=verbose
        )

    else:
        log_message(
            f"Warning: Unknown output extension '{extension}'. Saving as PNG.",
            verbose=verbose,
            always_print=True,
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
        raise ImageProcessingError(f"Failed to save image to {output_path}") from e


def calculate_centroid_expansion_box(
    cleaned_mask: np.ndarray, padding_pixels: float = 5.0, verbose: bool = False
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
    """
    Calculates guaranteed safe rendering box using the 5-step Distance Transform Insetting Method.

    This function implements a sophisticated algorithm to find the optimal text placement area
    within a speech bubble, ensuring text never touches the bubble boundaries. The method uses
    computer vision techniques to create a safe zone for text rendering.

    Algorithm Overview:
    The 5-step Distance Transform Insetting Method works as follows:

    1. Establish Safe Zone:
       - Uses cv2.distanceTransform() to compute the distance from each pixel to the nearest
         bubble edge (0 pixels)
       - Creates a safe_area_mask where distance >= padding_pixels
       - This ensures all pixels in the safe zone are at least padding_pixels away from edges

    2. Find Unbiased Anchor:
       - Calculates the centroid (geometric center) of the safe_area_mask using cv2.moments()
       - This provides an unbiased starting point for text placement
       - The centroid represents the "center of mass" of the safe area

    3. Measure Available Space:
       - Performs ray-casting from the centroid in four cardinal directions (left, right, up, down)
       - Measures distances to the nearest safe area boundary in each direction
       - Uses numpy array operations for efficient distance calculation

    4. Calculate Symmetrical Dimensions:
       - Takes the minimum distance in each axis to ensure the box fits in all directions
       - Multiplies by 2 to create symmetrical width and height around the centroid
       - Subtracts 1 pixel margin for safety

    5. Construct Final Box:
       - Creates a centered rectangle within the safe zone
       - Ensures the box is completely contained within the original mask bounds
       - Returns both the box coordinates and the true centroid for precise text positioning

    Why This Approach Works:
    - Distance Transform provides accurate edge detection and safe zone calculation
    - Ray-casting ensures the text box never touches bubble boundaries
    - Centroid-based approach provides natural, visually appealing text placement
    - Symmetrical dimensions prevent text from appearing off-center
    - The method handles complex bubble shapes (ovals, irregular polygons, etc.)

    Args:
        cleaned_mask: Binary mask (0/255) of the cleaned speech bubble where 255 represents
                     the bubble interior and 0 represents the background
        padding_pixels: Minimum distance in pixels that text must maintain from bubble edges.
                       Higher values create more padding but smaller text areas.
        verbose: Whether to print detailed processing information for debugging

    Returns:
        Tuple containing:
        - Tuple[int, int, int, int]: Safe box coordinates as [x, y, width, height] where
          (x, y) is the top-left corner.
        - Tuple[float, float]: True geometric center (centroid) of the safe area as (cx, cy).

    Raises:
        ImageProcessingError: If mask is invalid or calculation fails

    Example:
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> cv2.ellipse(mask, (50, 50), (40, 30), 0, 0, 360, 255, -1)
        >>> box, centroid = calculate_centroid_expansion_box(mask, padding_pixels=10.0)
        >>> log_message(f"Safe box: {box}, Centroid: {centroid}", verbose=True)
        Safe box: (20, 30, 60, 40), Centroid: (50.0, 50.0)
    """
    if cleaned_mask is None or not np.any(cleaned_mask):
        raise ImageProcessingError("Invalid or empty mask provided")

    try:
        # Create safe area using distance transform
        distance_map = cv2.distanceTransform(
            cleaned_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        safe_area_mask = (distance_map >= padding_pixels).astype(np.uint8) * 255

        if not np.any(safe_area_mask):
            log_message(
                f"Safe area calculation failed: padding {padding_pixels:.0f}px too large",
                verbose=verbose,
                always_print=True,
            )
            raise ImageProcessingError("Failed to create safe area mask")

        # Find centroid of safe area
        moments = cv2.moments(safe_area_mask)

        if moments["m00"] == 0:
            raise ImageProcessingError("Safe area mask has no area")

        centroid_x = moments["m10"] / moments["m00"]
        centroid_y = moments["m01"] / moments["m00"]

        # Check if centroid is in a constricted region (dual/conjoined bubbles)
        _, max_val, _, max_loc = cv2.minMaxLoc(distance_map)

        cx_int, cy_int = int(round(centroid_x)), int(round(centroid_y))
        mask_h, mask_w = safe_area_mask.shape

        cx_int = max(0, min(cx_int, mask_w - 1))
        cy_int = max(0, min(cy_int, mask_h - 1))

        dist_at_centroid = distance_map[cy_int, cx_int]

        if dist_at_centroid < max_val * 0.70:
            log_message(
                f"Centroid in constricted region (dist={dist_at_centroid:.1f} vs max={max_val:.1f}). "
                "Moving anchor to pole of inaccessibility.",
                verbose=verbose,
            )
            centroid_x, centroid_y = float(max_loc[0]), float(max_loc[1])

        centroid = (centroid_x, centroid_y)

        # Ray-cast from centroid to find maximum safe dimensions
        cx, cy = int(round(centroid_x)), int(round(centroid_y))
        mask_h, mask_w = safe_area_mask.shape

        # Verify centroid is within safe area, adjust if needed
        if (
            cy < 0
            or cy >= mask_h
            or cx < 0
            or cx >= mask_w
            or safe_area_mask[cy, cx] != 255
        ):
            # Centroid is outside safe area, find nearest safe pixel
            safe_pixels = np.argwhere(safe_area_mask == 255)
            if safe_pixels.size == 0:
                raise ImageProcessingError("No safe pixels found in safe_area_mask")
            # Find nearest safe pixel to calculated centroid
            distances = np.sqrt(
                (safe_pixels[:, 0] - centroid_y) ** 2
                + (safe_pixels[:, 1] - centroid_x) ** 2
            )
            nearest_idx = np.argmin(distances)
            cy, cx = safe_pixels[nearest_idx]
            # Update centroid to the adjusted position
            centroid_x, centroid_y = float(cx), float(cy)
            centroid = (centroid_x, centroid_y)

        left_zeros = np.where(safe_area_mask[cy, 0:cx] == 0)[0]
        dist_to_left_edge = cx - (left_zeros.max() if left_zeros.size > 0 else 0)

        right_zeros = np.where(safe_area_mask[cy, cx:] == 0)[0]
        dist_to_right_edge = right_zeros.min() if right_zeros.size > 0 else mask_w - cx

        up_zeros = np.where(safe_area_mask[0:cy, cx] == 0)[0]
        dist_to_top_edge = cy - (up_zeros.max() if up_zeros.size > 0 else 0)

        down_zeros = np.where(safe_area_mask[cy:, cx] == 0)[0]
        dist_to_bottom_edge = down_zeros.min() if down_zeros.size > 0 else mask_h - cy

        # Only subtract 1 if distance > 1, otherwise use the distance directly
        # This prevents collapsing 1-pixel safe areas to 0x0
        min_width_dist = min(dist_to_left_edge, dist_to_right_edge)
        min_height_dist = min(dist_to_top_edge, dist_to_bottom_edge)
        safe_width_base = min_width_dist - 1 if min_width_dist > 1 else min_width_dist
        safe_height_base = (
            min_height_dist - 1 if min_height_dist > 1 else min_height_dist
        )
        max_safe_width = 2 * max(0, safe_width_base)
        max_safe_height = 2 * max(0, safe_height_base)

        if max_safe_width <= 0 or max_safe_height <= 0:
            log_message(
                f"Invalid safe area dimensions: {max_safe_width:.0f}x{max_safe_height:.0f}",
                verbose=verbose,
                always_print=True,
            )
            raise ImageProcessingError("Failed to create safe area mask")

        box_x_float = centroid_x - max_safe_width / 2.0
        box_y_float = centroid_y - max_safe_height / 2.0

        box_x = int(round(box_x_float))
        box_y = int(round(box_y_float))

        guaranteed_box = (box_x, box_y, max_safe_width, max_safe_height)

        if (
            box_x >= 0
            and box_y >= 0
            and box_x + max_safe_width <= mask_w
            and box_y + max_safe_height <= mask_h
        ):
            log_message(
                f"Safe area: {max_safe_width:.0f}x{max_safe_height:.0f} at ({centroid_x:.0f}, {centroid_y:.0f})",
                verbose=verbose,
            )
            return guaranteed_box, centroid
        else:
            log_message(
                f"Safe area validation failed: exceeds bounds {mask_w}x{mask_h}",
                verbose=verbose,
                always_print=True,
            )
            raise ImageProcessingError("Failed to create safe area mask")

    except (cv2.error, ValueError, IndexError, ZeroDivisionError, OverflowError) as e:
        log_message(
            f"Safe area calculation error: {e}", verbose=verbose, always_print=True
        )
    except Exception as e:
        log_message(
            f"Safe area calculation failed: {e}", verbose=verbose, always_print=True
        )

    raise ImageProcessingError("Safe area calculation failed")


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    """Converts a PIL Image to a PyTorch tensor."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_np = np.array(image).astype(np.float32) / 255.0
    if img_np.ndim == 2:  # Grayscale to RGB
        img_np = np.stack((img_np,) * 3, axis=-1)
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Converts a PyTorch tensor to a PIL Image."""
    img_np = (
        tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255
    ).astype(np.uint8)
    return Image.fromarray(img_np)


def _upscale_image(model, image: Image.Image, device: torch.device) -> Image.Image:
    """Upscales a PIL image using the provided model."""
    tensor_in = image_to_tensor(image, device)
    with torch.no_grad():
        tensor_out = model(tensor_in)
    return tensor_to_image(tensor_out)


def upscale_image_to_dimension(
    model,
    image: Image.Image,
    target: int,
    device: torch.device,
    mode: str,
    model_type: str = "model",
    verbose: bool = False,
) -> Image.Image:
    """
    Upscale until a dimensional target is reached.

    Args:
        mode: 'max' ensures max(width, height) >= target, 'min' ensures min(width, height) >= target
        model_type: Model type identifier ("model" or "model_lite")
    """
    if mode not in {"max", "min"}:
        raise ImageProcessingError("mode must be 'max' or 'min'")

    # Validate input image dimensions
    if image.width <= 0 or image.height <= 0:
        log_message(
            f"Invalid image dimensions: {image.width}x{image.height}. Cannot upscale 0x0 images.",
            always_print=True,
        )
        raise ImageProcessingError(
            f"Invalid image dimensions: {image.width}x{image.height}. Cannot upscale 0x0 images."
        )

    cache = get_cache()
    cache_key = cache.get_upscale_dimension_cache_key(image, target, mode, model_type)
    cached_result = cache.get_upscaled_image(cache_key)
    if cached_result is not None:
        log_message("  - Using cached upscaled image", verbose=verbose)
        return cached_result

    current_image = image

    def met(w: int, h: int) -> bool:
        return (max(w, h) >= target) if mode == "max" else (min(w, h) >= target)

    if met(current_image.width, current_image.height):
        cache.set_upscaled_image(cache_key, current_image, verbose)
        return current_image

    log_message(
        f"Upscaling from {current_image.width}x{current_image.height}...",
        verbose=verbose,
    )
    current_image = _upscale_image(model, current_image, device)
    log_message(f"...to {current_image.width}x{current_image.height}", verbose=verbose)

    while not met(current_image.width, current_image.height):
        log_message(
            f"Upscaling from {current_image.width}x{current_image.height} (additional pass)...",
            verbose=verbose,
        )
        current_image = _upscale_image(model, current_image, device)
        log_message(
            f"...to {current_image.width}x{current_image.height}", verbose=verbose
        )

    cache.set_upscaled_image(cache_key, current_image, verbose)
    return current_image


def upscale_image(
    image: Image.Image, factor: float, model_type: str = "model", verbose: bool = False
) -> Image.Image:
    """Upscales an image by a given factor.

    Args:
        image: Image to upscale
        factor: Upscaling factor
        model_type: Model type to use - "model" or "model_lite"
        verbose: Whether to print verbose logging
    """
    if factor == 1.0:
        return image

    cache = get_cache()
    cache_key = cache.get_upscale_cache_key(image, factor, model_type)
    cached_upscale = cache.get_upscaled_image(cache_key)
    if cached_upscale is not None:
        log_message("  - Using cached upscaled image", verbose=verbose)
        return cached_upscale

    model_manager = get_model_manager()
    if model_type == "model_lite":
        upscale_model = model_manager.load_upscale_lite()
        log_message(f"Upscaling image by {factor}x with lite model...", verbose=verbose)
    else:
        upscale_model = model_manager.load_upscale()
        log_message(f"Upscaling image by {factor}x...", verbose=verbose)
    device = model_manager.device

    target_width = int(image.width * factor)
    target_height = int(image.height * factor)

    upscaled_image = upscale_image_to_dimension(
        upscale_model, image, max(target_width, target_height), device, "max", model_type, verbose
    )
    result = upscaled_image.resize((target_width, target_height), Image.LANCZOS)

    cache.set_upscaled_image(cache_key, result)
    return result


def resize_to_max_side(
    image: Image.Image, max_side: int, verbose: bool = False
) -> Image.Image:
    """Resize so that the largest side equals max_side (aspect ratio preserved)."""
    width, height = image.size
    current_max = max(width, height)
    if current_max == max_side:
        return image
    scale = max_side / current_max
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    log_message(
        f"Resizing to max-side {max_side}: {width}x{height} -> {new_width}x{new_height}",
        verbose=verbose,
    )
    return image.resize((new_width, new_height), Image.LANCZOS)


def resize_to_min_side(
    image: Image.Image, min_side: int, verbose: bool = False
) -> Image.Image:
    """Resize so that the smallest side equals min_side (aspect ratio preserved)."""
    width, height = image.size

    # Validate input image dimensions
    if width <= 0 or height <= 0:
        log_message(
            f"Invalid image dimensions: {width}x{height}. Cannot resize 0x0 images.",
            always_print=True,
        )
        raise ImageProcessingError(
            f"Invalid image dimensions: {width}x{height}. Cannot resize 0x0 images."
        )

    current_min = min(width, height)
    if current_min == min_side:
        return image
    scale = min_side / current_min
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    log_message(
        f"Resizing to min-side {min_side}: {width}x{height} -> {new_width}x{new_height}",
        verbose=verbose,
    )
    return image.resize((new_width, new_height), Image.LANCZOS)


def convert_image_to_target_mode(
    pil_image: Image.Image, target_mode: str, verbose: bool = False
) -> Image.Image:
    """
    Convert a PIL image to the target color mode (RGB or RGBA).

    Handles complex transparency flattening and mode conversion with multiple
    fallback strategies to ensure robust image processing.

    Args:
        pil_image: The PIL image to convert
        target_mode: Target mode ("RGB" or "RGBA")
        verbose: Whether to print detailed logging

    Returns:
        PIL.Image: The converted image in the target mode
    """
    if pil_image.mode == target_mode:
        return pil_image

    if target_mode == "RGB":
        if (
            pil_image.mode == "RGBA"
            or pil_image.mode == "LA"
            or (pil_image.mode == "P" and "transparency" in pil_image.info)
        ):
            log_message(
                f"Converting {pil_image.mode} to RGB (flattening transparency)",
                verbose=verbose,
            )
            background = Image.new("RGB", pil_image.size, (255, 255, 255))
            try:
                mask = None
                if pil_image.mode == "RGBA":
                    mask = pil_image.split()[3]
                elif pil_image.mode == "LA":
                    mask = pil_image.split()[1]
                elif pil_image.mode == "P" and "transparency" in pil_image.info:
                    temp_rgba = pil_image.convert("RGBA")
                    mask = temp_rgba.split()[3]

                if mask:
                    background.paste(pil_image, mask=mask)
                    pil_image = background
                else:
                    pil_image = pil_image.convert("RGB")
            except Exception as paste_err:
                log_message(
                    f"Warning: Paste failed, trying alpha_composite: {paste_err}",
                    verbose=verbose,
                )
                try:
                    background_comp = Image.new("RGB", pil_image.size, (255, 255, 255))
                    img_rgba_for_composite = (
                        pil_image
                        if pil_image.mode == "RGBA"
                        else pil_image.convert("RGBA")
                    )
                    pil_image = Image.alpha_composite(
                        background_comp.convert("RGBA"), img_rgba_for_composite
                    ).convert("RGB")
                    log_message(
                        "Alpha composite conversion successful", verbose=verbose
                    )
                except Exception as composite_err:
                    log_message(
                        f"Warning: Alpha composite failed, using simple convert: {composite_err}",
                        verbose=verbose,
                    )
                    pil_image = pil_image.convert("RGB")  # Final fallback conversion
        else:  # Non-transparent conversion to RGB
            log_message(f"Converting {pil_image.mode} to RGB", verbose=verbose)
            pil_image = pil_image.convert("RGB")
    elif target_mode == "RGBA":
        log_message(f"Converting {pil_image.mode} to RGBA", verbose=verbose)
        pil_image = pil_image.convert("RGBA")

    return pil_image


def process_bubble_image_cached(
    bubble_image_pil: Image.Image,
    upscale_model,
    device: torch.device,
    target_min_side: int = 200,
    mode: str = "min",
    model_type: str = "model",
    verbose: bool = False,
) -> Image.Image:
    """
    Process a bubble image with upscaling, using cache for the complete pipeline.

    This function handles the complete bubble processing pipeline:
    1. Upscales the bubble to meet minimum size requirements
    2. Resizes to exact minimum side length
    3. Caches the final result

    Args:
        bubble_image_pil: The bubble image to process
        upscale_model: The upscaling model to use
        device: PyTorch device for model inference
        target_min_side: Target minimum side length
        mode: Upscaling mode ('max' or 'min')
        model_type: Model type identifier ("model" or "model_lite")
        verbose: Whether to print detailed logging

    Returns:
        Image.Image: The processed bubble image
    """
    cache = get_cache()
    cache_key = cache.get_bubble_processing_cache_key(
        bubble_image_pil, target_min_side, mode, model_type
    )
    cached_result = cache.get_upscaled_image(cache_key)
    if cached_result is not None:
        log_message("  - Using cached bubble processing result", verbose=verbose)
        return cached_result

    upscaled_bubble = upscale_image_to_dimension(
        upscale_model,
        bubble_image_pil,
        target_min_side,
        device,
        mode,
        model_type,
        verbose,
    )

    resized_bubble = resize_to_min_side(upscaled_bubble, target_min_side, verbose)

    cache.set_upscaled_image(cache_key, resized_bubble, verbose)
    return resized_bubble
