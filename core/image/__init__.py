"""
Image processing and analysis modules for MangaTranslator.

This subpackage contains modules for:
- Speech bubble detection (YOLO, SAM)
- OCR text detection outside bubbles
- Image cleaning and preprocessing
- Inpainting for text removal
- General image utilities
"""

from .cleaning import clean_speech_bubbles
from .detection import detect_speech_bubbles
from .image_utils import (calculate_centroid_expansion_box,
                          convert_image_to_target_mode, cv2_to_pil, pil_to_cv2,
                          process_bubble_image_cached, resize_to_max_side,
                          save_image_with_compression, upscale_image,
                          upscale_image_to_dimension)
from .inpainting import FluxKontextInpainter
from .ocr_detection import OCR_LANGUAGE_MAP, OutsideTextDetector

__all__ = [
    "clean_speech_bubbles",
    "detect_speech_bubbles",
    "calculate_centroid_expansion_box",
    "convert_image_to_target_mode",
    "cv2_to_pil",
    "pil_to_cv2",
    "process_bubble_image_cached",
    "resize_to_max_side",
    "save_image_with_compression",
    "upscale_image",
    "upscale_image_to_dimension",
    "FluxKontextInpainter",
    "OCR_LANGUAGE_MAP",
    "OutsideTextDetector",
]
