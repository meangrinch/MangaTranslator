"""
MangaTranslator Core Package

This package contains the core functionality for translating manga/comic speech bubbles.
It uses YOLO for speech bubble detection and LLMs for text translation.
"""

from ._version import __version__, __version_info__
from .caching import UnifiedCache, get_cache
from .image.cleaning import clean_speech_bubbles
from .image.detection import detect_speech_bubbles
from .image.image_utils import cv2_to_pil, pil_to_cv2, save_image_with_compression
from .image.inpainting import FluxKleinInpainter, FluxKontextInpainter
from .image.ocr_detection import OutsideTextDetector
from .image.sorting import sort_bubbles_by_reading_order
from .ml.model_manager import ModelManager, get_model_manager
from .pipeline import batch_translate_images, translate_and_render
from .services.translation import call_translation_api_batch
from .text.text_renderer import render_text_skia

__all__ = [
    "__version__",
    "__version_info__",
    "get_cache",
    "UnifiedCache",
    "translate_and_render",
    "batch_translate_images",
    "render_text_skia",
    "detect_speech_bubbles",
    "clean_speech_bubbles",
    "call_translation_api_batch",
    "sort_bubbles_by_reading_order",
    "pil_to_cv2",
    "cv2_to_pil",
    "save_image_with_compression",
    "get_model_manager",
    "ModelManager",
    "OutsideTextDetector",
    "FluxKontextInpainter",
    "FluxKleinInpainter",
]
