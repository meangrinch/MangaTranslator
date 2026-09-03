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
    "FluxKleinInpainter",
    "FluxKontextInpainter",
    "ModelManager",
    "OutsideTextDetector",
    "UnifiedCache",
    "__version__",
    "__version_info__",
    "batch_translate_images",
    "call_translation_api_batch",
    "clean_speech_bubbles",
    "cv2_to_pil",
    "detect_speech_bubbles",
    "get_cache",
    "get_model_manager",
    "pil_to_cv2",
    "render_text_skia",
    "save_image_with_compression",
    "sort_bubbles_by_reading_order",
    "translate_and_render",
]
