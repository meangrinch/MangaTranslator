"""
MangaTranslator Core Package

This package contains the core functionality for translating manga/comic speech bubbles.
It uses YOLO for speech bubble detection and LLMs for text translation.
"""

from .caching import UnifiedCache, get_cache
from .image.cleaning import clean_speech_bubbles
from .image.detection import detect_speech_bubbles
from .image.image_utils import cv2_to_pil, pil_to_cv2, save_image_with_compression
from .image.inpainting import FluxKontextInpainter
from .image.ocr_detection import OutsideTextDetector
from .ml.model_manager import ModelManager, get_model_manager
from .pipeline import batch_translate_images, translate_and_render
from .services.translation import call_translation_api_batch
from .image.sorting import sort_bubbles_by_reading_order
from .text.text_renderer import render_text_skia

__version__ = "1.10.5"
__version_info__ = (1, 10, 5)
__author__ = "grinnch"
__copyright__ = "Copyright 2025-present grinnch"
__license__ = "Apache-2.0"
__description__ = "A tool for translating manga pages using AI"
__all__ = [
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
]
