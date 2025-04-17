"""
MangaTranslator Core Package

This package contains the core functionality for translating manga/comic speech bubbles.
It uses YOLO for speech bubble detection and LLMs for text translation.
"""

from .pipeline import translate_and_render, batch_translate_images
from .image_utils import pil_to_cv2, cv2_to_pil, save_image_with_compression
from .detection import detect_speech_bubbles
from .cleaning import clean_speech_bubbles
from .translation import sort_bubbles_by_reading_order, call_translation_api_batch
from .rendering import render_text_skia

__version__ = "1.0.0"
__version_info__ = (1, 0, 1)
__author__ = "grinnch"
__copyright__ = "Copyright 2025-present grinnch"
__license__ = "GPL-3.0"
__description__ = "A tool for translating manga pages using AI"
__all__ = [
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
]
