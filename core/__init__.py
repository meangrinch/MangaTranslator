"""
MangaTranslator Core Package

This package contains the core functionality for translating manga/comic speech bubbles.
It uses YOLO for speech bubble detection and Gemini API for text translation.
"""

from .generate_manga import (
    translate_and_render,
    batch_translate_images,
    choose_font,
    fit_text_to_bubble
)

from .utils import (
    detect_speech_bubbles,
    clean_speech_bubbles,
    call_gemini_api_batch,
    sort_bubbles_manga_order,
    pil_to_cv2,
    cv2_to_pil
)

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)
__author__ = "grinnch"
__copyright__ = "Copyright 2025 grinnch"
__license__ = "Apache-2.0"
__url__ = "https://github.com/meangrinch/MangaTranslator"
__description__ = "A tool for translating manga pages using AI"
__all__ = [
    'translate_and_render',
    'batch_translate_images',
    'choose_font',
    'fit_text_to_bubble',
    'detect_speech_bubbles',
    'clean_speech_bubbles',
    'call_gemini_api_batch',
    'sort_bubbles_manga_order',
    'pil_to_cv2',
    'cv2_to_pil'
]
