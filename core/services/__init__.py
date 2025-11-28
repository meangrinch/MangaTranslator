"""
External service integration modules for MangaTranslator.

This subpackage contains modules for:
- Translation API calls to various LLM providers
- External service communication
"""

from .translation import (
    call_translation_api_batch,
    prepare_bubble_images_for_translation,
    sort_bubbles_by_reading_order,
)

__all__ = [
    "call_translation_api_batch",
    "prepare_bubble_images_for_translation",
    "sort_bubbles_by_reading_order",
]
