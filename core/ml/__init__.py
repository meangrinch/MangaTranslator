"""
Machine learning model management for MangaTranslator.

This subpackage contains modules for:
- Centralized ML model loading and caching
- Model manager for YOLO, SAM, Flux, and other models
"""

from .model_manager import ModelManager, get_model_manager

__all__ = [
    "ModelManager",
    "get_model_manager",
]
