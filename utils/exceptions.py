class ValidationError(ValueError):
    """Custom exception for validation errors."""


class ModelError(RuntimeError):
    """Custom exception for model loading and inference failures."""


class FontError(RuntimeError):
    """Custom exception for font loading and resource failures."""


class RenderingError(RuntimeError):
    """Custom exception for text rendering and drawing failures."""


class ImageProcessingError(Exception):
    """Custom exception for image operations failures."""


class TranslationError(RuntimeError):
    """Custom exception for translation API and processing failures."""


class DetectionError(RuntimeError):
    """Custom exception for speech bubble detection failures."""


class CleaningError(Exception):
    """Custom exception for bubble cleaning failures."""


class CancellationError(Exception):
    pass
