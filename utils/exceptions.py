class ValidationError(ValueError):
    """Custom exception for validation errors."""

    pass


class ModelError(RuntimeError):
    """Custom exception for model loading and inference failures."""

    pass


class FontError(RuntimeError):
    """Custom exception for font loading and resource failures."""

    pass


class RenderingError(RuntimeError):
    """Custom exception for text rendering and drawing failures."""

    pass


class ImageProcessingError(Exception):
    """Custom exception for image operations failures."""

    pass


class TranslationError(RuntimeError):
    """Custom exception for translation API and processing failures."""

    pass


class DetectionError(RuntimeError):
    """Custom exception for speech bubble detection failures."""

    pass


class CleaningError(Exception):
    """Custom exception for bubble cleaning failures."""

    pass


class CancellationError(Exception):
    pass
