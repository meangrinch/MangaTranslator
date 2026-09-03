

def _normalize_scale(scale: float | None) -> float:
    if scale is None or scale <= 0:
        return 1.0
    return float(scale)


def _clamp(value: float, minimum: float | None, maximum: float | None) -> float:
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def scale_scalar(
    value: float,
    scale: float | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """
    Scale an arbitrary scalar (float) value by the processing scale.
    """
    effective_scale = _normalize_scale(scale)
    scaled = value * effective_scale
    return _clamp(scaled, minimum, maximum)


def scale_length(
    value: float,
    scale: float | None,
    *,
    minimum: float | None = 1.0,
    maximum: float | None = None,
) -> int:
    """
    Scale a pixel length and return an int with rounding and clamping.
    """
    scaled = scale_scalar(value, scale, minimum=minimum, maximum=maximum)
    # Round to nearest integer for pixel units
    return max(1, round(scaled))


def scale_area(
    value: float,
    scale: float | None,
    *,
    minimum: float | None = 1.0,
    maximum: float | None = None,
) -> int:
    """
    Scale an area-like value (square pixels). Uses scale^2.
    """
    effective_scale = _normalize_scale(scale)
    scaled = value * (effective_scale * effective_scale)
    scaled = _clamp(scaled, minimum, maximum)
    return max(1, round(scaled))


def scale_kernel(
    kernel: tuple[int, int],
    scale: float | None,
    *,
    minimum: int = 1,
    maximum: int = 63,
) -> tuple[int, int]:
    """
    Scale a 2D kernel size while ensuring odd dimensions (required for many morphology ops).
    """
    width, height = kernel
    effective_scale = _normalize_scale(scale)

    def _scale_dimension(base: int) -> int:
        dimension = scale_scalar(
            base,
            effective_scale,
            minimum=float(minimum),
            maximum=float(maximum),
        )
        dim_int = max(minimum, round(dimension))
        # Ensure result stays within bounds
        dim_int = min(maximum, dim_int)
        if dim_int % 2 == 0:
            # Prefer rounding up to keep padding generous, but clamp again
            dim_int = min(maximum, dim_int + 1)
            if dim_int % 2 == 0:
                dim_int = max(minimum, dim_int - 1)
                if dim_int % 2 == 0:
                    dim_int = max(minimum, dim_int + 1)
        return max(minimum, dim_int)

    return (_scale_dimension(width), _scale_dimension(height))


def scale_font_size(
    value: float,
    scale: float | None,
    *,
    minimum: int = 4,
    maximum: int = 256,
) -> int:
    """
    Scale a font size (int) using linear scaling with clamping.
    """
    return scale_length(value, scale, minimum=minimum, maximum=maximum)
