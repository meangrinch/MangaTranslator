import asyncio
import base64
import math
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import cv2
import numpy as np
from PIL import Image

from core.batch_coordinator import BatchRequestCoordinator
from core.caching import get_cache
from core.config import MangaTranslatorConfig, PreprocessingConfig, RenderingConfig
from core.scaling import scale_font_size, scale_length, scale_scalar
from core.terminology import default_glossary_path, load_terms
from core.validation import validate_config
from utils.archive import copy_comic_info
from utils.exceptions import (
    CancellationError,
    CleaningError,
    FontError,
    ImageProcessingError,
    RenderingError,
    TranslationError,
    ValidationError,
)
from utils.logging import log_message
from utils.path_list import record_failed_images, resolve_source_path

from .component_order_debug import (
    ENABLE_COMPONENT_ORDER_DEBUG,
    write_component_order_debug,
)
from .image.cleaning import clean_speech_bubbles, retry_cleaning_with_otsu
from .image.detection import detect_panels, detect_speech_bubbles
from .image.image_utils import (
    convert_image_to_target_mode,
    cv2_to_pil,
    estimate_glyph_size,
    pil_to_cv2,
    resize_to_max_side,
    save_image_with_compression,
    upscale_image,
    upscale_image_to_dimension,
)
from .image.sorting import sort_bubbles_by_reading_order
from .image.strip_splitting import TARGET_RATIO as STRIP_TARGET_RATIO
from .image.strip_splitting import find_cut_rows
from .image.tilt_detection import compute_effective_tilt_angle
from .ml.model_manager import get_model_manager
from .outside_text_processor import (
    finish_outside_text_work,
    prepare_outside_text_work,
    process_outside_text,
)
from .services.translation import (
    _pair_context_lines,
    call_translation_api_batch,
    prepare_bubble_images_for_translation,
)
from .text.font_manager import find_font_variants, get_cap_height_ratio
from .text.placeholders import generate_test_placeholders
from .text.text_processing import supports_long_word_breaking
from .text.text_renderer import render_text_skia

if TYPE_CHECKING:
    from utils.cancellation import CancellationManager


PREVIOUS_CONTEXT_CACHE_MAX_SIZE = 32
NATURAL_SORT_TOKEN_RE = re.compile(r"(\d+)")
# Translated cap height relative to the measured original lettering. CJK glyphs fill the em
# box and their translations run longer, so professional lettering sits near 0.71 of it.
# Other scripts are measured at about cap height already, so they are matched as is.
AUTO_FONT_CJK_GLYPH_RATIO = 0.71
AUTO_FONT_CJK_LANGUAGE_PREFIXES = ("japanese", "korean", "chinese")
# Typical translated cap height on a 1 MP page, assumed when a page has no measurable text
AUTO_FONT_TYPICAL_CAP_HEIGHT = 13.0
# Ink parts at which one bubble's measurement is fully trusted. Bubbles with fewer count for
# less, and a page that cannot reach it in total (e.g. a lone "..." bubble) is unmeasurable.
AUTO_FONT_TRUSTED_INK_PARTS = 10
# Translation errors that fail the page instead of leaving error text in its bubbles.
# utils/endpoints/_http.py words its errors to match these.
CRITICAL_TRANSLATION_ERRORS = (
    "429",
    "rate limit",
    "rate-limit",
    "auth",
    "unauthorized",
    "forbidden",
    "payment",
    "quota",
    "empty response",
    "api failed",
)
INVALID_TRANSLATION_VALUES = {"[OCR FAILED]", "[Empty response / no content]"}


def _page_glyph_size(
    page_image: np.ndarray, bubbles_info: list[dict[str, Any]], full_width: bool
) -> float | None:
    """Typical original lettering size across the page's speech bubbles, None if unmeasurable."""
    gray = (
        page_image
        if page_image.ndim == 2
        else cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    )
    measured = []
    for info in bubbles_info:
        mask = info.get("mask")
        if mask is None:
            continue
        size_and_parts = estimate_glyph_size(gray, info["bbox"], mask, full_width)
        if size_and_parts:
            size, parts = size_and_parts
            measured.append((size, min(parts, AUTO_FONT_TRUSTED_INK_PARTS)))
    total_weight = sum(weight for _, weight in measured)
    if total_weight < AUTO_FONT_TRUSTED_INK_PARTS:
        return None

    # Weighted median, so ellipses and shouts in a few giant strokes are outvoted
    measured.sort()
    weight_so_far = 0
    for i, (size, weight) in enumerate(measured):
        weight_so_far += weight
        if weight_so_far * 2 > total_weight:
            return size
        if weight_so_far * 2 == total_weight:
            return (size + measured[i + 1][0]) / 2


def _resolve_main_max_font(
    config: MangaTranslatorConfig,
    page_image: np.ndarray,
    bubbles_info: list[dict[str, Any]],
    min_font: int,
    max_font: int,
    processing_scale: float,
    strip_glyph_size: list[float] | None,
    verbose: bool,
) -> int:
    """Speech bubble max font size, lowered to the auto size when that is enabled."""
    if not config.rendering.auto_font_size:
        return max_font

    cjk_source = config.translation.input_language.lower().startswith(
        AUTO_FONT_CJK_LANGUAGE_PREFIXES
    )
    if strip_glyph_size:
        glyph_size = strip_glyph_size[0]
    else:
        glyph_size = _page_glyph_size(page_image, bubbles_info, cjk_source)
        if glyph_size is not None and strip_glyph_size is not None:
            strip_glyph_size.append(glyph_size)

    if glyph_size is None:
        cap_height = AUTO_FONT_TYPICAL_CAP_HEIGHT * processing_scale
    else:
        cap_height = glyph_size * (AUTO_FONT_CJK_GLYPH_RATIO if cjk_source else 1.0)
    regular_font_path = find_font_variants(config.rendering.font_dir, verbose=verbose)[
        "regular"
    ]
    auto_max = round(cap_height / get_cap_height_ratio(str(regular_font_path)))
    auto_max = max(min_font, min(max_font, auto_max))
    log_message(f"Auto font size: {auto_max}", verbose=verbose)
    return auto_max


def _should_overlap_llm_with_inpaint(config: MangaTranslatorConfig) -> bool:
    return (
        bool(getattr(config, "overlap_llm_with_inpaint", False))
        and not config.cleaning_only
        and not getattr(config, "test_mode", False)
    )


def _clean_speech_bubbles_for_page(
    pil_image_processed: Image.Image,
    bubble_data: list[dict[str, Any]],
    config: MangaTranslatorConfig,
    device,
    processing_scale: float,
    verbose: bool,
    fallback_cv_image: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Clean bubbles (including optional colored-bubble Flux). Returns (cv_image, info)."""
    try:
        use_otsu = config.cleaning.use_otsu_threshold
        if config.cleaning.inpaint_colored_bubbles:
            log_message(
                "Flux inpainting enabled for colored bubbles",
                verbose=verbose,
            )

        cleaned_image_cv, processed_bubbles_info = clean_speech_bubbles(
            pil_image_processed,
            config.yolo_model_path,
            config.detection.confidence,
            pre_computed_detections=bubble_data,
            device=device,
            thresholding_value=config.cleaning.thresholding_value,
            use_otsu_threshold=use_otsu,
            auto_threshold=config.cleaning.auto_threshold,
            roi_shrink_px=config.cleaning.roi_shrink_px,
            verbose=verbose,
            processing_scale=processing_scale,
            conjoined_confidence=config.detection.conjoined_confidence,
            inpaint_colored_bubbles=config.cleaning.inpaint_colored_bubbles,
            flux_hf_token=config.outside_text.huggingface_token,
            flux_num_inference_steps=config.outside_text.flux_num_inference_steps,
            flux_residual_diff_threshold=config.outside_text.flux_residual_diff_threshold,
            flux_seed=config.outside_text.seed,
            osb_text_verification=config.detection.use_osb_text_verification,
            osb_text_hf_token=config.outside_text.huggingface_token,
            inpaint_method=config.outside_text.inpainting_method,
            flux_backend=config.outside_text.flux_backend,
            flux_low_vram=config.outside_text.flux_low_vram,
            flux_sdcpp_cache_mode=config.outside_text.flux_sdcpp_cache_mode,
            flux_sdcpp_diffusion_quant=config.outside_text.flux_sdcpp_diffusion_quant,
            flux_sdcpp_text_encoder_quant=config.outside_text.flux_sdcpp_text_encoder_quant,
            flux_luminance_correction=config.outside_text.flux_luminance_correction,
            flux_upscale_small_crops=config.outside_text.flux_upscale_small_crops,
            bubble_detector_model=config.detection.bubble_detector_model,
            request_coordinator=getattr(config, "request_coordinator", None),
        )
        return cleaned_image_cv, processed_bubbles_info
    except CleaningError as e:
        log_message(f"Cleaning failed: {e}", always_print=True)
        return fallback_cv_image.copy(), []
    except CancellationError:
        raise
    except Exception as e:
        log_message(f"Error during cleaning: {e}", always_print=True)
        return fallback_cv_image.copy(), []


def _natural_text_sort_key(text: str) -> tuple[tuple[int, int | str, str], ...]:
    return tuple(
        (0, int(part), part) if part.isdigit() else (1, part.lower(), part)
        for part in NATURAL_SORT_TOKEN_RE.split(text)
        if part
    )


def natural_path_sort_key(path: Path):
    return tuple(_natural_text_sort_key(part) for part in path.parts)


def get_image_encoding_params(pil_image_format: str | None) -> tuple[str, str]:
    """Returns (mime_type, cv2_ext) for a given PIL image format."""
    if pil_image_format and pil_image_format.upper() == "PNG":
        return "image/png", ".png"
    return "image/jpeg", ".jpg"


def _normalize_context_image_mode(
    image: Image.Image,
    mime_type: str,
    verbose: bool = False,
) -> Image.Image:
    if mime_type == "image/jpeg":
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            return background
        if image.mode != "RGB":
            log_message(
                f"Converting {image.mode} previous context image to RGB",
                verbose=verbose,
            )
            return image.convert("RGB")
        return image

    if image.mode not in ("RGB", "RGBA", "L"):
        log_message(
            f"Converting {image.mode} previous context image to RGBA",
            verbose=verbose,
        )
        return image.convert("RGBA")
    return image


def _encode_context_image(
    image: Image.Image, config: MangaTranslatorConfig, cv2_ext: str
) -> str:
    """Resize a page for LLM context per the context upscale setting, base64-encoded."""
    verbose = config.verbose
    # processing_scale is intentionally not used for context_image_max_side_pixels
    max_side = scale_length(
        config.translation.context_image_max_side_pixels,
        None,
        minimum=512,
        maximum=4096,
    )
    # Disable upscaling in test_mode
    upscale_method = "none" if config.test_mode else config.translation.upscale_method

    if upscale_method in ("model", "model_lite"):
        model_manager = get_model_manager()
        if upscale_method == "model":
            upscale_model = model_manager.load_upscale(verbose=verbose)
        else:
            upscale_model = model_manager.load_upscale_lite(verbose=verbose)
        image = upscale_image_to_dimension(
            upscale_model,
            image,
            max_side,
            config.device,
            "max",
            upscale_method,
            verbose,
        )
        image = resize_to_max_side(image, max_side, verbose=verbose)
        model_manager.clear_cache()
    elif upscale_method == "lanczos":
        image = resize_to_max_side(image, max_side, verbose=verbose, antiring=True)

    is_success, buffer = cv2.imencode(cv2_ext, pil_to_cv2(image))
    if not is_success:
        raise ImageProcessingError(f"Context image encoding to {cv2_ext} failed")
    return base64.b64encode(buffer).decode("utf-8")


def _encode_previous_context_source_page(
    image_path: Path,
    config: MangaTranslatorConfig,
    verbose: bool = False,
) -> dict[str, str] | None:
    try:
        with Image.open(image_path) as source_image:
            image_format = source_image.format
            mime_type, cv2_ext = get_image_encoding_params(image_format)
            context_image_pil = _normalize_context_image_mode(
                source_image.copy(),
                mime_type,
                verbose,
            )
        return {
            "mime_type": mime_type,
            "data": _encode_context_image(context_image_pil, config, cv2_ext),
        }
    except Exception as e:
        log_message(
            f"Warning: Failed to encode previous context image {image_path}: {e}",
            always_print=True,
        )
        return None


def _previous_context_cache_key(
    image_path: Path,
    config: MangaTranslatorConfig,
) -> tuple[Any, ...]:
    stat = image_path.stat()
    context_upscale_method = (
        "none" if config.test_mode else config.translation.upscale_method
    )
    return (
        str(image_path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        config.translation.context_image_max_side_pixels,
        context_upscale_method,
    )


def _get_cached_previous_context_image(
    image_path: Path,
    config: MangaTranslatorConfig,
    context_cache: OrderedDict | None,
    context_cache_lock: threading.Lock | None,
) -> dict[str, str] | None:
    verbose = config.verbose
    if context_cache is None:
        return _encode_previous_context_source_page(image_path, config, verbose)

    try:
        cache_key = _previous_context_cache_key(image_path, config)
    except Exception:
        return _encode_previous_context_source_page(image_path, config, verbose)

    if context_cache_lock:
        with context_cache_lock:
            cached = context_cache.get(cache_key)
            if cached is not None:
                context_cache.move_to_end(cache_key)
                return cached

    encoded = _encode_previous_context_source_page(image_path, config, verbose)
    if encoded is None:
        return None

    if context_cache_lock:
        with context_cache_lock:
            context_cache[cache_key] = encoded
            context_cache.move_to_end(cache_key)
            while len(context_cache) > PREVIOUS_CONTEXT_CACHE_MAX_SIZE:
                context_cache.popitem(last=False)
    return encoded


def _build_previous_context_images(
    image_files: list[Path],
    image_index: int,
    config: MangaTranslatorConfig,
    context_cache: OrderedDict | None = None,
    context_cache_lock: threading.Lock | None = None,
) -> list[dict[str, str]]:
    if not getattr(config.translation, "send_full_page_context", False):
        return []
    if getattr(config.translation, "ocr_method", "LLM") != "LLM":
        return []

    requested_count = int(
        getattr(config.translation, "previous_context_image_count", 0) or 0
    )
    if requested_count <= 0:
        return []

    start_index = max(0, image_index - requested_count)
    previous_paths = image_files[start_index:image_index]
    previous_images = []
    for previous_path in previous_paths:
        encoded = _get_cached_previous_context_image(
            previous_path,
            config,
            context_cache,
            context_cache_lock,
        )
        if encoded is not None:
            previous_images.append(encoded)
    return previous_images


def _build_previous_context_texts(
    image_files: list[Path],
    image_index: int,
    config: MangaTranslatorConfig,
    ocr_text_history: dict[Path, list[str]] | None = None,
    ocr_text_history_lock: threading.Lock | None = None,
) -> list[list[str]]:
    """Collect OCR transcripts from up to N already-processed prior pages.

    Parallel callers that require deterministic prior-page text context should
    wait for the required previous pages before calling this helper.
    """
    requested_count = int(
        getattr(config.translation, "previous_context_text_count", 0) or 0
    )
    if requested_count <= 0 or ocr_text_history is None:
        return []

    start_index = max(0, image_index - requested_count)
    previous_paths = image_files[start_index:image_index]
    if not previous_paths:
        return []

    previous_texts: list[list[str]] = []
    if ocr_text_history_lock is not None:
        with ocr_text_history_lock:
            for previous_path in previous_paths:
                texts = ocr_text_history.get(previous_path)
                if texts:
                    previous_texts.append(list(texts))
    else:
        for previous_path in previous_paths:
            texts = ocr_text_history.get(previous_path)
            if texts:
                previous_texts.append(list(texts))
    return previous_texts


def _resolve_pre_upscale_factor(
    pre_cfg: PreprocessingConfig | None,
    verbose: bool = False,
) -> float:
    if pre_cfg is None or not pre_cfg.enabled:
        return 1.0

    factor = max(1.0, min(float(pre_cfg.factor or 1.0), 8.0))
    if factor <= 1.01:
        return 1.0

    log_message(f"Initial upscaling enabled: {factor:.2f}x", verbose=verbose)
    return factor


def _apply_pre_upscale_if_needed(
    image: Image.Image,
    config: MangaTranslatorConfig,
    verbose: bool = False,
) -> tuple[Image.Image, float]:
    factor = _resolve_pre_upscale_factor(
        getattr(config, "preprocessing", None), verbose
    )
    if factor == 1.0:
        return image, 1.0

    # Use the output upscale model setting for initial upscaling as well
    model_type = (
        getattr(config.output, "image_upscale_model", "model_lite")
        if hasattr(config, "output")
        else "model_lite"
    )
    upscaled = upscale_image(image, factor, model_type=model_type, verbose=verbose)
    return upscaled, factor


def _detect_strip_text_rows(
    crop: Image.Image, image_path: Path, config: MangaTranslatorConfig
) -> list[tuple[int, int]]:
    """Row ranges occupied by bubbles and text in a strip crop, for choosing cut rows."""
    bubbles, text_free_boxes = detect_speech_bubbles(
        image_path,
        config.yolo_model_path,
        config.detection.confidence,
        verbose=config.verbose,
        device=config.device,
        conjoined_detection=config.detection.conjoined_detection,
        conjoined_confidence=config.detection.conjoined_confidence,
        image_override=crop,
        bubble_detector_model=config.detection.bubble_detector_model,
    )
    boxes = [bubble["bbox"] for bubble in bubbles] + list(text_free_boxes)
    return [(int(box[1]), int(box[3])) for box in boxes]


def _translate_strip(
    image_path: Path,
    strip: Image.Image,
    cut_rows: list[int],
    config: MangaTranslatorConfig,
    output_path: str | Path | None,
    target_mode: str,
    cancellation_manager: Optional["CancellationManager"],
    previous_context_images: list[dict[str, str]],
    previous_context_texts: list[list[str]],
    previous_context_texts_provider: Callable[[], list[list[str]]] | None,
    ocr_texts_out: list[str] | None,
    empty_bubbles_out: list | None = None,
) -> Image.Image:
    """Translate a long strip section by section and stitch the results back together."""
    start_time = time.time()
    width, height = strip.size
    bounds = [0, *cut_rows, height]
    log_message(
        f"Long strip ({width}x{height}): splitting into {len(bounds) - 1} sections "
        f"at rows {cut_rows}",
        always_print=True,
    )

    section_config = replace(
        config,
        preprocessing=replace(config.preprocessing, split_long_strips=False),
    )
    if config.preprocessing.auto_scale:
        # Section heights vary, so scaling each by its own area would change font sizes
        # and kernels from one section to the next. All share a nominal section's scale.
        pre_upscale = _resolve_pre_upscale_factor(config.preprocessing)
        section_config = replace(
            section_config,
            processing_scale=math.sqrt(STRIP_TARGET_RATIO) * width * pre_upscale / 1000,
            preprocessing=replace(section_config.preprocessing, auto_scale=False),
        )
    # Sections of one image are one page to the reader, so they always share context.
    context_count = max(
        int(getattr(config.translation, "previous_context_text_count", 0) or 0), 1
    )
    section_texts: list[list[str]] = []
    strip_glyph_size: list[float] = []

    def context_texts() -> list[list[str]]:
        incoming = (
            previous_context_texts_provider()
            if previous_context_texts_provider is not None
            else previous_context_texts
        )
        return [*(incoming or []), *section_texts][-context_count:]

    sections: list[Image.Image] = []
    # Open image handles can outlive a section on Windows and block directory removal.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        for index, (top, bottom) in enumerate(pairwise(bounds)):
            log_message(
                f"Processing section {index + 1}/{len(bounds) - 1} (rows {top}-{bottom})",
                always_print=True,
            )
            section_path = Path(temp_dir) / f"{image_path.stem}_section{index + 1}.png"
            strip.crop((0, top, width, bottom)).save(section_path)
            texts: list[str] = []
            section = translate_and_render(
                section_path,
                section_config,
                cancellation_manager=cancellation_manager,
                previous_context_images=previous_context_images,
                previous_context_texts_provider=context_texts,
                ocr_texts_out=texts,
                strip_glyph_size=strip_glyph_size,
                empty_bubbles_out=empty_bubbles_out,
            )
            sections.append(_as_mode(section, target_mode))
            if texts:
                section_texts.append(texts)
                if ocr_texts_out is not None:
                    ocr_texts_out.extend(texts)

    stitched = Image.new(
        target_mode, (sections[0].width, sum(section.height for section in sections))
    )
    offset = 0
    for section in sections:
        stitched.paste(section, (0, offset))
        offset += section.height

    if output_path:
        save_image_with_compression(
            stitched,
            output_path,
            jpeg_quality=config.output.jpeg_quality,
            png_compression=config.output.png_compression,
            verbose=config.verbose,
        )

    log_message(
        f"Strip processing completed in {time.time() - start_time:.2f}s",
        always_print=True,
    )
    return stitched


class _RenderSizes(NamedTuple):
    """Font sizes and paddings scaled to the page, named as generate_test_placeholders takes them."""

    main_min_font: int
    main_max_font: int
    osb_min_font: int
    osb_max_font: int
    padding_pixels: float
    osb_padding_pixels: float
    osb_outline_ratio: float


def _scaled_render_sizes(
    config: MangaTranslatorConfig, processing_scale: float
) -> _RenderSizes:
    main_min_font = scale_font_size(
        config.rendering.min_font_size, processing_scale, minimum=4, maximum=256
    )
    osb_min_font = scale_font_size(
        config.outside_text.osb_min_font_size,
        processing_scale,
        minimum=4,
        maximum=512,
    )
    return _RenderSizes(
        main_min_font=main_min_font,
        main_max_font=scale_font_size(
            config.rendering.max_font_size,
            processing_scale,
            minimum=main_min_font,
            maximum=384,
        ),
        osb_min_font=osb_min_font,
        osb_max_font=scale_font_size(
            config.outside_text.osb_max_font_size,
            processing_scale,
            minimum=osb_min_font,
            maximum=640,
        ),
        padding_pixels=scale_scalar(
            config.rendering.padding_pixels,
            processing_scale,
            minimum=1.0,
            maximum=80.0,
        ),
        osb_padding_pixels=scale_scalar(
            config.outside_text.osb_padding_pixels,
            processing_scale,
            minimum=1.0,
            maximum=80.0,
        ),
        osb_outline_ratio=config.outside_text.osb_outline_ratio,
    )


def _raise_if_cancelled(cancellation_manager: Optional["CancellationManager"]) -> None:
    if cancellation_manager and cancellation_manager.is_cancelled():
        raise CancellationError("Process cancelled by user.")


def _as_mode(image: Image.Image, mode: str) -> Image.Image:
    return image if image.mode == mode else image.convert(mode)


def _open_source_image(image_path: Path) -> Image.Image:
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        log_message(f"Error: Input image not found at {image_path}", always_print=True)
        raise
    except Exception as e:
        log_message(f"Error opening image {image_path}: {e}", always_print=True)
        raise


def _resolve_target_mode(
    config: MangaTranslatorConfig, image_path: Path, output_path: str | Path | None
) -> str:
    """RGB for JPEG output, RGBA for PNG, WEBP, and other formats in auto mode."""
    desired_format = config.output.output_format
    output_ext = (
        Path(output_path).suffix.lower() if output_path else image_path.suffix.lower()
    )
    if desired_format == "jpeg" or (
        desired_format == "auto" and output_ext in (".jpg", ".jpeg")
    ):
        return "RGB"
    return "RGBA"


def _resolve_processing_scale(
    image: Image.Image, config: MangaTranslatorConfig
) -> float:
    """Scale relative to a 1 MP page when auto scale is on, else the configured scale."""
    if not config.preprocessing.auto_scale:
        return config.processing_scale
    width, height = image.size
    processing_scale = math.sqrt((width * height) / 1_000_000)
    log_message(
        f"Dynamic processing scale: {processing_scale:.2f}x", verbose=config.verbose
    )
    return processing_scale


def _finish_page(
    image: Image.Image,
    config: MangaTranslatorConfig,
    output_path: str | Path | None,
    target_mode: str,
    start_time: float,
) -> Image.Image:
    """Apply the optional final upscale and save the page if an output path is given."""
    verbose = config.verbose
    if config.output.upscale_final_image:
        log_message("Upscaling final image...", verbose=verbose, always_print=True)
        image = upscale_image(
            image,
            config.output.image_upscale_factor,
            model_type=config.output.image_upscale_model,
            verbose=verbose,
        )

    if output_path:
        if image.mode != target_mode:
            log_message(f"Converting final image to {target_mode}", verbose=verbose)
            image = image.convert(target_mode)
        try:
            save_image_with_compression(
                image,
                output_path,
                jpeg_quality=config.output.jpeg_quality,
                png_compression=config.output.png_compression,
                verbose=verbose,
            )
        except ImageProcessingError as e:
            log_message(f"Failed to save image: {e}", always_print=True)
            raise

    log_message(
        f"Processing completed in {time.time() - start_time:.2f}s", always_print=True
    )
    return image


def _detect_page_bubbles(
    image_path: Path, image: Image.Image, config: MangaTranslatorConfig
) -> tuple[list[dict[str, Any]], list]:
    """Returns (bubbles, text-free boxes), both empty if detection fails."""
    log_message("Detecting speech bubbles...", verbose=config.verbose)
    try:
        return detect_speech_bubbles(
            image_path,
            config.yolo_model_path,
            config.detection.confidence,
            verbose=config.verbose,
            device=config.device,
            seg_model=config.detection.seg_model,
            conjoined_detection=config.detection.conjoined_detection,
            conjoined_confidence=config.detection.conjoined_confidence,
            image_override=image,
            osb_enabled=config.outside_text.enabled,
            osb_text_verification=config.detection.use_osb_text_verification,
            osb_text_hf_token=config.outside_text.huggingface_token,
            bubble_detector_model=config.detection.bubble_detector_model,
        )
    except Exception as e:
        log_message(f"Error during detection: {e}", always_print=True)
        return [], []


def _detect_page_panels(
    image_path: Path, image: Image.Image, config: MangaTranslatorConfig
) -> tuple[list | None, list | None]:
    """Returns (panels for reading order, panels for the component order debug output)."""
    if not (config.detection.use_panel_sorting or ENABLE_COMPONENT_ORDER_DEBUG):
        return None, None

    verbose = config.verbose
    try:
        log_message("Detecting panels...", verbose=verbose)
        debug_panels = detect_panels(
            image_path,
            confidence=config.detection.panel_confidence,
            device=config.device,
            verbose=verbose,
            image_override=image,
        )
        if debug_panels:
            log_message(f"Detected {len(debug_panels)} panels", always_print=True)
        else:
            log_message("No panels detected", verbose=verbose)
    except Exception as e:
        log_message(
            f"Panel detection failed: {e}. Using global sorting.", always_print=True
        )
        debug_panels = None

    panels = debug_panels if config.detection.use_panel_sorting else None
    return panels, debug_panels


def _encode_full_page_context(
    image: Image.Image, config: MangaTranslatorConfig, cv2_ext: str
) -> str | None:
    try:
        encoded = _encode_context_image(image, config, cv2_ext)
    except Exception as e:
        log_message(
            f"Warning: Failed to encode full image context: {e}", always_print=True
        )
        return None
    log_message("Encoded full image for context", verbose=config.verbose)
    return encoded


def _clean_page(
    image: Image.Image,
    bubble_data: list[dict[str, Any]],
    config: MangaTranslatorConfig,
    processing_scale: float,
    target_mode: str,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Returns (page with bubbles cleaned, in target_mode, per-bubble cleaning info)."""
    if not bubble_data:
        return _as_mode(image, target_mode), []
    log_message("Cleaning speech bubbles...", verbose=config.verbose)
    cleaned_cv, processed_bubbles_info = _clean_speech_bubbles_for_page(
        image,
        bubble_data,
        config,
        config.device,
        processing_scale,
        config.verbose,
        pil_to_cv2(image),
    )
    return _as_mode(cv2_to_pil(cleaned_cv), target_mode), processed_bubbles_info


def _finish_deferred_cleaning(
    outside_work,
    image: Image.Image,
    bubble_data: list[dict[str, Any]],
    config: MangaTranslatorConfig,
    processing_scale: float,
    target_mode: str,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Finish the deferred outside text inpainting, then clean the bubbles on top of it."""
    if outside_work is not None:
        image, _ = finish_outside_text_work(outside_work)
    return _clean_page(image, bubble_data, config, processing_scale, target_mode)


def _prepare_bubble_crops(
    bubble_data: list[dict[str, Any]],
    processed_bubbles_info: list[dict[str, Any]],
    page_cv: np.ndarray,
    config: MangaTranslatorConfig,
    mime_type: str,
) -> list[dict[str, Any]]:
    """Attach the refined cleaning masks to the bubbles and encode their crops for the LLM."""
    verbose = config.verbose
    log_message("Preparing bubble images...", verbose=verbose)

    mask_by_bbox: dict[tuple, Any] = {}
    for info in processed_bubbles_info:
        bbox_key = tuple(round(v) for v in info.get("bbox", ()))
        if len(bbox_key) != 4:
            continue
        mask = info.get("mask")
        if mask is None:
            mask = info.get("base_mask")
        if mask is not None:
            mask_by_bbox[bbox_key] = mask
    for bubble in bubble_data:
        bbox_key = tuple(round(v) for v in bubble.get("bbox", ()))
        if bbox_key in mask_by_bbox:
            bubble["sam_mask"] = mask_by_bbox[bbox_key]

    # Disable upscaling in test_mode
    upscale_method = "none" if config.test_mode else config.translation.upscale_method

    model_manager = get_model_manager()
    upscale_model = None
    if upscale_method == "model":
        upscale_model = model_manager.load_upscale(verbose=verbose)
    elif upscale_method == "model_lite":
        upscale_model = model_manager.load_upscale_lite(verbose=verbose)

    bubble_data = prepare_bubble_images_for_translation(
        bubble_data,
        page_cv,
        upscale_model,
        config.device,
        mime_type,
        config.translation.bubble_min_side_pixels,
        upscale_method,
        config.translation.whiteout_conjoined_bubbles,
        verbose,
    )
    if upscale_model is not None:
        model_manager.clear_cache()

    action = "Upscaled" if upscale_method != "none" else "Prepared"
    log_message(
        f"{action} {len(bubble_data)} bubble images for translation",
        always_print=True,
    )
    return bubble_data


def _sort_text_elements(
    valid_bubble_data: list[dict[str, Any]],
    outside_text_data: list[dict[str, Any]],
    config: MangaTranslatorConfig,
    panels: list | None,
) -> list[dict[str, Any]]:
    """Sort speech bubbles and outside text together into reading order."""
    reading_direction = config.translation.reading_direction
    if outside_text_data:
        log_message(
            f"Including {len(outside_text_data)} outside text regions in reading order calculation",
            verbose=config.verbose,
        )
    log_message(
        f"Sorting all text elements ({reading_direction.upper()})",
        verbose=config.verbose,
    )
    return sort_bubbles_by_reading_order(
        valid_bubble_data + outside_text_data, reading_direction, panels=panels
    )


def _call_translation_api(
    config: MangaTranslatorConfig,
    sorted_bubble_data: list[dict[str, Any]],
    images_b64: list[str],
    mime_types: list[str],
    full_image_b64: str | None,
    full_image_mime_type: str,
    previous_context_images: list[dict[str, str]],
    previous_context_texts: list[list[str]],
    previous_context_texts_provider: Callable[[], list[list[str]]] | None,
) -> tuple[list[str], list[str]]:
    """Returns (translations, OCR transcripts) for the sorted page elements."""
    ocr_texts: list[str] = []
    if previous_context_texts_provider is not None:
        previous_context_texts = previous_context_texts_provider() or []

    context_parts = []
    if previous_context_images:
        context_parts.append(
            f"Previous Context Images: {len(previous_context_images)} page(s)"
        )
    usable_context_text_pages = sum(
        1
        for page_texts in previous_context_texts
        if any(
            (text or "").strip() and (text or "").strip() != "[OCR FAILED]"
            for text in (page_texts or [])
        )
    )
    if usable_context_text_pages:
        context_parts.append(
            f"Previous Context OCR Text: {usable_context_text_pages} page(s)"
        )
    context_suffix = f" ({', '.join(context_parts)})" if context_parts else ""
    log_message(
        f"Translating {len(images_b64)} bubbles: "
        f"{config.translation.input_language} → {config.translation.output_language}"
        f"{context_suffix}",
        always_print=True,
    )
    texts = call_translation_api_batch(
        config=config.translation,
        images_b64=images_b64,
        full_image_b64=full_image_b64 or "",
        mime_types=mime_types,
        full_image_mime_type=full_image_mime_type if full_image_b64 else "image/jpeg",
        bubble_metadata=sorted_bubble_data,
        previous_context_images=previous_context_images,
        previous_context_texts=previous_context_texts,
        ocr_texts_output=ocr_texts,
        debug=config.verbose,
    )
    return texts, ocr_texts


def _translate_or_error_texts(
    get_result: Callable[[], tuple[list[str], list[str]]],
    sorted_bubble_data: list[dict[str, Any]],
    image_count: int,
    config: MangaTranslatorConfig,
    ocr_texts_out: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Run the translation; non-critical failures become per-bubble error text."""
    try:
        texts, ocr_texts = get_result()
    except TranslationError as e:
        error_str = str(e).lower()
        if any(t in error_str for t in CRITICAL_TRANSLATION_ERRORS):
            raise
        log_message(f"Translation failed: {e}", always_print=True)
        texts = [f"[Translation Error: {e}]"] * image_count
        ocr_texts = []
    except CancellationError:
        raise
    except Exception as e:
        log_message(f"Translation API error: {e}", always_print=True)
        texts = [
            "[Translation Error: API call raised exception]" for _ in sorted_bubble_data
        ]
        ocr_texts = []

    if ocr_texts and ocr_texts_out is not None:
        ocr_texts_out.extend(_pair_context_lines(ocr_texts, texts))

    provider_tag = f"[{config.translation.provider}:"
    valid_translations = [
        t
        for t in texts
        if t
        and not t.startswith("[Translation Error")
        and not t.startswith("API Error")
        and not t.startswith(provider_tag)
        and t.strip() not in INVALID_TRANSLATION_VALUES
    ]
    if image_count and not valid_translations:
        raise TranslationError("All bubbles failed.")
    return texts, ocr_texts


def _attach_ocr_texts(
    sorted_bubble_data: list[dict[str, Any]], ocr_texts: list[str], verbose: bool
) -> None:
    if len(ocr_texts) == len(sorted_bubble_data):
        for bubble, ocr_text in zip(sorted_bubble_data, ocr_texts, strict=True):
            bubble["ocr_text"] = ocr_text
    else:
        log_message(
            "OCR/translation count mismatch; OSB unchanged-text restore disabled",
            verbose=verbose,
        )


def _paste_original_crop(image: Image.Image, bubble: dict[str, Any]) -> Image.Image:
    bbox = bubble["bbox"]
    restored = image.copy()
    restored.paste(bubble["original_crop_pil"], (round(bbox[0]), round(bbox[1])))
    return restored


def _text_render_config(
    config: MangaTranslatorConfig, sizes: _RenderSizes, is_outside_text: bool
) -> RenderingConfig:
    rendering = config.rendering
    osb = config.outside_text
    # Latin languages use hyphenation; Korean/Thai use
    # no-hyphen emergency breaks under the same user setting.
    should_hyphenate = (
        rendering.hyphenate_before_scaling
        and supports_long_word_breaking(config.translation.output_language)
    )
    return RenderingConfig(
        min_font_size=sizes.osb_min_font if is_outside_text else sizes.main_min_font,
        max_font_size=sizes.osb_max_font if is_outside_text else sizes.main_max_font,
        line_spacing_mult=(
            osb.osb_line_spacing if is_outside_text else rendering.line_spacing_mult
        ),
        use_ligatures=(
            osb.osb_use_ligatures if is_outside_text else rendering.use_ligatures
        ),
        hyphenate_before_scaling=should_hyphenate,
        hyphen_penalty=rendering.hyphen_penalty,
        hyphenation_min_word_length=rendering.hyphenation_min_word_length,
        badness_exponent=rendering.badness_exponent,
        padding_pixels=(
            sizes.osb_padding_pixels if is_outside_text else sizes.padding_pixels
        ),
        outline_ratio=sizes.osb_outline_ratio if is_outside_text else 0.0,
        supersampling_factor=rendering.supersampling_factor,
        detach_trailing_punctuation=rendering.detach_trailing_punctuation,
        auto_vertical_text=(
            osb.osb_auto_vertical_text
            if is_outside_text
            else rendering.auto_vertical_text
        ),
        vertical_line_spacing_mult=(
            osb.osb_vertical_line_spacing_mult
            if is_outside_text
            else rendering.vertical_line_spacing_mult
        ),
        vertical_font_size_mult=(
            osb.osb_vertical_font_size_mult
            if is_outside_text
            else rendering.vertical_font_size_mult
        ),
    )


def _render_outside_text(
    image: Image.Image,
    bubble: dict[str, Any],
    text: str,
    bubble_id: str,
    config: MangaTranslatorConfig,
    sizes: _RenderSizes,
) -> Image.Image | None:
    """Render one outside text translation, or None if it could not be placed."""
    verbose = config.verbose
    osb = config.outside_text
    bbox = bubble["bbox"]
    log_message(f"Rendering outside text {bbox}: '{text[:30]}...'", verbose=verbose)

    is_dark_text = bubble.get("is_dark_text", True)
    text_color_rgb = bubble.get("text_color_rgb", None)
    rotation_deg = 0.0
    if osb.osb_follow_tilt:
        rotation_deg = compute_effective_tilt_angle(
            tilt_deg=float(bubble.get("tilt_deg", 0.0) or 0.0),
            tilt_conf=float(bubble.get("tilt_conf", 0.0) or 0.0),
            follow_tilt=osb.osb_follow_tilt,
            confidence_threshold=osb.osb_tilt_confidence_threshold,
            min_tilt_deg=osb.osb_min_tilt_deg,
            max_tilt_deg=osb.osb_max_tilt_deg,
        )
        if abs(rotation_deg) > 0.01:
            conf = float(bubble.get("tilt_conf", 0.0) or 0.0)
            log_message(
                f"OSB tilt applied: {rotation_deg:.1f} deg (conf: {conf:.2f}, axis: {bubble.get('tilt_axis', 'none')})",
                verbose=verbose,
            )

    text_bg_rgb = None
    if bubble.get("needs_text_background"):
        if text_color_rgb:
            lum = (
                0.299 * text_color_rgb[0]
                + 0.587 * text_color_rgb[1]
                + 0.114 * text_color_rgb[2]
            )
            text_bg_rgb = (255, 255, 255) if lum < 128 else (0, 0, 0)
        else:
            text_bg_rgb = (0, 0, 0) if is_dark_text else (255, 255, 255)

    tilted_layout_dims = None
    if abs(rotation_deg) > 0.01:
        shrink = max(0.70, math.cos(math.radians(rotation_deg)))
        avail_w = max(1.0, (bbox[2] - bbox[0]) - 2 * sizes.osb_padding_pixels)
        avail_h = max(1.0, (bbox[3] - bbox[1]) - 2 * sizes.osb_padding_pixels)
        tilted_layout_dims = (avail_w * shrink, avail_h * shrink)

    # OSB renders default to horizontal; vertical stacking is fallback-only. Try the
    # detected tilt first, then vertical, then upright in each orientation.
    attempts = [(rotation_deg, False, tilted_layout_dims)]
    if abs(rotation_deg) > 0.01:
        attempts += [(rotation_deg, True, tilted_layout_dims), (0.0, False, None)]
    attempts.append((0.0, True, None))

    render_config = _text_render_config(config, sizes, is_outside_text=True)
    for attempt_rotation, attempt_vertical, layout_dims in attempts:
        try:
            return render_text_skia(
                pil_image=image,
                text=text,
                bbox=bbox,
                font_dir=osb.osb_font_dir or config.rendering.font_dir,
                # Outside text was inpainted, no mask needed
                cleaned_mask=None,
                bubble_color_bgr=(50, 50, 50) if is_dark_text else (255, 255, 255),
                config=render_config,
                verbose=verbose,
                bubble_id=bubble_id,
                rotation_deg=attempt_rotation,
                vertical_stack=attempt_vertical,
                text_color_rgb=text_color_rgb,
                raise_on_safe_error=False,
                text_background_color=text_bg_rgb,
                fallback_padding_pixels=sizes.osb_padding_pixels,
                layout_dimensions=layout_dims,
            )
        except Exception as e:
            log_message(
                f"OSB render failed at {attempt_rotation:.1f} deg "
                f"(vertical_stack={attempt_vertical}): {e}",
                verbose=verbose,
            )

    # Last resort: restore the original OSB patch if available
    if "original_crop_pil" in bubble:
        log_message(
            f"Restoring original OSB patch for {bbox}",
            verbose=verbose,
            always_print=True,
        )
        return _paste_original_crop(image, bubble)
    return None


def _render_bubble(
    image: Image.Image,
    bubble: dict[str, Any],
    text: str,
    bubble_id: str,
    render_info: dict[str, Any] | None,
    page_cv: np.ndarray,
    config: MangaTranslatorConfig,
    sizes: _RenderSizes,
    processing_scale: float,
) -> Image.Image | None:
    """Render one speech bubble translation, or None if it could not be placed.

    When the cleaning mask leaves no safe area, the mask is rebuilt with Otsu and the
    render retried, then a padded bbox render is the final fallback.
    """
    verbose = config.verbose
    bbox = bubble["bbox"]
    log_message(f"Rendering bubble {bbox}: '{text[:30]}...'", verbose=verbose)

    bubble_color_bgr = (255, 255, 255)
    cleaned_mask = None
    base_mask = None
    is_sam_mask = False
    text_color_rgb = None
    if render_info:
        bubble_color_bgr = render_info["color"]
        cleaned_mask = render_info.get("mask")
        base_mask = render_info.get("base_mask")
        is_sam_mask = render_info.get("is_sam", False)
        text_color_bgr = render_info.get("text_color_bgr")
        if text_color_bgr:
            text_color_rgb = (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0])

    render_config = _text_render_config(config, sizes, is_outside_text=False)

    def render(mask, color, raise_on_safe_error: bool) -> Image.Image:
        return render_text_skia(
            pil_image=image,
            text=text,
            bbox=bbox,
            font_dir=config.rendering.font_dir,
            cleaned_mask=mask,
            bubble_color_bgr=color,
            config=render_config,
            verbose=verbose,
            bubble_id=bubble_id,
            rotation_deg=0.0,
            vertical_stack=False,
            text_color_rgb=text_color_rgb,
            raise_on_safe_error=raise_on_safe_error,
        )

    try:
        return render(cleaned_mask, bubble_color_bgr, raise_on_safe_error=True)
    except ImageProcessingError as e:
        safe_area_failed = "Safe area calculation failed" in str(e)
        retry_result = None
        if safe_area_failed and base_mask is not None:
            log_message(
                f"Safe area failed for bubble {bbox}, retrying mask with Otsu",
                verbose=verbose,
                always_print=True,
            )
            retry_result = retry_cleaning_with_otsu(
                page_cv,
                {
                    "base_mask": base_mask,
                    "bbox": bbox,
                    "is_sam": is_sam_mask,
                    "is_colored": (
                        render_info.get("is_colored", False) if render_info else False
                    ),
                    "text_bbox": render_info.get("text_bbox") if render_info else None,
                    "text_color_bgr": (
                        render_info.get("text_color_bgr") if render_info else None
                    ),
                },
                config.cleaning.thresholding_value,
                config.cleaning.roi_shrink_px,
                processing_scale,
                verbose=verbose,
                classify_colored=config.cleaning.inpaint_colored_bubbles,
            )

        if retry_result and retry_result.get("mask") is not None:
            cleaned_mask = retry_result["mask"]
            bubble_color_bgr = retry_result.get("color", bubble_color_bgr)
            base_mask = retry_result.get("base_mask", base_mask)
            if render_info is not None:
                render_info.update(
                    {
                        "mask": cleaned_mask,
                        "color": bubble_color_bgr,
                        "base_mask": base_mask,
                        "is_colored": retry_result.get(
                            "is_colored", render_info.get("is_colored", False)
                        ),
                        "text_bbox": retry_result.get(
                            "text_bbox", render_info.get("text_bbox")
                        ),
                    }
                )
            try:
                return render(cleaned_mask, bubble_color_bgr, raise_on_safe_error=False)
            except (RenderingError, FontError, ImageProcessingError) as e2:
                log_message(
                    f"Text rendering failed after Otsu retry: {e2}", verbose=verbose
                )

        log_message(
            f"Safe area calculation failed for {bbox}, using padded bbox fallback"
            if safe_area_failed
            else f"Rendering retry fallback for {bbox}, using padded bbox method",
            verbose=verbose,
        )
        try:
            return render(cleaned_mask, bubble_color_bgr, raise_on_safe_error=False)
        except (RenderingError, FontError) as e2:
            log_message(f"Text rendering failed: {e2}", verbose=verbose)
            return None
    except (RenderingError, FontError) as e:
        log_message(f"Text rendering failed: {e}", verbose=verbose)
        return None


def _render_translations(
    image: Image.Image,
    sorted_bubble_data: list[dict[str, Any]],
    translated_texts: list[str],
    processed_bubbles_info: list[dict[str, Any]],
    page_cv: np.ndarray,
    config: MangaTranslatorConfig,
    sizes: _RenderSizes,
    processing_scale: float,
    empty_bubbles: list,
    cancellation_manager: Optional["CancellationManager"],
) -> Image.Image:
    """Render each translation onto the cleaned page, recording bubbles left empty."""
    verbose = config.verbose
    render_info_by_bbox = {
        tuple(info["bbox"]): {
            "color": info["color"],
            "mask": info.get("mask"),
            "base_mask": info.get("base_mask"),
            "is_sam": info.get("is_sam", False),
            "is_colored": info.get("is_colored", False),
            "text_bbox": info.get("text_bbox"),
            "text_color_bgr": info.get("text_color_bgr"),
        }
        for info in processed_bubbles_info
        if "bbox" in info and "color" in info and "mask" in info
    }
    log_message("Rendering translations...", verbose=verbose)
    if len(translated_texts) != len(sorted_bubble_data):
        log_message(
            f"Warning: Bubble/translation count mismatch "
            f"({len(sorted_bubble_data)}/{len(translated_texts)})",
            always_print=True,
        )
        empty_bubbles.extend(b["bbox"] for b in sorted_bubble_data)
        return image

    provider_tag = f"[{config.translation.provider}:"
    for i, (bubble, text) in enumerate(
        zip(sorted_bubble_data, translated_texts, strict=True)
    ):
        _raise_if_cancelled(cancellation_manager)
        bubble["translation"] = text
        bbox = bubble["bbox"]
        ocr_text = (bubble.get("ocr_text") or "").strip()
        trimmed_text = (text or "").strip()
        is_outside_text = bubble.get("is_outside_text", False)
        entry_type = "outside text" if is_outside_text else "bubble"

        is_ocr_failed = (
            trimmed_text.upper() == "[OCR FAILED]"
            or ocr_text.upper() == "[OCR FAILED]"
            or ": OCR FAILED]" in trimmed_text.upper()
        )
        if is_ocr_failed:
            if "original_crop_pil" in bubble:
                log_message(
                    f"Restoring original {entry_type} patch due to OCR failure for {bbox}",
                    verbose=verbose,
                    always_print=True,
                )
                image = _paste_original_crop(image, bubble)
            else:
                log_message(
                    f"Skipping {entry_type} {bbox} - OCR failed and no original crop available",
                    verbose=verbose,
                )
                empty_bubbles.append(bbox)
            continue

        if (
            not text
            or text.startswith(
                (
                    "API Error",
                    "[Translation Error]",
                    "[Translation Error:",
                    provider_tag,
                )
            )
            or trimmed_text in INVALID_TRANSLATION_VALUES
        ):
            log_message(
                f"Skipping {entry_type} {bbox} - invalid translation", verbose=verbose
            )
            empty_bubbles.append(bbox)
            continue

        if is_outside_text:
            if (
                ocr_text
                and ocr_text not in INVALID_TRANSLATION_VALUES
                and ocr_text == trimmed_text
                and "original_crop_pil" in bubble
            ):
                log_message(
                    "Restoring original OSB patch because OCR matches translation "
                    f"for {bbox}",
                    verbose=verbose,
                    always_print=True,
                )
                image = _paste_original_crop(image, bubble)
                continue

            text = text.upper()
            bubble["translation"] = text
            rendered = _render_outside_text(
                image, bubble, text, str(i + 1), config, sizes
            )
        else:
            rendered = _render_bubble(
                image,
                bubble,
                text,
                str(i + 1),
                render_info_by_bbox.get(tuple(bbox)),
                page_cv,
                config,
                sizes,
                processing_scale,
            )

        if rendered is None:
            log_message(f"Failed to render bubble {bbox}", verbose=verbose)
            empty_bubbles.append(bbox)
        else:
            image = rendered
    return image


def translate_and_render(
    image_path: str | Path,
    config: MangaTranslatorConfig,
    output_path: str | Path | None = None,
    cancellation_manager: Optional["CancellationManager"] = None,
    previous_context_images: list[dict[str, str]] | None = None,
    previous_context_texts: list[list[str]] | None = None,
    previous_context_texts_provider: Callable[[], list[list[str]]] | None = None,
    ocr_texts_out: list[str] | None = None,
    strip_glyph_size: list[float] | None = None,
    empty_bubbles_out: list | None = None,
):
    """
    Main function to translate manga speech bubbles and render translations using a config object.

    Args:
        image_path (str or Path): Path to input image
        config (MangaTranslatorConfig): Configuration object containing all settings.
        output_path (str or Path, optional): Path to save the final image. If None, image is not saved.
        previous_context_images: Batch-only previous source page images for LLM reference.
        previous_context_texts: Batch-only previous source page OCR transcripts for LLM reference
            (oldest-to-newest, one inner list per page).
        previous_context_texts_provider: Optional callback used to fetch previous-page
            OCR transcripts immediately before the translation API call.
        ocr_texts_out: Optional mutable list. When provided, the current page's OCR transcripts
            (in reading order, each joined to its translation as `source → translation`) are
            appended so the orchestrator can chain them as previous-page text context for
            subsequent pages.
        strip_glyph_size: Optional mutable list shared by the sections of one long strip. The
            first section with measurable lettering stores its glyph size, and later sections
            reuse it so auto font size stays the same along the strip.
        empty_bubbles_out: Optional mutable list. The bbox of every bubble that was cleaned
            but left without rendered text is appended, so batch runs can fail the page.

    Returns:
        PIL.Image: Final translated image
    """
    start_time = time.time()
    validate_config(config)
    image_path = Path(image_path)
    verbose = config.verbose
    previous_context_images = previous_context_images or []
    previous_context_texts = previous_context_texts or []
    empty_bubbles = empty_bubbles_out if empty_bubbles_out is not None else []
    config.translation.request_coordinator = getattr(
        config, "request_coordinator", None
    )

    log_message(f"Using device: {config.device}", verbose=verbose)

    # Set global HF token for model downloads
    get_model_manager().set_hf_token(config.outside_text.huggingface_token)

    pil_original = _open_source_image(image_path)
    image_format = pil_original.format
    mime_type, cv2_ext = get_image_encoding_params(image_format)
    log_message(
        f"Original image format: {image_format} -> MIME: {mime_type}", verbose=verbose
    )
    _raise_if_cancelled(cancellation_manager)

    target_mode = _resolve_target_mode(config, image_path, output_path)
    log_message(f"Target mode: {target_mode}", verbose=verbose)
    page = convert_image_to_target_mode(pil_original, target_mode, verbose)

    if config.preprocessing.split_long_strips and not config.upscaling_only:
        cut_rows = find_cut_rows(
            page, lambda crop: _detect_strip_text_rows(crop, image_path, config)
        )
        if cut_rows:
            return _translate_strip(
                image_path,
                page,
                cut_rows,
                config,
                output_path,
                target_mode,
                cancellation_manager,
                previous_context_images,
                previous_context_texts,
                previous_context_texts_provider,
                ocr_texts_out,
                empty_bubbles_out,
            )

    page, _ = _apply_pre_upscale_if_needed(page, config, verbose)

    if config.upscaling_only:
        log_message(
            "Upscaling only mode - skipping detection and translation",
            always_print=True,
        )
        return _finish_page(page, config, output_path, target_mode, start_time)

    processing_scale = _resolve_processing_scale(page, config)
    get_cache().set_current_image(page, verbose)
    full_page_context_source = page.copy()

    # Detect speech bubbles first so OSB processing can respect bubble regions
    bubble_data, text_free_boxes = _detect_page_bubbles(image_path, page, config)
    _raise_if_cancelled(cancellation_manager)
    panels, debug_panels = _detect_page_panels(image_path, page, config)
    _raise_if_cancelled(cancellation_manager)

    # Outside text is always detected here, but with overlap its inpainting (and bubble
    # cleaning) is deferred to run alongside the LLM call
    overlap = _should_overlap_llm_with_inpaint(config)
    outside_work = None
    if overlap:
        outside_work = prepare_outside_text_work(
            page,
            config,
            image_path,
            image_format,
            verbose,
            bubble_data=bubble_data,
            text_free_boxes=text_free_boxes,
            panels=panels,
        )
        outside_text_data = (
            outside_work.outside_text_data if outside_work is not None else []
        )
    else:
        page, outside_text_data = process_outside_text(
            page,
            config,
            image_path,
            image_format,
            verbose,
            bubble_data=bubble_data,
            text_free_boxes=text_free_boxes,
            panels=panels,
        )
    # With overlap this is the pre-inpaint page, which the bubble and OSB LLM crops use
    page_cv = pil_to_cv2(page)

    full_image_b64 = None
    if config.translation.send_full_page_context:
        full_image_b64 = _encode_full_page_context(
            full_page_context_source, config, cv2_ext
        )
    _raise_if_cancelled(cancellation_manager)

    if not bubble_data and not outside_text_data:
        log_message("No speech bubbles or outside text detected", always_print=True)
        return _finish_page(page, config, output_path, target_mode, start_time)
    if bubble_data:
        log_message(f"Detected {len(bubble_data)} bubbles", verbose=verbose)
    if outside_text_data:
        log_message(
            f"Detected {len(outside_text_data)} outside text regions", verbose=verbose
        )

    if overlap:
        cleaned, processed_bubbles_info = _as_mode(page, target_mode), []
    else:
        cleaned, processed_bubbles_info = _clean_page(
            page, bubble_data, config, processing_scale, target_mode
        )

    if config.cleaning_only:
        log_message("Cleaning only mode - skipping translation", always_print=True)
        return _finish_page(cleaned, config, output_path, target_mode, start_time)

    sizes = _scaled_render_sizes(config, processing_scale)
    bubble_data = _prepare_bubble_crops(
        bubble_data, processed_bubbles_info, page_cv, config, mime_type
    )
    finish_deferred_cleaning = partial(
        _finish_deferred_cleaning,
        outside_work,
        page,
        bubble_data,
        config,
        processing_scale,
        target_mode,
    )
    valid_bubble_data = [b for b in bubble_data if b.get("image_b64")]
    if not valid_bubble_data and not outside_text_data:
        log_message(
            "No valid bubble images or outside text for translation", always_print=True
        )
        if overlap:
            cleaned, _ = finish_deferred_cleaning()
        return _finish_page(cleaned, config, output_path, target_mode, start_time)
    _raise_if_cancelled(cancellation_manager)

    sorted_bubble_data = _sort_text_elements(
        valid_bubble_data, outside_text_data, config, panels
    )
    if ENABLE_COMPONENT_ORDER_DEBUG:
        write_component_order_debug(
            sorted_bubble_data,
            processed_bubbles_info,
            page.size,
            debug_panels,
            config.translation.reading_direction,
            image_path,
            output_path,
            verbose=verbose,
        )

    bubble_images_b64 = [
        bubble["image_b64"] for bubble in sorted_bubble_data if "image_b64" in bubble
    ]
    bubble_mime_types = [
        bubble["mime_type"]
        for bubble in sorted_bubble_data
        if "image_b64" in bubble and "mime_type" in bubble
    ]
    translated_texts: list[str] = []
    current_ocr_texts: list[str] = []
    if not bubble_images_b64:
        log_message("No valid bubbles after sorting", always_print=True)
        if overlap:
            cleaned, processed_bubbles_info = finish_deferred_cleaning()
    elif config.test_mode:
        # Size placeholders against the same max the renderer will use
        sizes = sizes._replace(
            main_max_font=_resolve_main_max_font(
                config,
                page_cv,
                processed_bubbles_info,
                sizes.main_min_font,
                sizes.main_max_font,
                processing_scale,
                strip_glyph_size,
                verbose,
            )
        )
        translated_texts = generate_test_placeholders(
            sorted_bubble_data=sorted_bubble_data,
            processed_bubbles_info=processed_bubbles_info,
            config=config,
            verbose=verbose,
            **sizes._asdict(),
        )
    else:
        call_api = partial(
            _call_translation_api,
            config,
            sorted_bubble_data,
            bubble_images_b64,
            bubble_mime_types,
            full_image_b64,
            mime_type,
            previous_context_images,
            previous_context_texts,
            previous_context_texts_provider,
        )
        translate = partial(
            _translate_or_error_texts,
            sorted_bubble_data=sorted_bubble_data,
            image_count=len(bubble_images_b64),
            config=config,
            ocr_texts_out=ocr_texts_out,
        )
        if overlap:
            log_message(
                "Running LLM translation concurrently with inpainting",
                always_print=True,
            )
            with ThreadPoolExecutor(max_workers=2) as overlap_executor:
                inpaint_future = overlap_executor.submit(finish_deferred_cleaning)
                translate_future = overlap_executor.submit(call_api)
                try:
                    cleaned, processed_bubbles_info = inpaint_future.result()
                except Exception:
                    translate_future.cancel()
                    raise
                translated_texts, current_ocr_texts = translate(translate_future.result)
        else:
            translated_texts, current_ocr_texts = translate(call_api)

    if current_ocr_texts:
        _attach_ocr_texts(sorted_bubble_data, current_ocr_texts, verbose)

    sizes = sizes._replace(
        main_max_font=_resolve_main_max_font(
            config,
            page_cv,
            processed_bubbles_info,
            sizes.main_min_font,
            sizes.main_max_font,
            processing_scale,
            strip_glyph_size,
            verbose,
        )
    )
    cleaned = _render_translations(
        cleaned,
        sorted_bubble_data,
        translated_texts,
        processed_bubbles_info,
        page_cv,
        config,
        sizes,
        processing_scale,
        empty_bubbles,
        cancellation_manager,
    )
    return _finish_page(cleaned, config, output_path, target_mode, start_time)


def _resolve_output_path(
    img_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: MangaTranslatorConfig,
    preserve_structure: bool,
) -> tuple[Path, str, str]:
    """Compute output path, display name, and error key for a single image."""
    if preserve_structure:
        relative_path = img_path.relative_to(input_dir)
        output_subdir = output_dir / relative_path.parent
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f"{relative_path.stem}_translated"
        display_path = str(relative_path)
        error_key = str(relative_path)
    else:
        output_subdir = output_dir
        output_filename = f"{img_path.stem}_translated"
        display_path = img_path.name
        error_key = img_path.name

    original_ext = img_path.suffix.lower()
    desired_format = config.output.output_format
    if desired_format == "jpeg":
        output_ext = ".jpg"
    elif desired_format == "png":
        output_ext = ".png"
    elif desired_format == "auto":
        output_ext = original_ext
    else:
        output_ext = original_ext
        log_message(
            f"Warning: Invalid output_format '{desired_format}' in config. "
            f"Using original extension '{original_ext}'.",
            always_print=True,
        )

    return output_subdir / f"{output_filename}{output_ext}", display_path, error_key


def _check_glossary_path(path: Path) -> None:
    """Fail the batch once up front instead of on every page."""
    if path.is_dir():
        raise ValidationError(f"Glossary file '{path}' is a folder, not a file.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        load_terms(path)
    except (OSError, UnicodeDecodeError) as e:
        raise ValidationError(f"Cannot use glossary file '{path}': {e}") from e


def _translate_batch_page(*args, **kwargs) -> None:
    """Translate one batch page, failing it if any bubble was left empty.

    The partial result is still saved; a later retry of the page overwrites it.
    """
    empty_bubbles: list = []
    translate_and_render(*args, empty_bubbles_out=empty_bubbles, **kwargs)
    if empty_bubbles:
        raise TranslationError(f"{len(empty_bubbles)} bubble(s) left empty")


def _should_run_failed_retry(
    config: MangaTranslatorConfig,
    failed_jobs: list[dict[str, Any]],
    cancellation_manager: Optional["CancellationManager"] = None,
) -> bool:
    if not getattr(config, "retry_failed_once", False):
        return False
    if not failed_jobs:
        return False
    return not (
        cancellation_manager is not None and cancellation_manager.is_cancelled()
    )


def _retry_failed_batch_images(
    failed_jobs: list[dict[str, Any]],
    results: dict[str, Any],
    config: MangaTranslatorConfig,
    input_dir: Path,
    output_dir: Path,
    preserve_structure: bool = False,
    progress_callback: Callable[[float, str], None] | None = None,
    cancellation_manager: Optional["CancellationManager"] = None,
    source_path_map: dict[str, str] | None = None,
) -> None:
    """Retry failed batch images once. Mutates *results* in place.

    Each entry in *failed_jobs* must include:
      - error_key: key used in results["errors"]
      - img_path: processing Path passed to translate_and_render
      - source_path: path stored in results["failed_image_paths"]
    """
    if not failed_jobs:
        results.setdefault("retry_attempted_count", 0)
        results.setdefault("retry_success_count", 0)
        results.setdefault("retry_failed_count", 0)
        return

    if cancellation_manager is not None and cancellation_manager.is_cancelled():
        results.setdefault("retry_attempted_count", 0)
        results.setdefault("retry_success_count", 0)
        results.setdefault("retry_failed_count", 0)
        return

    total = len(failed_jobs)
    attempted = 0
    retry_success = 0
    retry_failed = 0

    log_message(
        f"Retrying {total} failed image(s) once...",
        always_print=True,
    )

    for i, job in enumerate(failed_jobs):
        if cancellation_manager is not None and cancellation_manager.is_cancelled():
            log_message(
                "Batch retry cancelled by user; remaining retries skipped.",
                always_print=True,
            )
            break

        error_key = job["error_key"]
        img_path = Path(job["img_path"])
        source_path = job.get("source_path")
        if source_path is None:
            source_path = resolve_source_path(img_path, source_path_map)

        output_path, display_path, _ = _resolve_output_path(
            img_path, input_dir, output_dir, config, preserve_structure
        )

        attempted += 1
        try:
            if progress_callback:
                # Keep retries in the high progress band so UI still looks near-complete.
                frac = 0.95 + 0.05 * (i / max(total, 1))
                progress_callback(
                    frac,
                    f"Retrying failed image {i + 1}/{total}: {display_path}",
                )

            log_message(
                f"Retrying {i + 1}/{total}: {display_path}",
                always_print=True,
            )

            if cancellation_manager is not None and cancellation_manager.is_cancelled():
                attempted -= 1
                break
            _translate_batch_page(
                img_path,
                config,
                output_path,
                cancellation_manager=cancellation_manager,
            )
            results["success_count"] = int(results.get("success_count", 0)) + 1
            results["error_count"] = max(0, int(results.get("error_count", 0)) - 1)
            errors = results.setdefault("errors", {})
            errors.pop(error_key, None)
            failed_paths = list(results.get("failed_image_paths") or [])
            before_len = len(failed_paths)
            failed_paths = [p for p in failed_paths if p != source_path]
            if len(failed_paths) == before_len:
                log_message(
                    f"Retry success path cleanup: source_path not in "
                    f"failed_image_paths ({source_path!r})",
                    always_print=True,
                )
            results["failed_image_paths"] = failed_paths
            retry_success += 1
            log_message(
                f"Retry succeeded: {display_path}",
                always_print=True,
            )
        except CancellationError:
            attempted -= 1
            log_message(
                f"Batch retry cancelled during {display_path}; "
                "remaining retries skipped.",
                always_print=True,
            )
            break
        except Exception as e:
            results.setdefault("errors", {})[error_key] = str(e)
            retry_failed += 1
            log_message(
                f"Retry failed for {display_path}: {e}",
                always_print=True,
            )

    results["retry_attempted_count"] = attempted
    results["retry_success_count"] = retry_success
    results["retry_failed_count"] = retry_failed

    if progress_callback and attempted > 0:
        try:
            progress_callback(
                0.99,
                f"Retry complete: {retry_success} recovered, {retry_failed} still failed",
            )
        except CancellationError:
            log_message(
                "Batch retry progress callback cancelled after stats flush.",
                always_print=True,
            )


async def _batch_translate_pages(
    image_files: list[Path],
    input_dir: Path,
    config: MangaTranslatorConfig,
    output_dir: Path,
    preserve_structure: bool,
    progress_callback: Callable[[float, str], None] | None,
    cancellation_manager: Optional["CancellationManager"],
    source_path_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Process images with up to `config.parallel_requests` pages in flight.

    The first image is processed alone to warm up all ML models (triggering
    lazy loading and YOLO layer fusing on the main thread). Remaining images are
    then processed in parallel with models already initialized; with one worker
    they run in order.
    """
    total_images = len(image_files)
    n_workers = config.parallel_requests
    results = {
        "success_count": 0,
        "error_count": 0,
        "errors": {},
        "failed_image_paths": [],
        "_failed_jobs": [],
    }
    previous_context_cache = OrderedDict()
    previous_context_cache_lock = threading.Lock()
    ocr_text_history: dict[Path, list[str]] = {}
    ocr_text_history_lock = threading.Lock()
    ocr_text_ready_events = [threading.Event() for _ in image_files]
    requested_text_context_count = int(
        getattr(config.translation, "previous_context_text_count", 0) or 0
    )

    log_message(
        f"Starting batch processing: {total_images} images, {n_workers} worker(s)",
        always_print=True,
    )

    def _record_failure(img_path: Path, display_path: str, error_key: str, error):
        log_message(f"Error processing {display_path}: {error!s}", always_print=True)
        source_path = resolve_source_path(img_path, source_path_map)
        results["error_count"] += 1
        results["errors"][error_key] = str(error)
        results["failed_image_paths"].append(source_path)
        results["_failed_jobs"].append(
            {"error_key": error_key, "img_path": img_path, "source_path": source_path}
        )

    if n_workers > 1 and getattr(config, "batch_parallel_within_pages", False):
        request_coordinator = BatchRequestCoordinator(
            n_workers, cancellation_manager=cancellation_manager
        )
        config.request_coordinator = request_coordinator
        config.translation.request_coordinator = request_coordinator
        log_message(
            "Intra-page parallel requests enabled",
            always_print=True,
        )
    else:
        config.request_coordinator = None
        config.translation.request_coordinator = None

    # -- Phase 1: process the first image sequentially to warm up models --
    first_img = image_files[0]
    first_output, first_display, first_key = _resolve_output_path(
        first_img, input_dir, output_dir, config, preserve_structure
    )
    log_message(
        f"Processing 1/{total_images}: {first_display} (warming up models)",
        always_print=True,
    )
    try:
        if cancellation_manager and cancellation_manager.is_cancelled():
            raise CancellationError("Batch process cancelled by user.")
        if progress_callback:
            progress_callback(
                0.0, f"Processing image 1/{total_images}: {first_display}"
            )
        first_previous_context_images = _build_previous_context_images(
            image_files,
            0,
            config,
            previous_context_cache,
            previous_context_cache_lock,
        )
        first_previous_context_texts = _build_previous_context_texts(
            image_files,
            0,
            config,
            ocr_text_history,
            ocr_text_history_lock,
        )
        first_ocr_texts: list[str] = []
        _translate_batch_page(
            first_img,
            config,
            first_output,
            cancellation_manager=cancellation_manager,
            previous_context_images=first_previous_context_images,
            previous_context_texts=first_previous_context_texts,
            ocr_texts_out=first_ocr_texts,
        )
        if first_ocr_texts:
            with ocr_text_history_lock:
                ocr_text_history[first_img] = first_ocr_texts
        results["success_count"] += 1
    except CancellationError:
        raise
    except Exception as e:
        _record_failure(first_img, first_display, first_key, e)
    finally:
        ocr_text_ready_events[0].set()

    completed_count = 1
    if progress_callback:
        has_errors = results["error_count"] > 0
        suffix = " (with errors)" if has_errors else ""
        progress_callback(
            1 / total_images, f"Completed 1/{total_images} images{suffix}"
        )

    # -- Phase 2: process remaining images in parallel --
    remaining = image_files[1:]
    if not remaining:
        return results

    if cancellation_manager and cancellation_manager.is_cancelled():
        raise CancellationError("Batch process cancelled by user.")

    sem = asyncio.Semaphore(n_workers)
    results_lock = threading.Lock()
    cancelled = False

    def _wait_for_required_previous_ocr(index: int) -> None:
        if requested_text_context_count <= 0:
            return
        start_index = max(0, index - requested_text_context_count)
        for previous_index in range(start_index, index):
            if cancellation_manager and cancellation_manager.is_cancelled():
                raise CancellationError("Batch process cancelled by user.")
            while not ocr_text_ready_events[previous_index].wait(timeout=0.2):
                if cancellation_manager and cancellation_manager.is_cancelled():
                    raise CancellationError("Batch process cancelled by user.")
            if cancellation_manager and cancellation_manager.is_cancelled():
                raise CancellationError("Batch process cancelled by user.")

    def _process_single(img_path: Path, index: int) -> tuple[str, str]:
        """Run translate_and_render for a single image. Returns (display_path, error_key)."""
        output_path, display_path, error_key = _resolve_output_path(
            img_path, input_dir, output_dir, config, preserve_structure
        )
        log_message(
            f"Processing {index + 1}/{total_images}: {display_path}",
            always_print=True,
        )
        previous_context_images = _build_previous_context_images(
            image_files,
            index,
            config,
            previous_context_cache,
            previous_context_cache_lock,
        )

        def previous_context_texts_provider() -> list[list[str]]:
            _wait_for_required_previous_ocr(index)
            return _build_previous_context_texts(
                image_files,
                index,
                config,
                ocr_text_history,
                ocr_text_history_lock,
            )

        captured_ocr_texts: list[str] = []
        _translate_batch_page(
            img_path,
            config,
            output_path,
            cancellation_manager=cancellation_manager,
            previous_context_images=previous_context_images,
            previous_context_texts_provider=previous_context_texts_provider,
            ocr_texts_out=captured_ocr_texts,
        )
        if captured_ocr_texts:
            with ocr_text_history_lock:
                ocr_text_history[img_path] = captured_ocr_texts
        return display_path, error_key

    async def _worker(img_path: Path, index: int, executor: ThreadPoolExecutor):
        nonlocal completed_count, cancelled
        try:
            if cancelled or (
                cancellation_manager and cancellation_manager.is_cancelled()
            ):
                cancelled = True
                return

            async with sem:
                if cancelled or (
                    cancellation_manager and cancellation_manager.is_cancelled()
                ):
                    cancelled = True
                    return

                if progress_callback:
                    _, display_name, _ = _resolve_output_path(
                        img_path, input_dir, output_dir, config, preserve_structure
                    )
                    progress_callback(
                        completed_count / total_images,
                        f"Processing image {index + 1}/{total_images}: {display_name}",
                    )

                loop = asyncio.get_running_loop()
                try:
                    if n_workers == 1:
                        # On the main thread, so Ctrl+C stops the batch at once
                        # instead of waiting for a worker thread's page to finish
                        _process_single(img_path, index)
                    else:
                        await loop.run_in_executor(
                            executor, _process_single, img_path, index
                        )
                    with results_lock:
                        results["success_count"] += 1
                        completed_count += 1
                        count = completed_count
                except CancellationError:
                    cancelled = True
                    raise
                except Exception as e:
                    _, display_path, error_key = _resolve_output_path(
                        img_path, input_dir, output_dir, config, preserve_structure
                    )
                    with results_lock:
                        _record_failure(img_path, display_path, error_key, e)
                        completed_count += 1
                        count = completed_count

                if progress_callback:
                    progress = count / total_images
                    has_errors = results["error_count"] > 0
                    suffix = " (with errors)" if has_errors else ""
                    progress_callback(
                        progress, f"Completed {count}/{total_images} images{suffix}"
                    )
        except CancellationError:
            cancelled = True
            raise
        finally:
            ocr_text_ready_events[index].set()

            if cancelled:
                for event in ocr_text_ready_events:
                    event.set()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        tasks = [_worker(img, i, executor) for i, img in enumerate(remaining, start=1)]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

    for exc in gathered:
        if isinstance(exc, CancellationError):
            raise exc

    return results


def batch_translate_images(
    input_dir: str | Path,
    config: MangaTranslatorConfig,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    preserve_structure: bool = False,
    cancellation_manager: Optional["CancellationManager"] = None,
    source_path_map: dict[str, str] | None = None,
    glossary_name: str | None = None,
    glossary_path: str | None = None,
) -> dict[str, Any]:
    """
    Process all images in a directory using a configuration object.

    Args:
        input_dir (str or Path): Directory containing images to process
        config (MangaTranslatorConfig): Configuration object containing all settings.
        output_dir (str or Path, optional): Directory to save translated images.
                                            If None, uses input_dir / "output_translated".
        progress_callback (callable, optional): Function to call with progress updates (0.0-1.0, message).
        preserve_structure (bool): If True, recursively process subdirectories and preserve folder structure
                                   in the output. If False, only processes files in the root directory.
        source_path_map: Optional mapping from processing paths to original source paths
            (used when UI/CLI copies inputs into a temp directory).
        glossary_name: Name for a new terminology memory file (the input folder or ZIP
            name). Defaults to the first image's stem.
        glossary_path: Existing terminology memory file to reuse (a retried run's
            glossary) when the config doesn't set one; takes precedence over
            glossary_name.

    Returns:
        dict: Processing results with keys:
            - "success_count": Number of successfully processed images
            - "error_count": Number of images that failed to process
            - "errors": Dictionary mapping filenames to error messages
            - "failed_image_paths": Absolute source paths of failed images
            - "failed_paths_file": Path to failed_paths.txt when written
    """
    empty_results = {
        "success_count": 0,
        "error_count": 0,
        "errors": {},
        "failed_image_paths": [],
    }

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        log_message(f"Input path '{input_dir}' is not a directory", always_print=True)
        return empty_results

    if output_dir:
        output_dir = Path(output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./output") / timestamp

    os.makedirs(output_dir, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    if preserve_structure:
        image_files = []
        for root, _dirs, files in os.walk(input_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
    else:
        image_files = [
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

    def _batch_sort_key(path: Path):
        try:
            sort_path = (
                path.relative_to(input_dir) if preserve_structure else Path(path.name)
            )
        except ValueError:
            sort_path = path
        return natural_path_sort_key(sort_path)

    image_files.sort(key=_batch_sort_key)

    if not image_files:
        log_message(f"No image files found in '{input_dir}'", always_print=True)
        return empty_results

    translation_cfg = config.translation
    translates = not (config.test_mode or config.cleaning_only or config.upscaling_only)
    if not (translates and translation_cfg.terminology_memory):
        glossary_path = None
    elif translation_cfg.terminology_memory_path:
        glossary_path = translation_cfg.terminology_memory_path
    else:
        glossary_path = glossary_path or str(
            default_glossary_path(glossary_name or image_files[0].stem)
        )
        config = replace(
            config,
            translation=replace(translation_cfg, terminology_memory_path=glossary_path),
        )
    if glossary_path:
        _check_glossary_path(Path(glossary_path))
        log_message(f"Terminology memory: {glossary_path}", always_print=True)

    total_images = len(image_files)
    start_batch_time = time.time()

    if progress_callback:
        progress_callback(0.0, f"Starting batch processing of {total_images} images...")

    results = asyncio.run(
        _batch_translate_pages(
            image_files=image_files,
            input_dir=input_dir,
            config=config,
            output_dir=output_dir,
            preserve_structure=preserve_structure,
            progress_callback=progress_callback,
            cancellation_manager=cancellation_manager,
            source_path_map=source_path_map,
        )
    )
    failed_jobs = results.pop("_failed_jobs", [])

    if _should_run_failed_retry(config, failed_jobs, cancellation_manager):
        _retry_failed_batch_images(
            failed_jobs=failed_jobs,
            results=results,
            config=config,
            input_dir=input_dir,
            output_dir=output_dir,
            preserve_structure=preserve_structure,
            progress_callback=progress_callback,
            cancellation_manager=cancellation_manager,
            source_path_map=source_path_map,
        )

    copy_comic_info(
        input_dir, output_dir, config.translation.output_language, preserve_structure
    )

    if progress_callback:
        progress_callback(1.0, "Processing complete")

    end_batch_time = time.time()
    total_batch_time = end_batch_time - start_batch_time
    seconds_per_image = total_batch_time / total_images if total_images > 0 else 0

    log_message(
        f"Batch complete: {results['success_count']}/{total_images} images in "
        f"{total_batch_time:.2f}s ({seconds_per_image:.2f}s/image)",
        always_print=True,
    )
    if results.get("retry_attempted_count"):
        log_message(
            f"Retry pass: {results.get('retry_success_count', 0)} recovered, "
            f"{results.get('retry_failed_count', 0)} still failed "
            f"(of {results['retry_attempted_count']} attempted)",
            always_print=True,
        )
    if results["error_count"] > 0:
        log_message(f"Failed: {results['error_count']} images", always_print=True)
        for filename, error_msg in results["errors"].items():
            log_message(f"  - {filename}: {error_msg}", always_print=True)

    if glossary_path:
        results["glossary_path"] = glossary_path

    still_failed = set(results.get("failed_image_paths") or [])
    failures = [
        (
            Path(job["img_path"]),
            Path(job["img_path"]).relative_to(input_dir)
            if preserve_structure
            else Path(Path(job["img_path"]).name),
        )
        for job in failed_jobs
        if job["source_path"] in still_failed
    ]
    succeeded = [
        source
        for source in (resolve_source_path(p, source_path_map) for p in image_files)
        if source not in still_failed
    ]
    failed_file = record_failed_images(output_dir, failures, succeeded, glossary_path)
    if failed_file:
        results["failed_paths_file"] = str(failed_file)
        log_message(
            f"Failed image paths saved to: {failed_file}",
            always_print=True,
        )

    return results
