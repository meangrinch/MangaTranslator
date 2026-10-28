from dataclasses import dataclass, field
from functools import partial

import cv2
import numpy as np
from PIL import Image

from core.config import MangaTranslatorConfig
from core.image.glyph_mask import (
    UpscaledGlyphMap,
    composite_glyphs,
    glyph_dilation_radius,
    page_glyph_probability,
    refine_box_mask_to_glyphs,
    refine_mask_to_glyphs,
)
from core.ml.hi_sam import predict_text_probability
from core.ml.model_manager import get_model_manager
from utils.exceptions import CancellationError
from utils.logging import log_message


@dataclass
class PageGlyphMap:
    """Text probability of the original page, computed on first use.

    Bubble cleaning, OSB tilt detection, LaMa's glyph masks, and the edit models' glyph
    composites share it, so the text segmentation model runs at most once per page, and
    not at all on pages that need none of them. The model stays on the device like the
    app's other models until a diffusion inpainter parks it to make room.
    """

    image: Image.Image
    verbose: bool = False
    # LaMa masks add text found in each box at twice the text scale
    upscale_boxes: bool = False
    # Off by the user's choice; consumers then get no map, as when the model fails
    enabled: bool = True
    prob: np.ndarray | None = None
    failed: bool = False
    _upscaled: UpscaledGlyphMap | None = field(default=None, init=False, repr=False)

    @classmethod
    def for_page(
        cls, image: Image.Image, config: MangaTranslatorConfig, verbose: bool
    ) -> "PageGlyphMap":
        return cls(
            image,
            verbose=verbose,
            upscale_boxes=config.outside_text.inpainting_method == "lama_manga",
            enabled=config.cleaning.text_segmentation,
        )

    def get(self) -> np.ndarray | None:
        """The probability map, or None when the text model can't be loaded or run."""
        if self.prob is None and self.enabled and not self.failed:
            manager = get_model_manager()
            with manager.flux_inference_lock:
                # Another caller may have built it while this one waited
                if self.prob is None and not self.failed:
                    self._compute(manager)
        return self.prob

    def _compute(self, manager) -> None:
        try:
            # No local reference to the model, so unloading it frees its memory
            self.prob = page_glyph_probability(
                self.image,
                partial(
                    predict_text_probability,
                    manager.load_text_sam(verbose=self.verbose),
                ),
            )
        except CancellationError:
            raise
        except Exception as e:
            self.failed = True
            log_message(
                f"Text segmentation unavailable ({e}); cleaning and inpainting "
                "without glyph masks, and using heuristic tilt",
                always_print=True,
            )

    def box_probability(self, box: tuple[int, int, int, int]) -> np.ndarray | None:
        """Text probability inside `box` at twice the text scale, or None when that
        pass doesn't run for this page."""
        if not self.upscale_boxes or self.get() is None:
            return None
        manager = get_model_manager()
        with manager.flux_inference_lock:
            try:
                if self._upscaled is None:
                    self._upscaled = UpscaledGlyphMap(self.image)
                return self._upscaled.box_probability(
                    box,
                    partial(
                        predict_text_probability,
                        manager.load_text_sam(verbose=self.verbose),
                    ),
                )
            except CancellationError:
                raise
            except Exception as e:
                self.upscale_boxes = False
                self._upscaled = None
                log_message(
                    f"Upscaled text segmentation failed ({e}); LaMa masks use the "
                    "page map alone",
                    always_print=True,
                )
                return None

    def lama_mask(self, box_mask: np.ndarray) -> np.ndarray:
        """`box_mask` shrunk to its dilated glyphs, including those found inside its
        bounds at twice the text scale; `box_mask` itself without a map."""
        glyph_prob = self.get()
        if glyph_prob is None:
            return box_mask
        radius = glyph_dilation_radius(self.image.size)
        x, y, w, h = cv2.boundingRect(np.asarray(box_mask, np.uint8))
        box = (x, y, x + w, y + h)
        box_prob = self.box_probability(box) if w and h else None
        if box_prob is None:
            return refine_mask_to_glyphs(box_mask, glyph_prob, radius)
        return refine_box_mask_to_glyphs(box_mask, glyph_prob, box, box_prob, radius)

    def composite(
        self, before: Image.Image, after: Image.Image, box_mask: np.ndarray
    ) -> Image.Image:
        """`after` pasted over `before` only on the glyphs of each `box_mask` region;
        `after` itself without a map."""
        glyph_prob = self.get()
        if glyph_prob is None:
            return after
        out = composite_glyphs(
            np.asarray(before),
            np.asarray(after),
            box_mask,
            glyph_prob,
            glyph_dilation_radius(self.image.size),
        )
        return Image.fromarray(out)
