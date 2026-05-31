from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from PIL import Image

from utils.exceptions import CancellationError

T = TypeVar("T")
R = TypeVar("R")
BBox = Tuple[int, int, int, int]


class BatchRequestCoordinator:
    """Shared request budget for batch LLM and Flux work."""

    def __init__(self, max_requests: int, cancellation_manager=None):
        self.max_requests = max(1, int(max_requests or 1))
        self._semaphore = threading.BoundedSemaphore(self.max_requests)
        self._cancellation_manager = cancellation_manager
        self._local = threading.local()

    def _check_cancelled(self) -> None:
        if (
            self._cancellation_manager is not None
            and self._cancellation_manager.is_cancelled()
        ):
            raise CancellationError("Batch process cancelled by user.")

    def in_slot(self) -> bool:
        return getattr(self._local, "depth", 0) > 0

    @contextmanager
    def slot(self):
        """Acquire one shared request slot, with same-thread re-entry allowed."""
        if self.in_slot():
            yield
            return

        self._check_cancelled()
        self._semaphore.acquire()
        self._local.depth = 1
        try:
            self._check_cancelled()
            yield
        finally:
            self._local.depth = 0
            self._semaphore.release()

    def run(self, fn: Callable[..., R], *args, **kwargs) -> R:
        with self.slot():
            return fn(*args, **kwargs)

    def map_ordered(self, jobs: Sequence[Callable[[], R]]) -> List[R]:
        """Run jobs under the shared budget and return results in input order."""
        if not jobs:
            return []
        if len(jobs) == 1:
            return [self.run(jobs[0])]

        max_workers = min(len(jobs), self.max_requests)
        results: List[Optional[R]] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                (index, executor.submit(self.run, job))
                for index, job in enumerate(jobs)
            ]
            for index, future in futures:
                results[index] = future.result()

        return [result for result in results]  # type: ignore[list-item]


def bboxes_overlap(first: BBox, second: BBox) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def expanded_mask_bbox(
    mask: np.ndarray,
    image_size: Tuple[int, int],
    padding_ratio: float = 0.5,
    max_padding: int = 160,
    min_padding: int = 64,
    extra_padding: int = 16,
) -> Optional[BBox]:
    """Return a conservative bbox for Flux context and compositing."""
    mask_np = np.asarray(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    mask_bool = mask_np.astype(bool)
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return None

    img_w, img_h = image_size
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    max_side = max(x2 - x1, y2 - y1)
    padding = max(min_padding, int(min(max_side * padding_ratio, max_padding)))
    padding += extra_padding

    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(img_w, x2 + padding),
        min(img_h, y2 + padding),
    )


def partition_non_overlapping_waves(
    items: Iterable[T],
    get_bbox: Callable[[T], Optional[BBox]],
) -> List[List[T]]:
    """Partition items into ordered waves with no overlapping bboxes per wave."""
    waves: List[List[T]] = []
    current_wave: List[T] = []
    current_bboxes: List[BBox] = []

    for item in items:
        bbox = get_bbox(item)
        if bbox is None:
            if current_wave:
                waves.append(current_wave)
                current_wave = []
                current_bboxes = []
            waves.append([item])
            continue

        if current_bboxes and any(
            bboxes_overlap(bbox, other) for other in current_bboxes
        ):
            waves.append(current_wave)
            current_wave = []
            current_bboxes = []

        current_wave.append(item)
        current_bboxes.append(bbox)

    if current_wave:
        waves.append(current_wave)

    return waves


def paste_image_region(
    target: Image.Image,
    source: Image.Image,
    bbox: BBox,
) -> Image.Image:
    """Paste one source bbox into a copy of target."""
    next_image = target.copy()
    next_image.paste(source.crop(bbox), (bbox[0], bbox[1]))
    return next_image
