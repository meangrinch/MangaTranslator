import threading
import time

import numpy as np
import pytest
from PIL import Image

from core.batch_coordinator import (
    BatchRequestCoordinator,
    bboxes_overlap,
    expanded_mask_bbox,
    partition_non_overlapping_waves,
    paste_image_region,
)
from utils.exceptions import CancellationError


def test_bboxes_overlap_excludes_touching_edges():
    assert bboxes_overlap((0, 0, 10, 10), (5, 5, 15, 15))
    assert not bboxes_overlap((0, 0, 10, 10), (10, 0, 20, 10))
    assert not bboxes_overlap((0, 0, 10, 10), (0, 10, 10, 20))


def test_partition_waves_splits_on_overlap_and_preserves_order():
    boxes = {"a": (0, 0, 10, 10), "b": (20, 0, 30, 10), "c": (5, 5, 15, 15)}
    waves = partition_non_overlapping_waves(["a", "b", "c"], boxes.get)
    assert waves == [["a", "b"], ["c"]]


def test_partition_waves_isolates_items_without_bbox():
    boxes = {"a": (0, 0, 10, 10), "b": (20, 0, 30, 10)}
    waves = partition_non_overlapping_waves(["a", "x", "b"], boxes.get)
    assert waves == [["a"], ["x"], ["b"]]


def test_map_ordered_keeps_input_order_and_respects_budget():
    coordinator = BatchRequestCoordinator(max_requests=2)
    lock = threading.Lock()
    active = 0
    peak = 0

    def make_job(i):
        def job():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02 * (5 - i))
            with lock:
                active -= 1
            return i

        return job

    assert coordinator.map_ordered([make_job(i) for i in range(5)]) == list(range(5))
    assert peak <= 2


def test_slot_is_reentrant_on_same_thread():
    coordinator = BatchRequestCoordinator(max_requests=1)
    assert coordinator.run(lambda: coordinator.run(lambda: "inner")) == "inner"


def test_run_raises_when_cancelled():
    class Cancelled:
        def is_cancelled(self):
            return True

    coordinator = BatchRequestCoordinator(2, cancellation_manager=Cancelled())
    with pytest.raises(CancellationError):
        coordinator.run(lambda: None)


@pytest.mark.parametrize("value", [0, None, -3])
def test_max_requests_floor_is_one(value):
    assert BatchRequestCoordinator(value).max_requests == 1


def test_expanded_mask_bbox_pads_and_clamps():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:60, 50:60] = 255
    # min_padding (64) + extra_padding (16) = 80 on each side, clamped to image.
    assert expanded_mask_bbox(mask, (200, 200)) == (0, 0, 140, 140)
    assert expanded_mask_bbox(np.stack([mask] * 3, axis=-1), (200, 200)) == (
        0,
        0,
        140,
        140,
    )


def test_expanded_mask_bbox_empty_mask():
    assert expanded_mask_bbox(np.zeros((10, 10)), (10, 10)) is None


def test_paste_image_region_copies_only_bbox():
    target = Image.new("RGB", (10, 10), "black")
    source = Image.new("RGB", (10, 10), "white")
    result = paste_image_region(target, source, (2, 2, 5, 5))
    assert result.getpixel((3, 3)) == (255, 255, 255)
    assert result.getpixel((6, 6)) == (0, 0, 0)
    assert target.getpixel((3, 3)) == (0, 0, 0)
