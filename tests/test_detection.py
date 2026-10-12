import cv2
import numpy as np

from core.image.detection import _pad_parent_mask_with_uncovered_boxes


def test_conjoined_child_box_is_padded_only_when_parent_mask_barely_covers_it():
    parent = np.zeros((100, 200), np.uint8)
    cv2.ellipse(parent, (50, 50), (40, 40), 0, 0, 360, 255, -1)
    covered_box = [10, 10, 90, 90]
    uncovered_box = [120, 10, 190, 90]

    padded = _pad_parent_mask_with_uncovered_boxes(
        parent, [covered_box, uncovered_box], 100, 200
    )

    assert padded[50, 50]
    assert not padded[12, 12]
    assert padded[50, 150]
