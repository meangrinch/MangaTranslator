import cv2
import numpy as np
import pytest

from core.image.cleaning import _build_adaptive_shrink_mask, process_single_bubble
from utils.exceptions import CleaningError


@pytest.mark.parametrize("processing_scale", [1.0, 1.04, 2.0])
def test_junction_zone_keeps_roi_edge_at_any_processing_scale(processing_scale):
    roi = np.zeros((100, 100), np.uint8)
    roi[20:80, 20:60] = 255

    shrunk = _build_adaptive_shrink_mask(
        roi, 5.0, (20, 20, 60, 80), [(55, 20, 95, 80)], processing_scale
    )

    assert shrunk[50, 59] == 255  # edge pixel facing the conjoined neighbor
    assert shrunk[50, 20] == 0  # far edge still gets the normal shrink


def test_auto_threshold_scales_fixed_value_to_a_gray_bubble_background():
    gray = np.full((200, 200), 60, np.uint8)
    cv2.ellipse(gray, (100, 100), (70, 80), 0, 0, 360, 170, -1)
    cv2.ellipse(gray, (100, 100), (70, 80), 0, 0, 360, 0, 3)
    cv2.rectangle(gray, (90, 70), (110, 130), 0, -1)
    base = np.zeros_like(gray)
    cv2.ellipse(base, (100, 100), (70, 80), 0, 0, 360, 255, -1)

    with pytest.raises(CleaningError):
        process_single_bubble(base, gray, 200, 200, 200, False, 5, False)

    mask = process_single_bubble(
        base, gray, 200, 200, 200, False, 5, False, auto_threshold=True
    )[0]

    assert mask[100, 100] == 255
    assert mask[100, 50] == 255
