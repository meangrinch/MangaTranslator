import cv2
import numpy as np
import pytest

from core.image.cleaning import process_single_bubble
from utils.exceptions import CleaningError


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
