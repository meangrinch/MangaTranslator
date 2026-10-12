import cv2
import numpy as np

from core.image.tilt_detection import detect_crop_tilt_angle


def test_steep_ink_in_horizontal_crop_is_not_clamped_into_a_tilt():
    crop = np.full((120, 120, 3), 255, np.uint8)
    t = np.radians(70)
    dx, dy = int(45 * np.cos(t)), int(45 * np.sin(t))
    cv2.line(crop, (60 - dx, 60 - dy), (60 + dx, 60 + dy), (0, 0, 0), 9)

    angle, conf, _, _ = detect_crop_tilt_angle(crop, max_tilt_deg=20.0)

    assert (angle, conf) == (0.0, 0.0)
