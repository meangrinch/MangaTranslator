import os

import cv2
import numpy as np
import torch

from core.image.detection import (
    _pad_parent_mask_with_uncovered_boxes,
    _remove_contained_boxes,
)
from core.ml.model_manager import ModelType, get_model_manager


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


def test_near_duplicate_boxes_keep_the_earlier_higher_confidence_one():
    boxes = torch.tensor(
        [
            [10.0, 10.0, 110.0, 110.0],
            [11.0, 11.0, 111.0, 111.0],
            [20.0, 20.0, 40.0, 40.0],
        ]
    )

    _, kept = _remove_contained_boxes(boxes, [("secondary", i) for i in range(3)])

    assert kept == [("secondary", 0)]


def test_relative_bubble_model_path_resolves_to_its_own_model_type():
    manager = get_model_manager()
    for model_type in (ModelType.YOLO_SPEECH_BUBBLE, ModelType.YOLO_SPEECH_BUBBLE_2):
        relative = os.path.relpath(manager.model_paths[model_type])
        assert manager._resolve_speech_bubble_model_type(relative) is model_type
