import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from core.image.detection import (
    _pad_parent_mask_with_uncovered_boxes,
    _remove_contained_boxes,
)
from core.ml.model_manager import ModelType, get_model_manager
from core.ml.rtdetr_adapter import RTDetrYOLOAdapter


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


def test_rtdetr_boxes_are_clipped_to_the_image_like_yolo():
    class FakeProcessor:
        def __call__(self, **_kwargs):
            return {"pixel_values": torch.zeros(1)}

        def post_process_object_detection(self, *_args, **_kwargs):
            return [
                {
                    "boxes": torch.tensor([[-2.4, -1.0, 50.0, 130.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            ]

    class FakeModel:
        training = False
        config = SimpleNamespace(id2label={0: "bubble"})

        def eval(self):
            pass

        def __call__(self, **_kwargs):
            return None

    adapter = RTDetrYOLOAdapter(FakeModel(), FakeProcessor(), torch.device("cpu"))

    boxes = adapter(np.zeros((120, 80, 3), np.uint8))[0].boxes

    assert boxes.xyxy.tolist() == [[0.0, 0.0, 50.0, 120.0]]
