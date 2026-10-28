import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
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


class _FakeOsbModel:
    """Finds fixed page-space boxes, translated into each crop, at every size."""

    def __init__(self, page_boxes):
        self.page_boxes = page_boxes  # (box, conf) pairs
        self.calls = []

    def __call__(self, crop, conf, device, verbose, imgsz):
        self.calls.append((crop.shape[:2], imgsz))
        ox, oy = crop.origin
        boxes = [
            [b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy] for b, _ in self.page_boxes
        ]
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=torch.tensor(boxes, dtype=torch.float32),
                    conf=torch.tensor([c for _, c in self.page_boxes]),
                )
            )
        ]


class _Page(np.ndarray):
    """Image whose crops remember where they were cut from."""

    def __getitem__(self, key):
        crop = np.asarray(self)[key]
        view = crop.view(_Page)
        view.origin = (key[1].start, key[0].start)
        return view


def _detector_with(model, monkeypatch):
    import core.image.ocr_detection as ocr

    manager = SimpleNamespace(load_yolo_osbtext=lambda token=None: model)
    monkeypatch.setattr(ocr, "get_model_manager", lambda: manager)
    return ocr.OutsideTextDetector(device=torch.device("cpu"))


def test_text_near_boxes_reruns_the_osb_model_on_crops(monkeypatch):
    image = np.zeros((1000, 1000, 3), np.uint8).view(_Page)
    near_first = (390, 395, 480, 450)
    model = _FakeOsbModel([(near_first, 0.6), ((300, 300, 340, 330), 0.9)])
    detector = _detector_with(model, monkeypatch)

    found = detector.detect_text_near(
        image, [(400, 400, 500, 440), (700, 700, 720, 720)], confidence=0.2
    )

    # Three pads (0.25, 0.5, 1.5 of the long side, at least 48 px) around each region,
    # each at two input sizes
    assert [size for _, size in model.calls] == [640, 320] * 6
    assert [shape for shape, _ in model.calls[:6:2]] == [
        (136, 196),
        (140, 200),
        (340, 400),
    ]
    # Detections come back whole in page space, so one can span several regions; the
    # one between the regions belongs to neither
    assert found == [[((390.0, 395.0, 480.0, 450.0), pytest.approx(0.6))] * 6, []]


def test_text_near_boxes_drops_weak_detections(monkeypatch):
    image = np.zeros((1000, 1000, 3), np.uint8).view(_Page)
    weak = _FakeOsbModel([((400, 400, 500, 440), 0.1)])

    found = _detector_with(weak, monkeypatch).detect_text_near(
        image, [(400, 400, 500, 440)], 0.2
    )

    assert found == [[]]
