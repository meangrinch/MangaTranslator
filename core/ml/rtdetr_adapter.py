from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image


class _RTDetrBoxes:
    """Minimal stand-in for ultralytics Results.boxes."""

    def __init__(
        self,
        xyxy: torch.Tensor,
        conf: torch.Tensor,
        cls: torch.Tensor,
    ):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _RTDetrResults:
    """Minimal stand-in for ultralytics Results."""

    def __init__(self, boxes: _RTDetrBoxes, names: Dict[int, str]):
        self.boxes = boxes
        self.names = names


class RTDetrYOLOAdapter:
    """Callable wrapper that mimics YOLO's predict API for RT-DETR-v2.

    Usage matches ultralytics YOLO:
        results = adapter(image_cv, conf=0.35, device=device, imgsz=640)[0]
        results.boxes.xyxy / .conf / .cls
        adapter.names  # {id: class_name}
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: torch.device,
        names: Optional[Dict[int, str]] = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        if names is not None:
            self.names = {int(k): str(v) for k, v in names.items()}
        else:
            id2label = getattr(model.config, "id2label", None) or {}
            self.names = {int(k): str(v) for k, v in id2label.items()}

    def __call__(
        self,
        source: Union[np.ndarray, Image.Image, str],
        conf: float = 0.35,
        device: Optional[Union[torch.device, str]] = None,
        verbose: bool = False,
        imgsz: Optional[int] = None,
        **_kwargs,
    ) -> List[_RTDetrResults]:
        del verbose  # accepted for YOLO API compatibility
        run_device = torch.device(device) if device is not None else self.device
        image_pil, orig_h, orig_w = self._to_pil(source)

        processor_kwargs = {"images": image_pil, "return_tensors": "pt"}
        if imgsz is not None:
            size = int(imgsz)
            processor_kwargs["size"] = {"height": size, "width": size}

        inputs = self.processor(**processor_kwargs)
        inputs = {k: v.to(run_device) for k, v in inputs.items()}

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
                results = self.processor.post_process_object_detection(
                    outputs,
                    threshold=float(conf),
                    target_sizes=[(orig_h, orig_w)],
                    use_focal_loss=bool(
                        getattr(self.model.config, "use_focal_loss", True)
                    ),
                )[0]
        finally:
            if was_training:
                self.model.train()

        boxes = results["boxes"].detach()
        scores = results["scores"].detach()
        labels = results["labels"].detach()

        # Keep tensors on the inference device so they can be concatenated with YOLO boxes.
        if boxes.numel() == 0:
            xyxy = torch.empty((0, 4), dtype=torch.float32, device=run_device)
            conf_t = torch.empty((0,), dtype=torch.float32, device=run_device)
            cls_t = torch.empty((0,), dtype=torch.float32, device=run_device)
        else:
            xyxy = boxes.to(device=run_device, dtype=torch.float32)
            conf_t = scores.to(device=run_device, dtype=torch.float32)
            cls_t = labels.to(device=run_device, dtype=torch.float32)

        return [_RTDetrResults(_RTDetrBoxes(xyxy, conf_t, cls_t), self.names)]

    @staticmethod
    def _to_pil(
        source: Union[np.ndarray, Image.Image, str],
    ) -> tuple[Image.Image, int, int]:
        if isinstance(source, Image.Image):
            image = source.convert("RGB") if source.mode != "RGB" else source
            w, h = image.size
            return image, h, w

        if isinstance(source, str):
            image = Image.open(source).convert("RGB")
            w, h = image.size
            return image, h, w

        # Remaining accepted type: OpenCV BGR ndarray
        arr = source
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(arr)
        h, w = arr.shape[:2]
        return image, h, w
