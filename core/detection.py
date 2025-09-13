from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor
from ultralytics import YOLO

from utils.logging import log_message

# Global cache
_model_cache = {}
_sam2_cache = {}


@lru_cache(maxsize=1)
def get_yolo_model(model_path):
    """
    Get a cached YOLO model or load a new one.

    Args:
        model_path (str): Path to the YOLO model file

    Returns:
        YOLO: Loaded model instance
    """
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)

    return _model_cache[model_path]


def _get_sam2(model_id: str, device: torch.device):
    """Load and cache SAM2.1 model + processor."""
    cache_key = (model_id, device.type)
    if cache_key not in _sam2_cache:
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        processor = Sam2Processor.from_pretrained(model_id, cache_dir=str(models_dir))
        model = Sam2Model.from_pretrained(model_id, cache_dir=str(models_dir)).to(device)
        _sam2_cache[cache_key] = (processor, model)
    return _sam2_cache[cache_key]


def detect_speech_bubbles(
    image_path: Path,
    model_path,
    confidence=0.35,
    verbose=False,
    device=None,
    use_sam2: bool = False,
    sam2_model_id: str = "facebook/sam2.1-hiera-large",
):
    """
    Detect speech bubbles in the given image using YOLO model.

    Args:
        image_path (Path): Path to the input image
        model_path (str): Path to the YOLO segmentation model
        confidence (float): Confidence threshold for detections
        verbose (bool): Whether to show detailed YOLO output
        device (torch.device, optional): The device to run the model on. Autodetects if None.

    Returns:
        list: List of dictionaries containing detection information (bbox, class, confidence)
    """
    detections = []

    _device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = get_yolo_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading image: {e}")

    results = model(image, conf=confidence, device=_device, verbose=verbose)

    if len(results) == 0:
        return detections

    boxes_for_sam = []
    for _, result in enumerate(results[0]):
        boxes = result.boxes

        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            conf = float(box.conf[0])

            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            detection = {"bbox": (x1, y1, x2, y2), "confidence": conf, "class": cls_name}

            detections.append(detection)
            boxes_for_sam.append([x1, y1, x2, y2])

            if hasattr(result, "masks") and result.masks is not None:
                try:
                    masks = result.masks
                    if len(masks) > j:
                        mask_points = masks[j].xy[0]
                        detection["mask_points"] = (
                            mask_points.tolist() if hasattr(mask_points, "tolist") else mask_points
                        )
                except (IndexError, AttributeError) as e:
                    log_message(f"Warning: Could not extract mask for detection {j}: {e}", verbose=verbose)
                    detection["mask_points"] = None

    # Optional: SAM 2.1 segmentation guided by YOLO boxes
    try:
        if use_sam2 and len(detections) > 0:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            processor, sam_model = _get_sam2(sam2_model_id, _device)

            # SAM expects a batch of images; input_boxes is list per image
            sam_inputs = processor(images=pil_image, input_boxes=[boxes_for_sam], return_tensors="pt").to(_device)

            with torch.no_grad():
                sam_outputs = sam_model(multimask_output=False, **sam_inputs)

            # Post-process to original resolution masks
            masks = processor.post_process_masks(sam_outputs.pred_masks, sam_inputs["original_sizes"])[0][:, 0]
            sam_masks_np = (masks > 0.5).detach().cpu().numpy()  # [N, H, W]

            img_h, img_w = image.shape[:2]
            n_sam = sam_masks_np.shape[0]
            for i, det in enumerate(detections):
                if i >= n_sam:
                    break
                x0, y0, x1, y1 = boxes_for_sam[i]
                # Clip to image bounds
                x0 = max(0, min(int(x0), img_w))
                y0 = max(0, min(int(y0), img_h))
                x1 = max(0, min(int(x1), img_w))
                y1 = max(0, min(int(y1), img_h))
                if x1 <= x0 or y1 <= y0:
                    continue

                m = sam_masks_np[i].astype(bool)
                bbox_mask = np.zeros((img_h, img_w), dtype=bool)
                bbox_mask[y0:y1, x0:x1] = True
                clipped = np.logical_and(m, bbox_mask)
                det["sam_mask"] = (clipped.astype(np.uint8) * 255)
    except Exception as e:
        log_message(f"Warning: SAM2.1 segmentation failed: {e}", always_print=True)

    return detections
