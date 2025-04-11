from functools import lru_cache
import cv2
import torch
from ultralytics import YOLO

from .common_utils import log_message

# Global cache
_model_cache = {}


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


def detect_speech_bubbles(image_path, model_path, confidence=0.35, verbose=False, device=None):
    """
    Detect speech bubbles in the given image using YOLO model.

    Args:
        image_path (str): Path to the input image
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
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading image: {e}")

    results = model(image, conf=confidence, device=_device, verbose=verbose)

    if len(results) == 0:
        return detections

    for _, result in enumerate(results[0]):
        boxes = result.boxes

        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            conf = float(box.conf[0])

            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            detection = {"bbox": (x1, y1, x2, y2), "confidence": conf, "class": cls_name}

            detections.append(detection)

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

    return detections
