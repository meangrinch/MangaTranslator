from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor
from ultralytics import YOLO

from utils.logging import log_message

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
        model.eval()
        _sam2_cache[cache_key] = (processor, model)
    return _sam2_cache[cache_key]


def detect_speech_bubbles(
    image_path: Path,
    model_path,
    confidence=0.35,
    verbose=False,
    device=None,
    use_sam2: bool = True,
    sam2_model_id: str = "facebook/sam2.1-hiera-large",
):
    """
    Detect speech bubbles in the given image using YOLO model.

    Args:
        image_path (Path): Path to the input image
        model_path (str): Path to the YOLO segmentation model
        confidence (float): Confidence threshold for detections
        verbose (bool): Whether to show detailed processing information
        device (torch.device, optional): The device to run the model on. Autodetects if None.
        use_sam2 (bool): Whether to use SAM2.1 for enhanced segmentation
        sam2_model_id (str): SAM2.1 model identifier

    Returns:
        list: List of dictionaries containing detection information (bbox, class, confidence)
    """
    detections = []

    _device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {_device}", verbose=verbose)

    try:
        model = get_yolo_model(model_path)
        log_message(f"Loaded YOLO model: {model_path}", verbose=verbose)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        log_message(f"Processing image: {image_path.name} ({image.shape[1]}x{image.shape[0]})", verbose=verbose)
    except Exception as e:
        raise RuntimeError(f"Error loading image: {e}")

    results = model(image, conf=confidence, device=_device, verbose=verbose)

    if len(results) == 0:
        log_message("No detections found", verbose=verbose)
        return detections

    result = results[0]
    boxes = result.boxes

    # Float boxes for SAM prompts; keep int bbox for returned detections
    sam_input_boxes = []

    if boxes is not None:
        for j, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            x0_f, y0_f, x1_f, y1_f = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            detection = {
                "bbox": (int(round(x0_f)), int(round(y0_f)), int(round(x1_f)), int(round(y1_f))),
                "confidence": conf,
                "class": cls_name,
            }

            detections.append(detection)
            sam_input_boxes.append([x0_f, y0_f, x1_f, y1_f])

            if getattr(result, "masks", None) is not None:
                try:
                    masks = result.masks
                    if len(masks) > j:
                        mask_points = masks[j].xy[0]
                        detection["mask_points"] = (
                            mask_points.tolist() if hasattr(mask_points, "tolist") else mask_points
                        )
                except (IndexError, AttributeError) as e:
                    log_message(f"Could not extract YOLO mask for detection {j}: {e}", always_print=True)
                    detection["mask_points"] = None

    log_message(f"Found {len(detections)} speech bubbles", verbose=verbose)

    # SAM 2.1 segmentation guided by YOLO boxes for more precise masks
    if use_sam2 and len(detections) > 0:
        try:
            log_message("Applying SAM2.1 segmentation refinement", verbose=verbose)
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            processor, sam_model = _get_sam2(sam2_model_id, _device)

            sam_inputs = processor(images=pil_image, input_boxes=[sam_input_boxes], return_tensors="pt").to(_device)

            with torch.no_grad():
                sam_outputs = sam_model(multimask_output=False, **sam_inputs)

            masks = processor.post_process_masks(sam_outputs.pred_masks, sam_inputs["original_sizes"])[0][:, 0]
            sam_masks_np = (masks > 0.5).detach().cpu().numpy()

            img_h, img_w = image.shape[:2]
            n_sam = sam_masks_np.shape[0]
            for i, det in enumerate(detections):
                if i >= n_sam:
                    break
                x0_f, y0_f, x1_f, y1_f = sam_input_boxes[i]
                # Clip to image bounds, floor starts and ceil ends to avoid off-by-one errors
                x0 = int(np.floor(max(0, min(x0_f, img_w))))
                y0 = int(np.floor(max(0, min(y0_f, img_h))))
                x1 = int(np.ceil(max(0, min(x1_f, img_w))))
                y1 = int(np.ceil(max(0, min(y1_f, img_h))))
                if x1 <= x0 or y1 <= y0:
                    continue

                m = sam_masks_np[i].astype(bool)
                bbox_mask = np.zeros((img_h, img_w), dtype=bool)
                bbox_mask[y0:y1, x0:x1] = True
                clipped = np.logical_and(m, bbox_mask)
                det["sam_mask"] = (clipped.astype(np.uint8) * 255)

            log_message("SAM2.1 segmentation completed successfully", verbose=verbose)
        except Exception as e:
            log_message(f"SAM2.1 segmentation failed: {e}", always_print=True)

    return detections
