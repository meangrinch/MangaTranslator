import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from core.caching import get_cache
from core.device import get_best_device
from core.ml.model_manager import ModelType, get_model_manager
from utils.exceptions import ImageProcessingError
from utils.logging import log_message


class OutsideTextDetector:
    """Detects text outside speech bubbles to isolate SFX/captions from dialogue."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        hf_token: Optional[str] = None,
    ):
        """Initialize the outside text detector.

        Args:
            device: PyTorch device to use. Auto-detects if None.
            hf_token: Hugging Face token for gated repo access.
        """
        self.device = device if device is not None else get_best_device()
        self.hf_token = hf_token
        self.manager = get_model_manager()
        self.cache = get_cache()

    def boxes_overlap(self, box1, box2):
        """Check if two bounding boxes overlap (have non-zero intersection).

        Args:
            box1: Bounding box in [x_min, y_min, x_max, y_max] format.
            box2: Bounding box in YOLO format [x_min, y_min, x_max, y_max].

        Returns:
            bool: True if boxes overlap, False otherwise.
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        return not (
            x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min
        )

    def box_is_inside(self, box1, box2):
        """Check if box1 is completely inside box2.

        Args:
            box1: Bounding box in [x1, y1, x2, y2] format.
            box2: Bounding box in [x1, y1, x2, y2] format.

        Returns:
            bool: True if box1 is completely inside box2, False otherwise.
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        return (
            x1_min >= x2_min
            and x1_max <= x2_max
            and y1_min >= y2_min
            and y1_max <= y2_max
        )

    def filter_nested_detections(self, results):
        """Remove detections fully contained in larger ones to avoid duplicates.

        Args:
            results: List of detection results (bbox, text, confidence).

        Returns:
            list: Filtered results with nested detections removed.
        """
        if len(results) <= 1:
            return results

        # Prioritize larger detections to avoid removing important text
        def get_area(result):
            bbox = result[0]
            x_min, y_min, x_max, y_max = bbox
            return (x_max - x_min) * (y_max - y_min)

        sorted_results = sorted(results, key=get_area, reverse=True)
        filtered_results = []

        for i, current_result in enumerate(sorted_results):
            is_nested = False
            current_bbox = current_result[0]

            for kept_result in filtered_results:
                kept_bbox = kept_result[0]
                if self.box_is_inside(current_bbox, kept_bbox):
                    is_nested = True
                    break

            if not is_nested:
                filtered_results.append(current_result)

        return filtered_results

    def unload_models(self):
        """Unload OCR models via model manager to free GPU/CPU memory."""
        self.manager.unload_ocr_models()

    def detect_outside_text(
        self,
        image_path: str,
        yolo_model_path: Optional[str] = None,
        confidence: float = 0.6,
        conjoined_confidence: float = 0.35,
        verbose: bool = False,
        image_override: Optional[Image.Image] = None,
        existing_bubbles: Optional[List] = None,
        text_free_boxes: Optional[List] = None,
    ):
        """Detect non-dialogue text by subtracting YOLO speech bubbles from OCR results.

        Args:
            image_path: Path to the input image.
            yolo_model_path: Optional custom YOLO model path.
            confidence: Confidence threshold for primary YOLO model detections.
            conjoined_confidence: Confidence threshold for secondary YOLO model (conjoined bubble detection).
            verbose: If True, logs intermediate steps.
            text_free_boxes: Optional list of text_free regions to use as fallback OSB detections.

        Returns:
            list: Detected regions outside bubbles as (bbox, confidence).
        """
        if image_override is None and not os.path.exists(image_path):
            raise FileNotFoundError(f"Error: The file '{image_path}' was not found.")

        try:
            if image_override is not None:
                image_pil = (
                    image_override
                    if image_override.mode == "RGB"
                    else image_override.convert("RGB")
                )
                image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            else:
                image_cv = cv2.imread(str(image_path))
                if image_cv is None:
                    raise ImageProcessingError(f"Could not read image at {image_path}")
                image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            image_name = image_path if image_override is None else "override"
            log_message(
                f"Processing image: {image_name} "
                f"({image_cv.shape[1]}x{image_cv.shape[0]})",
                verbose=verbose,
            )
        except Exception as e:
            raise ImageProcessingError(f"Error loading image: {e}")

        provided_bubble_boxes = None
        if existing_bubbles is not None:
            try:
                provided_bubble_boxes = []
                for b in existing_bubbles:
                    bbox = b.get("bbox") if isinstance(b, dict) else b
                    if bbox is None or len(bbox) != 4:
                        continue
                    x0, y0, x1, y1 = bbox
                    provided_bubble_boxes.append(
                        [float(x0), float(y0), float(x1), float(y1)]
                    )
                if provided_bubble_boxes:
                    log_message(
                        f"Using {len(provided_bubble_boxes)} provided bubble boxes for OSB filtering",
                        verbose=verbose,
                    )
            except Exception as e:
                log_message(
                    f"Warning: Failed to parse provided bubbles: {e}. Falling back to YOLO.",
                    always_print=True,
                )
                provided_bubble_boxes = None

        text_free_boxes = list(text_free_boxes) if text_free_boxes else []

        if provided_bubble_boxes:
            yolo_boxes = torch.tensor(
                provided_bubble_boxes, device=self.device, dtype=torch.float32
            )
            num_yolo_boxes = len(yolo_boxes)
            log_message(
                f"Skipping YOLO; using provided bubbles ({num_yolo_boxes})",
                verbose=verbose,
            )
        else:
            log_message("Running YOLO detection for speech bubbles...", verbose=verbose)

            sb_model_path = (
                str(self.manager.model_paths[ModelType.YOLO_SPEECH_BUBBLE])
                if yolo_model_path is None
                else yolo_model_path
            )
            sb_cache_key = self.cache.get_yolo_cache_key(
                image_pil, sb_model_path, confidence
            )
            cached_sb = self.cache.get_yolo_detection(sb_cache_key)

            if cached_sb is not None:
                log_message("Using cached Speech Bubble detections", verbose=verbose)
                yolo_results, yolo_boxes = cached_sb
            else:
                yolo_model = self.manager.load_yolo_speech_bubble(yolo_model_path)
                yolo_results = yolo_model(
                    image_cv, conf=confidence, device=self.device, verbose=False
                )[0]
                yolo_boxes = (
                    yolo_results.boxes.xyxy
                    if yolo_results.boxes is not None
                    else torch.tensor([])
                )
                self.cache.set_yolo_detection(sb_cache_key, (yolo_results, yolo_boxes))

            num_yolo_boxes = len(yolo_boxes) if yolo_boxes.nelement() > 0 else 0
            log_message(
                f"YOLO detected {num_yolo_boxes} speech bubbles", verbose=verbose
            )

            log_message(
                "Running Secondary YOLO to catch missed bubbles...", verbose=verbose
            )
            try:
                sec_model = self.manager.load_yolo_conjoined_bubble()
                sec_results = sec_model(
                    image_cv,
                    conf=conjoined_confidence,
                    device=self.device,
                    verbose=False,
                )[0]

                sec_boxes = (
                    sec_results.boxes.xyxy
                    if sec_results.boxes is not None
                    else torch.tensor([])
                )
                sec_cls = (
                    sec_results.boxes.cls
                    if sec_results.boxes is not None
                    else torch.tensor([])
                )

                # Find text_bubble and text_free classes
                tb_id = None
                tf_id = None
                if hasattr(sec_model, "names"):
                    for cid, cname in sec_model.names.items():
                        if cname == "text_bubble":
                            tb_id = cid
                        elif cname == "text_free":
                            tf_id = cid

                if tf_id is not None and len(sec_boxes) > 0:
                    for i, cls_id in enumerate(sec_cls):
                        if int(cls_id) == tf_id:
                            text_free_boxes.append(sec_boxes[i].detach().cpu().numpy())

                if tb_id is not None and len(sec_boxes) > 0:
                    boxes_to_add = []
                    for i, cls_id in enumerate(sec_cls):
                        if int(cls_id) == tb_id:
                            boxes_to_add.append(sec_boxes[i])

                    if boxes_to_add:
                        log_message(
                            f"Secondary YOLO found {len(boxes_to_add)} potential bubbles",
                            verbose=verbose,
                        )
                        boxes_to_add_tensor = torch.stack(boxes_to_add)
                        if yolo_boxes.nelement() > 0:
                            yolo_boxes = torch.cat(
                                (yolo_boxes, boxes_to_add_tensor), dim=0
                            )
                        else:
                            yolo_boxes = boxes_to_add_tensor
            except Exception as e:
                log_message(f"Secondary YOLO failed: {e}", verbose=verbose)

        log_message("Running YOLO OSB Text...", always_print=True)

        osbtext_boxes = None
        osbtext_confs = None
        try:
            osbtext_model_path = str(self.manager.model_paths[ModelType.YOLO_OSBTEXT])
            osbtext_cache_key = self.cache.get_yolo_cache_key(
                image_pil, osbtext_model_path, confidence
            )

            cached_osbtext = self.cache.get_yolo_detection(osbtext_cache_key)

            if cached_osbtext is not None:
                log_message("Using cached OSBText detections", verbose=verbose)
                osbtext_results, osbtext_boxes, osbtext_confs = cached_osbtext
            else:
                osbtext_model = self.manager.load_yolo_osbtext(token=self.hf_token)
                osbtext_results = osbtext_model(
                    image_cv, conf=confidence, device=self.device, verbose=False
                )[0]
                osbtext_boxes = (
                    osbtext_results.boxes.xyxy
                    if osbtext_results.boxes is not None
                    else None
                )
                osbtext_confs = (
                    osbtext_results.boxes.conf
                    if osbtext_results.boxes is not None
                    else None
                )
                self.cache.set_yolo_detection(
                    osbtext_cache_key, (osbtext_results, osbtext_boxes, osbtext_confs)
                )
        except Exception as e:
            log_message(
                f"OSB text model unavailable: {e}. Using text_free fallback if available.",
                always_print=True,
            )
            if text_free_boxes:
                log_message(
                    f"Using {len(text_free_boxes)} text_free boxes as OSB fallback",
                    always_print=True,
                )
                osbtext_boxes = torch.tensor(
                    text_free_boxes, device=self.device, dtype=torch.float32
                )
                osbtext_confs = torch.ones(
                    len(text_free_boxes), device=self.device, dtype=torch.float32
                )
            else:
                log_message(
                    "No text_free fallback available; skipping OSB text detections",
                    always_print=True,
                )

        base_results = []
        if osbtext_boxes is not None:
            boxes_np = osbtext_boxes.detach().cpu().numpy()
            confs_np = osbtext_confs.detach().cpu().numpy()

            for i, box in enumerate(boxes_np):
                conf = confs_np[i]
                base_results.append((box, float(conf)))

        final_results = list(base_results)

        log_message("Filtering out nested detections...", verbose=verbose)
        before_nested_filter = len(final_results)
        final_results = self.filter_nested_detections(final_results)
        after_nested_filter = len(final_results)
        nested_removed = before_nested_filter - after_nested_filter
        log_message(
            f"Nested detections removed: {nested_removed}. Remaining detections: {after_nested_filter}.",
            verbose=verbose,
        )

        if yolo_boxes is not None and yolo_boxes.nelement() > 0:
            log_message(
                "Filtering OCR results to keep text outside speech bubbles...",
                verbose=verbose,
            )
            filtered_results = []
            yolo_boxes_np = yolo_boxes.detach().cpu().numpy()

            for ocr_result in final_results:
                bbox, _ = ocr_result

                overlaps_any_bubble = False

                for yolo_box in yolo_boxes_np:
                    if self.boxes_overlap(bbox, yolo_box):
                        # Check if this bubble is actually a text_free region
                        is_text_free_bubble = False
                        if text_free_boxes:
                            for tf_box in text_free_boxes:
                                # We check if the YOLO bubble overlaps with a text_free detection
                                if self.boxes_overlap(yolo_box, tf_box):
                                    is_text_free_bubble = True
                                    break

                        if not is_text_free_bubble:
                            overlaps_any_bubble = True
                            break

                if not overlaps_any_bubble:
                    filtered_results.append(ocr_result)

            filtered_out = len(final_results) - len(filtered_results)
            log_message(
                f"Filtered out {filtered_out} OCR results that overlapped with speech bubbles",
                verbose=verbose,
            )
            final_results = filtered_results

        log_message(
            f"Found {len(final_results)} outside text regions", always_print=True
        )

        return final_results

    def get_text_masks(
        self,
        image_path: str,
        bbox_expansion_percent: float = 0.0,
        text_box_proximity_ratio: float = 0.02,
        verbose: bool = False,
        image_override: Optional[Image.Image] = None,
        existing_results: Optional[List] = None,
    ) -> Tuple[Optional[List], Optional[Image.Image]]:
        """Create rectangular masks from OCR bounding boxes for inpainting.

        Args:
            image_path: Path to the input image.
            bbox_expansion_percent: Percentage to expand bounding boxes.
            text_box_proximity_ratio: Ratio for grouping nearby text boxes (as fraction of image dimension).
            verbose: Whether to print verbose output.

        Returns:
            tuple: (groups, image_pil) where groups is a list of dicts with:
                {
                    'combined_mask': np.array[H,W,bool],
                    'bbox': dict,
                    'individual_masks': [np.array],
                    'mask_indices': [int],
                    'confidence': float,
                }.
        """
        results = (
            existing_results
            if existing_results is not None
            else self.detect_outside_text(
                image_path,
                verbose=verbose,
                image_override=image_override,
            )
        )

        if not results:
            return None, None

        if image_override is not None:
            image_pil = (
                image_override.convert("RGB")
                if image_override.mode != "RGB"
                else image_override
            )
        else:
            image_pil = Image.open(image_path).convert("RGB")
        img_w, img_h = image_pil.size

        log_message("Converting OCR results to axis-aligned boxes...", verbose=verbose)
        boxes = [[int(c) for c in result[0]] for result in results]

        expanded_boxes = []
        for box in boxes:
            x0, y0, x1, y1 = box
            width = x1 - x0
            height = y1 - y0
            expand_x = width * bbox_expansion_percent
            expand_y = height * bbox_expansion_percent
            x0e = int(np.floor(max(0, x0 - expand_x)))
            y0e = int(np.floor(max(0, y0 - expand_y)))
            x1e = int(np.ceil(min(img_w, x1 + expand_x)))
            y1e = int(np.ceil(min(img_h, y1 + expand_y)))
            if x1e > x0e and y1e > y0e:
                expanded_boxes.append([x0e, y0e, x1e, y1e])

        log_message(
            f"Grouping {len(expanded_boxes)} text boxes spatially...",
            verbose=verbose,
        )

        grouped_boxes = self._group_text_boxes_spatially(
            expanded_boxes, results, img_w, img_h, text_box_proximity_ratio, verbose
        )

        groups = []
        for group_boxes, group_results in grouped_boxes:
            combined_mask = np.zeros((img_h, img_w), dtype=bool)
            individual_masks = []
            mask_indices = []
            avg_confidence = 0.0

            min_x = min(box[0] for box in group_boxes)
            min_y = min(box[1] for box in group_boxes)
            max_x = max(box[2] for box in group_boxes)
            max_y = max(box[3] for box in group_boxes)

            # Ensure combined region doesn't exceed Flux Kontext preferred resolutions
            max_dimension = 1568
            if max_x - min_x > max_dimension or max_y - min_y > max_dimension:
                log_message(
                    f"  - Group too large ({max_x - min_x}x{max_y - min_y}), splitting...",
                    verbose=verbose,
                )
                for i, (box, result) in enumerate(zip(group_boxes, group_results)):
                    x0, y0, x1, y1 = box
                    mask = np.zeros((img_h, img_w), dtype=bool)
                    mask[y0:y1, x0:x1] = True

                    bbox = {
                        "x": int(x0),
                        "y": int(y0),
                        "width": int(x1 - x0),
                        "height": int(y1 - y0),
                    }

                    raw_box = [int(c) for c in result[0]]
                    raw_x0, raw_y0, raw_x1, raw_y1 = raw_box
                    original_bbox = {
                        "x": raw_x0,
                        "y": raw_y0,
                        "width": raw_x1 - raw_x0,
                        "height": raw_y1 - raw_y0,
                    }

                    _, conf = result

                    groups.append(
                        {
                            "combined_mask": mask,
                            "bbox": bbox,
                            "original_bbox": original_bbox,
                            "individual_masks": [mask],
                            "mask_indices": [i],
                            "confidence": conf,
                        }
                    )
                continue

            raw_boxes = [[int(c) for c in res[0]] for res in group_results]

            for i, (box, result, raw_box) in enumerate(
                zip(group_boxes, group_results, raw_boxes)
            ):
                x0, y0, x1, y1 = box
                mask = np.zeros((img_h, img_w), dtype=bool)
                mask[y0:y1, x0:x1] = True

                combined_mask |= mask
                individual_masks.append(mask)
                mask_indices.append(i)

                _, conf = result
                avg_confidence += conf

            raw_min_x = min(box[0] for box in raw_boxes)
            raw_min_y = min(box[1] for box in raw_boxes)
            raw_max_x = max(box[2] for box in raw_boxes)
            raw_max_y = max(box[3] for box in raw_boxes)

            bbox = {
                "x": int(min_x),
                "y": int(min_y),
                "width": int(max_x - min_x),
                "height": int(max_y - min_y),
            }

            original_bbox = {
                "x": int(raw_min_x),
                "y": int(raw_min_y),
                "width": int(raw_max_x - raw_min_x),
                "height": int(raw_max_y - raw_min_y),
            }

            groups.append(
                {
                    "combined_mask": combined_mask,
                    "bbox": bbox,
                    "original_bbox": original_bbox,
                    "individual_masks": individual_masks,
                    "mask_indices": mask_indices,
                    "confidence": avg_confidence / len(group_results),
                }
            )

        log_message(
            f"Created {len(groups)} grouped text regions for inpainting",
            verbose=verbose,
        )

        return groups, image_pil

    def _group_text_boxes_spatially(
        self, boxes, results, img_w, img_h, text_box_proximity_ratio=0.02, verbose=False
    ):
        """
        Group nearby text boxes based on spatial proximity.

        Args:
            boxes: List of bounding boxes [x0, y0, x1, y1]
            results: List of OCR results corresponding to boxes
            img_w: Image width
            img_h: Image height
            text_box_proximity_ratio: Ratio for grouping nearby text boxes (as fraction of image dimension).
            verbose: Whether to print detailed logs

        Returns:
            List of tuples (group_boxes, group_results) where each group contains
            spatially related text boxes
        """
        if not boxes:
            return []

        proximity_threshold = min(img_w, img_h) * text_box_proximity_ratio

        parent = list(range(len(boxes)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if self._boxes_are_nearby(boxes[i], boxes[j], proximity_threshold):
                    union(i, j)

        groups = {}
        for i in range(len(boxes)):
            root = find(i)
            if root not in groups:
                groups[root] = ([], [])
            groups[root][0].append(boxes[i])
            groups[root][1].append(results[i])

        grouped_boxes = list(groups.values())

        log_message(
            f"  - Grouped {len(boxes)} boxes into {len(grouped_boxes)} spatial groups",
            verbose=verbose,
        )

        return grouped_boxes

    def _boxes_are_nearby(self, box1, box2, threshold):
        """
        Check if two bounding boxes are spatially close enough to be grouped.

        Args:
            box1: First bounding box [x0, y0, x1, y1]
            box2: Second bounding box [x0, y0, x1, y1]
            threshold: Maximum distance for boxes to be considered nearby

        Returns:
            True if boxes are nearby, False otherwise
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        cx1 = (x1_min + x1_max) / 2
        cy1 = (y1_min + y1_max) / 2
        cx2 = (x2_min + x2_max) / 2
        cy2 = (y2_min + y2_max) / 2

        distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

        return distance <= threshold


def extract_text_with_manga_ocr(
    images: List[Image.Image], verbose: bool = False
) -> List[str]:
    """Extract text from images using manga-ocr library.

    Args:
        images: List of PIL Images to process
        verbose: Whether to print verbose output

    Returns:
        List of extracted text strings (one per image). Returns [OCR FAILED] on errors.
    """
    if not images:
        return []

    try:
        model_manager = get_model_manager()
        manga_ocr_instance = model_manager.get_manga_ocr(verbose=verbose)

        extracted_texts = []
        for i, img in enumerate(images):
            try:
                if img is None:
                    log_message(
                        f"Image {i + 1} is None (decode failure), skipping",
                        always_print=True,
                    )
                    extracted_texts.append("[OCR FAILED]")
                    continue

                log_message(
                    f"Processing image {i + 1}/{len(images)} with manga-ocr",
                    verbose=verbose,
                )
                text = manga_ocr_instance(img)

                extracted_texts.append(text.strip() if text else "")

            except Exception as e:
                log_message(
                    f"manga-ocr failed for image {i + 1}: {e}", always_print=True
                )
                extracted_texts.append("[OCR FAILED]")

        return extracted_texts

    except Exception as e:
        log_message(f"Error with manga-ocr: {e}", always_print=True)
        return ["[OCR FAILED]"] * len(images)
