import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from core.caching import get_cache
from core.ml.model_manager import get_model_manager
from utils.logging import log_message

# OCR Detection Parameters
IOU_THRESHOLD = 0.5  # IoU threshold for box overlap detection
EASYOCR_MAG_RATIO = 1.0  # EasyOCR magnification ratio
EASYOCR_LINK_THRESHOLD = 0.1  # EasyOCR link threshold for connecting text components
EASYOCR_BEAM_WIDTH = 10  # Beam search width for EasyOCR decoder
EASYOCR_ROTATION_ANGLE = 30  # Text rotation angle to attempt (degrees)
TEXT_BOX_PROXIMITY_RATIO = (
    0.05  # Text box spatial grouping uses 5% of image dimension as proximity threshold
)

OCR_LANGUAGE_MAP = {
    "Afrikaans": "af",
    "Albanian": "sq",
    "Arabic": "ar",
    "Belarusian": "be",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Simplified Chinese": "ch_sim",
    "Traditional Chinese": "ch_tra",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Estonian": "et",
    "Persian (Farsi)": "fa",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Malay": "ms",
    "Maltese": "mt",
    "Marathi": "mr",
    "Nepali": "ne",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian (cyrillic)": "rs_cyrillic",
    "Serbian (latin)": "rs_latin",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Tagalog": "tl",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
}


class OutsideTextDetector:
    """Detects text outside speech bubbles to isolate SFX/captions from dialogue."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        languages: Optional[List[str]] = None,
    ):
        """Initialize the outside text detector.

        Args:
            device: PyTorch device to use. Auto-detects if None.
            languages: List of language codes for EasyOCR. Defaults to ['ja'].
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.languages = languages if languages is not None else ["ja"]
        self.manager = get_model_manager()
        self.cache = get_cache()
        self.yolo_model = None
        self.reader = None
        self.iou_threshold = IOU_THRESHOLD

    def calculate_iou(self, box1, box2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1: Bounding box from EasyOCR (list of 4 corner points).
            box2: Bounding box from EasyOCR (list of 4 corner points).

        Returns:
            float: The IoU score, between 0 and 1.
        """
        x1_min, y1_min = np.min(box1, axis=0)
        x1_max, y1_max = np.max(box1, axis=0)

        x2_min, y2_min = np.min(box2, axis=0)
        x2_max, y2_max = np.max(box2, axis=0)

        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        inter_width = max(0, x_inter_max - x_inter_min)
        inter_height = max(0, y_inter_max - y_inter_min)
        intersection_area = inter_width * inter_height

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = area1 + area2 - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def boxes_overlap(self, box1, box2):
        """Check if two bounding boxes overlap (have non-zero intersection).

        Args:
            box1: Bounding box in EasyOCR format (list of 4 points).
            box2: Bounding box in YOLO format [x_min, y_min, x_max, y_max].

        Returns:
            bool: True if boxes overlap, False otherwise.
        """
        x1_min, y1_min = np.min(box1, axis=0)
        x1_max, y1_max = np.max(box1, axis=0)

        x2_min, y2_min, x2_max, y2_max = box2

        return not (
            x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min
        )

    def box_is_inside(self, box1, box2):
        """Check if box1 is completely inside box2.

        Args:
            box1: Bounding box in EasyOCR format (list of 4 points).
            box2: Bounding box in EasyOCR format (list of 4 points).

        Returns:
            bool: True if box1 is completely inside box2, False otherwise.
        """
        x1_min, y1_min = np.min(box1, axis=0)
        x1_max, y1_max = np.max(box1, axis=0)

        x2_min, y2_min = np.min(box2, axis=0)
        x2_max, y2_max = np.max(box2, axis=0)

        return (
            x1_min >= x2_min
            and x1_max <= x2_max
            and y1_min >= y2_min
            and y1_max <= y2_max
        )

    def filter_nested_detections(self, results):
        """Remove detections fully contained in larger ones to avoid duplicates.

        Args:
            results: List of OCR results (bbox, text, confidence).

        Returns:
            list: Filtered results with nested detections removed.
        """
        if len(results) <= 1:
            return results

        # Prioritize larger detections to avoid removing important text
        def get_area(result):
            bbox = result[0]
            x_min, y_min = np.min(bbox, axis=0)
            x_max, y_max = np.max(bbox, axis=0)
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

    def initialize_models(self, yolo_model_path: Optional[str] = None):
        """Initialize YOLO and EasyOCR models via the shared model manager."""
        if self.yolo_model is None:
            self.yolo_model = self.manager.load_yolo_speech_bubble(yolo_model_path)

        if self.reader is None:
            self.reader = self.manager.load_easyocr(self.languages)

    def unload_models(self):
        """Unload OCR models via model manager to free GPU/CPU memory."""
        self.yolo_model = None
        self.reader = None
        self.manager.unload_ocr_models()

    def convert_easyocr_to_bboxes(self, easyocr_results):
        """Convert EasyOCR quadrilaterals to axis-aligned [x_min, y_min, x_max, y_max] boxes."""
        boxes = []
        for result in easyocr_results:
            if len(result) >= 3:
                bbox = result[0]
            else:
                continue

            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            boxes.append([x_min, y_min, x_max, y_max])

        return boxes

    def detect_outside_text(
        self,
        image_path: str,
        yolo_model_path: Optional[str] = None,
        verbose: bool = False,
        min_size: int = 200,
        image_override: Optional[Image.Image] = None,
    ):
        """Detect non-dialogue text by subtracting YOLO speech bubbles from OCR results.

        Args:
            image_path: Path to the input image.
            yolo_model_path: Optional custom YOLO model path.
            verbose: If True, logs intermediate steps.
            min_size: Minimum text region size in pixels for OCR detection.

        Returns:
            list: Detected regions outside bubbles as (bbox, text, confidence).
        """
        if image_override is None and not os.path.exists(image_path):
            raise FileNotFoundError(f"Error: The file '{image_path}' was not found.")

        self.initialize_models(yolo_model_path)

        log_message(f"Using device: {self.device}", verbose=verbose)

        log_message("Running YOLO detection for speech bubbles...", verbose=verbose)
        raw_image = (
            image_override.convert("RGB")
            if (image_override is not None and image_override.mode != "RGB")
            else (image_override or Image.open(image_path).convert("RGB"))
        )
        yolo_results = self.yolo_model(raw_image, device=self.device)
        yolo_boxes = (
            yolo_results[0].boxes.xyxy if yolo_results[0].boxes is not None else None
        )

        image_np = np.array(raw_image)

        num_yolo_boxes = (
            len(yolo_boxes)
            if yolo_boxes is not None and yolo_boxes.nelement() > 0
            else 0
        )
        log_message(f"YOLO detected {num_yolo_boxes} speech bubbles", verbose=verbose)

        log_message("Running EasyOCR...", always_print=True)
        cache_key = self.cache.get_ocr_cache_key(
            raw_image,
            self.languages,
            mag_ratio=EASYOCR_MAG_RATIO,
            link_threshold=EASYOCR_LINK_THRESHOLD,
            decoder="beamsearch",
            beamWidth=EASYOCR_BEAM_WIDTH,
            rotation_info=[EASYOCR_ROTATION_ANGLE],
            min_size=min_size,
        )
        cached_ocr = self.cache.get_ocr_detection(cache_key)

        if cached_ocr is not None:
            log_message("  - Using cached OCR results", verbose=verbose)
            base_results = cached_ocr
        else:
            base_results_raw = self.reader.readtext(
                image_np,
                mag_ratio=EASYOCR_MAG_RATIO,
                link_threshold=EASYOCR_LINK_THRESHOLD,
                decoder="beamsearch",
                beamWidth=EASYOCR_BEAM_WIDTH,
                rotation_info=[EASYOCR_ROTATION_ANGLE],
                min_size=min_size,
            )
            base_results = [(bbox, text, conf) for bbox, text, conf in base_results_raw]
            self.cache.set_ocr_detection(cache_key, base_results)

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
                bbox, _, _ = ocr_result

                overlaps_any = False

                for yolo_box in yolo_boxes_np:
                    if self.boxes_overlap(bbox, yolo_box):
                        overlaps_any = True
                        break

                if not overlaps_any:
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
        verbose: bool = False,
        min_size: int = 200,
        image_override: Optional[Image.Image] = None,
        existing_results: Optional[List] = None,
    ) -> Tuple[Optional[List], Optional[Image.Image]]:
        """Create rectangular masks from OCR bounding boxes for inpainting.

        Args:
            image_path: Path to the input image.
            bbox_expansion_percent: Percentage to expand bounding boxes.
            verbose: Whether to print verbose output.
            min_size: Minimum text region size in pixels for OCR detection.

        Returns:
            tuple: (groups, image_pil) where groups is a list of dicts with:
                {
                    'combined_mask': np.array[H,W,bool],
                    'bbox': dict,
                    'individual_masks': [np.array],
                    'mask_indices': [int],
                    'text': str,
                    'confidence': float,
                }.
        """
        results = (
            existing_results
            if existing_results is not None
            else self.detect_outside_text(
                image_path,
                verbose=verbose,
                min_size=min_size,
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

        log_message(
            "Converting EasyOCR results to axis-aligned boxes...", verbose=verbose
        )
        boxes = self.convert_easyocr_to_bboxes(results)

        # Expand and clip boxes
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

        # Group nearby text boxes spatially
        log_message(
            f"Grouping {len(expanded_boxes)} text boxes spatially...",
            verbose=verbose,
        )

        grouped_boxes = self._group_text_boxes_spatially(
            expanded_boxes, results, img_w, img_h, verbose
        )

        groups = []
        for group_boxes, group_results in grouped_boxes:
            # Create combined mask for the group
            combined_mask = np.zeros((img_h, img_w), dtype=bool)
            individual_masks = []
            mask_indices = []
            combined_text = []
            avg_confidence = 0.0

            # Calculate combined bounding box
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
                # Split large groups into smaller ones
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

                    _, text, conf = result

                    groups.append(
                        {
                            "combined_mask": mask,
                            "bbox": bbox,
                            "individual_masks": [mask],
                            "mask_indices": [i],
                            "text": text,
                            "confidence": conf,
                        }
                    )
                continue

            # Process each box in the group
            for i, (box, result) in enumerate(zip(group_boxes, group_results)):
                x0, y0, x1, y1 = box
                mask = np.zeros((img_h, img_w), dtype=bool)
                mask[y0:y1, x0:x1] = True

                combined_mask |= mask
                individual_masks.append(mask)
                mask_indices.append(i)

                _, text, conf = result
                combined_text.append(text)
                avg_confidence += conf

            # Create combined bounding box
            bbox = {
                "x": int(min_x),
                "y": int(min_y),
                "width": int(max_x - min_x),
                "height": int(max_y - min_y),
            }

            groups.append(
                {
                    "combined_mask": combined_mask,
                    "bbox": bbox,
                    "individual_masks": individual_masks,
                    "mask_indices": mask_indices,
                    "text": " ".join(combined_text),
                    "confidence": avg_confidence / len(group_results),
                }
            )

        log_message(
            f"Created {len(groups)} grouped text regions for inpainting",
            verbose=verbose,
        )

        return groups, image_pil

    def _group_text_boxes_spatially(self, boxes, results, img_w, img_h, verbose=False):
        """
        Group nearby text boxes based on spatial proximity.

        Args:
            boxes: List of bounding boxes [x0, y0, x1, y1]
            results: List of OCR results corresponding to boxes
            img_w: Image width
            img_h: Image height
            verbose: Whether to print detailed logs

        Returns:
            List of tuples (group_boxes, group_results) where each group contains
            spatially related text boxes
        """
        if not boxes:
            return []

        # Calculate proximity threshold based on image size
        proximity_threshold = min(img_w, img_h) * TEXT_BOX_PROXIMITY_RATIO

        # Use Union-Find to group nearby boxes
        parent = list(range(len(boxes)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Check proximity between all pairs of boxes
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if self._boxes_are_nearby(boxes[i], boxes[j], proximity_threshold):
                    union(i, j)

        # Group boxes by their root parent
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

        # Calculate center points
        cx1 = (x1_min + x1_max) / 2
        cy1 = (y1_min + y1_max) / 2
        cx2 = (x2_min + x2_max) / 2
        cy2 = (y2_min + y2_max) / 2

        # Calculate distance between centers
        distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

        return distance <= threshold
