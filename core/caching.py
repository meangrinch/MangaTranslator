import hashlib
import pickle
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image

from utils.logging import log_message


class UnifiedCache:
    """Unified cache for various MangaTranslator operations."""

    def __init__(self):
        """Initialize the unified cache."""
        from core.text.font_manager import LRUCache

        self._ocr_cache = LRUCache(max_size=1)
        self._yolo_cache = LRUCache(max_size=1)
        self._sam_cache = LRUCache(max_size=1)
        self._translation_cache = LRUCache(max_size=1)
        self._upscale_cache = LRUCache(max_size=20)
        self._inpaint_cache = LRUCache(max_size=20)
        self._current_image_hash = None

    def _hash_image(self, image: Image.Image) -> str:
        """Compute perceptual hash (pHash) of PIL Image.

        Args:
            image: PIL Image to hash

        Returns:
            str: Hash string (16 chars)
        """
        if image.mode == "RGBA":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])
            cv_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        elif image.mode == "L":
            cv_image = np.array(image)
        else:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        hasher = cv2.img_hash.PHash_create()
        hash_bytes = hasher.compute(cv_image)

        hash_hex = "".join(f"{byte:02x}" for byte in hash_bytes.flatten())

        # Include image metadata for additional uniqueness
        metadata = f"{image.mode}_{image.size[0]}_{image.size[1]}"
        combined_data = metadata.encode() + hash_hex.encode()

        return hashlib.blake2b(combined_data, digest_size=8).hexdigest()

    def _hash_numpy(self, array: np.ndarray) -> str:
        """Compute hash of numpy array using sampling for better performance.

        Args:
            array: Numpy array to hash

        Returns:
            str: Hash string (16 chars)
        """
        # Sample array data instead of hashing entire array
        if array.size == 0:
            return hashlib.blake2b(b"empty_array", digest_size=8).hexdigest()

        # Sample every nth element based on array size
        sample_size = min(1024, array.size // 4)  # Adaptive sample size
        step = max(1, array.size // sample_size)

        # Flatten and sample
        flat_array = array.flatten()
        sampled_data = flat_array[::step]

        # Include array metadata for uniqueness
        metadata = f"{array.shape}_{array.dtype}_{len(sampled_data)}"
        combined_data = metadata.encode() + sampled_data.tobytes()

        return hashlib.blake2b(combined_data, digest_size=8).hexdigest()

    def _hash_dict(self, data: Dict) -> str:
        """Compute hash of dictionary.

        Args:
            data: Dictionary to hash

        Returns:
            str: Hash string (16 chars)
        """
        data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def get_ocr_cache_key(
        self,
        image: Image.Image,
        languages: list,
        mag_ratio: float = 1.0,
        link_threshold: float = 0.1,
        decoder: str = "beamsearch",
        beamWidth: int = 10,
        rotation_info: list = None,
        min_size: int = 100,
    ) -> str:
        """Compute cache key for EasyOCR detection.

        Args:
            image: Input image
            languages: List of language codes
            mag_ratio: Magnification ratio
            link_threshold: Link threshold
            decoder: Decoder type
            beamWidth: Beam width for decoder
            rotation_info: Rotation information
            min_size: Minimum text size

        Returns:
            str: Cache key
        """
        image_hash = self._hash_image(image)
        params_str = (
            f"langs_{'_'.join(sorted(languages))}_"
            f"mag{mag_ratio:.2f}_link{link_threshold:.2f}_"
            f"dec{decoder}_beam{beamWidth}_"
            f"rot{rotation_info}_min{min_size}"
        )
        key_string = f"ocr_{image_hash}_{params_str}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_ocr_detection(self, cache_key: str) -> Optional[Any]:
        """Get cached OCR detection result.

        Args:
            cache_key: Cache key

        Returns:
            Cached OCR results or None if not found
        """
        return self._ocr_cache.get(cache_key)

    def set_ocr_detection(
        self, cache_key: str, results: Any, verbose: bool = False
    ) -> None:
        """Cache OCR detection result.

        Args:
            cache_key: Cache key
            results: OCR detection results to cache
            verbose: Whether to print verbose logging
        """
        self._ocr_cache.put(cache_key, results)
        log_message(
            f"  - Cached OCR detection (cache size: {len(self._ocr_cache.cache)})",
            verbose=verbose,
        )

    def get_yolo_cache_key(
        self, image: Image.Image, model_path: str, confidence: float
    ) -> str:
        """Compute cache key for YOLO detection.

        Args:
            image: Input image
            model_path: Path to YOLO model
            confidence: Confidence threshold

        Returns:
            str: Cache key
        """
        image_hash = self._hash_image(image)
        model_hash = hashlib.sha256(model_path.encode()).hexdigest()[:16]
        key_string = f"yolo_{image_hash}_{model_hash}_conf{confidence:.3f}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_yolo_detection(self, cache_key: str) -> Optional[Any]:
        """Get cached YOLO detection result.

        Args:
            cache_key: Cache key

        Returns:
            Cached YOLO results or None if not found
        """
        return self._yolo_cache.get(cache_key)

    def set_yolo_detection(
        self, cache_key: str, results: Any, verbose: bool = False
    ) -> None:
        """Cache YOLO detection result.

        Args:
            cache_key: Cache key
            results: YOLO detection results to cache
            verbose: Whether to print verbose logging
        """
        self._yolo_cache.put(cache_key, results)
        log_message(
            f"  - Cached YOLO detection (cache size: {len(self._yolo_cache.cache)})",
            verbose=verbose,
        )

    def get_sam_cache_key(
        self,
        image: Image.Image,
        yolo_boxes: Any,
        use_sam2: bool = True,
        conjoined_detection: bool = True,
    ) -> str:
        """Compute cache key for SAM segmentation.

        Args:
            image: Input image
            yolo_boxes: YOLO detection boxes (tensor or list)
            use_sam2: Whether SAM2 is enabled
            conjoined_detection: Whether conjoined detection is enabled

        Returns:
            str: Cache key
        """
        image_hash = self._hash_image(image)

        if hasattr(yolo_boxes, "cpu"):
            boxes_np = yolo_boxes.cpu().numpy()
        else:
            boxes_np = np.array(yolo_boxes)
        boxes_hash = self._hash_numpy(boxes_np)

        sam_model_id = "facebook/sam2.1-hiera-large"
        model_hash = hashlib.sha256(sam_model_id.encode()).hexdigest()[:8]
        key_string = (
            f"sam_{image_hash}_{boxes_hash}_{model_hash}_sam2{int(use_sam2)}"
            f"_conjoined{int(conjoined_detection)}"
        )
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_sam_masks(self, cache_key: str) -> Optional[Any]:
        """Get cached SAM masks.

        Args:
            cache_key: Cache key

        Returns:
            Cached SAM masks or None if not found
        """
        return self._sam_cache.get(cache_key)

    def set_sam_masks(self, cache_key: str, masks: Any, verbose: bool = False) -> None:
        """Cache SAM masks.

        Args:
            cache_key: Cache key
            masks: SAM masks to cache
            verbose: Whether to print verbose logging
        """
        self._sam_cache.put(cache_key, masks)
        log_message(
            f"  - Cached SAM masks (cache size: {len(self._sam_cache.cache)})",
            verbose=verbose,
        )

    def _is_deterministic(self, config) -> bool:
        """Check if translation config is deterministic.

        Args:
            config: TranslationConfig object

        Returns:
            bool: True if translation is deterministic
        """
        return config.temperature == 0.0 or config.top_k == 1 or config.top_p == 0.0

    def get_translation_cache_key(
        self,
        images_b64: list,
        full_image_b64: str,
        config,
    ) -> Optional[str]:
        """Compute cache key for LLM translation.

        Only returns a key if the config is deterministic.

        Args:
            images_b64: List of base64 encoded bubble images
            full_image_b64: Base64 encoded full page image
            config: TranslationConfig object

        Returns:
            str: Cache key, or None if not deterministic
        """
        if not self._is_deterministic(config):
            return None

        images_hash = hashlib.sha256("".join(images_b64).encode()).hexdigest()[:16]
        full_hash = hashlib.sha256(full_image_b64.encode()).hexdigest()[:16]

        cache_params = {
            "provider": config.provider,
            "model_name": config.model_name,
            "input_language": config.input_language,
            "output_language": config.output_language,
            "reading_direction": config.reading_direction,
            "translation_mode": config.translation_mode,
            "send_full_page_context": config.send_full_page_context,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        config_hash = self._hash_dict(cache_params)

        key_string = f"trans_{images_hash}_{full_hash}_{config_hash}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_translation(self, cache_key: Optional[str]) -> Optional[list]:
        """Get cached translation results.

        Args:
            cache_key: Cache key (can be None if not deterministic)

        Returns:
            Cached translations or None if not found
        """
        if cache_key is None:
            return None
        return self._translation_cache.get(cache_key)

    def set_translation(
        self, cache_key: Optional[str], translations: list, verbose: bool = False
    ) -> None:
        """Cache translation results.

        Args:
            cache_key: Cache key (can be None if not deterministic)
            translations: Translation results to cache
            verbose: Whether to print verbose logging
        """
        if cache_key is None:
            return
        self._translation_cache.put(cache_key, translations)
        log_message(
            f"  - Cached translation (cache size: {len(self._translation_cache.cache)})",
            verbose=verbose,
        )

    def get_upscale_cache_key(
        self, image: Image.Image, factor: float, model_type: str = "model"
    ) -> str:
        """Compute cache key for image upscaling.

        Args:
            image: Input image
            factor: Upscaling factor
            model_type: Model type identifier ("model" or "model_lite")

        Returns:
            str: Cache key
        """
        image_hash = self._hash_image(image)
        key_string = f"upscale_{image_hash}_factor{factor:.3f}_model{model_type}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_upscale_dimension_cache_key(
        self, image: Image.Image, target: int, mode: str, model_type: str = "model"
    ) -> str:
        """Compute cache key for image upscaling to dimension.

        Args:
            image: Input image
            target: Target dimension
            mode: Upscaling mode ('max' or 'min')
            model_type: Model type identifier ("model" or "model_lite")

        Returns:
            str: Cache key
        """
        image_hash = self._hash_image(image)
        key_string = (
            f"upscale_dim_{image_hash}_target{target}_mode{mode}_model{model_type}"
        )
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_bubble_processing_cache_key(
        self, image: Image.Image, target: int, mode: str, model_type: str = "model"
    ) -> str:
        """Compute cache key for complete bubble processing (upscale + color match).

        Args:
            image: Input image
            target: Target dimension
            mode: Upscaling mode ('max' or 'min')
            model_type: Model type identifier ("model" or "model_lite")

        Returns:
            str: Cache key
        """
        image_hash = self._hash_image(image)
        key_string = (
            f"bubble_proc_{image_hash}_target{target}_mode{mode}_model{model_type}"
        )
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_upscaled_image(self, cache_key: str) -> Optional[Image.Image]:
        """Get cached upscaled image.

        Args:
            cache_key: Cache key

        Returns:
            Cached upscaled image or None if not found
        """
        return self._upscale_cache.get(cache_key)

    def set_upscaled_image(
        self, cache_key: str, image: Image.Image, verbose: bool = False
    ) -> None:
        """Cache upscaled image.

        Args:
            cache_key: Cache key
            image: Upscaled image to cache
            verbose: Whether to print verbose logging
        """
        self._upscale_cache.put(cache_key, image)
        log_message(
            f"  - Cached upscaled image (cache size: {len(self._upscale_cache.cache)})",
            verbose=verbose,
        )

    def get_inpaint_cache_key(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int,
        num_inference_steps: int,
        residual_diff_threshold: float,
        guidance_scale: float,
        prompt: str,
        ocr_params: Optional[Dict] = None,
    ) -> str:
        """Compute cache key for Flux inpainting.

        Args:
            image: Input image
            mask: Mask array
            seed: Random seed
            num_inference_steps: Number of inference steps
            residual_diff_threshold: Residual diff threshold
            guidance_scale: Guidance scale
            prompt: Inpainting prompt
            ocr_params: Optional OCR parameters dict (e.g., {'min_size': 200})

        Returns:
            str: Cache key
        """
        image_hash = self._hash_image(image)
        mask_hash = self._hash_numpy(mask)

        # Include OCR parameters in cache key if provided
        ocr_params_str = ""
        if ocr_params:
            ocr_params_str = "_" + "_".join(
                f"{k}{v}" for k, v in sorted(ocr_params.items())
            )

        key_string = (
            f"inpaint_{image_hash}_{mask_hash}_"
            f"seed{seed}_steps{num_inference_steps}_"
            f"thresh{residual_diff_threshold:.3f}_"
            f"guide{guidance_scale:.2f}_"
            f"{prompt}{ocr_params_str}"
        )
        return hashlib.sha256(key_string.encode()).hexdigest()

    def should_use_inpaint_cache(self, seed: int) -> bool:
        """Determine if inpainting caching should be used.

        Args:
            seed: Random seed value

        Returns:
            bool: True if caching is enabled (seed != -1)
        """
        return seed != -1

    def get_inpainted_image(self, cache_key: str) -> Optional[Image.Image]:
        """Get cached inpainted image.

        Args:
            cache_key: Cache key

        Returns:
            Cached inpainted image or None if not found
        """
        return self._inpaint_cache.get(cache_key)

    def set_inpainted_image(
        self, cache_key: str, image: Image.Image, verbose: bool = False
    ) -> None:
        """Cache inpainted image.

        Args:
            cache_key: Cache key
            image: Inpainted image to cache
            verbose: Whether to print verbose logging
        """
        self._inpaint_cache.put(cache_key, image)
        log_message(
            f"  - Cached inpainted image (cache size: {len(self._inpaint_cache.cache)})",
            verbose=verbose,
        )

    def clear_ocr_cache(self, verbose: bool = False) -> None:
        """Clear OCR detection cache."""
        self._ocr_cache.cache.clear()
        log_message("OCR cache cleared", verbose=verbose)

    def clear_yolo_cache(self, verbose: bool = False) -> None:
        """Clear YOLO detection cache."""
        self._yolo_cache.cache.clear()
        log_message("YOLO cache cleared", verbose=verbose)

    def clear_sam_cache(self, verbose: bool = False) -> None:
        """Clear SAM masks cache."""
        self._sam_cache.cache.clear()
        log_message("SAM cache cleared", verbose=verbose)

    def clear_translation_cache(self, verbose: bool = False) -> None:
        """Clear translation cache."""
        self._translation_cache.cache.clear()
        log_message("Translation cache cleared", verbose=verbose)

    def clear_upscale_cache(self, verbose: bool = False) -> None:
        """Clear upscaling cache."""
        self._upscale_cache.cache.clear()
        log_message("Upscale cache cleared", verbose=verbose)

    def clear_inpaint_cache(self, verbose: bool = False) -> None:
        """Clear inpainting cache."""
        self._inpaint_cache.cache.clear()
        log_message("Inpaint cache cleared", verbose=verbose)

    def clear_all(self) -> None:
        """Clear all caches."""
        self.clear_ocr_cache(verbose=False)
        self.clear_yolo_cache(verbose=False)
        self.clear_sam_cache(verbose=False)
        self.clear_translation_cache(verbose=False)
        self.clear_upscale_cache(verbose=False)
        self.clear_inpaint_cache(verbose=False)
        log_message("All caches cleared", always_print=True)

    def set_current_image(self, image: Image.Image, verbose: bool = False) -> None:
        """Set the current image being processed and clear caches if different.

        Args:
            image: The current image being processed
            verbose: Whether to print verbose logging
        """
        image_hash = self._hash_image(image)

        if self._current_image_hash is None:
            # First image
            self._current_image_hash = image_hash
            log_message("Cache initialized for new image", verbose=verbose)
        elif self._current_image_hash != image_hash:
            # Different image detected - clear all caches
            log_message(
                "Different image detected - clearing all caches", verbose=verbose
            )
            self.clear_all()
            self._current_image_hash = image_hash
        else:
            # Same image - no action needed
            log_message("Same image detected - reusing caches", verbose=verbose)

    def get_cache_stats(self) -> dict:
        """Get statistics about cache sizes.

        Returns:
            dict: Cache statistics
        """
        return {
            "ocr": len(self._ocr_cache.cache),
            "yolo": len(self._yolo_cache.cache),
            "sam": len(self._sam_cache.cache),
            "translation": len(self._translation_cache.cache),
            "upscale": len(self._upscale_cache.cache),
            "inpaint": len(self._inpaint_cache.cache),
        }


_global_cache = None


def get_cache() -> UnifiedCache:
    """Get the global cache instance.

    Returns:
        UnifiedCache: The global cache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = UnifiedCache()
    return _global_cache
