import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from core.caching import get_cache
from core.device import empty_cache, get_best_device, get_best_dtype
from core.ml.model_manager import get_model_manager
from utils.logging import log_message

# Blur Parameters
BLUR_SCALE_FACTOR = (
    0.1  # Multiplier for bounding box dimensions to calculate blur radius
)
MIN_BLUR_RADIUS = 1  # Minimum blur radius in pixels
MAX_BLUR_RADIUS = 10  # Maximum blur radius in pixels

# Inpainting Parameters
FLUX_GUIDANCE_SCALE = 2.5  # Flux Kontext guidance scale
CONTEXT_PADDING_RATIO = 0.5  # Context padding is 50% of detection size
MAX_CONTEXT_PADDING = 80  # Context padding capped at 80 pixels


class FluxKontextInpainter:
    """Inpainter using Flux Kontext models for text removal."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        huggingface_token: str = "",
        num_inference_steps: int = 8,
        residual_diff_threshold: float = 0.15,
        backend: str = "nunchaku",
        low_vram: bool = False,
    ):
        """Initialize the Flux Kontext Inpaint class.

        Args:
            device: PyTorch device to use. Auto-detects if None.
            huggingface_token: HuggingFace token for model downloads (Nunchaku only).
            num_inference_steps: Number of denoising steps for inference.
            residual_diff_threshold: Residual diff threshold for Flux caching (Nunchaku only).
            backend: 'nunchaku' (CUDA + Nunchaku + HF token) or 'sdnq' (cross-platform).
            low_vram: If True, use sequential CPU offload (SDNQ only).
        """
        self.DEVICE = device if device is not None else get_best_device()
        self.DTYPE = get_best_dtype(self.DEVICE)
        self.huggingface_token = huggingface_token
        self.num_inference_steps = num_inference_steps
        self.residual_diff_threshold = residual_diff_threshold
        self.backend = backend
        self.low_vram = low_vram
        self.manager = get_model_manager()
        self.cache = get_cache()

        # Preferred resolutions for optimal Flux performance
        self.PREFERED_KONTEXT_RESOLUTIONS = [
            (672, 1568),
            (688, 1504),
            (720, 1456),
            (752, 1392),
            (800, 1328),
            (832, 1248),
            (880, 1184),
            (944, 1104),
            (1024, 1024),
            (1104, 944),
            (1184, 880),
            (1248, 832),
            (1328, 800),
            (1392, 752),
            (1456, 720),
            (1504, 688),
            (1568, 672),
        ]

        self.pipeline = None
        self.transformer = None
        self.text_encoder_2 = None

        # Fixed parameters optimized for text removal
        self.guidance_scale = FLUX_GUIDANCE_SCALE
        self.prompt = "Remove all text."
        self.context_padding_ratio = CONTEXT_PADDING_RATIO
        self.max_context_padding = MAX_CONTEXT_PADDING

    def load_models(self):
        """Load Flux Kontext models via model manager."""
        if self.pipeline is not None:
            return

        if self.backend == "sdnq":
            # SDNQ: cross-platform, no token required
            self.pipeline = self.manager.load_flux_kontext_sdnq(
                low_vram=self.low_vram,
                verbose=True,
            )
            # SDNQ uses CPU offload, no separate transformer/text_encoder management
            self.transformer = None
            self.text_encoder_2 = None
        else:
            # Nunchaku: CUDA-only, requires token
            if self.huggingface_token:
                self.manager.set_flux_hf_token(self.huggingface_token)

            self.manager.set_flux_residual_diff_threshold(self.residual_diff_threshold)

            self.transformer, self.text_encoder_2, self.pipeline = (
                self.manager.load_flux_models()
            )

    def unload_models(self):
        """Unload Flux models via model manager to free up memory."""
        self.pipeline = None
        self.transformer = None
        self.text_encoder_2 = None

        if self.backend == "sdnq":
            self.manager.unload_flux_kontext_sdnq_models()
        else:
            self.manager.unload_flux_models()

    def convert_mask_to_tensor(self, mask_np):
        """Convert a numpy mask to the tensor format expected by the pipeline.

        Args:
            mask_np: Numpy mask array (H, W) with True/False values

        Returns:
            torch.Tensor: Mask tensor in CHW format (1.0 for areas to keep, 0.0 for areas to inpaint)
        """
        # Invert mask: True = inpaint (0.0), False = keep (1.0)
        mask_float = mask_np.astype(np.float32)
        mask_inverted = 1.0 - mask_float
        mask_tensor = torch.from_numpy(mask_inverted).unsqueeze(0)

        return mask_tensor

    def flux_kontext_image_scale(self, image_pil):
        """Find the closest preferred resolution and resize the image.

        Args:
            image_pil (PIL.Image): Input image to scale

        Returns:
            PIL.Image: Scaled image at the closest preferred resolution
        """
        w_in, h_in = image_pil.size
        if w_in == 0 or h_in == 0:
            return image_pil

        ar = w_in / h_in
        # Find resolution with minimum aspect ratio difference
        _, w_opt, h_opt = min(
            (abs(ar - w / h), w, h) for (w, h) in self.PREFERED_KONTEXT_RESOLUTIONS
        )

        log_message(
            f"  - Original image size: {w_in}x{h_in} (AR: {ar:.2f})", always_print=True
        )
        log_message(
            f"  - Scaling to nearest preferred resolution: {w_opt}x{h_opt}",
            always_print=True,
        )

        if (w_in, h_in) == (w_opt, h_opt):
            return image_pil

        # Use LANCZOS for high-quality downscaling
        image_scaled = image_pil.resize((w_opt, h_opt), Image.Resampling.LANCZOS)

        return image_scaled

    def compute_mask_bbox_aspect_ratio(
        self,
        mask_chw,
        padding,
        blur_radius,
        target_ar=None,
        transpose=False,
        preferred_resolutions=None,
        verbose=False,
    ):
        """Compute an optimized bounding box for the mask with aspect ratio adjustment.

        Args:
            mask_chw (torch.Tensor): Input mask tensor in CHW format
            padding (int): Padding around the mask bounding box
            blur_radius (int): Radius for edge blur effect
            target_ar (float, optional): Target aspect ratio
            transpose (bool): Whether to transpose the aspect ratio logic
            preferred_resolutions (list, optional): List of preferred resolutions
            verbose (bool): Whether to print verbose output

        Returns:
            tuple: (mask_for_composite, x, y, width, height)
        """
        if mask_chw.dim() == 4:
            mask = mask_chw[0, 0]
        else:
            mask = mask_chw[0]

        H, W = mask.shape[0], mask.shape[1]
        hard = mask.clone().unsqueeze(0)
        if blur_radius > 0:
            # Create smooth falloff at mask edges for better blending
            m_bool = hard[0].cpu().to(torch.float32).numpy().astype(bool)
            d_out = distance_transform_edt(~m_bool)
            d_in = distance_transform_edt(m_bool)
            alpha = np.zeros_like(d_out, np.float32)
            alpha[d_in > 0] = 1.0
            ramp = np.clip(1.0 - (d_out / blur_radius), 0.0, 1.0)
            alpha[d_out > 0] = ramp[d_out > 0]
            mask_blur_full = torch.from_numpy(alpha)[None, ...].to(hard.device)
        else:
            mask_blur_full = hard.clone()

        ys, xs = torch.where(hard[0] > 0)
        if len(ys) == 0:
            return (
                torch.zeros((1, H, W), device=mask_chw.device, dtype=mask_chw.dtype),
                0,
                0,
                W,
                H,
            )

        x1 = max(0, int(xs.min()) - padding)
        x2 = min(W, int(xs.max()) + 1 + padding)
        y1 = max(0, int(ys.min()) - padding)
        y2 = min(H, int(ys.max()) + 1 + padding)
        w0 = x2 - x1
        h0 = y2 - y1

        if preferred_resolutions:
            if h0 == 0:
                initial_ar = W / H
            else:
                initial_ar = w0 / h0
            log_message(
                f"  - Initial mask bounding box AR: {initial_ar:.2f}",
                verbose=verbose,
            )

            # Snap to closest preferred aspect ratio
            _, w_opt, h_opt = min(
                (abs(initial_ar - w / h), w, h) for (w, h) in preferred_resolutions
            )
            ar = w_opt / h_opt
            log_message(
                f"  - Snapping to closest preferred AR: {ar:.2f} ({w_opt}x{h_opt})",
                verbose=verbose,
            )
        else:
            ar = target_ar

        req_w = math.ceil(h0 * ar)
        req_h = math.floor(w0 / ar)

        new_x1, new_x2 = x1, x2
        new_y1, new_y2 = y1, y2

        flush_left = x1 == 0
        flush_right = x2 == W
        flush_top = y1 == 0
        flush_bot = y2 == H

        if not transpose:
            if req_w > w0:
                target_w = min(W, req_w)
                delta = target_w - w0
                if flush_right:
                    new_x1, new_x2 = W - target_w, W
                elif flush_left:
                    new_x1, new_x2 = 0, target_w
                else:
                    off = delta // 2
                    new_x1 = max(0, x1 - off)
                    new_x2 = new_x1 + target_w
                    if new_x2 > W:
                        new_x2 = W
                        new_x1 = W - target_w

            elif req_h > h0:
                target_h = min(H, req_h)
                delta = target_h - h0
                if flush_bot:
                    new_y1, new_y2 = H - target_h, H
                elif flush_top:
                    new_y1, new_y2 = 0, target_h
                else:
                    off = delta // 2
                    new_y1 = max(0, y1 - off)
                    new_y2 = new_y1 + target_h
                    if new_y2 > H:
                        new_y2 = H
                        new_y1 = H - target_h

        else:  # Transpose logic
            if req_h > h0:
                target_h = min(H, req_h)
                delta = target_h - h0
                if flush_bot:
                    new_y1, new_y2 = H - target_h, H
                elif flush_top:
                    new_y1, new_y2 = 0, target_h
                else:
                    off = delta // 2
                    new_y1 = max(0, y1 - off)
                    new_y2 = new_y1 + target_h
                    if new_y2 > H:
                        new_y2 = H
                        new_y1 = H - target_h

            elif req_w > w0:
                target_w = min(W, req_w)
                delta = target_w - w0
                if flush_right:
                    new_x1, new_x2 = W - target_w, W
                elif flush_left:
                    new_x1, new_x2 = 0, target_w
                else:
                    off = delta // 2
                    new_x1 = max(0, x1 - off)
                    new_x2 = new_x1 + target_w
                    if new_x2 > W:
                        new_x2 = W
                        new_x1 = W - target_w

        final_w = new_x2 - new_x1
        final_h = new_y2 - new_y1

        # Return cropped mask for compositing
        mask_for_composite = mask_blur_full[:, new_y1:new_y2, new_x1:new_x2]

        return (
            mask_for_composite.to(mask_chw.device, dtype=mask_chw.dtype),
            int(new_x1),
            int(new_y1),
            int(final_w),
            int(final_h),
        )

    def image_alpha_fix(self, destination, source):
        """Ensure destination and source tensors have compatible channel dimensions.

        Args:
            destination (torch.Tensor): Destination tensor
            source (torch.Tensor): Source tensor

        Returns:
            tuple: (destination, source) with compatible dimensions
        """
        dest_channels = destination.shape[-1]
        source_channels = source.shape[-1]

        if dest_channels == source_channels:
            return destination, source

        if dest_channels > source_channels:
            # Pad source to match destination's channel count
            padding = torch.ones(
                (*source.shape[:-1], dest_channels - source_channels),
                device=source.device,
                dtype=source.dtype,
            )
            source = torch.cat([source, padding], dim=-1)
        else:  # source_channels > dest_channels
            # Truncate source to match destination's channel count
            source = source[..., :dest_channels]

        return destination, source

    def repeat_to_batch_size(self, tensor, batch_size):
        """Adjust tensor batch size by repeating or truncating as needed.

        Args:
            tensor (torch.Tensor): Input tensor
            batch_size (int): Target batch size

        Returns:
            torch.Tensor: Tensor with the specified batch size
        """
        if tensor.shape[0] > batch_size:
            return tensor[:batch_size]
        elif tensor.shape[0] < batch_size:
            return tensor.repeat(batch_size, 1, 1, 1)
        return tensor

    def composite(
        self, destination, source, x, y, mask=None, multiplier=1, resize_source=False
    ):
        """Composite source image onto destination at specified coordinates.

        Args:
            destination (torch.Tensor): Destination image tensor
            source (torch.Tensor): Source image tensor
            x (int): X coordinate for placement
            y (int): Y coordinate for placement
            mask (torch.Tensor, optional): Alpha mask for blending
            multiplier (int): Coordinate multiplier
            resize_source (bool): Whether to resize source to match destination

        Returns:
            torch.Tensor: Composited image tensor
        """
        source = source.to(destination.device)
        if resize_source:
            source = torch.nn.functional.interpolate(
                source,
                size=(destination.shape[2], destination.shape[3]),
                mode="bilinear",
            )

        source = self.repeat_to_batch_size(source, destination.shape[0])

        x = max(
            -source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier)
        )
        y = max(
            -source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier)
        )

        left, top = (x // multiplier, y // multiplier)

        if mask is None:
            mask = torch.ones_like(source)
        else:
            mask = mask.to(destination.device, copy=True)
            mask = torch.nn.functional.interpolate(
                mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                size=(source.shape[2], source.shape[3]),
                mode="bilinear",
            )
            mask = self.repeat_to_batch_size(mask, source.shape[0])

        visible_width = max(0, min(source.shape[3], destination.shape[3] - left))
        visible_height = max(0, min(source.shape[2], destination.shape[2] - top))

        if visible_width == 0 or visible_height == 0:
            return destination

        source_portion = source[:, :, :visible_height, :visible_width]
        mask_portion = mask[:, :, :visible_height, :visible_width]
        inverse_mask_portion = torch.ones_like(mask_portion) - mask_portion

        destination_portion = destination[
            :, :, top : top + visible_height, left : left + visible_width
        ]
        # Alpha blend source and destination using mask
        blended_portion = (source_portion * mask_portion) + (
            destination_portion * inverse_mask_portion
        )
        destination[:, :, top : top + visible_height, left : left + visible_width] = (
            blended_portion
        )

        return destination

    def image_composite_masked(
        self, destination, source, x, y, resize_source, mask=None
    ):
        """Wrapper function that handles channel dimension compatibility.

        Args:
            destination (torch.Tensor): Destination image tensor
            source (torch.Tensor): Source image tensor
            x (int): X coordinate for placement
            y (int): Y coordinate for placement
            resize_source (bool): Whether to resize source to match destination
            mask (torch.Tensor, optional): Alpha mask for blending

        Returns:
            torch.Tensor: Composited image tensor
        """
        destination, source = self.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        output = self.composite(
            destination, source.movedim(-1, 1), x, y, mask, 1, resize_source
        ).movedim(1, -1)
        return output

    def inpaint_mask(
        self,
        image_pil: Image.Image,
        mask_np: np.ndarray,
        seed: int = 1,
        verbose: bool = False,
        ocr_params: Optional[Dict] = None,
        strict_mask_clipping: bool = False,
        composite_clip_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Image.Image:
        """Inpaint a specific mask region in the image.

        Args:
            image_pil: PIL Image to inpaint
            mask_np: Numpy mask array (H, W) with True for areas to inpaint
            seed: Random seed for inference
            verbose: Whether to print verbose output
            ocr_params: Optional OCR parameters dict for cache key generation
            strict_mask_clipping: When True, ensure compositing is limited to the
                original mask extent (no bleed from padding/blur)
            composite_clip_bbox: Optional (x1, y1, x2, y2) bbox to clip the final
                composite mask to, in original image coordinates.

        Returns:
            PIL.Image: The inpainted image
        """
        mask_np = np.asarray(mask_np)
        if mask_np.dtype != bool:
            mask_np = mask_np.astype(bool)

        if not np.any(mask_np):
            return image_pil

        log_message(
            "  - Computing optimized mask bounding box with blur and aspect ratio...",
            verbose=verbose,
        )

        ys, xs = np.where(mask_np)
        if len(ys) == 0 or len(xs) == 0:
            return image_pil

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        padding_pixels = int(max(bbox_width, bbox_height) * self.context_padding_ratio)
        padding = min(padding_pixels, self.max_context_padding)
        log_message(
            f"  - Proportional context padding: {padding_pixels}px, capped to: {padding}px",
            verbose=verbose,
        )

        blur_radius = int(max(bbox_width, bbox_height) * BLUR_SCALE_FACTOR)
        blur_radius = max(
            MIN_BLUR_RADIUS, min(blur_radius, MAX_BLUR_RADIUS)
        )  # clamp between MIN and MAX
        log_message(f"  - Dynamic blur radius set to: {blur_radius}", verbose=verbose)

        mask_tensor = (
            torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        )

        mask_for_composite, x, y, width, height = self.compute_mask_bbox_aspect_ratio(
            mask_chw=mask_tensor,
            padding=padding,
            blur_radius=blur_radius,
            preferred_resolutions=self.PREFERED_KONTEXT_RESOLUTIONS,
            transpose=False,
            verbose=verbose,
        )

        # Quantize bbox to improve cache stability against minor detection jitter
        quant = 2
        img_h, img_w = mask_np.shape
        qx1 = max(0, min(img_w, int(round(x / quant) * quant)))
        qy1 = max(0, min(img_h, int(round(y / quant) * quant)))
        qx2 = max(qx1 + 1, min(img_w, int(round((x + width) / quant) * quant)))
        qy2 = max(qy1 + 1, min(img_h, int(round((y + height) / quant) * quant)))
        qwidth = max(1, qx2 - qx1)
        qheight = max(1, qy2 - qy1)

        # Adjust mask_for_composite to the quantized bbox via pad/crop
        dx_left = x - qx1
        dy_top = y - qy1
        dx_right = (qx1 + qwidth) - (x + width)
        dy_bottom = (qy1 + qheight) - (y + height)

        if dx_left > 0 or dx_right > 0 or dy_top > 0 or dy_bottom > 0:
            pad_l = max(dx_left, 0)
            pad_r = max(dx_right, 0)
            pad_t = max(dy_top, 0)
            pad_b = max(dy_bottom, 0)
            mask_for_composite = torch.nn.functional.pad(
                mask_for_composite, (pad_l, pad_r, pad_t, pad_b)
            )

        if dx_left < 0:
            mask_for_composite = mask_for_composite[:, :, -dx_left:]
        if dy_top < 0:
            mask_for_composite = mask_for_composite[:, -dy_top:, :]
        if mask_for_composite.shape[-1] > qwidth:
            mask_for_composite = mask_for_composite[:, :, :qwidth]
        if mask_for_composite.shape[-2] > qheight:
            mask_for_composite = mask_for_composite[:, :qheight, :]

        x, y, width, height = qx1, qy1, qwidth, qheight

        if strict_mask_clipping:
            original_mask_crop = mask_tensor[0, 0, y : y + height, x : x + width]
            mask_for_composite = mask_for_composite * original_mask_crop

        if composite_clip_bbox is not None:
            clip_x1, clip_y1, clip_x2, clip_y2 = composite_clip_bbox

            img_h, img_w = mask_np.shape
            clip_x1 = max(0, min(img_w, clip_x1))
            clip_x2 = max(0, min(img_w, clip_x2))
            clip_y1 = max(0, min(img_h, clip_y1))
            clip_y2 = max(0, min(img_h, clip_y2))

            start_x = max(0, clip_x1 - x)
            end_x = min(width, clip_x2 - x)
            start_y = max(0, clip_y1 - y)
            end_y = min(height, clip_y2 - y)

            if end_x <= start_x or end_y <= start_y:
                mask_for_composite = torch.zeros_like(mask_for_composite)
            else:
                clipped_mask = torch.zeros_like(mask_for_composite)
                clipped_mask[:, start_y:end_y, start_x:end_x] = mask_for_composite[
                    :, start_y:end_y, start_x:end_x
                ]
                mask_for_composite = clipped_mask

        log_message(
            f"  - Optimized bbox found at ({x}, {y}) with size {width}x{height}",
            verbose=verbose,
        )

        image_cropped_pil = image_pil.crop((x, y, x + width, y + height))
        mask_crop_np = mask_np[y : y + height, x : x + width]

        cache_params = {
            "bbox": (x, y, width, height),
            "padding": padding,
            "blur": blur_radius,
        }
        if strict_mask_clipping:
            cache_params["strict_clip"] = True
        if composite_clip_bbox is not None:
            cache_params["clip_bbox"] = tuple(composite_clip_bbox)
        if ocr_params:
            cache_params.update(ocr_params)

        cache_key = None
        cached_patch = None
        if self.cache.should_use_inpaint_cache(seed):
            # Downsample mask signature to reduce sensitivity to minor jitter
            if mask_crop_np.size > 0:
                sig_h = min(64, max(4, mask_crop_np.shape[0]))
                sig_w = min(64, max(4, mask_crop_np.shape[1]))
                mask_sig = (
                    torch.from_numpy(mask_crop_np.astype(np.float32))
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                mask_sig = torch.nn.functional.interpolate(
                    mask_sig, size=(sig_h, sig_w), mode="bilinear", align_corners=False
                )
                mask_sig_np = (mask_sig > 0.5).cpu().numpy().astype(np.uint8)[0, 0]
            else:
                mask_sig_np = mask_crop_np

            cache_key = self.cache.get_inpaint_cache_key(
                image_cropped_pil,
                mask_sig_np,
                seed,
                self.num_inference_steps,
                self.residual_diff_threshold,
                self.guidance_scale,
                self.prompt,
                cache_params,
            )
            cached_patch = self.cache.get_inpainted_image(cache_key)
            if cached_patch is not None:
                log_message("  - Using cached inpainting patch", verbose=verbose)

        patch_pil = cached_patch

        if patch_pil is None:
            self.load_models()

            if self.pipeline is None:
                log_message(
                    "Warning: Flux Kontext pipeline not available. Skipping inpainting.",
                    always_print=True,
                )
                return image_pil

            image_scaled_for_inference_pil = self.flux_kontext_image_scale(
                image_cropped_pil
            )
            inference_width, inference_height = image_scaled_for_inference_pil.size

            if image_scaled_for_inference_pil.mode == "RGBA":
                image_scaled_for_inference_pil = image_scaled_for_inference_pil.convert(
                    "RGB"
                )

            log_message("  - Running inference...", verbose=verbose)

            required_area = inference_width * inference_height

            if self.backend == "sdnq":
                # SDNQ: CPU offload handles device management automatically
                with torch.inference_mode():
                    gen_device = "cpu" if self.low_vram else self.DEVICE
                    gen = torch.Generator(device=gen_device).manual_seed(seed)
                    out = self.pipeline(
                        prompt=self.prompt,
                        image=image_scaled_for_inference_pil,
                        width=inference_width,
                        height=inference_height,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        generator=gen,
                        output_type="pt",
                        max_area=required_area,
                    )
                    img = out.images[0]
                    torch.nan_to_num_(img, nan=0.0, posinf=1.0, neginf=0.0)
                    img.clamp_(0, 1)
                    generated_patch_pil = Image.fromarray(
                        (
                            img.mul(255)
                            .round()
                            .to(torch.uint8)
                            .permute(1, 2, 0)
                            .cpu()
                            .numpy()
                        )
                    )
            else:
                # Nunchaku: Manual device management for memory efficiency
                self.pipeline.text_encoder_2.to(self.DEVICE)

                prompt_embeds, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                    prompt=self.prompt,
                    prompt_2=None,
                    device=self.DEVICE,
                )

                self.pipeline.text_encoder_2.to("cpu")
                empty_cache(self.DEVICE)

                self.pipeline.transformer.to(self.DEVICE)

                with torch.inference_mode():
                    gen = torch.Generator(device=self.DEVICE).manual_seed(seed)
                    out = self.pipeline(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        image=image_scaled_for_inference_pil,
                        width=inference_width,
                        height=inference_height,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        generator=gen,
                        output_type="pt",
                        max_area=required_area,
                    )
                    img = out.images[0]
                    torch.nan_to_num_(img, nan=0.0, posinf=1.0, neginf=0.0)
                    img.clamp_(0, 1)
                    generated_patch_pil = Image.fromarray(
                        (
                            img.mul(255)
                            .round()
                            .to(torch.uint8)
                            .permute(1, 2, 0)
                            .cpu()
                            .numpy()
                        )
                    )

                self.pipeline.transformer.to("cpu")
                empty_cache(self.DEVICE)

            patch_pil = generated_patch_pil.resize(
                (width, height), Image.Resampling.LANCZOS
            )

        dest_tensor = torch.from_numpy(
            np.asarray(image_pil, dtype=np.float32) / 255.0
        ).unsqueeze(0)
        src_tensor = torch.from_numpy(
            np.asarray(patch_pil, dtype=np.float32) / 255.0
        ).unsqueeze(0)

        composited_tensor = self.image_composite_masked(
            destination=dest_tensor,
            source=src_tensor,
            x=x,
            y=y,
            resize_source=False,
            mask=mask_for_composite,
        )

        composited_pil = Image.fromarray(
            (composited_tensor[0].cpu().numpy() * 255).astype("uint8")
        )

        if (
            self.cache.should_use_inpaint_cache(seed)
            and cache_key is not None
            and cached_patch is None
        ):
            self.cache.set_inpainted_image(cache_key, patch_pil)

        return composited_pil


class FluxKleinInpainter:
    """Inpainter using Flux.2 Klein models for text removal.

    Unlike FluxKontextInpainter, this works on CPU/ROCm/MPS without Nunchaku.
    Uses FP8 quantized models for reduced memory usage.
    """

    # Parameters for Klein models (distilled, optimized)
    KLEIN_MAX_STEPS = 12  # Max steps for Klein models
    KLEIN_DEFAULT_STEPS = 4  # Recommended default
    KLEIN_GUIDANCE_SCALE = 1.0  # Fixed CFG for Klein
    KLEIN_PROMPT = (
        "Remove all text. Preserve all character line art, screentones, panel borders, "
        "and background details exactly as they appear. Maintain the original black-and-white "
        "contrast and shading, ensuring character expressions and environmental textures "
        "remain unchanged while leaving the text areas completely blank."
    )

    # Resolution constraints: 64x64 to 2048x2048, multiple of 16
    MIN_RESOLUTION = 64
    MAX_RESOLUTION = 2048
    RESOLUTION_MULTIPLE = 16
    KLEIN_PADDING_MULTIPLIER = 2.0  # Double padding vs Kontext for more context

    def __init__(
        self,
        variant: str = "4b",
        device: Optional[torch.device] = None,
        huggingface_token: str = "",
        num_inference_steps: int = 4,
        low_vram: bool = False,
        verbose: bool = False,
    ):
        """Initialize the Flux Klein Inpainter.

        Args:
            variant: Model variant - "9b" or "4b" (default: "4b")
            device: PyTorch device to use. Auto-detects if None.
            huggingface_token: HuggingFace token (required for 9B gated repo).
            num_inference_steps: Number of denoising steps (1-12, default: 4).
            low_vram: If True, use sequential CPU offload (slower but lower VRAM).
            verbose: Whether to print verbose logging.
        """
        self.variant = variant.lower()
        if self.variant not in ("9b", "4b"):
            raise ValueError(f"Invalid variant '{variant}'. Must be '9b' or '4b'.")

        self.num_inference_steps = num_inference_steps
        self.low_vram = low_vram
        self.verbose = verbose

        self.DEVICE = device if device is not None else get_best_device()
        self.DTYPE = get_best_dtype(self.DEVICE)
        self.huggingface_token = huggingface_token
        self.manager = get_model_manager()
        self.cache = get_cache()
        self.pipeline = None

    def load_models(self):
        """Load Flux Klein models via model manager."""
        if self.pipeline is not None:
            return

        if self.huggingface_token:
            self.manager.set_flux_hf_token(self.huggingface_token)

        if self.variant == "9b":
            self.pipeline = self.manager.load_flux_klein_9b(
                low_vram=self.low_vram, verbose=self.verbose
            )
        else:
            self.pipeline = self.manager.load_flux_klein_4b(
                low_vram=self.low_vram, verbose=self.verbose
            )

    def unload_models(self):
        """Unload Flux Klein models via model manager to free up memory."""
        self.pipeline = None
        self.manager.unload_flux_klein_models()

    def _quantize_dimension(self, dim: int) -> int:
        """Quantize dimension to be a multiple of 16 within allowed range."""
        dim = max(self.MIN_RESOLUTION, min(self.MAX_RESOLUTION, dim))
        return (dim // self.RESOLUTION_MULTIPLE) * self.RESOLUTION_MULTIPLE

    def _compute_luminance(self, image_np: np.ndarray, mask_np: np.ndarray) -> float:
        """Compute mean luminance (LAB L channel) of masked region.

        Args:
            image_np: RGB image array (H, W, 3), uint8
            mask_np: Boolean mask where True = pixels to include

        Returns:
            Mean luminance value (0-255 scale, cv2 LAB)
        """
        if not np.any(mask_np):
            return 127.5  # Neutral fallback

        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        return float(np.mean(l_channel[mask_np]))

    def _match_luminance(
        self,
        generated_pil: Image.Image,
        original_crop_pil: Image.Image,
        mask_crop_np: np.ndarray,
        padded_bounds: Tuple[int, int, int, int],
        original_bounds: Tuple[int, int, int, int],
    ) -> Image.Image:
        """Match luminance of generated patch to original context ring.

        Samples luminance from the "context ring" - the area between the mask
        bounds and the padded crop bounds, excluding the mask itself. This
        captures the local lighting context that the inpainted region should
        blend into.

        Args:
            generated_pil: Generated patch from Flux Klein (at crop size)
            original_crop_pil: Original cropped region
            mask_crop_np: Boolean mask of inpaint region within crop
            padded_bounds: (x1, y1, x2, y2) of padded bbox in original image
            original_bounds: (x_min, y_min, x_max, y_max) of mask in original image

        Returns:
            Luminance-corrected generated patch
        """
        original_np = np.asarray(original_crop_pil)
        generated_np = np.asarray(generated_pil).copy()

        crop_h, crop_w = mask_crop_np.shape
        pad_x1, pad_y1, pad_x2, pad_y2 = padded_bounds
        orig_x_min, orig_y_min, orig_x_max, orig_y_max = original_bounds

        ring_x1 = max(0, orig_x_min - pad_x1)
        ring_y1 = max(0, orig_y_min - pad_y1)
        ring_x2 = min(crop_w, orig_x_max + 1 - pad_x1)
        ring_y2 = min(crop_h, orig_y_max + 1 - pad_y1)

        context_ring_mask = np.ones((crop_h, crop_w), dtype=bool)
        context_ring_mask[ring_y1:ring_y2, ring_x1:ring_x2] = False
        context_ring_mask[mask_crop_np] = False

        if not np.any(context_ring_mask):
            return generated_pil

        original_lum = self._compute_luminance(original_np, context_ring_mask)
        generated_lum = self._compute_luminance(generated_np, context_ring_mask)

        lum_shift = original_lum - generated_lum

        # Threshold ~1.3 in 0-255 scale ≈ 0.5 in 0-100 scale
        if abs(lum_shift) < 1.3:
            return generated_pil

        log_message(
            f"  - Luminance correction: {lum_shift:+.1f} (orig={original_lum:.1f}, gen={generated_lum:.1f})",
            verbose=True,
        )

        lab = cv2.cvtColor(generated_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + lum_shift, 0, 255)
        lab = lab.astype(np.uint8)
        corrected_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(corrected_np)

    def _prepare_image_for_inference(
        self, image_pil: Image.Image, verbose: bool = False
    ) -> Tuple[Image.Image, int, int]:
        """Prepare image for Klein inference by scaling to ~1MP.

        Scales the image to approximately 1 megapixel while maintaining
        aspect ratio and ensuring dimensions are multiples of 16.

        Args:
            image_pil: Input PIL image
            verbose: Whether to print verbose output

        Returns:
            Tuple of (prepared image, original width, original height)
        """
        orig_w, orig_h = image_pil.size
        current_pixels = orig_w * orig_h
        target_pixels = 1_048_576  # 2^20 = 1024x1024

        if current_pixels > 0:
            scale = math.sqrt(target_pixels / current_pixels)
        else:
            scale = 1.0

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        new_w = self._quantize_dimension(new_w)
        new_h = self._quantize_dimension(new_h)

        if (new_w, new_h) != (orig_w, orig_h):
            log_message(
                f"  - Scaling {orig_w}x{orig_h} -> {new_w}x{new_h} (~1MP, multiples of 16)",
                verbose=verbose,
            )
            image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return image_pil, orig_w, orig_h

    def inpaint_mask(
        self,
        image_pil: Image.Image,
        mask_np: np.ndarray,
        seed: int = 1,
        verbose: bool = False,
        strict_mask_clipping: bool = False,
        composite_clip_bbox: Optional[Tuple[int, int, int, int]] = None,
        ocr_params: Optional[Dict] = None,
    ) -> Image.Image:
        """Inpaint a specific mask region in the image using Flux.2 Klein.

        Args:
            image_pil: PIL Image to inpaint
            mask_np: Numpy mask array (H, W) with True for areas to inpaint
            seed: Random seed for inference
            verbose: Whether to print verbose output
            strict_mask_clipping: When True, ensure compositing is limited to the
                original mask extent
            composite_clip_bbox: Optional (x1, y1, x2, y2) bbox to clip the final
                composite mask to
            ocr_params: Optional OCR parameters dict for cache key generation

        Returns:
            PIL.Image: The inpainted image
        """
        from scipy.ndimage import distance_transform_edt

        mask_np = np.asarray(mask_np)
        if mask_np.dtype != bool:
            mask_np = mask_np.astype(bool)

        if not np.any(mask_np):
            return image_pil

        log_message(
            f"  - Flux.2 Klein {self.variant.upper()} inpainting...", verbose=verbose
        )

        ys, xs = np.where(mask_np)
        if len(ys) == 0 or len(xs) == 0:
            return image_pil

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        # Calculate context padding (doubled for Klein vs Kontext)
        padding_pixels = int(max(bbox_width, bbox_height) * CONTEXT_PADDING_RATIO)
        padding = int(
            min(padding_pixels, MAX_CONTEXT_PADDING) * self.KLEIN_PADDING_MULTIPLIER
        )

        blur_radius = int(max(bbox_width, bbox_height) * BLUR_SCALE_FACTOR)
        blur_radius = max(MIN_BLUR_RADIUS, min(blur_radius, MAX_BLUR_RADIUS))

        img_h, img_w = mask_np.shape
        x1 = max(0, x_min - padding)
        y1 = max(0, y_min - padding)
        x2 = min(img_w, x_max + 1 + padding)
        y2 = min(img_h, y_max + 1 + padding)

        width = self._quantize_dimension(x2 - x1)
        height = self._quantize_dimension(y2 - y1)

        x2 = min(img_w, x1 + width)
        y2 = min(img_h, y1 + height)
        width = x2 - x1
        height = y2 - y1

        if width < self.MIN_RESOLUTION or height < self.MIN_RESOLUTION:
            log_message(
                f"  - Region too small ({width}x{height}), skipping", verbose=verbose
            )
            return image_pil

        log_message(
            f"  - Processing region at ({x1}, {y1}) size {width}x{height}",
            verbose=verbose,
        )

        image_cropped_pil = image_pil.crop((x1, y1, x2, y2))
        mask_crop_np = mask_np[y1:y2, x1:x2]

        # Build cache parameters
        cache_params = {
            "bbox": (x1, y1, width, height),
            "padding": padding,
            "blur": blur_radius,
            "variant": self.variant,
        }
        if strict_mask_clipping:
            cache_params["strict_clip"] = True
        if composite_clip_bbox is not None:
            cache_params["clip_bbox"] = tuple(composite_clip_bbox)
        if ocr_params:
            cache_params.update(ocr_params)

        cache_key = None
        cached_patch = None
        if self.cache.should_use_inpaint_cache(seed):
            # Downsample mask signature to reduce sensitivity to minor jitter
            if mask_crop_np.size > 0:
                sig_h = min(64, max(4, mask_crop_np.shape[0]))
                sig_w = min(64, max(4, mask_crop_np.shape[1]))
                mask_sig = (
                    torch.from_numpy(mask_crop_np.astype(np.float32))
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                mask_sig = torch.nn.functional.interpolate(
                    mask_sig, size=(sig_h, sig_w), mode="bilinear", align_corners=False
                )
                mask_sig_np = (mask_sig > 0.5).cpu().numpy().astype(np.uint8)[0, 0]
            else:
                mask_sig_np = mask_crop_np

            cache_key = self.cache.get_inpaint_cache_key(
                image_cropped_pil,
                mask_sig_np,
                seed,
                self.num_inference_steps,
                0.0,  # Klein doesn't use residual_diff_threshold
                self.KLEIN_GUIDANCE_SCALE,
                self.KLEIN_PROMPT,
                cache_params,
            )
            cached_patch = self.cache.get_inpainted_image(cache_key)
            if cached_patch is not None:
                log_message("  - Using cached inpainting patch", verbose=verbose)

        mask_float = mask_crop_np.astype(np.float32)
        if blur_radius > 0:
            d_out = distance_transform_edt(~mask_crop_np)
            d_in = distance_transform_edt(mask_crop_np)
            alpha = np.zeros_like(d_out, np.float32)
            alpha[d_in > 0] = 1.0
            ramp = np.clip(1.0 - (d_out / blur_radius), 0.0, 1.0)
            alpha[d_out > 0] = ramp[d_out > 0]
            mask_for_composite = torch.from_numpy(alpha)[None, ...]
        else:
            mask_for_composite = torch.from_numpy(mask_float)[None, ...]

        if strict_mask_clipping:
            original_mask_crop = torch.from_numpy(mask_crop_np.astype(np.float32))
            mask_for_composite = mask_for_composite * original_mask_crop

        if composite_clip_bbox is not None:
            clip_x1, clip_y1, clip_x2, clip_y2 = composite_clip_bbox
            clip_x1 = max(0, min(img_w, clip_x1))
            clip_x2 = max(0, min(img_w, clip_x2))
            clip_y1 = max(0, min(img_h, clip_y1))
            clip_y2 = max(0, min(img_h, clip_y2))

            start_x = max(0, clip_x1 - x1)
            end_x = min(width, clip_x2 - x1)
            start_y = max(0, clip_y1 - y1)
            end_y = min(height, clip_y2 - y1)

            if end_x <= start_x or end_y <= start_y:
                mask_for_composite = torch.zeros_like(mask_for_composite)
            else:
                clipped_mask = torch.zeros_like(mask_for_composite)
                clipped_mask[:, start_y:end_y, start_x:end_x] = mask_for_composite[
                    :, start_y:end_y, start_x:end_x
                ]
                mask_for_composite = clipped_mask

        patch_pil = cached_patch

        if patch_pil is None:
            inference_image, _, _ = self._prepare_image_for_inference(
                image_cropped_pil, verbose=verbose
            )
            inference_w, inference_h = inference_image.size

            if inference_image.mode == "RGBA":
                inference_image = inference_image.convert("RGB")

            self.load_models()

            if self.pipeline is None:
                log_message(
                    f"Warning: Flux Klein {self.variant.upper()} pipeline unavailable.",
                    always_print=True,
                )
                return image_pil

            log_message("  - Running inference...", verbose=verbose)

            with torch.inference_mode():
                gen = torch.Generator(device=self.DEVICE).manual_seed(seed)
                out = self.pipeline(
                    prompt=self.KLEIN_PROMPT,
                    image=inference_image,
                    height=inference_h,
                    width=inference_w,
                    guidance_scale=self.KLEIN_GUIDANCE_SCALE,
                    num_inference_steps=self.num_inference_steps,
                    generator=gen,
                )
                generated_patch_pil = out.images[0]

            if (inference_w, inference_h) != (width, height):
                patch_pil = generated_patch_pil.resize(
                    (width, height), Image.Resampling.LANCZOS
                )
            else:
                patch_pil = generated_patch_pil

            patch_pil = self._match_luminance(
                generated_pil=patch_pil,
                original_crop_pil=image_cropped_pil,
                mask_crop_np=mask_crop_np,
                padded_bounds=(x1, y1, x2, y2),
                original_bounds=(x_min, y_min, x_max, y_max),
            )

        dest_tensor = torch.from_numpy(
            np.asarray(image_pil, dtype=np.float32) / 255.0
        ).unsqueeze(0)
        src_tensor = torch.from_numpy(
            np.asarray(patch_pil, dtype=np.float32) / 255.0
        ).unsqueeze(0)

        # Fix channel mismatch (e.g. RGBA destination vs RGB source)
        dest_channels = dest_tensor.shape[-1]
        src_channels = src_tensor.shape[-1]

        if dest_channels > src_channels:
            padding = torch.ones(
                (*src_tensor.shape[:-1], dest_channels - src_channels),
                device=src_tensor.device,
                dtype=src_tensor.dtype,
            )
            src_tensor = torch.cat([src_tensor, padding], dim=-1)
        elif src_channels > dest_channels:
            src_tensor = src_tensor[..., :dest_channels]

        # Use FluxKontextInpainter's composite method (same logic)
        dest_tensor = dest_tensor.movedim(-1, 1)
        src_tensor = src_tensor.movedim(-1, 1)

        mask_interp = torch.nn.functional.interpolate(
            mask_for_composite.reshape(
                (-1, 1, mask_for_composite.shape[-2], mask_for_composite.shape[-1])
            ),
            size=(src_tensor.shape[2], src_tensor.shape[3]),
            mode="bilinear",
        )

        composited = dest_tensor.clone()
        visible_h = min(src_tensor.shape[2], dest_tensor.shape[2] - y1)
        visible_w = min(src_tensor.shape[3], dest_tensor.shape[3] - x1)

        if visible_h > 0 and visible_w > 0:
            src_portion = src_tensor[:, :, :visible_h, :visible_w]
            mask_portion = mask_interp[:, :, :visible_h, :visible_w]
            dest_portion = composited[:, :, y1 : y1 + visible_h, x1 : x1 + visible_w]

            blended = src_portion * mask_portion + dest_portion * (1 - mask_portion)
            composited[:, :, y1 : y1 + visible_h, x1 : x1 + visible_w] = blended

        composited = composited.movedim(1, -1)
        composited_pil = Image.fromarray(
            (composited[0].cpu().numpy() * 255).astype("uint8")
        )

        # Save to cache if generated (not from cache)
        if (
            self.cache.should_use_inpaint_cache(seed)
            and cache_key is not None
            and cached_patch is None
        ):
            self.cache.set_inpainted_image(cache_key, patch_pil)

        return composited_pil
