import gc
import math
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from core.caching import get_cache
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
        num_inference_steps: int = 15,
        residual_diff_threshold: float = 0.12,
    ):
        """Initialize the Flux Kontext Inpaint class.

        Args:
            device: PyTorch device to use. Auto-detects if None.
            huggingface_token: HuggingFace token for model downloads.
            num_inference_steps: Number of denoising steps for inference.
            residual_diff_threshold: Residual diff threshold for Flux caching (0.0-1.0).
        """
        self.DEVICE = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.DTYPE = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.huggingface_token = huggingface_token
        self.num_inference_steps = num_inference_steps
        self.residual_diff_threshold = residual_diff_threshold
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
    ) -> Image.Image:
        """Inpaint a specific mask region in the image.

        Args:
            image_pil: PIL Image to inpaint
            mask_np: Numpy mask array (H, W) with True for areas to inpaint
            seed: Random seed for inference
            verbose: Whether to print verbose output
            ocr_params: Optional OCR parameters dict for cache key generation

        Returns:
            PIL.Image: The inpainted image
        """
        # Check cache first (if seed is not random)
        if self.cache.should_use_inpaint_cache(seed):
            cache_key = self.cache.get_inpaint_cache_key(
                image_pil,
                mask_np,
                seed,
                self.num_inference_steps,
                self.residual_diff_threshold,
                self.guidance_scale,
                self.prompt,
                ocr_params if ocr_params else None,
            )
            cached_result = self.cache.get_inpainted_image(cache_key)
            if cached_result is not None:
                log_message("  - Using cached inpainting result", verbose=verbose)
                return cached_result

        self.load_models()

        # Check if models loaded successfully
        if self.pipeline is None:
            log_message(
                "Warning: Flux Kontext pipeline not available. Skipping inpainting.",
                always_print=True,
            )
            return image_pil

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

        # Ensure a minimum context padding proportional to the detection size
        padding_pixels = int(max(bbox_width, bbox_height) * self.context_padding_ratio)
        padding = min(padding_pixels, self.max_context_padding)  # Cap padding
        log_message(
            f"  - Proportional context padding: {padding_pixels}px, capped to: {padding}px",
            verbose=verbose,
        )

        # Dynamic blur radius calculation
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

        log_message(
            f"  - Optimized bbox found at ({x}, {y}) with size {width}x{height}",
            verbose=verbose,
        )

        image_cropped_pil = image_pil.crop((x, y, x + width, y + height))

        # Scale to preferred resolution for optimal Flux performance
        image_scaled_for_inference_pil = self.flux_kontext_image_scale(
            image_cropped_pil
        )
        inference_width, inference_height = image_scaled_for_inference_pil.size

        # Ensure image is RGB for the model
        if image_scaled_for_inference_pil.mode == "RGBA":
            image_scaled_for_inference_pil = image_scaled_for_inference_pil.convert(
                "RGB"
            )

        log_message("  - Running inference...", verbose=verbose)

        self.pipeline.text_encoder_2.to(self.DEVICE)

        prompt_embeds, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=self.prompt,
            prompt_2=None,
            device=self.DEVICE,
        )

        # Offload text encoder to free VRAM before inference
        self.pipeline.text_encoder_2.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        self.pipeline.transformer.to(self.DEVICE)

        required_area = inference_width * inference_height
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
                (img.mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy())
            )

        # Offload transformer to free VRAM
        self.pipeline.transformer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        src_resized_for_composite_pil = generated_patch_pil.resize(
            (width, height), Image.Resampling.LANCZOS
        )
        dest_tensor = torch.from_numpy(
            np.asarray(image_pil, dtype=np.float32) / 255.0
        ).unsqueeze(0)
        src_tensor = torch.from_numpy(
            np.asarray(src_resized_for_composite_pil, dtype=np.float32) / 255.0
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

        # Cache the result if caching is enabled
        if self.cache.should_use_inpaint_cache(seed):
            cache_key = self.cache.get_inpaint_cache_key(
                image_pil,
                mask_np,
                seed,
                self.num_inference_steps,
                self.residual_diff_threshold,
                self.guidance_scale,
                self.prompt,
            )
            self.cache.set_inpainted_image(cache_key, composited_pil)

        return composited_pil
