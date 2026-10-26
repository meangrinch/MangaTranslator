import math

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from core.caching import get_cache
from core.device import empty_cache, get_best_device, get_best_dtype
from core.ml.model_manager import get_model_manager
from core.ml.sdcpp_server import pil_to_base64_png, run_image_job
from utils.cancellation import is_cancelled, raise_if_cancelled
from utils.exceptions import ModelError
from utils.logging import log_message
from utils.model_metadata import (
    flux_method_backends,
    flux_sdcpp_quant_default,
    flux_sdcpp_text_encoder_default,
    flux_steps_range,
)

# Dynamic edge feathering scaled to bubble size; clamped to prevent sharp seams on small text or bleed on large text
BLUR_SCALE_FACTOR = 0.1
MIN_BLUR_RADIUS = 1
MAX_BLUR_RADIUS = 10

# Guidance scale balanced for Kontext text erasure
FLUX_GUIDANCE_SCALE = 2.5

# Surrounding context ratio and pixel ceiling to ensure clean blending without diluting inpaint resolution
CONTEXT_PADDING_RATIO = 0.5
MAX_CONTEXT_PADDING = 80


def _interrupt_on_cancel(pipeline, step, timestep, callback_kwargs):
    """Diffusers step callback that stops denoising once the user cancels."""
    if is_cancelled():
        pipeline._interrupt = True
    return callback_kwargs


def _prompt_value_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, tuple):
        return tuple(_prompt_value_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_prompt_value_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _prompt_value_to_cpu(item) for key, item in value.items()}
    return value


def _prompt_value_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_prompt_value_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_prompt_value_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _prompt_value_to_device(item, device) for key, item in value.items()
        }
    return value


def _pipeline_execution_device(pipeline, fallback: torch.device) -> torch.device:
    execution_device = getattr(pipeline, "_execution_device", None)
    if execution_device is None:
        return fallback
    return torch.device(execution_device)


def _encode_flux_prompt(pipeline, prompt: str, device: torch.device):
    try:
        encoded = pipeline.encode_prompt(prompt=prompt, prompt_2=None, device=device)
    except TypeError:
        encoded = pipeline.encode_prompt(prompt=prompt, device=device)

    if not isinstance(encoded, tuple) or len(encoded) < 2:
        raise RuntimeError("Flux prompt encoder returned an unexpected result.")

    return encoded[0], encoded[1]


def _flux_prompt_kwargs(
    prompt_embeds, pooled_prompt_embeds, include_pooled: bool = True
) -> dict:
    kwargs = {"prompt_embeds": prompt_embeds}
    if include_pooled and pooled_prompt_embeds is not None:
        kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
    return kwargs


def make_flux_inpainter(
    method: str,
    *,
    device: torch.device | None,
    huggingface_token: str,
    num_inference_steps: int,
    backend: str,
    low_vram: bool,
    residual_diff_threshold: float,
    luminance_correction: bool,
    upscale_small_crops: bool,
    sdcpp_cache_mode: str,
    sdcpp_diffusion_quant: str,
    sdcpp_text_encoder_quant: str,
    verbose: bool = False,
):
    """Build the inpainter for an inpainting method key; Flux.1 Kontext otherwise."""
    if method == "lama_manga":
        return LamaInpainter(device=device, verbose=verbose)
    edit_model_kwargs = {
        "device": device,
        "huggingface_token": huggingface_token,
        "num_inference_steps": int(num_inference_steps),
        "low_vram": low_vram,
        "luminance_correction": luminance_correction,
        "upscale_small_crops": upscale_small_crops,
        "backend": backend,
        "sdcpp_cache_mode": sdcpp_cache_mode,
        "sdcpp_diffusion_quant": sdcpp_diffusion_quant,
        "sdcpp_text_encoder_quant": sdcpp_text_encoder_quant,
        "verbose": verbose,
    }
    if method in ("flux_klein_9b", "flux_klein_4b"):
        return FluxKleinInpainter(
            variant=method.removeprefix("flux_klein_"), **edit_model_kwargs
        )
    if method in _EDIT_MODEL_INPAINTERS:
        return _EDIT_MODEL_INPAINTERS[method](**edit_model_kwargs)
    return FluxKontextInpainter(
        device=device,
        huggingface_token=huggingface_token,
        num_inference_steps=int(num_inference_steps),
        residual_diff_threshold=float(residual_diff_threshold),
        backend=backend,
        # Only SDNQ offloads to the CPU
        low_vram=low_vram if backend == "sdnq" else False,
        sdcpp_cache_mode=sdcpp_cache_mode,
        sdcpp_diffusion_quant=sdcpp_diffusion_quant,
        sdcpp_text_encoder_quant=sdcpp_text_encoder_quant,
    )


def _inpaint_cache_options(
    inpainter,
    strict_mask_clipping: bool,
    composite_clip_bbox: tuple[int, int, int, int] | None,
    ocr_params: dict | None,
) -> dict:
    """Cache-key fields shared by the Flux inpainters (backend quants and clip options)."""
    options = {}
    if inpainter.backend == "sdcpp":
        options["sdcpp_cache"] = inpainter.sdcpp_cache_mode
        options["sdcpp_diffusion_quant"] = inpainter.sdcpp_diffusion_quant
        options["sdcpp_text_encoder_quant"] = inpainter.sdcpp_text_encoder_quant
    if strict_mask_clipping:
        options["strict_clip"] = True
    if composite_clip_bbox is not None:
        options["clip_bbox"] = tuple(composite_clip_bbox)
    if ocr_params:
        options.update(ocr_params)
    return options


def _lookup_inpaint_patch(
    cache,
    image_cropped_pil: Image.Image,
    mask_crop_np: np.ndarray,
    seed: int,
    num_inference_steps: int,
    residual_diff_threshold: float,
    guidance_scale: float,
    prompt: str,
    cache_params: dict,
) -> tuple[str | None, Image.Image | None]:
    """Return (cache key, cached patch); (None, None) when the seed disables caching."""
    if not cache.should_use_inpaint_cache(seed):
        return None, None

    # Downsample mask signature to reduce sensitivity to minor jitter
    if mask_crop_np.size > 0:
        sig_h = min(64, max(4, mask_crop_np.shape[0]))
        sig_w = min(64, max(4, mask_crop_np.shape[1]))
        mask_sig = (
            torch.from_numpy(mask_crop_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        )
        mask_sig = torch.nn.functional.interpolate(
            mask_sig, size=(sig_h, sig_w), mode="bilinear", align_corners=False
        )
        mask_sig_np = (mask_sig > 0.5).cpu().numpy().astype(np.uint8)[0, 0]
    else:
        mask_sig_np = mask_crop_np

    cache_key = cache.get_inpaint_cache_key(
        image_cropped_pil,
        mask_sig_np,
        seed,
        num_inference_steps,
        residual_diff_threshold,
        guidance_scale,
        prompt,
        cache_params,
    )
    cached_patch = cache.get_inpainted_image(cache_key)
    if cached_patch is not None:
        log_message("  - Using cached inpainting patch", always_print=True)
    return cache_key, cached_patch


def _clip_composite_mask(
    mask_for_composite: torch.Tensor,
    composite_clip_bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> torch.Tensor:
    """Zero the crop's composite mask outside `composite_clip_bbox` (image coordinates).

    `crop_box` is (x, y, width, height) of the crop within the image.
    """
    img_w, img_h = image_size
    x, y, width, height = crop_box
    clip_x1, clip_y1, clip_x2, clip_y2 = composite_clip_bbox
    clip_x1 = max(0, min(img_w, clip_x1))
    clip_x2 = max(0, min(img_w, clip_x2))
    clip_y1 = max(0, min(img_h, clip_y1))
    clip_y2 = max(0, min(img_h, clip_y2))

    start_x = max(0, clip_x1 - x)
    end_x = min(width, clip_x2 - x)
    start_y = max(0, clip_y1 - y)
    end_y = min(height, clip_y2 - y)

    clipped_mask = torch.zeros_like(mask_for_composite)
    if end_x > start_x and end_y > start_y:
        clipped_mask[:, start_y:end_y, start_x:end_x] = mask_for_composite[
            :, start_y:end_y, start_x:end_x
        ]
    return clipped_mask


def _feathered_crop_mask(mask_crop_np: np.ndarray, blur_radius: int) -> torch.Tensor:
    """Composite alpha for a crop: 1 inside the mask, fading to 0 over `blur_radius` px."""
    if blur_radius <= 0:
        return torch.from_numpy(mask_crop_np.astype(np.float32))[None, ...]
    d_out = distance_transform_edt(~mask_crop_np)
    d_in = distance_transform_edt(mask_crop_np)
    alpha = np.zeros_like(d_out, np.float32)
    alpha[d_in > 0] = 1.0
    ramp = np.clip(1.0 - (d_out / blur_radius), 0.0, 1.0)
    alpha[d_out > 0] = ramp[d_out > 0]
    return torch.from_numpy(alpha)[None, ...]


def _paste_patch(
    image_pil: Image.Image,
    patch_pil: Image.Image,
    mask_for_composite: torch.Tensor,
    x1: int,
    y1: int,
) -> Image.Image:
    """Blend `patch_pil` into `image_pil` at (x1, y1) through the composite mask."""
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
        alpha_fill = torch.ones(
            (*src_tensor.shape[:-1], dest_channels - src_channels),
            device=src_tensor.device,
            dtype=src_tensor.dtype,
        )
        src_tensor = torch.cat([src_tensor, alpha_fill], dim=-1)
    elif src_channels > dest_channels:
        src_tensor = src_tensor[..., :dest_channels]

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
    return Image.fromarray((composited[0].cpu().numpy() * 255).astype("uint8"))


class FluxKontextInpainter:
    """Inpainter using Flux Kontext models for text removal."""

    def __init__(
        self,
        device: torch.device | None = None,
        huggingface_token: str = "",
        num_inference_steps: int = 8,
        residual_diff_threshold: float = 0.15,
        backend: str = "nunchaku",
        low_vram: bool = False,
        sdcpp_cache_mode: str = "none",
        sdcpp_diffusion_quant: str = "",
        sdcpp_text_encoder_quant: str = "",
    ):
        """Initialize the Flux Kontext Inpaint class.

        Args:
            device: PyTorch device to use. Auto-detects if None.
            huggingface_token: HuggingFace token for model downloads (Nunchaku only).
            num_inference_steps: Number of denoising steps for inference.
            residual_diff_threshold: Residual diff threshold for Flux caching (Nunchaku only).
            backend: "nunchaku" (CUDA + Nunchaku + HF token), "sdnq", or "sdcpp".
            low_vram: If True, use sequential CPU offload for SDNQ.
            sdcpp_cache_mode: sd.cpp cache mode to use when backend is "sdcpp".
            sdcpp_diffusion_quant: Flux sd.cpp diffusion model quant.
            sdcpp_text_encoder_quant: Flux sd.cpp T5 text encoder quant.
        """
        self.DEVICE = device if device is not None else get_best_device()
        self.DTYPE = get_best_dtype(self.DEVICE)
        self.huggingface_token = huggingface_token
        self.num_inference_steps = num_inference_steps
        self.residual_diff_threshold = residual_diff_threshold
        self.backend = backend.lower()
        if self.backend not in ("nunchaku", "sdnq", "sdcpp"):
            raise ValueError(
                f"Invalid Kontext backend '{backend}'. "
                "Must be 'nunchaku', 'sdnq', or 'sdcpp'."
            )
        self.low_vram = low_vram
        self.sdcpp_cache_mode = sdcpp_cache_mode
        self.sdcpp_diffusion_quant = sdcpp_diffusion_quant or flux_sdcpp_quant_default(
            "flux_kontext", "diffusion_model"
        )
        self.sdcpp_text_encoder_quant = (
            sdcpp_text_encoder_quant or flux_sdcpp_text_encoder_default("flux_kontext")
        )
        self.manager = get_model_manager()
        self.cache = get_cache()

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
        self.sdcpp_assets = None
        self._prompt_embeds_cpu = None
        self._pooled_prompt_embeds_cpu = None

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

        if self.backend == "sdnq":
            self.pipeline = self.manager.load_flux_kontext_sdnq(
                low_vram=self.low_vram,
                verbose=True,
            )
            # CPU offload handled internally; separate model references not needed
            self.transformer = None
            self.text_encoder_2 = None
        elif self.backend == "sdcpp":
            self.sdcpp_assets = self.manager.ensure_flux_sdcpp_server(
                "flux_kontext",
                cache_mode=self.sdcpp_cache_mode,
                diffusion_quant=self.sdcpp_diffusion_quant,
                text_encoder_quant=self.sdcpp_text_encoder_quant,
                num_inference_steps=self.num_inference_steps,
                verbose=True,
            )
            self.pipeline = self.sdcpp_assets
            self.transformer = None
            self.text_encoder_2 = None
        else:
            self.manager.set_flux_residual_diff_threshold(self.residual_diff_threshold)

            self.transformer, self.text_encoder_2, self.pipeline = (
                self.manager.load_flux_models()
            )

    def unload_models(self):
        """Unload Flux.1 Kontext models via model manager to free up memory."""
        self.pipeline = None
        self.transformer = None
        self.text_encoder_2 = None
        self.sdcpp_assets = None
        self._prompt_embeds_cpu = None
        self._pooled_prompt_embeds_cpu = None

        if self.backend == "sdnq":
            self.manager.unload_flux_kontext_sdnq_models()
        elif self.backend == "nunchaku":
            self.manager.unload_flux_kontext_models()
        elif self.backend == "sdcpp":
            self.manager.shutdown_sdcpp_server("flux_kontext")

    def _get_prompt_embeddings(self, device: torch.device, verbose: bool = False):
        if self._prompt_embeds_cpu is None:
            log_message("  - Encoding Flux prompt embeddings", verbose=verbose)
            prompt_embeds, pooled_prompt_embeds = _encode_flux_prompt(
                self.pipeline, self.prompt, device
            )
            self._prompt_embeds_cpu = _prompt_value_to_cpu(prompt_embeds)
            self._pooled_prompt_embeds_cpu = _prompt_value_to_cpu(pooled_prompt_embeds)
        else:
            log_message("  - Reusing Flux prompt embeddings", verbose=verbose)

        return (
            _prompt_value_to_device(self._prompt_embeds_cpu, device),
            _prompt_value_to_device(self._pooled_prompt_embeds_cpu, device),
        )

    def _run_sdcpp_inference(
        self,
        image_pil: Image.Image,
        width: int,
        height: int,
        seed: int,
        verbose: bool = False,
    ) -> Image.Image:
        if self.sdcpp_assets is None:
            raise ModelError("Flux.1 Kontext sd.cpp server is not loaded.")

        payload = {
            "prompt": self.prompt,
            "negative_prompt": "",
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "batch_count": 1,
            "ref_images": [pil_to_base64_png(image_pil)],
            "sample_params": {
                "sample_method": "euler",
                "sample_steps": int(self.num_inference_steps),
                "guidance": {
                    "txt_cfg": 1.0,
                    "img_cfg": 1.0,
                    "distilled_guidance": float(self.guidance_scale),
                },
            },
            "output_format": "png",
            "output_compression": 100,
        }
        return run_image_job(
            self.sdcpp_assets, payload, verbose=verbose, timeout_sec=900
        )

    def convert_mask_to_tensor(self, mask_np):
        """Convert a numpy mask to the tensor format expected by the pipeline.

        Args:
            mask_np: Numpy mask array (H, W) with True/False values

        Returns:
            torch.Tensor: Mask tensor in CHW format (1.0 for areas to keep, 0.0 for areas to inpaint)
        """
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
        mask = mask_chw[0, 0] if mask_chw.dim() == 4 else mask_chw[0]

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
            initial_ar = W / H if h0 == 0 else w0 / h0
            log_message(
                f"  - Initial mask bounding box AR: {initial_ar:.2f}",
                verbose=verbose,
            )

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

        else:
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
            padding = torch.ones(
                (*source.shape[:-1], dest_channels - source_channels),
                device=source.device,
                dtype=source.dtype,
            )
            source = torch.cat([source, padding], dim=-1)
        else:
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

    def _generate_patch(
        self, image: Image.Image, width: int, height: int, seed: int, verbose: bool
    ) -> Image.Image:
        """Run the loaded backend on the scaled crop; the caller holds the inference lock."""
        if self.backend == "sdcpp":
            return self._run_sdcpp_inference(
                image, width, height, seed, verbose=verbose
            )

        if self.backend == "sdnq":
            with torch.inference_mode():
                gen_device = "cpu" if self.low_vram else self.DEVICE
                gen = torch.Generator(device=gen_device).manual_seed(seed)
                prompt_device = _pipeline_execution_device(self.pipeline, self.DEVICE)
                prompt_embeds, pooled_prompt_embeds = self._get_prompt_embeddings(
                    prompt_device, verbose=verbose
                )
                return self._run_pipeline(
                    image, width, height, gen, prompt_embeds, pooled_prompt_embeds
                )

        should_encode_prompt = self._prompt_embeds_cpu is None
        if should_encode_prompt:
            self.pipeline.text_encoder_2.to(self.DEVICE)

        prompt_embeds, pooled_prompt_embeds = self._get_prompt_embeddings(
            self.DEVICE, verbose=verbose
        )

        if should_encode_prompt:
            self.pipeline.text_encoder_2.to("cpu")
            empty_cache(self.DEVICE)

        self.pipeline.transformer.to(self.DEVICE)

        with torch.inference_mode():
            gen = torch.Generator(device=self.DEVICE).manual_seed(seed)
            patch = self._run_pipeline(
                image, width, height, gen, prompt_embeds, pooled_prompt_embeds
            )

        self.pipeline.transformer.to("cpu")
        empty_cache(self.DEVICE)
        return patch

    def _run_pipeline(
        self,
        image: Image.Image,
        width: int,
        height: int,
        generator: torch.Generator,
        prompt_embeds,
        pooled_prompt_embeds,
    ) -> Image.Image:
        """Diffusers Kontext call; returns the patch as an image with NaNs zeroed."""
        out = self.pipeline(
            **_flux_prompt_kwargs(prompt_embeds, pooled_prompt_embeds),
            image=image,
            width=width,
            height=height,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            output_type="pt",
            max_area=width * height,
            callback_on_step_end=_interrupt_on_cancel,
        )
        img = out.images[0]
        torch.nan_to_num_(img, nan=0.0, posinf=1.0, neginf=0.0)
        img.clamp_(0, 1)
        return Image.fromarray(
            img.mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        )

    def inpaint_mask(
        self,
        image_pil: Image.Image,
        mask_np: np.ndarray,
        seed: int = 1,
        verbose: bool = False,
        ocr_params: dict | None = None,
        strict_mask_clipping: bool = False,
        composite_clip_bbox: tuple[int, int, int, int] | None = None,
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
        blur_radius = max(MIN_BLUR_RADIUS, min(blur_radius, MAX_BLUR_RADIUS))
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
            img_h, img_w = mask_np.shape
            mask_for_composite = _clip_composite_mask(
                mask_for_composite,
                composite_clip_bbox,
                (img_w, img_h),
                (x, y, width, height),
            )

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
            "backend": self.backend,
            **_inpaint_cache_options(
                self, strict_mask_clipping, composite_clip_bbox, ocr_params
            ),
        }
        cache_key, cached_patch = _lookup_inpaint_patch(
            self.cache,
            image_cropped_pil,
            mask_crop_np,
            seed,
            self.num_inference_steps,
            self.residual_diff_threshold,
            self.guidance_scale,
            self.prompt,
            cache_params,
        )

        patch_pil = cached_patch

        if patch_pil is None:
            image_scaled_for_inference_pil = self.flux_kontext_image_scale(
                image_cropped_pil
            )
            inference_width, inference_height = image_scaled_for_inference_pil.size

            if image_scaled_for_inference_pil.mode == "RGBA":
                image_scaled_for_inference_pil = image_scaled_for_inference_pil.convert(
                    "RGB"
                )

            log_message("  - Running inference...", verbose=verbose)

            # CPU offload moves weights between devices; serialize access across threads
            with self.manager.flux_inference_lock:
                raise_if_cancelled()  # Jobs queued on the lock skip their inference
                self.load_models()

                if self.pipeline is None:
                    log_message(
                        "Warning: Flux Kontext pipeline not available. Skipping inpainting.",
                        always_print=True,
                    )
                    return image_pil

                generated_patch_pil = self._generate_patch(
                    image_scaled_for_inference_pil,
                    inference_width,
                    inference_height,
                    seed,
                    verbose,
                )

            raise_if_cancelled()  # An interrupted run returns a half-denoised patch
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


class _CropInpainter:
    """Inpaints a padded crop around the mask and blends the patch back in.

    Subclasses set `model_key` and `label` and provide load_models, unload_models,
    and _generate_patch.
    """

    model_key = ""
    label = ""
    PROMPT = ""
    GUIDANCE_SCALE = 1.0

    # DiT patch requirements (multiples of 16) and VRAM cap (~4MP max inference)
    MIN_RESOLUTION = 64
    MAX_RESOLUTION = 2048
    RESOLUTION_MULTIPLE = 16
    MAX_INFERENCE_PIXELS = 4_000_000
    PADDING_MULTIPLIER = (
        2.0  # Wider context helps preserve surrounding line art and details
    )

    def __init__(
        self,
        *,
        backend: str,
        device: torch.device | None,
        num_inference_steps: int,
        luminance_correction: bool,
        upscale_small_crops: bool,
        verbose: bool,
    ):
        self.backend = backend
        self.num_inference_steps = num_inference_steps
        self.luminance_correction = luminance_correction
        self.upscale_small_crops = upscale_small_crops
        self.verbose = verbose

        self.DEVICE = device if device is not None else get_best_device()
        self.DTYPE = get_best_dtype(self.DEVICE)
        self.manager = get_model_manager()
        self.cache = get_cache()
        self.pipeline = None

    def load_models(self):
        raise NotImplementedError

    def unload_models(self):
        raise NotImplementedError

    def _generate_patch(
        self,
        image: Image.Image,
        mask: np.ndarray,
        width: int,
        height: int,
        seed: int,
        verbose: bool,
    ) -> Image.Image:
        """Inpaint the scaled crop; `mask` is the crop's mask at its original size."""
        raise NotImplementedError

    def _snap_dimension(self, dim: int) -> int:
        """Round dimension down to a multiple of RESOLUTION_MULTIPLE, at least MIN_RESOLUTION."""
        dim = max(self.MIN_RESOLUTION, dim)
        return (dim // self.RESOLUTION_MULTIPLE) * self.RESOLUTION_MULTIPLE

    def _quantize_dimension(self, dim: int) -> int:
        """Quantize dimension to be a multiple of RESOLUTION_MULTIPLE within allowed range."""
        return self._snap_dimension(min(self.MAX_RESOLUTION, dim))

    def _expand_bounds_to_min_size(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_w: int,
        img_h: int,
    ) -> tuple[int, int, int, int]:
        target_w = min(self.MIN_RESOLUTION, img_w)
        target_h = min(self.MIN_RESOLUTION, img_h)

        width = x2 - x1
        if width < target_w:
            extra = target_w - width
            x1 = max(0, x1 - extra // 2)
            x2 = min(img_w, x2 + extra - extra // 2)
            if x2 - x1 < target_w:
                if x1 == 0:
                    x2 = min(img_w, target_w)
                else:
                    x1 = max(0, img_w - target_w)

        height = y2 - y1
        if height < target_h:
            extra = target_h - height
            y1 = max(0, y1 - extra // 2)
            y2 = min(img_h, y2 + extra - extra // 2)
            if y2 - y1 < target_h:
                if y1 == 0:
                    y2 = min(img_h, target_h)
                else:
                    y1 = max(0, img_h - target_h)

        return x1, y1, x2, y2

    def _compute_luminance_stats(
        self, image_np: np.ndarray, mask_np: np.ndarray
    ) -> tuple[float, float]:
        """Compute mean and std of luminance (LAB L channel) for masked pixels.

        Args:
            image_np: RGB image array (H, W, 3), uint8
            mask_np: Boolean mask where True = pixels to include

        Returns:
            (mean, std) of L channel values (0-255 scale, cv2 LAB)
        """
        if not np.any(mask_np):
            return 127.5, 30.0  # Neutral fallback

        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_values = lab[:, :, 0][mask_np].astype(np.float32)

        return float(np.mean(l_values)), float(np.std(l_values)) + 1e-6

    def _match_luminance(
        self,
        generated_pil: Image.Image,
        original_crop_pil: Image.Image,
        mask_crop_np: np.ndarray,
        verbose: bool = False,
        boundary_width: int = 12,
    ) -> Image.Image:
        """Match patch luminance and color to surrounding context.

        Uses a local boundary band around the mask to avoid distant margin/gutter
        contamination, applies an additive luminance shift to preserve internal
        contrast without false scaling artifacts, and forces true neutral chroma
        on monochrome manga to eliminate diffusion color casts.

        Args:
            generated_pil: Generated patch (at crop size)
            original_crop_pil: Original cropped region
            mask_crop_np: Boolean mask of inpaint region within crop
            verbose: Whether to print verbose output
            boundary_width: Width in pixels of the local reference boundary ring

        Returns:
            Luminance-corrected generated patch
        """
        mask_bool = np.asarray(mask_crop_np, dtype=bool)
        if not np.any(mask_bool):
            return generated_pil

        # Prefer immediate boundary ring to avoid distant gutters/borders skewing statistics
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (boundary_width * 2 + 1, boundary_width * 2 + 1)
        )
        dilated_mask = cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0
        boundary_band = dilated_mask & (~mask_bool)

        ref_mask = (
            boundary_band if np.count_nonzero(boundary_band) >= 32 else (~mask_bool)
        )
        if not np.any(ref_mask):
            return generated_pil

        if original_crop_pil.mode != "RGB":
            original_crop_pil = original_crop_pil.convert("RGB")
        if generated_pil.mode != "RGB":
            generated_pil = generated_pil.convert("RGB")

        original_np = np.asarray(original_crop_pil)
        generated_np = np.asarray(generated_pil).copy()

        orig_lab = cv2.cvtColor(original_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        gen_lab = cv2.cvtColor(generated_np, cv2.COLOR_RGB2LAB).astype(np.float32)

        orig_l_mean = float(np.mean(orig_lab[:, :, 0][ref_mask]))
        gen_l_mean = float(np.mean(gen_lab[:, :, 0][ref_mask]))
        l_shift = orig_l_mean - gen_l_mean

        if abs(l_shift) >= 1.0:
            log_message(
                f"  - Luminance correction: L shift {l_shift:+.1f} (mean {gen_l_mean:.1f}->{orig_l_mean:.1f})",
                verbose=verbose,
            )
            gen_lab[:, :, 0][mask_bool] = np.clip(
                gen_lab[:, :, 0][mask_bool] + l_shift, 0, 255
            )

        # In OpenCV LAB, neutral gray is a=128, b=128
        a_dev = float(np.mean(np.abs(orig_lab[:, :, 1][ref_mask] - 128.0)))
        b_dev = float(np.mean(np.abs(orig_lab[:, :, 2][ref_mask] - 128.0)))
        is_monochrome = (a_dev < 3.0) and (b_dev < 3.0)

        if is_monochrome:
            gen_lab[:, :, 1][mask_bool] = 128.0
            gen_lab[:, :, 2][mask_bool] = 128.0
        else:
            for ch in (1, 2):
                orig_ch_mean = float(np.mean(orig_lab[:, :, ch][ref_mask]))
                gen_ch_mean = float(np.mean(gen_lab[:, :, ch][ref_mask]))
                ch_shift = orig_ch_mean - gen_ch_mean
                if abs(ch_shift) > 1.0:
                    gen_lab[:, :, ch][mask_bool] = np.clip(
                        gen_lab[:, :, ch][mask_bool] + ch_shift, 0, 255
                    )

        corrected_np = cv2.cvtColor(gen_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return Image.fromarray(corrected_np)

    def _prepare_image_for_inference(
        self,
        image_pil: Image.Image,
        verbose: bool = False,
    ) -> tuple[Image.Image, int, int]:
        """Prepare image for inference.

        Optionally scales the image to approximately 1 megapixel while maintaining
        aspect ratio. When that scaling is disabled, large crops are still capped
        to 4MP. Dimensions are always kept within model constraints.

        Args:
            image_pil: Input PIL image
            verbose: Whether to print verbose output

        Returns:
            Tuple of (prepared image, original width, original height)
        """
        orig_w, orig_h = image_pil.size
        current_pixels = orig_w * orig_h

        if current_pixels <= 0:
            scale = 1.0
            reason = "model-compatible dimensions"
        elif self.upscale_small_crops:
            target_pixels = 1_048_576  # 2^20 = 1024x1024
            scale = math.sqrt(target_pixels / current_pixels)
            reason = f"~1MP, multiples of {self.RESOLUTION_MULTIPLE}"
        elif current_pixels > self.MAX_INFERENCE_PIXELS:
            scale = math.sqrt(self.MAX_INFERENCE_PIXELS / current_pixels)
            reason = f"4MP cap, multiples of {self.RESOLUTION_MULTIPLE}"
        else:
            scale = 1.0
            reason = "model-compatible dimensions"
        if current_pixels > 0:
            # Fit the longest side here; clamping each side later would distort the shape
            scale = min(
                scale, self.MAX_RESOLUTION / orig_w, self.MAX_RESOLUTION / orig_h
            )

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        new_w = self._quantize_dimension(new_w)
        new_h = self._quantize_dimension(new_h)
        while new_w * new_h > self.MAX_INFERENCE_PIXELS:
            if new_w >= new_h and new_w > self.MIN_RESOLUTION:
                new_w -= self.RESOLUTION_MULTIPLE
            elif new_h > self.MIN_RESOLUTION:
                new_h -= self.RESOLUTION_MULTIPLE
            else:
                break

        if (new_w, new_h) != (orig_w, orig_h):
            log_message(
                f"  - Scaling {orig_w}x{orig_h} -> {new_w}x{new_h} ({reason})",
                verbose=verbose,
            )
            image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return image_pil, orig_w, orig_h

    def _cache_options(
        self,
        strict_mask_clipping: bool,
        composite_clip_bbox: tuple[int, int, int, int] | None,
        ocr_params: dict | None,
    ) -> dict:
        options = {}
        if strict_mask_clipping:
            options["strict_clip"] = True
        if composite_clip_bbox is not None:
            options["clip_bbox"] = tuple(composite_clip_bbox)
        if ocr_params:
            options.update(ocr_params)
        return options

    def inpaint_mask(
        self,
        image_pil: Image.Image,
        mask_np: np.ndarray,
        seed: int = 1,
        verbose: bool = False,
        strict_mask_clipping: bool = False,
        composite_clip_bbox: tuple[int, int, int, int] | None = None,
        ocr_params: dict | None = None,
    ) -> Image.Image:
        """Inpaint a specific mask region in the image.

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

        mask_np = np.asarray(mask_np)
        if mask_np.dtype != bool:
            mask_np = mask_np.astype(bool)

        if not np.any(mask_np):
            return image_pil

        log_message(f"  - {self.label} inpainting...", verbose=verbose)

        ys, xs = np.where(mask_np)
        if len(ys) == 0 or len(xs) == 0:
            return image_pil

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        padding_pixels = int(max(bbox_width, bbox_height) * CONTEXT_PADDING_RATIO)
        padding = int(
            min(padding_pixels, MAX_CONTEXT_PADDING) * self.PADDING_MULTIPLIER
        )

        blur_radius = int(max(bbox_width, bbox_height) * BLUR_SCALE_FACTOR)
        blur_radius = max(MIN_BLUR_RADIUS, min(blur_radius, MAX_BLUR_RADIUS))

        img_h, img_w = mask_np.shape
        x1 = max(0, x_min - padding)
        y1 = max(0, y_min - padding)
        x2 = min(img_w, x_max + 1 + padding)
        y2 = min(img_h, y_max + 1 + padding)
        x1, y1, x2, y2 = self._expand_bounds_to_min_size(x1, y1, x2, y2, img_w, img_h)

        # Only snap to the patch grid here; _prepare_image_for_inference caps the size
        width = min(self._snap_dimension(x2 - x1), img_w)
        height = min(self._snap_dimension(y2 - y1), img_h)

        if x1 + width > img_w:
            x1 = max(0, img_w - width)
        if y1 + height > img_h:
            y1 = max(0, img_h - height)
        x2 = x1 + width
        y2 = y1 + height
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            log_message(
                f"  - Region has invalid size ({width}x{height}), skipping",
                verbose=verbose,
            )
            return image_pil

        log_message(
            f"  - Processing region at ({x1}, {y1}) size {width}x{height}",
            verbose=verbose,
        )

        image_cropped_pil = image_pil.crop((x1, y1, x2, y2))
        mask_crop_np = mask_np[y1:y2, x1:x2]

        cache_params = {
            "bbox": (x1, y1, width, height),
            "padding": padding,
            "blur": blur_radius,
            "model": self.model_key,
            "lum_corr": self.luminance_correction,
            "upscale_small": self.upscale_small_crops,
            "max_pixels": self.MAX_INFERENCE_PIXELS,
            "min_size": (self.MIN_RESOLUTION, self.MIN_RESOLUTION),
            "backend": self.backend,
            **self._cache_options(
                strict_mask_clipping, composite_clip_bbox, ocr_params
            ),
        }
        cache_key, cached_patch = _lookup_inpaint_patch(
            self.cache,
            image_cropped_pil,
            mask_crop_np,
            seed,
            self.num_inference_steps,
            0.0,  # residual_diff_threshold is Kontext/Nunchaku only
            self.GUIDANCE_SCALE,
            self.PROMPT,
            cache_params,
        )

        mask_for_composite = _feathered_crop_mask(mask_crop_np, blur_radius)

        if strict_mask_clipping:
            original_mask_crop = torch.from_numpy(mask_crop_np.astype(np.float32))
            mask_for_composite = mask_for_composite * original_mask_crop

        if composite_clip_bbox is not None:
            mask_for_composite = _clip_composite_mask(
                mask_for_composite,
                composite_clip_bbox,
                (img_w, img_h),
                (x1, y1, width, height),
            )

        patch_pil = cached_patch

        if patch_pil is None:
            inference_image, _, _ = self._prepare_image_for_inference(
                image_cropped_pil,
                verbose=verbose,
            )
            inference_w, inference_h = inference_image.size

            if inference_image.mode == "RGBA":
                inference_image = inference_image.convert("RGB")

            log_message("  - Running inference...", verbose=verbose)

            with self.manager.flux_inference_lock:
                raise_if_cancelled()  # Jobs queued on the lock skip their inference
                self.load_models()

                if self.pipeline is None:
                    log_message(
                        f"Warning: {self.label} pipeline unavailable.",
                        always_print=True,
                    )
                    return image_pil

                generated_patch_pil = self._generate_patch(
                    inference_image,
                    mask_crop_np,
                    inference_w,
                    inference_h,
                    seed,
                    verbose,
                )

            raise_if_cancelled()  # An interrupted run returns a half-denoised patch
            if (inference_w, inference_h) != (width, height):
                patch_pil = generated_patch_pil.resize(
                    (width, height), Image.Resampling.LANCZOS
                )
            else:
                patch_pil = generated_patch_pil

            if self.luminance_correction:
                patch_pil = self._match_luminance(
                    generated_pil=patch_pil,
                    original_crop_pil=image_cropped_pil,
                    mask_crop_np=mask_crop_np,
                    verbose=verbose,
                )

        composited_pil = _paste_patch(image_pil, patch_pil, mask_for_composite, x1, y1)

        if (
            self.cache.should_use_inpaint_cache(seed)
            and cache_key is not None
            and cached_patch is None
        ):
            self.cache.set_inpainted_image(cache_key, patch_pil)

        return composited_pil


class _EditModelInpainter(_CropInpainter):
    """A distilled reference-image edit model run through sd.cpp or Diffusers/SDNQ."""

    PROMPT = (
        "Erase only the Japanese lettering and sound effect characters. "
        "Do not remove or alter any other lines or marks in the artwork."
    )
    # Distilled models bake guidance into weights and require CFG 1.0
    GUIDANCE_SCALE = 1.0

    def __init__(
        self,
        device: torch.device | None = None,
        huggingface_token: str = "",
        num_inference_steps: int = 4,
        low_vram: bool = False,
        luminance_correction: bool = True,
        upscale_small_crops: bool = True,
        backend: str = "sdnq",
        sdcpp_cache_mode: str = "none",
        sdcpp_diffusion_quant: str = "",
        sdcpp_text_encoder_quant: str = "",
        verbose: bool = False,
    ):
        """Initialize the inpainter.

        Args:
            device: PyTorch device to use. Auto-detects if None.
            huggingface_token: HuggingFace token for model downloads.
            num_inference_steps: Number of denoising steps.
            low_vram: If True, use the low-VRAM CPU offload for SDNQ.
            luminance_correction: If True, match patch luminance to surrounding context.
            upscale_small_crops: If True, scale small crops to ~1MP before inference.
            backend: "sdnq" for Diffusers/SDNQ or "sdcpp" for stable-diffusion.cpp.
            sdcpp_cache_mode: sd.cpp cache mode to use when backend is "sdcpp".
            sdcpp_diffusion_quant: sd.cpp diffusion model quant.
            sdcpp_text_encoder_quant: sd.cpp text encoder quant.
            verbose: Whether to print verbose logging.
        """
        backends = flux_method_backends(self.model_key)
        if backend.lower() not in backends:
            raise ValueError(
                f"Invalid {self.label} backend '{backend}'. "
                f"Must be one of: {', '.join(backends)}."
            )
        super().__init__(
            backend=backend.lower(),
            device=device,
            num_inference_steps=num_inference_steps,
            luminance_correction=luminance_correction,
            upscale_small_crops=upscale_small_crops,
            verbose=verbose,
        )
        self.low_vram = low_vram
        self.huggingface_token = huggingface_token
        self.sdcpp_cache_mode = sdcpp_cache_mode
        self.sdcpp_diffusion_quant = sdcpp_diffusion_quant or flux_sdcpp_quant_default(
            self.model_key, "diffusion_model"
        )
        self.sdcpp_text_encoder_quant = (
            sdcpp_text_encoder_quant or flux_sdcpp_text_encoder_default(self.model_key)
        )
        self.sdcpp_assets = None

    def load_models(self):
        """Start the sd.cpp server or load the SDNQ pipeline via the model manager."""
        if self.pipeline is not None:
            return

        if self.huggingface_token:
            self.manager.set_flux_hf_token(self.huggingface_token)

        if self.backend == "sdcpp":
            self.sdcpp_assets = self.manager.ensure_flux_sdcpp_server(
                self.model_key,
                cache_mode=self.sdcpp_cache_mode,
                diffusion_quant=self.sdcpp_diffusion_quant,
                text_encoder_quant=self.sdcpp_text_encoder_quant,
                num_inference_steps=self.num_inference_steps,
                verbose=self.verbose,
            )
            self.pipeline = self.sdcpp_assets
            return

        self.pipeline = self._load_sdnq_pipeline()

    def unload_models(self):
        """Unload the model via the model manager to free up memory."""
        self.pipeline = None
        self.sdcpp_assets = None
        self._unload_backend()

    def _load_sdnq_pipeline(self):
        raise NotImplementedError

    def _unload_backend(self):
        self.manager.shutdown_sdcpp_server(self.model_key)

    def _run_sdnq_inference(
        self, image: Image.Image, width: int, height: int, seed: int, verbose: bool
    ) -> Image.Image:
        raise NotImplementedError

    def _cache_options(
        self,
        strict_mask_clipping: bool,
        composite_clip_bbox: tuple[int, int, int, int] | None,
        ocr_params: dict | None,
    ) -> dict:
        return _inpaint_cache_options(
            self, strict_mask_clipping, composite_clip_bbox, ocr_params
        )

    def _run_sdcpp_inference(
        self,
        image_pil: Image.Image,
        width: int,
        height: int,
        seed: int,
        verbose: bool = False,
    ) -> Image.Image:
        if self.sdcpp_assets is None:
            raise ModelError(f"{self.label} sd.cpp server is not loaded.")

        payload = {
            "prompt": self.PROMPT,
            "negative_prompt": "",
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "batch_count": 1,
            "ref_images": [pil_to_base64_png(image_pil)],
            "sample_params": {
                "sample_method": "euler",
                "sample_steps": int(self.num_inference_steps),
                "guidance": {
                    "txt_cfg": 1.0,
                    "img_cfg": 1.0,
                    "distilled_guidance": float(self.GUIDANCE_SCALE),
                },
            },
            "output_format": "png",
            "output_compression": 100,
        }
        return run_image_job(
            self.sdcpp_assets, payload, verbose=verbose, timeout_sec=900
        )

    def _generate_patch(
        self,
        image: Image.Image,
        mask: np.ndarray,
        width: int,
        height: int,
        seed: int,
        verbose: bool,
    ) -> Image.Image:
        """Run the loaded backend on the scaled crop; the caller holds the inference lock."""
        if self.backend == "sdcpp":
            return self._run_sdcpp_inference(
                image,
                width,
                height,
                seed,
                verbose=verbose or self.verbose,
            )
        return self._run_sdnq_inference(
            image, width, height, seed, verbose or self.verbose
        )


class FluxKleinInpainter(_EditModelInpainter):
    """Inpainter using Flux.2 Klein models for text removal.

    Unlike FluxKontextInpainter, this works on CPU/ROCm/MPS without Nunchaku.
    """

    def __init__(self, variant: str = "4b", **kwargs):
        """Initialize the Flux Klein Inpainter.

        Args:
            variant: Model variant - "9b" or "4b" (default: "4b")
            **kwargs: See _EditModelInpainter (huggingface_token is required for 9B).
        """
        self.variant = variant.lower()
        if self.variant not in ("9b", "4b"):
            raise ValueError(f"Invalid variant '{variant}'. Must be '9b' or '4b'.")
        self.model_key = f"flux_klein_{self.variant}"
        self.label = f"Flux.2 Klein {self.variant.upper()}"
        super().__init__(**kwargs)
        self._prompt_embeds_cpu = None
        self._pooled_prompt_embeds_cpu = None

    def _load_sdnq_pipeline(self):
        if self.variant == "9b":
            return self.manager.load_flux_klein_9b(
                low_vram=self.low_vram, verbose=self.verbose
            )
        return self.manager.load_flux_klein_4b(
            low_vram=self.low_vram, verbose=self.verbose
        )

    def unload_models(self):
        """Unload Flux.2 Klein models via model manager to free up memory."""
        self._prompt_embeds_cpu = None
        self._pooled_prompt_embeds_cpu = None
        super().unload_models()

    def _unload_backend(self):
        if self.backend == "sdcpp":
            self.manager.shutdown_sdcpp_server(self.model_key)
        else:
            self.manager.unload_flux_klein_models()

    def _get_prompt_embeddings(self, device: torch.device, verbose: bool = False):
        if self._prompt_embeds_cpu is None:
            log_message("  - Encoding Flux.2 Klein prompt embeddings", verbose=verbose)
            prompt_embeds, pooled_prompt_embeds = _encode_flux_prompt(
                self.pipeline, self.PROMPT, device
            )
            self._prompt_embeds_cpu = _prompt_value_to_cpu(prompt_embeds)
            self._pooled_prompt_embeds_cpu = _prompt_value_to_cpu(pooled_prompt_embeds)
        else:
            log_message("  - Reusing Flux.2 Klein prompt embeddings", verbose=verbose)

        return (
            _prompt_value_to_device(self._prompt_embeds_cpu, device),
            _prompt_value_to_device(self._pooled_prompt_embeds_cpu, device),
        )

    def _run_sdnq_inference(
        self, image: Image.Image, width: int, height: int, seed: int, verbose: bool
    ) -> Image.Image:
        with torch.inference_mode():
            gen = torch.Generator(device=self.DEVICE).manual_seed(seed)
            prompt_device = _pipeline_execution_device(self.pipeline, self.DEVICE)
            prompt_embeds, pooled_prompt_embeds = self._get_prompt_embeddings(
                prompt_device, verbose=verbose
            )
            out = self.pipeline(
                **_flux_prompt_kwargs(
                    prompt_embeds,
                    pooled_prompt_embeds,
                    include_pooled=False,
                ),
                image=image,
                height=height,
                width=width,
                guidance_scale=self.GUIDANCE_SCALE,
                num_inference_steps=self.num_inference_steps,
                generator=gen,
                callback_on_step_end=_interrupt_on_cancel,
            )
            return out.images[0]


class Flux3KleinInpainter(_EditModelInpainter):
    """Inpainter using Flux.3 Image Klein (16-step distilled) for text removal."""

    model_key = "flux3_klein"
    label = "Flux.3 Image Klein"

    def _load_sdnq_pipeline(self):
        return self.manager.load_flux3_klein_sdnq(
            low_vram=self.low_vram, verbose=self.verbose
        )

    def _unload_backend(self):
        self.manager.unload_flux3_klein_models()

    def _run_sdnq_inference(
        self, image: Image.Image, width: int, height: int, seed: int, verbose: bool
    ) -> Image.Image:
        with torch.inference_mode():
            images = self.pipeline(
                prompt=self.PROMPT,
                image=image,
                height=height,
                width=width,
                num_inference_steps=self.num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                output="images",
            )
        return images[0].convert("RGB")


class Fibo2Inpainter(_EditModelInpainter):
    """Inpainter using Fibo 2.0 Edit Turbo (sd.cpp only) for text removal."""

    model_key = "fibo2_edit_turbo"
    label = "Fibo 2.0 Edit Turbo"


class QwenImage21Inpainter(_EditModelInpainter):
    """Inpainter using Qwen-Image 2.1 Turbo for text removal."""

    model_key = "qwen_image_2_1_turbo"
    label = "Qwen-Image 2.1 Turbo"
    # sd.cpp and the Diffusers pipeline both need multiples of 32
    RESOLUTION_MULTIPLE = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_inference_steps = flux_steps_range(self.model_key)[0]

    def _load_sdnq_pipeline(self):
        return self.manager.load_qwen_image_21_sdnq(
            low_vram=self.low_vram, verbose=self.verbose
        )

    def _unload_backend(self):
        self.manager.unload_qwen_image_21_models()

    def _run_sdnq_inference(
        self, image: Image.Image, width: int, height: int, seed: int, verbose: bool
    ) -> Image.Image:
        with torch.inference_mode():
            out = self.pipeline(
                prompt=self.PROMPT,
                image=image,
                width=width,
                height=height,
                # The pipeline resizes references to this area at the crop's aspect
                # ratio, so the crop's own area keeps output and crop pixel-aligned
                output_resolution=math.sqrt(width * height),
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=_interrupt_on_cancel,
            )
        return out.images[0].convert("RGB")


class LamaInpainter(_CropInpainter):
    """Inpainter using the manga-finetuned Big-LaMa, a fast non-diffusion model."""

    model_key = "lama_manga"
    label = "LaMa Manga"

    def __init__(self, device: torch.device | None = None, verbose: bool = False):
        # LaMa fills from the crop's own context, so it runs at native resolution
        super().__init__(
            backend="torch",
            device=device,
            num_inference_steps=1,
            luminance_correction=False,
            upscale_small_crops=False,
            verbose=verbose,
        )

    def load_models(self):
        if self.pipeline is None:
            self.pipeline = self.manager.load_lama_manga(verbose=self.verbose)

    def unload_models(self):
        self.pipeline = None
        self.manager.unload_lama_manga()

    def _generate_patch(
        self,
        image: Image.Image,
        mask: np.ndarray,
        width: int,
        height: int,
        seed: int,
        verbose: bool,
    ) -> Image.Image:
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        device = self.pipeline.device
        image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)[None].to(device)
        mask_t = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
        with torch.inference_mode():
            output = self.pipeline(image_t, mask_t)
        output_np = output[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return Image.fromarray((output_np * 255).round().astype(np.uint8))


_EDIT_MODEL_INPAINTERS = {
    cls.model_key: cls
    for cls in (Flux3KleinInpainter, Fibo2Inpainter, QwenImage21Inpainter)
}
