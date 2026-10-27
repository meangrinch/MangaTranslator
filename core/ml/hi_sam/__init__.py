"""Inference-only SAM-TS text segmentation from Hi-SAM, for Koharu's manga fine-tune.

The modules keep Hi-SAM's names so `mayocream/koharu-text-sam-ts-l` loads with
`strict=True`. Upstream: https://github.com/ymy-k/Hi-SAM (Apache-2.0).
"""

from functools import partial

import numpy as np
import torch
from torch import nn

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .modal_aligner import ModalAligner
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer

IMAGE_SIZE = 1024
PAD_VALUE = 128


class TextSamTS(nn.Module):
    """SAM-TS: ViT image encoder, self-prompting modal aligner, high-res mask head."""

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        modal_aligner: ModalAligner,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.modal_aligner = modal_aligner
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.register_buffer("pixel_mean", mean, False)
        self.register_buffer("pixel_std", std, False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Text logits for B x 3 x 1024 x 1024 RGB images in 0-255, as B x 1 x 1024 x 1024."""
        embeddings = self.image_encoder((images - self.pixel_mean) / self.pixel_std)
        prompts = self.modal_aligner(embeddings)
        dense_pe = self.prompt_encoder.get_dense_pe()
        # The decoder repeats image embeddings per prompt set, so decode one image at a time
        return torch.cat(
            [
                self.mask_decoder(emb.unsqueeze(0), dense_pe, prompt.unsqueeze(0))
                for emb, prompt in zip(embeddings, prompts)
            ],
            dim=0,
        )


def build_text_sam_ts_l() -> TextSamTS:
    """Build the ViT-L SAM-TS architecture (attn_layers=1, prompt_len=12)."""
    prompt_embed_dim = 256
    vit_patch_size = 16
    embedding_size = IMAGE_SIZE // vit_patch_size
    return TextSamTS(
        image_encoder=ImageEncoderViT(
            depth=24,
            embed_dim=1024,
            img_size=IMAGE_SIZE,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=16,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[5, 11, 17, 23],
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        modal_aligner=ModalAligner(prompt_embed_dim, attn_layers=1, prompt_len=12),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(embedding_size, embedding_size),
            input_image_size=(IMAGE_SIZE, IMAGE_SIZE),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    )


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """Reduced precision the model runs in on `device`, or None to run in fp32.

    On 14 sample pages fp16 came about 6x closer to fp32 than bf16, with no overflow,
    and Recover Missed Text did as well or better; Apple (MPS) and Intel (XPU) GPUs are
    untested on real hardware.
    """
    return torch.float16 if device.type in ("cuda", "mps", "xpu") else None


def store_matmul_weights(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Store Linear and conv weights in `dtype`, which autocast casts them to on every
    call anyway, so outputs are unchanged and their memory halves. Norms and position
    embeddings stay fp32, as autocast runs them."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.to(dtype)
    return model


def to_inference_device(model: nn.Module, device: torch.device) -> nn.Module:
    """Move the model to `device` for predict_text_probability."""
    dtype = autocast_dtype(device)
    if dtype is not None:
        # Before the move, so the fp32 copies never occupy device memory
        store_matmul_weights(model, dtype)
    return model.to(device)


def predict_text_probability(model: nn.Module, window_rgb: np.ndarray) -> np.ndarray:
    """Per-pixel text probability for an RGB uint8 window of at most 1024 x 1024."""
    height, width = window_rgb.shape[:2]
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:height, :width] = window_rgb
    device = next(model.parameters()).device
    images = torch.from_numpy(canvas).to(device).permute(2, 0, 1)[None].float()
    dtype = autocast_dtype(device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None),
    ):
        logits = model(images)
    prob = torch.sigmoid(logits[0, 0, :height, :width].float())
    return prob.cpu().numpy().astype(np.float32)
