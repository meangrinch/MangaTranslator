# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Vendored from Hi-SAM (https://github.com/ymy-k/Hi-SAM, commit 6900943, Apache-2.0; see
# the LICENSE file in the repository root). Changed: HiDecoder is dropped, and forward
# returns only the high-resolution text logits. The low-resolution and IoU heads are
# kept so checkpoints load strictly.

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        hr_dim = transformer_dim // 16
        self.output_upscaling_hr = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim // 8, hr_dim, kernel_size=2, stride=2),
            LayerNorm2d(hr_dim),
            activation(),
            nn.ConvTranspose2d(hr_dim, hr_dim, kernel_size=2, stride=2),
            LayerNorm2d(hr_dim),
            activation(),
            nn.Conv2d(hr_dim, hr_dim, kernel_size=3, padding=1),
            LayerNorm2d(hr_dim),
            activation(),
            nn.Conv2d(hr_dim, hr_dim, kernel_size=3, padding=1),
            LayerNorm2d(hr_dim),
            activation(),
            nn.Conv2d(hr_dim, hr_dim, kernel_size=3, padding=1),
            LayerNorm2d(hr_dim),
            activation(),
            nn.Conv2d(hr_dim, hr_dim, kernel_size=3, padding=1),
            activation(),
        )
        self.output_hypernetworks_mlps_hr = MLP(
            transformer_dim, transformer_dim, hr_dim, 3
        )
        self.iou_prediction_head_hr = MLP(
            transformer_dim, iou_head_hidden_dim, 1, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """High-resolution (4x the embedding grid x 4) text logits, B x 1 x H x W."""
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        mask_token_out = hs[:, 1, :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled = self.output_upscaling_hr(self.output_upscaling(src))
        hyper_in_hr = self.output_hypernetworks_mlps_hr(mask_token_out)
        b, c, h, w = upscaled.shape
        return (hyper_in_hr.unsqueeze(1) @ upscaled.view(b, c, h * w)).view(b, -1, h, w)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
