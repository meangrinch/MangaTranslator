# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Vendored from Hi-SAM (https://github.com/ymy-k/Hi-SAM, commit 6900943, Apache-2.0; see
# the LICENSE file in the repository root). Changed: the einops rearrange is replaced
# with the equivalent flatten/transpose, and the unused CrossModalMLP and init are
# dropped.

import torch
import torch.nn.functional as F
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, output_dim, nhead, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            output_dim, nhead, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(output_dim)

        self.cross_attn = nn.MultiheadAttention(
            output_dim, nhead, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(output_dim)

        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Linear(output_dim * 4, output_dim),
        )
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, sparse_embeddings, image_embeddings):
        sparse_embeddings = sparse_embeddings + self.dropout1(
            self.self_attn(
                query=sparse_embeddings, key=sparse_embeddings, value=sparse_embeddings
            )[0]
        )
        sparse_embeddings = self.norm1(sparse_embeddings)
        image_tokens = image_embeddings.flatten(2).transpose(1, 2)
        sparse_embeddings = sparse_embeddings + self.dropout2(
            self.cross_attn(
                query=sparse_embeddings, key=image_tokens, value=image_tokens
            )[0]
        )
        return sparse_embeddings + self.dropout3(
            self.ffn(self.norm2(sparse_embeddings))
        )


class ModalAligner(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        act: type[nn.Module] = nn.GELU,
        nhead: int = 8,
        dropout: float = 0.1,
        attn_layers: int = 1,
        prompt_len: int = 12,
    ) -> None:
        super().__init__()
        self.prompt_len = prompt_len
        self.conv = nn.Sequential(
            nn.Conv2d(
                transformer_dim, prompt_len, kernel_size=3, padding=1, bias=False
            ),
            act(),
            nn.Conv2d(prompt_len, prompt_len, kernel_size=3, padding=1, bias=False),
            act(),
            nn.Conv2d(prompt_len, prompt_len, kernel_size=3, padding=1, bias=False),
            act(),
            nn.Conv2d(prompt_len, prompt_len, kernel_size=3, padding=1, bias=False),
        )
        self.transformer_layers = nn.ModuleList(
            [
                AttentionBlock(transformer_dim, nhead, dropout)
                for _ in range(attn_layers)
            ]
        )

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        bs, c, _, _ = image_embeddings.shape
        spatial_attention = self.conv(image_embeddings).reshape(bs, self.prompt_len, -1)
        spatial_attention = F.sigmoid(spatial_attention)[..., None]
        feat = image_embeddings.reshape(bs, c, -1).permute(0, 2, 1)[:, None, ...]
        sparse_embeddings = (feat * spatial_attention).mean(dim=2)
        for layer in self.transformer_layers:
            sparse_embeddings = layer(sparse_embeddings, image_embeddings)
        return sparse_embeddings
