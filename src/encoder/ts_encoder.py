"""Time-series encoder: patch embedding + Transformer → 256-dim CLS vector."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class TSEncoder(nn.Module):
    """
    Encodes patch sequences into a fixed-size CLS representation.

    Input:  (B, n_patches, patch_dim)
    Output: (B, embed_dim)
    """

    def __init__(self, patch_dim: int, embed_dim: int = 256, n_heads: int = 8,
                 n_layers: int = 4, ffn_dim: int = 512, dropout: float = 0.1,
                 max_seq_len: int = 7) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, n_patches, patch_dim)
        Returns:
            cls_out: (B, embed_dim)
        """
        B = patches.size(0)
        x = self.patch_proj(patches)                          # (B, n_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)                        # (B, n_patches+1, embed_dim)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0, :]                                     # (B, embed_dim) — CLS token

    @classmethod
    def from_config(cls, cfg: dict) -> "TSEncoder":
        """Instantiate from base_config dict."""
        return cls(
            patch_dim=cfg["tokenizer"]["patch_dim"],
            embed_dim=cfg["encoder"]["embed_dim"],
            n_heads=cfg["encoder"]["n_heads"],
            n_layers=cfg["encoder"]["n_layers"],
            ffn_dim=cfg["encoder"]["ffn_dim"],
            dropout=cfg["encoder"]["dropout"],
            max_seq_len=cfg["encoder"]["max_seq_len"],
        )
