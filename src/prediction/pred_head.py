"""Prediction head: 2-layer MLP decoding LLM hidden state to future prices."""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    Decodes LLM CLS hidden state into N future close prices.

    Input:  (B, 768)
    Output: (B, output_steps)
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 output_steps: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_steps),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, input_dim). Returns: (B, output_steps)."""
        return self.net(x)

    @classmethod
    def from_config(cls, cfg: dict) -> "PredictionHead":
        ph = cfg["prediction_head"]
        return cls(
            input_dim=ph["input_dim"],
            hidden_dim=ph["hidden_dim"],
            output_steps=ph["output_steps"],
            dropout=ph["dropout"],
        )
