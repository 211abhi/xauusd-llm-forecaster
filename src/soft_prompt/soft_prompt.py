"""Trainable soft prompt prefix tensor, shape (n_tokens, token_dim)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SoftPrompt(nn.Module):
    """
    A learnable prefix in the LLM's embedding space.

    Shape: (n_tokens, token_dim) = (32, 768) by default.
    Optimized via CMA-ES (not backprop) — but stored as nn.Parameter
    so it can be moved to device and serialized uniformly.
    """

    def __init__(self, n_tokens: int = 32, token_dim: int = 768,
                 init_std: float = 0.01) -> None:
        super().__init__()
        data = torch.randn(n_tokens, token_dim) * init_std
        self.prompt = nn.Parameter(data, requires_grad=False)
        self.n_tokens = n_tokens
        self.token_dim = token_dim

    @classmethod
    def from_config(cls, cfg: dict) -> "SoftPrompt":
        return cls(
            n_tokens=cfg["soft_prompt"]["n_tokens"],
            token_dim=cfg["soft_prompt"]["token_dim"],
            init_std=cfg["soft_prompt"]["init_std"],
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """Return soft prompt expanded for a batch: (B, n_tokens, token_dim)."""
        return self.prompt.unsqueeze(0).expand(batch_size, -1, -1)

    def set_from_numpy(self, arr: np.ndarray) -> None:
        """Update prompt weights from a numpy array (used by CMA-ES)."""
        assert arr.shape == (self.n_tokens, self.token_dim), \
            f"Expected shape {(self.n_tokens, self.token_dim)}, got {arr.shape}"
        with torch.no_grad():
            self.prompt.copy_(torch.tensor(arr, dtype=torch.float32,
                                            device=self.prompt.device))

    def to_numpy(self) -> np.ndarray:
        """Return current prompt as numpy array."""
        return self.prompt.detach().cpu().numpy()

    def save(self, path: str) -> None:
        """Save prompt weights as .npy file."""
        np.save(path, self.to_numpy())

    @classmethod
    def load(cls, path: str, cfg: dict) -> "SoftPrompt":
        """Load prompt weights from .npy file."""
        arr = np.load(path)
        sp = cls.from_config(cfg)
        sp.set_from_numpy(arr)
        return sp
