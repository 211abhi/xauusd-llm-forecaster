"""Patch-based tokenizer: segments OHLCV windows into fixed-size patches."""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple


def window_to_patches(window: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Convert a single (window_size, n_features) window into patches.

    Returns shape (n_patches, patch_dim) where patch_dim = patch_size * n_features.
    """
    window_size, n_features = window.shape
    assert window_size % patch_size == 0, \
        f"window_size {window_size} must be divisible by patch_size {patch_size}"
    n_patches = window_size // patch_size
    patches = window.reshape(n_patches, patch_size * n_features)
    return patches


def batch_to_patches(windows: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Convert batch of windows (B, window_size, n_features) to patches (B, n_patches, patch_dim).
    """
    B, window_size, n_features = windows.shape
    n_patches = window_size // patch_size
    patch_dim = patch_size * n_features
    return windows.reshape(B, n_patches, patch_dim)


def patches_to_tensor(patches: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy patches array to float32 tensor on device."""
    return torch.tensor(patches, dtype=torch.float32, device=device)


class PatchTokenizer:
    """Converts OHLCV windows into patch sequences for the TS encoder."""

    def __init__(self, window_size: int, patch_size: int, n_features: int, device: str = "cpu") -> None:
        assert window_size % patch_size == 0, \
            f"window_size {window_size} must be divisible by patch_size {patch_size}"
        self.window_size = window_size
        self.patch_size = patch_size
        self.n_patches = window_size // patch_size
        self.n_features = n_features
        self.patch_dim = patch_size * n_features
        self.device = torch.device(device)

    @classmethod
    def from_config(cls, cfg: dict) -> "PatchTokenizer":
        """Instantiate from base_config dict."""
        return cls(
            window_size=cfg["tokenizer"]["window_size"],
            patch_size=cfg["tokenizer"]["patch_size"],
            n_features=cfg["tokenizer"]["n_features"],
            device=cfg["project"]["device"],
        )

    def __call__(self, windows: np.ndarray) -> torch.Tensor:
        """
        Tokenize a batch of windows.

        Args:
            windows: (B, window_size, n_features) numpy array
        Returns:
            Tensor of shape (B, n_patches, patch_dim)
        """
        patches = batch_to_patches(windows, self.patch_size)
        return patches_to_tensor(patches, self.device)

    def output_shape(self) -> Tuple[int, int]:
        """Return (n_patches, patch_dim)."""
        return self.n_patches, self.patch_dim
