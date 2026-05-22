"""Builds (ts_patch, text_embedding) positive pairs for contrastive training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple


class PairDataset(Dataset):
    """
    Dataset of (patch_sequence, text_embedding) pairs.

    Each item is a positive pair: a time-series window and the embedding of
    its regime's text description. Negatives are handled implicitly by InfoNCE
    (all other items in the batch are negatives).
    """

    def __init__(self, patches: np.ndarray, regimes: List[str],
                 regime_embeddings: Dict[str, torch.Tensor]) -> None:
        """
        Args:
            patches:           (N, n_patches, patch_dim) float32 array
            regimes:           list of N regime label strings
            regime_embeddings: dict mapping regime → (D,) text embedding tensor
        """
        assert len(patches) == len(regimes), "patches and regimes must be same length"
        self.patches = torch.tensor(patches, dtype=torch.float32)
        self.regimes = regimes
        self.regime_embeddings = regime_embeddings

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx]
        text_embed = self.regime_embeddings[self.regimes[idx]]
        return patch, text_embed

    @classmethod
    def from_arrays(cls, patches: np.ndarray, regimes: List[str],
                    regime_embeddings: Dict[str, torch.Tensor]) -> "PairDataset":
        """Convenience constructor."""
        return cls(patches, regimes, regime_embeddings)

    def regime_counts(self) -> Dict[str, int]:
        """Return count per regime label (useful for balance check)."""
        from collections import Counter
        return dict(Counter(self.regimes))
