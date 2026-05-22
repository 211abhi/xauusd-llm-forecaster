"""InfoNCE / NT-Xent contrastive loss for TS encoder alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    NT-Xent (InfoNCE) loss.

    Expects ts_embeds and text_embeds of the same shape (B, D).
    Each sample i is a positive pair; all other j≠i are negatives.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, ts_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ts_embeds:   (B, D) — projected encoder outputs
            text_embeds: (B, D) — frozen LLM text embeddings
        Returns:
            scalar loss
        """
        ts_norm = F.normalize(ts_embeds, dim=-1)
        text_norm = F.normalize(text_embeds, dim=-1)

        logits = torch.matmul(ts_norm, text_norm.T) / self.temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric loss: ts→text and text→ts
        loss_ts = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)
        return (loss_ts + loss_text) / 2.0
