"""Full inference pipeline: patches → encoder → LLM → prediction head → prices."""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from src.encoder.ts_encoder import TSEncoder
from src.encoder.encoder_trainer import ProjectionHead
from src.llm.frozen_llm import FrozenLLM
from src.soft_prompt.soft_prompt import SoftPrompt
from src.prediction.pred_head import PredictionHead
from src.tokenizer.patch_tokenizer import PatchTokenizer
from src.utils.checkpoint import load_all


class Forecaster:
    """End-to-end inference pipeline for XAUUSD price forecasting."""

    def __init__(self, encoder: TSEncoder, proj_head: ProjectionHead, llm: FrozenLLM,
                 soft_prompt: SoftPrompt, pred_head: PredictionHead,
                 tokenizer: PatchTokenizer, device: torch.device) -> None:
        self.encoder = encoder.to(device)
        self.proj_head = proj_head.to(device)
        self.llm = llm
        self.soft_prompt = soft_prompt.to(device)
        self.pred_head = pred_head.to(device)
        self.tokenizer = tokenizer
        self.device = device

        for m in [self.encoder, self.proj_head, self.pred_head]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    @classmethod
    def from_config(cls, cfg: dict) -> "Forecaster":
        """Load all components from checkpoints specified in config."""
        device = torch.device(cfg["project"]["device"])
        encoder, proj_head, pred_head, soft_prompt = load_all(cfg)
        llm = FrozenLLM.from_config(cfg)
        tokenizer = PatchTokenizer.from_config(cfg)
        return cls(encoder, proj_head, llm, soft_prompt, pred_head, tokenizer, device)

    @torch.no_grad()
    def predict(self, windows: np.ndarray) -> np.ndarray:
        """
        Predict future prices for a batch of input windows.

        Args:
            windows: (B, window_size, n_features) numpy array (already normalized)
        Returns:
            predictions: (B, output_steps) numpy array of normalized prices
        """
        patches = self.tokenizer(windows)              # (B, n_patches, patch_dim)
        B = patches.size(0)

        ts_embed = self.encoder(patches)               # (B, 256)
        ts_proj = self.proj_head(ts_embed).unsqueeze(1)  # (B, 1, 768)

        soft = self.soft_prompt(B)                     # (B, 32, 768)
        inputs = torch.cat([soft, ts_proj], dim=1)     # (B, 33, 768)

        hidden = self.llm.get_hidden_state(inputs)     # (B, 768)
        preds = self.pred_head(hidden)                 # (B, output_steps)
        return preds.cpu().numpy()
