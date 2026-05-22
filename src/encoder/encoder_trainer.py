"""Contrastive training loop for the TS encoder (Phase 2)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

from src.encoder.ts_encoder import TSEncoder
from src.alignment.contrastive_loss import InfoNCELoss
from src.alignment.pair_dataset import PairDataset
from src.utils.checkpoint import save_encoder, load_encoder


class ProjectionHead(nn.Module):
    """Projects encoder output (256) into LLM embedding space (768)."""

    def __init__(self, input_dim: int = 256, output_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cosine_schedule_with_warmup(optimizer: optim.Optimizer, warmup_steps: int,
                                  total_steps: int) -> optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EncoderTrainer:
    """Manages Phase 2: contrastive training of TSEncoder + ProjectionHead."""

    def __init__(self, encoder: TSEncoder, proj_head: ProjectionHead,
                 loss_fn: InfoNCELoss, cfg: dict, device: torch.device) -> None:
        self.encoder = encoder.to(device)
        self.proj_head = proj_head.to(device)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device

        tc = cfg["encoder_training"]
        params = list(encoder.parameters()) + list(proj_head.parameters())
        self.optimizer = optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])

        total_steps = tc["epochs"] * 1000  # rough estimate; updated at train time
        self.scheduler = cosine_schedule_with_warmup(
            self.optimizer, tc["warmup_steps"], total_steps
        )
        self.patience = tc["patience"]
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch, return average loss."""
        self.encoder.train()
        self.proj_head.train()
        total_loss = 0.0
        for patches, text_embeddings in loader:
            patches = patches.to(self.device)
            text_embeddings = text_embeddings.to(self.device)

            ts_embed = self.encoder(patches)           # (B, 256)
            ts_proj = self.proj_head(ts_embed)         # (B, 768)

            loss = self.loss_fn(ts_proj, text_embeddings)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.proj_head.parameters()), 1.0
            )
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> float:
        """Run validation epoch, return average loss."""
        self.encoder.eval()
        self.proj_head.eval()
        total_loss = 0.0
        for patches, text_embeddings in loader:
            patches = patches.to(self.device)
            text_embeddings = text_embeddings.to(self.device)
            ts_proj = self.proj_head(self.encoder(patches))
            total_loss += self.loss_fn(ts_proj, text_embeddings).item()
        return total_loss / max(1, len(loader))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            checkpoint_dir: str) -> None:
        """Full training loop with early stopping."""
        epochs = self.cfg["encoder_training"]["epochs"]
        val_every = self.cfg["encoder_training"]["val_every"]
        ckpt_path = Path(checkpoint_dir) / "best_encoder.pt"

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)

            if epoch % val_every == 0:
                val_loss = self.val_epoch(val_loader)
                print(f"Epoch {epoch:3d}/{epochs} | train={train_loss:.4f} val={val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    save_encoder(self.encoder, self.proj_head, str(ckpt_path))
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                print(f"Epoch {epoch:3d}/{epochs} | train={train_loss:.4f}")
