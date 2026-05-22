"""Save and load model checkpoints."""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import Tuple


def save_encoder(encoder: torch.nn.Module, proj_head: torch.nn.Module,
                 path: str) -> None:
    """Save encoder + projection head state dicts."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "proj_head": proj_head.state_dict(),
    }, path)


def load_encoder(encoder: torch.nn.Module, proj_head: torch.nn.Module,
                 path: str, device: torch.device) -> None:
    """Load encoder + projection head state dicts in-place."""
    ckpt = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    proj_head.load_state_dict(ckpt["proj_head"])


def save_pred_head(pred_head: torch.nn.Module, path: str) -> None:
    """Save prediction head state dict."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pred_head": pred_head.state_dict()}, path)


def load_pred_head(pred_head: torch.nn.Module, path: str, device: torch.device) -> None:
    """Load prediction head state dict in-place."""
    ckpt = torch.load(path, map_location=device)
    pred_head.load_state_dict(ckpt["pred_head"])


def load_all(cfg: dict) -> Tuple:
    """Load encoder, proj_head, pred_head, soft_prompt from config checkpoint paths."""
    from src.encoder.ts_encoder import TSEncoder
    from src.encoder.encoder_trainer import ProjectionHead
    from src.prediction.pred_head import PredictionHead
    from src.soft_prompt.soft_prompt import SoftPrompt

    device = torch.device(cfg["project"]["device"])
    encoder = TSEncoder.from_config(cfg)
    proj_head = ProjectionHead(
        input_dim=cfg["encoder"]["output_dim"],
        output_dim=cfg["alignment"]["projection_dim"],
    )
    load_encoder(encoder, proj_head, cfg["encoder"]["checkpoint_path"], device)
    encoder.to(device)
    proj_head.to(device)

    pred_head = PredictionHead.from_config(cfg)
    load_pred_head(pred_head, cfg["prediction_head"]["checkpoint_path"], device)
    pred_head.to(device)

    soft_prompt = SoftPrompt.load(cfg["soft_prompt"]["checkpoint_path"], cfg)
    soft_prompt.to(device)

    return encoder, proj_head, pred_head, soft_prompt
