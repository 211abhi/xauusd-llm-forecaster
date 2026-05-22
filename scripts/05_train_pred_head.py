"""Phase 4 — Train prediction head (encoder + LLM + soft prompt all frozen)."""

from __future__ import annotations

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.encoder.ts_encoder import TSEncoder
from src.encoder.encoder_trainer import ProjectionHead
from src.llm.frozen_llm import FrozenLLM
from src.soft_prompt.soft_prompt import SoftPrompt
from src.prediction.pred_head import PredictionHead
from src.tokenizer.patch_tokenizer import batch_to_patches
from src.utils.data_loader import get_feature_columns, build_windows, build_targets
from src.utils.checkpoint import load_encoder, save_pred_head


def make_loader(arr: np.ndarray, cfg: dict, stride: int) -> DataLoader:
    """Build a DataLoader from a feature array."""
    feature_cols = get_feature_columns(cfg)
    window_size = cfg["tokenizer"]["window_size"]
    patch_size = cfg["tokenizer"]["patch_size"]
    n_steps = cfg["prediction_head"]["output_steps"]
    close_idx = feature_cols.index("close") if isinstance(feature_cols[0], str) else 3

    windows = build_windows(arr, window_size, stride)
    targets = build_targets(arr[:, close_idx], window_size, stride, n_steps)
    min_len = min(len(windows), len(targets))
    windows, targets = windows[:min_len], targets[:min_len]

    patches = batch_to_patches(windows, patch_size).astype(np.float32)
    dataset = TensorDataset(
        torch.tensor(patches, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    bs = cfg["pred_head_training"]["batch_size"]
    return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg["project"]["device"])

    processed_dir = Path(cfg["data"]["processed_dir"])
    df = pd.read_csv(processed_dir / "xau_with_regimes.csv", parse_dates=["datetime"])
    feature_cols = get_feature_columns(cfg)

    train_ratio = cfg["data"]["split_ratios"]["train"]
    val_ratio = cfg["data"]["split_ratios"]["val"]
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_arr = df[feature_cols].values[:train_end]
    val_arr   = df[feature_cols].values[train_end:val_end]

    train_loader = make_loader(train_arr, cfg, cfg["tokenizer"]["train_stride"])
    val_loader   = make_loader(val_arr,   cfg, cfg["tokenizer"]["val_stride"])
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    print("Loading frozen components...")
    encoder = TSEncoder.from_config(cfg)
    proj_head = ProjectionHead(cfg["encoder"]["output_dim"], cfg["alignment"]["projection_dim"])
    load_encoder(encoder, proj_head, cfg["encoder"]["checkpoint_path"], device)

    llm = FrozenLLM.from_config(cfg)
    soft_prompt = SoftPrompt.load(cfg["soft_prompt"]["checkpoint_path"], cfg).to(device)

    for m in [encoder, proj_head]:
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False

    pred_head_raw = PredictionHead.from_config(cfg).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        pred_head = torch.nn.DataParallel(pred_head_raw)
    else:
        pred_head = pred_head_raw
    optimizer = optim.AdamW(
        pred_head_raw.parameters(),
        lr=cfg["pred_head_training"]["lr"],
        weight_decay=cfg["pred_head_training"]["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = cfg["pred_head_training"]["patience"]
    epochs = cfg["pred_head_training"]["epochs"]
    ckpt_path = cfg["prediction_head"]["checkpoint_path"]
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    print("Training prediction head...")
    for epoch in range(1, epochs + 1):
        pred_head.train()
        train_loss = 0.0
        for patches, targets in train_loader:
            patches, targets = patches.to(device), targets.to(device)
            B = patches.size(0)
            with torch.no_grad():
                ts_embed = encoder(patches)
                ts_proj = proj_head(ts_embed).unsqueeze(1)
                soft = soft_prompt(B)
                inputs = torch.cat([soft, ts_proj], dim=1)
                hidden = llm.get_hidden_state(inputs)
            preds = pred_head(hidden)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pred_head.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        pred_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for patches, targets in val_loader:
                patches, targets = patches.to(device), targets.to(device)
                B = patches.size(0)
                ts_proj = proj_head(encoder(patches)).unsqueeze(1)
                soft = soft_prompt(B)
                inputs = torch.cat([soft, ts_proj], dim=1)
                hidden = llm.get_hidden_state(inputs)
                val_loss += loss_fn(pred_head(hidden), targets).item()

        tl = train_loss / len(train_loader)
        vl = val_loss / len(val_loader)
        print(f"Epoch {epoch:3d}/{epochs} | train={tl:.6f} val={vl:.6f}")

        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            save_pred_head(pred_head_raw, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best val loss: {best_val_loss:.6f}")
    print("Phase 4 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
