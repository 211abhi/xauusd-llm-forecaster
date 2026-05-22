"""Phase 2 — Contrastive training of the TS encoder."""

from __future__ import annotations

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.encoder.ts_encoder import TSEncoder
from src.encoder.encoder_trainer import EncoderTrainer, ProjectionHead
from src.alignment.contrastive_loss import InfoNCELoss
from src.alignment.pair_dataset import PairDataset
from src.alignment.text_encoder import TextEncoder
from src.tokenizer.patch_tokenizer import PatchTokenizer, batch_to_patches
from src.utils.data_loader import get_feature_columns
from src.utils.regime_templates import REGIME_TEMPLATES


def load_split_patches(df: pd.DataFrame, regimes: pd.Series, cfg: dict,
                        split: str) -> tuple:
    """Extract patch arrays and regime list for a data split."""
    feature_cols = get_feature_columns(cfg)
    arr = df[feature_cols].values
    window_size = cfg["tokenizer"]["window_size"]
    patch_size = cfg["tokenizer"]["patch_size"]
    stride = cfg["tokenizer"]["train_stride"] if split == "train" else cfg["tokenizer"]["val_stride"]

    windows, idxs = [], []
    for start in range(0, len(arr) - window_size + 1, stride):
        windows.append(arr[start: start + window_size])
        idxs.append(start)
    windows = np.stack(windows, axis=0)                              # (N, W, F)
    patches = batch_to_patches(windows, patch_size).astype(np.float32)  # (N, P, D)

    regime_list = regimes.values[:len(patches)]
    return patches, list(regime_list)


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg["project"]["device"])

    processed_dir = Path(cfg["data"]["processed_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])

    ckpt_path = Path(cfg["encoder"]["checkpoint_path"])
    if ckpt_path.exists():
        print(f"Encoder checkpoint found at {ckpt_path} — skipping training.")
        print("Phase 2 complete (skipped).")
        return

    print("Loading processed data...")
    df = pd.read_csv(processed_dir / "xau_with_regimes.csv", parse_dates=["datetime"])
    train_ratio = cfg["data"]["split_ratios"]["train"]
    val_ratio = cfg["data"]["split_ratios"]["val"]
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df   = df.iloc[train_end:val_end].reset_index(drop=True)

    train_regimes = pd.read_csv(splits_dir / "train_regimes.csv")["regime"]
    val_regimes   = pd.read_csv(splits_dir / "val_regimes.csv")["regime"]

    print("Building patch arrays...")
    train_patches, train_reg_list = load_split_patches(train_df, train_regimes, cfg, "train")
    val_patches,   val_reg_list   = load_split_patches(val_df,   val_regimes,   cfg, "val")
    print(f"  train patches: {train_patches.shape}, val patches: {val_patches.shape}")

    print("Precomputing regime text embeddings (frozen LLM)...")
    text_enc = TextEncoder(
        model_name=cfg["llm"]["model_name"],
        cache_dir=cfg["llm"]["cache_dir"],
        device=cfg["project"]["device"],
    )
    regime_embeddings = text_enc.precompute_regime_embeddings(REGIME_TEMPLATES)
    print(f"  Computed embeddings for {len(regime_embeddings)} regimes")

    train_dataset = PairDataset(train_patches, train_reg_list, regime_embeddings)
    val_dataset   = PairDataset(val_patches,   val_reg_list,   regime_embeddings)
    print("  Regime counts (train):", train_dataset.regime_counts())

    tc = cfg["encoder_training"]
    train_loader = DataLoader(train_dataset, batch_size=tc["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=tc["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=False)

    print("Building encoder + projection head...")
    encoder   = TSEncoder.from_config(cfg)
    proj_head = ProjectionHead(
        input_dim=cfg["encoder"]["output_dim"],
        output_dim=cfg["alignment"]["projection_dim"],
    )
    loss_fn = InfoNCELoss(temperature=cfg["alignment"]["temperature"])

    trainer = EncoderTrainer(encoder, proj_head, loss_fn, cfg, device)
    ckpt_dir = str(Path(cfg["encoder"]["checkpoint_path"]).parent)

    print("Starting contrastive training...")
    trainer.fit(train_loader, val_loader, ckpt_dir)
    print("Phase 2 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
