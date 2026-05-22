"""Phase 3 — CMA-ES soft prompt optimization."""

from __future__ import annotations

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.encoder.ts_encoder import TSEncoder
from src.encoder.encoder_trainer import ProjectionHead
from src.llm.frozen_llm import FrozenLLM
from src.soft_prompt.soft_prompt import SoftPrompt
from src.soft_prompt.cmaes_optimizer import CMAESOptimizer
from src.prediction.pred_head import PredictionHead
from src.tokenizer.patch_tokenizer import batch_to_patches
from src.utils.data_loader import get_feature_columns, build_windows, build_targets
from src.utils.checkpoint import load_encoder, save_pred_head


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

    val_arr = df[feature_cols].values[train_end:val_end]
    window_size = cfg["tokenizer"]["window_size"]
    patch_size = cfg["tokenizer"]["patch_size"]
    val_stride = cfg["tokenizer"]["val_stride"]
    n_steps = cfg["prediction_head"]["output_steps"]
    close_idx = feature_cols.index("close")

    val_windows = build_windows(val_arr, window_size, val_stride)
    val_targets = build_targets(val_arr[:, close_idx], window_size, val_stride, n_steps)
    min_len = min(len(val_windows), len(val_targets))
    val_windows, val_targets = val_windows[:min_len], val_targets[:min_len]

    val_patches = batch_to_patches(val_windows, patch_size).astype(np.float32)
    val_dataset = TensorDataset(
        torch.tensor(val_patches, dtype=torch.float32),
        torch.tensor(val_targets, dtype=torch.float32),
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["cmaes"]["eval_batch_size"],
                            shuffle=False, num_workers=0)

    print("Loading frozen encoder...")
    encoder = TSEncoder.from_config(cfg)
    proj_head = ProjectionHead(cfg["encoder"]["output_dim"], cfg["alignment"]["projection_dim"])
    load_encoder(encoder, proj_head, cfg["encoder"]["checkpoint_path"], device)
    encoder.to(device).eval()
    proj_head.to(device).eval()

    print("Loading frozen LLM...")
    llm = FrozenLLM.from_config(cfg)

    print("Initializing soft prompt and prediction head...")
    soft_prompt = SoftPrompt.from_config(cfg).to(device)
    pred_head = PredictionHead.from_config(cfg).to(device)

    optimizer = CMAESOptimizer(soft_prompt, encoder, llm, pred_head, proj_head, cfg, device)

    ckpt_path = cfg["soft_prompt"]["checkpoint_path"]
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    print("Starting CMA-ES optimization...")
    best_prompt = optimizer.optimize(val_loader, ckpt_path)
    print(f"Optimization complete. Best prompt saved → {ckpt_path}")
    print("Phase 3 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
