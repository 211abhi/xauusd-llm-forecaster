"""Phase 5 — Full evaluation on test split."""

from __future__ import annotations

import argparse
import yaml
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

from src.prediction.forecaster import Forecaster
from src.tokenizer.patch_tokenizer import batch_to_patches
from src.utils.data_loader import get_feature_columns, build_windows, build_targets
from src.utils.metrics import compute_all_metrics
from src.tokenizer.regime_labeler import assign_window_regimes


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])
    device = torch.device(cfg["project"]["device"])

    processed_dir = Path(cfg["data"]["processed_dir"])
    df = pd.read_csv(processed_dir / "xau_with_regimes.csv", parse_dates=["datetime"])
    feature_cols = get_feature_columns(cfg)

    train_ratio = cfg["data"]["split_ratios"]["train"]
    val_ratio = cfg["data"]["split_ratios"]["val"]
    n = len(df)
    val_end = int(n * (train_ratio + val_ratio))
    test_df = df.iloc[val_end:].reset_index(drop=True)
    test_arr = test_df[feature_cols].values

    window_size = cfg["tokenizer"]["window_size"]
    patch_size = cfg["tokenizer"]["patch_size"]
    val_stride = cfg["tokenizer"]["val_stride"]
    n_steps = cfg["prediction_head"]["output_steps"]
    close_idx = feature_cols.index("close")

    test_windows = build_windows(test_arr, window_size, val_stride)
    test_targets = build_targets(test_arr[:, close_idx], window_size, val_stride, n_steps)
    min_len = min(len(test_windows), len(test_targets))
    test_windows, test_targets = test_windows[:min_len], test_targets[:min_len]

    print(f"Test windows: {test_windows.shape}, targets: {test_targets.shape}")

    print("Loading forecaster...")
    forecaster = Forecaster.from_config(cfg)

    print("Running inference...")
    batch_size = 64
    all_preds = []
    for i in range(0, len(test_windows), batch_size):
        batch = test_windows[i: i + batch_size]
        preds = forecaster.predict(batch)
        all_preds.append(preds)
    all_preds = np.concatenate(all_preds, axis=0)

    last_known = test_windows[:, -1, close_idx]
    metrics = compute_all_metrics(all_preds, test_targets, last_known)
    print("\n=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Per-regime breakdown
    if cfg["evaluation"]["per_regime"]:
        test_regimes = assign_window_regimes(test_df, window_size, val_stride)
        test_regimes = test_regimes[:min_len]
        regime_metrics: dict = defaultdict(lambda: {"preds": [], "targets": [], "last": []})
        for i, regime in enumerate(test_regimes):
            regime_metrics[regime]["preds"].append(all_preds[i])
            regime_metrics[regime]["targets"].append(test_targets[i])
            regime_metrics[regime]["last"].append(last_known[i])

        print("\n=== Per-Regime Metrics ===")
        regime_results = {}
        for regime, data in sorted(regime_metrics.items()):
            p = np.array(data["preds"])
            t = np.array(data["targets"])
            l = np.array(data["last"])
            m = compute_all_metrics(p, t, l)
            regime_results[regime] = m
            print(f"  {regime:15s} | n={len(p):5d} | MAE={m['mae']:.4f} "
                  f"| RMSE={m['rmse']:.4f} | DirAcc={m['directional_accuracy']:.1f}%")

    results = {"overall": metrics}
    if cfg["evaluation"]["per_regime"]:
        results["per_regime"] = regime_results

    out_path = Path(cfg["project"]["log_dir"]) / "evaluation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print("Phase 5 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
