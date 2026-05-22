"""Phase 1 — Data preparation: clean, normalize, and split OHLCV data."""

from __future__ import annotations

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.data_loader import (
    load_raw_ohlcv, forward_fill_gaps, add_derived_features, add_indicators,
    compute_train_norm_stats, apply_minmax_norm, chronological_split,
    save_split_indices, get_feature_columns,
)


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["project"]["seed"])

    print("Loading raw data...")
    df = load_raw_ohlcv(cfg["data"]["raw_path"])
    print(f"  {len(df)} rows loaded, date range: {df['datetime'].iloc[0]} → {df['datetime'].iloc[-1]}")

    print("Forward-filling gaps...")
    df = forward_fill_gaps(df)
    print(f"  {len(df)} rows after gap fill")

    print("Adding derived features...")
    df = add_derived_features(df)

    print("Adding technical indicators...")
    df = add_indicators(
        df,
        atr_period=cfg["data"]["indicators"]["atr_period"],
        rsi_period=cfg["data"]["indicators"]["rsi_period"],
        ema_fast=cfg["data"]["indicators"]["ema_fast"],
        ema_slow=cfg["data"]["indicators"]["ema_slow"],
    )
    df = df.dropna().reset_index(drop=True)
    print(f"  {len(df)} rows after indicator warmup drop")

    feature_cols = get_feature_columns(cfg)
    train_ratio = cfg["data"]["split_ratios"]["train"]
    val_ratio = cfg["data"]["split_ratios"]["val"]
    train_end = int(len(df) * train_ratio)

    print("Computing normalization stats on train split only...")
    norm_stats = compute_train_norm_stats(df, feature_cols, train_end)

    print("Applying normalization...")
    df_norm = apply_minmax_norm(df, feature_cols, norm_stats)

    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_norm.to_csv(processed_dir / "xau_processed.csv", index=False)
    print(f"Saved processed data → {processed_dir / 'xau_processed.csv'}")

    # Save norm stats
    import json
    stats_path = processed_dir / "norm_stats.json"
    serializable = {k: list(v) for k, v in norm_stats.items()}
    with open(stats_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved normalization stats → {stats_path}")

    print("Saving split indices...")
    save_split_indices(df_norm, train_ratio, val_ratio, cfg["data"]["splits_dir"])
    print(f"Saved split indices → {cfg['data']['splits_dir']}")

    train_df, val_df, test_df = chronological_split(df_norm, train_ratio, val_ratio)
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("Phase 1 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
