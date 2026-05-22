"""Phase 1b — Label market regimes on processed data."""

from __future__ import annotations

import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from src.tokenizer.regime_labeler import label_dataframe, assign_window_regimes


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["project"]["seed"])

    processed_dir = Path(cfg["data"]["processed_dir"])
    df = pd.read_csv(processed_dir / "xau_processed.csv", parse_dates=["datetime"])
    print(f"Loaded {len(df)} rows from processed data")

    print("Labeling market regimes...")
    df = label_dataframe(df, cfg)
    counts = Counter(df["regime"])
    print("  Regime distribution (per-candle):")
    for regime, cnt in sorted(counts.items()):
        print(f"    {regime:15s}: {cnt:6d} ({cnt / len(df) * 100:.1f}%)")

    df.to_csv(processed_dir / "xau_with_regimes.csv", index=False)
    print(f"Saved → {processed_dir / 'xau_with_regimes.csv'}")

    # Assign window-level regimes for train/val splits
    train_ratio = cfg["data"]["split_ratios"]["train"]
    val_ratio = cfg["data"]["split_ratios"]["val"]
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    window_size = cfg["tokenizer"]["window_size"]
    train_stride = cfg["tokenizer"]["train_stride"]
    val_stride = cfg["tokenizer"]["val_stride"]

    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df, stride in [
        ("train", df.iloc[:train_end], train_stride),
        ("val",   df.iloc[train_end:val_end], val_stride),
        ("test",  df.iloc[val_end:], val_stride),
    ]:
        window_regimes = assign_window_regimes(split_df.reset_index(drop=True), window_size, stride)
        regime_counts = Counter(window_regimes)
        print(f"\n  {split_name} window regimes ({len(window_regimes)} windows):")
        for r, c in sorted(regime_counts.items()):
            print(f"    {r:15s}: {c}")
        pd.Series(window_regimes, name="regime").to_csv(
            splits_dir / f"{split_name}_regimes.csv", index=False
        )

    print("\nPhase 1b complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
