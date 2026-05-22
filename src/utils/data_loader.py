"""OHLCV data loading, feature engineering, normalization, and windowing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import ta


def load_raw_ohlcv(path: str) -> pd.DataFrame:
    """Load raw CSV, parse datetime, sort, and drop duplicates."""
    df = pd.read_csv(path, sep=";", parse_dates=["Date"])
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "datetime"})
    df = df.sort_values("datetime").drop_duplicates(subset="datetime").reset_index(drop=True)
    return df


def forward_fill_gaps(df: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    """Fill missing candles (weekends/holidays) via forward-fill."""
    full_idx = pd.date_range(df["datetime"].iloc[0], df["datetime"].iloc[-1], freq=freq)
    df = df.set_index("datetime").reindex(full_idx, method="ffill").reset_index()
    df = df.rename(columns={"index": "datetime"})
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add returns, log_returns, hl_range, body_ratio."""
    df = df.copy()
    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df["hl_range"] = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()
    df["body_ratio"] = (body / df["hl_range"].replace(0, np.nan)).fillna(0.0)
    return df


def add_indicators(df: pd.DataFrame, atr_period: int = 14, rsi_period: int = 14,
                   ema_fast: int = 20, ema_slow: int = 50) -> pd.DataFrame:
    """Add ATR, RSI, EMA20, EMA50 indicators."""
    df = df.copy()
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"],
        window=atr_period, fillna=True
    ).average_true_range()
    df["rsi"] = ta.momentum.RSIIndicator(
        close=df["close"], window=rsi_period, fillna=True
    ).rsi()
    df["ema20"] = ta.trend.EMAIndicator(
        close=df["close"], window=ema_fast, fillna=True
    ).ema_indicator()
    df["ema50"] = ta.trend.EMAIndicator(
        close=df["close"], window=ema_slow, fillna=True
    ).ema_indicator()
    return df


def compute_train_norm_stats(df: pd.DataFrame, feature_cols: List[str],
                              train_end_idx: int) -> Dict[str, Tuple[float, float]]:
    """Compute min/max per feature on train split only (no lookahead)."""
    stats: Dict[str, Tuple[float, float]] = {}
    train_data = df.iloc[:train_end_idx]
    for col in feature_cols:
        col_min = float(train_data[col].min())
        col_max = float(train_data[col].max())
        stats[col] = (col_min, col_max)
    return stats


def apply_minmax_norm(df: pd.DataFrame, feature_cols: List[str],
                      stats: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Apply train-derived min/max normalization. Values outside train range are clipped."""
    df = df.copy()
    for col in feature_cols:
        col_min, col_max = stats[col]
        rng = col_max - col_min if col_max != col_min else 1.0
        df[col] = (df[col] - col_min) / rng
        df[col] = df[col].clip(0.0, 1.0)
    return df


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.70,
                         val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train/val/test. No shuffling."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def save_split_indices(df: pd.DataFrame, train_ratio: float, val_ratio: float,
                       splits_dir: str) -> None:
    """Save train/val/test index ranges to CSV files."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"start": [0], "end": [train_end]}).to_csv(
        Path(splits_dir) / "train_indices.csv", index=False)
    pd.DataFrame({"start": [train_end], "end": [val_end]}).to_csv(
        Path(splits_dir) / "val_indices.csv", index=False)
    pd.DataFrame({"start": [val_end], "end": [n]}).to_csv(
        Path(splits_dir) / "test_indices.csv", index=False)


def build_windows(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Create sliding windows of shape (N, window_size, n_features)."""
    n_steps = data.shape[0]
    indices = range(0, n_steps - window_size + 1, stride)
    windows = np.stack([data[i: i + window_size] for i in indices], axis=0)
    return windows


def build_targets(close: np.ndarray, window_size: int, stride: int,
                  n_steps: int) -> np.ndarray:
    """Build target arrays: next n_steps close prices after each window."""
    n_total = close.shape[0]
    indices = range(0, n_total - window_size - n_steps + 1, stride)
    targets = np.stack([close[i + window_size: i + window_size + n_steps]
                        for i in indices], axis=0)
    return targets


def get_feature_columns(cfg: dict) -> List[str]:
    """Derive full feature column list from config."""
    base = list(cfg["data"]["feature_cols"])
    derived = list(cfg["data"]["derived_features"])
    indicators = ["atr", "rsi", "ema20", "ema50"]
    # Match n_features=9: open,high,low,close,volume,returns,log_returns,atr,rsi
    return ["open", "high", "low", "close", "volume", "returns", "log_returns", "atr", "rsi"]


def preprocess_pipeline(cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray, np.ndarray,
                                             pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Full preprocessing pipeline. Returns (X_train, y_train, X_val, y_val, X_test, y_test, full_df, norm_stats)."""
    raw_path = cfg["data"]["raw_path"]
    df = load_raw_ohlcv(raw_path)
    df = forward_fill_gaps(df)
    df = add_derived_features(df)
    df = add_indicators(
        df,
        atr_period=cfg["data"]["indicators"]["atr_period"],
        rsi_period=cfg["data"]["indicators"]["rsi_period"],
        ema_fast=cfg["data"]["indicators"]["ema_fast"],
        ema_slow=cfg["data"]["indicators"]["ema_slow"],
    )
    df = df.dropna().reset_index(drop=True)

    feature_cols = get_feature_columns(cfg)
    train_ratio = cfg["data"]["split_ratios"]["train"]
    val_ratio = cfg["data"]["split_ratios"]["val"]
    n = len(df)
    train_end = int(n * train_ratio)

    norm_stats = compute_train_norm_stats(df, feature_cols, train_end)
    df_norm = apply_minmax_norm(df, feature_cols, norm_stats)

    window_size = cfg["tokenizer"]["window_size"]
    n_steps = cfg["prediction_head"]["output_steps"]
    train_stride = cfg["tokenizer"]["train_stride"]
    val_stride = cfg["tokenizer"]["val_stride"]

    val_end = int(n * (train_ratio + val_ratio))
    train_arr = df_norm[feature_cols].values[:train_end]
    val_arr = df_norm[feature_cols].values[train_end:val_end]
    test_arr = df_norm[feature_cols].values[val_end:]

    close_idx = feature_cols.index("close")
    X_train = build_windows(train_arr, window_size, train_stride)
    y_train = build_targets(train_arr[:, close_idx], window_size, train_stride, n_steps)

    # Trim to matching length
    min_len = min(len(X_train), len(y_train))
    X_train, y_train = X_train[:min_len], y_train[:min_len]

    X_val = build_windows(val_arr, window_size, val_stride)
    y_val = build_targets(val_arr[:, close_idx], window_size, val_stride, n_steps)
    min_len = min(len(X_val), len(y_val))
    X_val, y_val = X_val[:min_len], y_val[:min_len]

    X_test = build_windows(test_arr, window_size, val_stride)
    y_test = build_targets(test_arr[:, close_idx], window_size, val_stride, n_steps)
    min_len = min(len(X_test), len(y_test))
    X_test, y_test = X_test[:min_len], y_test[:min_len]

    return X_train, y_train, X_val, y_val, X_test, y_test, df_norm, norm_stats
