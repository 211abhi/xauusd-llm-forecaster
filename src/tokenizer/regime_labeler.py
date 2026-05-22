"""Rule-based market regime labeler for XAUUSD 30m data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List


REGIME_LABELS = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "BREAKOUT"]


def label_regime(row: pd.Series, atr_mean: float, cfg: dict) -> str:
    """
    Assign a single regime label to one row using deterministic rules.

    Priority order: BREAKOUT > VOLATILE > TRENDING_UP > TRENDING_DOWN > RANGING
    """
    rc = cfg["regime_labeling"]

    is_breakout = bool(row.get("breakout", False))
    if is_breakout:
        return "BREAKOUT"

    if row["atr"] > rc["atr_spike_mult"] * atr_mean and \
       row["hl_range"] > rc["hl_range_spike_mult"] * row["atr"]:
        return "VOLATILE"

    if row["close"] > row["ema20"] > row["ema50"] and row["rsi"] > rc["rsi_high"]:
        return "TRENDING_UP"

    if row["close"] < row["ema20"] < row["ema50"] and row["rsi"] < rc["rsi_low"]:
        return "TRENDING_DOWN"

    if abs(row["close"] - row["ema20"]) < rc["ranging_atr_mult"] * row["atr"] and \
       rc["rsi_mid_low"] < row["rsi"] < rc["rsi_mid_high"]:
        return "RANGING"

    # Default fallback based on RSI midpoint
    if row["rsi"] >= 50:
        return "TRENDING_UP"
    return "TRENDING_DOWN"


def detect_breakouts(df: pd.DataFrame, lookback: int = 20,
                     volume_spike_mult: float = 1.5) -> pd.Series:
    """
    Detect breakout candles: close crosses recent high/low with volume spike.
    Returns boolean Series.
    """
    recent_high = df["high"].rolling(lookback).max().shift(1)
    recent_low = df["low"].rolling(lookback).min().shift(1)
    vol_mean = df["volume"].rolling(lookback).mean().shift(1)

    breaks_up = (df["close"] > recent_high)
    breaks_down = (df["close"] < recent_low)
    vol_spike = (df["volume"] > volume_spike_mult * vol_mean)

    return ((breaks_up | breaks_down) & vol_spike).fillna(False)


def label_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add 'regime' column to a fully-featured DataFrame.

    Expects columns: close, ema20, ema50, rsi, atr, hl_range, volume.
    """
    df = df.copy()
    rc = cfg["regime_labeling"]
    atr_mean = float(df["atr"].mean())
    df["breakout"] = detect_breakouts(
        df,
        lookback=rc["breakout_lookback"],
        volume_spike_mult=rc["volume_spike_mult"],
    )
    df["regime"] = df.apply(lambda row: label_regime(row, atr_mean, cfg), axis=1)
    df = df.drop(columns=["breakout"])
    return df


def get_window_regime(regimes: List[str]) -> str:
    """Return the majority regime for a window of regime labels."""
    from collections import Counter
    return Counter(regimes).most_common(1)[0][0]


def assign_window_regimes(df: pd.DataFrame, window_size: int, stride: int) -> List[str]:
    """
    For each sliding window, assign a single regime label (majority vote).

    Returns list of regime strings aligned to windows.
    """
    regime_series = df["regime"].values
    n = len(regime_series)
    window_regimes: List[str] = []
    for start in range(0, n - window_size + 1, stride):
        window_labels = regime_series[start: start + window_size].tolist()
        window_regimes.append(get_window_regime(window_labels))
    return window_regimes
