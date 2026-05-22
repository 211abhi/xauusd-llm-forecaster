"""Evaluation metrics: MAE, RMSE, directional accuracy."""

from __future__ import annotations

import numpy as np


def mae(preds: np.ndarray, targets: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(preds - targets)))


def rmse(preds: np.ndarray, targets: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def directional_accuracy(preds: np.ndarray, targets: np.ndarray,
                          last_known: np.ndarray) -> float:
    """
    Percentage of predictions with correct direction vs last known price.

    Args:
        preds:      (N, steps) predicted prices
        targets:    (N, steps) actual future prices
        last_known: (N,) last price in each input window
    """
    last = last_known[:, None]
    pred_dir = np.sign(preds - last)
    true_dir = np.sign(targets - last)
    correct = (pred_dir == true_dir).mean()
    return float(correct) * 100.0


def compute_all_metrics(preds: np.ndarray, targets: np.ndarray,
                         last_known: np.ndarray) -> dict:
    """Return dict with mae, rmse, directional_accuracy."""
    return {
        "mae": mae(preds, targets),
        "rmse": rmse(preds, targets),
        "directional_accuracy": directional_accuracy(preds, targets, last_known),
    }
