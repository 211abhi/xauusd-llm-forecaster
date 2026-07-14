"""Text templates describing each market regime for contrastive alignment."""

from __future__ import annotations

REGIME_TEMPLATES: dict[str, str] = {
    "TRENDING_UP": (
        "Transformer oil temperature is in a sustained uptrend, running above both "
        "moving averages with rising thermal load."
    ),
    "TRENDING_DOWN": (
        "Transformer oil temperature is in a sustained downtrend, running below both "
        "moving averages as thermal load eases."
    ),
    "RANGING": (
        "Transformer oil temperature is holding steady in a narrow band, "
        "oscillating near the moving average with little net change."
    ),
    "VOLATILE": (
        "Transformer oil temperature is swinging sharply with wide short-term "
        "moves and an elevated average true range."
    ),
    "BREAKOUT": (
        "Transformer oil temperature is breaking out of a recent stable band "
        "alongside a spike in transformer load, signaling a new thermal trend."
    ),
}


def get_template(regime: str) -> str:
    """Return the text description for a given regime label."""
    return REGIME_TEMPLATES[regime]
