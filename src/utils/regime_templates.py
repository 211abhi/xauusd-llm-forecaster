"""Text templates describing each market regime for contrastive alignment."""

from __future__ import annotations

REGIME_TEMPLATES: dict[str, str] = {
    "TRENDING_UP": (
        "Gold price is in a strong uptrend, trading above both moving averages "
        "with bullish momentum and rising price action."
    ),
    "TRENDING_DOWN": (
        "Gold price is in a downtrend, below key moving averages "
        "with bearish pressure and declining price action."
    ),
    "RANGING": (
        "Gold price is consolidating in a sideways range, "
        "oscillating near the moving average with neutral momentum."
    ),
    "VOLATILE": (
        "Gold price is experiencing high volatility with wide candles, "
        "large price swings, and elevated average true range."
    ),
    "BREAKOUT": (
        "Gold price is breaking out of a recent consolidation zone "
        "on elevated volume, signaling a potential new directional move."
    ),
}


def get_template(regime: str) -> str:
    """Return the text description for a given regime label."""
    return REGIME_TEMPLATES[regime]
