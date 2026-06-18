"""Regime detection and regime-switching signal combiner."""

from __future__ import annotations

import pandas as pd

from src.strategy.indicators import compute_adx


def classify_regime(
    ohlcv: pd.DataFrame,
    adx_period: int = 14,
    adx_trend_threshold: float = 25.0,
) -> pd.Series:
    """
    Return regime label per bar: 'trend' or 'range'.
    """
    adx = compute_adx(ohlcv, period=adx_period)
    regime = pd.Series("range", index=ohlcv.index)
    regime[adx >= adx_trend_threshold] = "trend"
    return regime


def regime_switch_signal(
    ohlcv: pd.DataFrame,
    momentum_signal: pd.Series,
    mr_signal: pd.Series,
    adx_period: int = 14,
    adx_trend_threshold: float = 25.0,
) -> pd.DataFrame:
    """
    Use momentum in trending regimes, mean reversion in range-bound regimes.
    """
    regime = classify_regime(ohlcv, adx_period, adx_trend_threshold)
    adx = compute_adx(ohlcv, period=adx_period)

    combined = pd.Series(0.0, index=ohlcv.index)
    trend_mask = regime == "trend"
    combined[trend_mask] = momentum_signal.reindex(ohlcv.index).fillna(0.0)[trend_mask]
    combined[~trend_mask] = mr_signal.reindex(ohlcv.index).fillna(0.0)[~trend_mask]

    return pd.DataFrame(
        {
            "regime": regime,
            "adx": adx,
            "momentum_signal": momentum_signal,
            "mr_signal": mr_signal,
            "combined_signal": combined,
        },
        index=ohlcv.index,
    )
