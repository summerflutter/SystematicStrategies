"""Momentum-based strategy signals."""

from __future__ import annotations

import pandas as pd

from src.strategy.indicators import ema


def ts_momentum_signal(
    close: pd.Series,
    lookback: int = 24,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Time-series momentum: long if N-bar return > threshold, short if < -threshold.
    """
    mom = close.pct_change(lookback)
    signal = pd.Series(0.0, index=close.index)
    signal[mom > threshold] = 1.0
    signal[mom < -threshold] = -1.0
    return signal


def dual_ma_crossover_signal(
    close: pd.Series,
    fast: int = 20,
    slow: int = 60,
) -> pd.Series:
    """Classic dual moving-average crossover on close."""
    fast_ma = ema(close, fast)
    slow_ma = ema(close, slow)
    signal = pd.Series(0.0, index=close.index)
    signal[fast_ma > slow_ma] = 1.0
    signal[fast_ma < slow_ma] = -1.0
    return signal
