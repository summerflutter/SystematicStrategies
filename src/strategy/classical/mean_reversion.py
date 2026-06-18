"""Mean-reversion strategy signals."""

from __future__ import annotations

import pandas as pd

from src.strategy.indicators import bollinger_bands, rsi


def bollinger_mr_signal(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """
    Fade Bollinger band extremes; exit toward middle band.
    """
    bb = bollinger_bands(close, period=period, num_std=num_std)
    signal = pd.Series(0.0, index=close.index)
    position = 0.0

    for i, t in enumerate(close.index):
        c = close.iloc[i]
        upper = bb["bb_upper"].iloc[i]
        lower = bb["bb_lower"].iloc[i]
        mid = bb["bb_middle"].iloc[i]

        if pd.isna(upper) or pd.isna(lower):
            signal.iloc[i] = 0.0
            continue

        if position == 0:
            if c <= lower:
                position = 1.0
            elif c >= upper:
                position = -1.0
        elif position > 0 and c >= mid:
            position = 0.0
        elif position < 0 and c <= mid:
            position = 0.0

        signal.iloc[i] = position

    return signal


def rsi_mr_signal(
    close: pd.Series,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    exit_level: float = 50.0,
) -> pd.Series:
    """RSI mean reversion with mid-line exit."""
    rsi_vals = rsi(close, period=period)
    signal = pd.Series(0.0, index=close.index)
    position = 0.0

    for i, t in enumerate(close.index):
        r = rsi_vals.iloc[i]
        if pd.isna(r):
            signal.iloc[i] = 0.0
            continue

        if position == 0:
            if r <= oversold:
                position = 1.0
            elif r >= overbought:
                position = -1.0
        elif position > 0 and r >= exit_level:
            position = 0.0
        elif position < 0 and r <= exit_level:
            position = 0.0

        signal.iloc[i] = position

    return signal
