"""Shared indicators and OHLCV utilities for classical strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np


def prepare_ohlcv(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Extract a single-symbol OHLCV series indexed by open_time."""
    out = df.copy()
    if symbol is not None and "symbol" in out.columns:
        out = out[out["symbol"] == symbol]
    if "open_time" in out.columns:
        out = out.set_index("open_time")
    out = out.sort_index()
    return out[["open", "high", "low", "close", "volume"]].astype(float)


def interval_to_pandas_freq(interval: str) -> str:
    """Convert Binance-style interval to pandas resample frequency."""
    unit = interval[-1]
    value = interval[:-1]
    if unit == "m":
        return f"{value}min"
    if unit == "h":
        return f"{value}h"
    if unit == "d":
        return f"{value}D"
    if unit == "w":
        return f"{value}W"
    raise ValueError(f"Unsupported interval: {interval}")


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample 1m (or any) OHLCV to a higher timeframe."""
    freq = interval_to_pandas_freq(interval)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(freq).agg(agg).dropna()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"bb_upper": upper, "bb_middle": mid, "bb_lower": lower})


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window=window).std()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window=period).mean()
