"""Unified trading cost configuration for backtests and paper-trading reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.data.data_download import ensure_timestamp

DEFAULT_BASE_PATH = Path(__file__).resolve().parents[2] / "data"

# Binance USDT-M perp funding settles every 8 hours.
FUNDING_INTERVAL_HOURS = 8


@dataclass(frozen=True)
class CostConfig:
    """Shared fee, slippage, and funding assumptions."""

    trading_fee_bps: float = 5.0
    slippage_bps: float = 2.0
    stress_multiplier: float = 1.0

    @property
    def directional_cost_bps(self) -> float:
        """All-in per-turnover cost for single-asset directional strategies."""
        return (self.trading_fee_bps + self.slippage_bps) * self.stress_multiplier

    @property
    def stat_arb_fee_bps(self) -> float:
        """Taker fee per leg (entry or exit)."""
        return self.trading_fee_bps * self.stress_multiplier

    @property
    def stat_arb_slippage_bps(self) -> float:
        """Slippage per leg (entry or exit)."""
        return self.slippage_bps * self.stress_multiplier

    @property
    def stat_arb_leg_cost_bps(self) -> float:
        """Fee + slippage per leg."""
        return self.stat_arb_fee_bps + self.stat_arb_slippage_bps

    def with_stress(self, multiplier: float = 1.5) -> "CostConfig":
        return CostConfig(
            trading_fee_bps=self.trading_fee_bps,
            slippage_bps=self.slippage_bps,
            stress_multiplier=multiplier,
        )


DEFAULT_COSTS = CostConfig()
STRESS_COSTS = CostConfig().with_stress(1.5)


def _load_funding_table(
    symbols: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_path: Path = DEFAULT_BASE_PATH,
) -> pd.DataFrame:
    """Load funding rates from parquet; return empty frame if unavailable."""
    try:
        from src.data.funding_load import load_funding_rates

        return load_funding_rates(symbols, start, end, base_path=base_path)
    except Exception:
        return pd.DataFrame(columns=["symbol", "funding_time", "funding_rate"])


def build_funding_series_for_symbol(
    symbol: str,
    bar_index: pd.DatetimeIndex,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    base_path: Path = DEFAULT_BASE_PATH,
) -> pd.Series:
    """
    Map 8h funding events onto bar timestamps.

    Returns signed funding rate applied at each bar (0 except at funding timestamps).
    Positive rate => longs pay shorts on Binance perps.
    """
    if len(bar_index) == 0:
        return pd.Series(dtype=float)

    if start is None:
        start = bar_index[0]
    if end is None:
        end = bar_index[-1]

    start_ts = ensure_timestamp(start)
    end_ts = ensure_timestamp(end)

    df = _load_funding_table([symbol], start_ts, end_ts, base_path)
    out = pd.Series(0.0, index=bar_index, dtype=float)
    if df.empty:
        return out

    sym_df = df[df["symbol"] == symbol].copy()
    if sym_df.empty:
        return out

    sym_df = sym_df.sort_values("funding_time")
    idx = pd.DatetimeIndex(bar_index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    funding_times = pd.DatetimeIndex(sym_df["funding_time"]).tz_convert("UTC")
    rates = sym_df["funding_rate"].astype(float).values

    # Apply each funding event to the first bar at or after funding_time.
    pos = idx.searchsorted(funding_times, side="left")
    for i, bar_pos in enumerate(pos):
        if bar_pos < len(idx):
            out.iloc[bar_pos] += rates[i]

    return out


def compute_directional_funding_cost(
    exposure: pd.Series,
    price: pd.Series,
    funding_rate: pd.Series,
    nav: pd.Series | None = None,
) -> pd.Series:
    """
    Funding PnL drag on a directional perp position.

    exposure: signed fraction of NAV (positive = long).
    funding_rate: 8h rate applied at funding timestamps.
    """
    if nav is None:
        nav = pd.Series(1.0, index=exposure.index)

    notional = exposure.abs() * nav
    # Longs pay when rate > 0; shorts receive.
    signed_notional = exposure * notional
    return signed_notional * funding_rate


def compute_leg_funding_cost(
    units: pd.Series,
    price: pd.Series,
    funding_rate: pd.Series,
) -> pd.Series:
    """
    Funding PnL for one leg of a spread.

    units: signed position size (positive = long).
    Positive funding rate => longs pay.
    """
    notional = units * price
    return notional * funding_rate
