"""Trade log extraction and backtest result packaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.backtest.directional import (
    BacktestMetrics,
    backtest_directional,
    compute_backtest_metrics,
)


@dataclass
class BacktestRunResult:
    """Bar-level simulation plus trade log and summary metrics."""

    bars: pd.DataFrame
    trades: pd.DataFrame
    pnl: pd.Series
    nav: pd.Series
    metrics: BacktestMetrics


def _exposure_direction(exposure: float, min_exposure: float) -> int:
    if exposure > min_exposure:
        return 1
    if exposure < -min_exposure:
        return -1
    return 0


def _leg_pnl_components(leg: pd.DataFrame, init_nav: float) -> tuple[float, float, float, float]:
    gross = float(leg["gross_pnl"].sum())
    net = float(leg["net_pnl"].sum())
    fees = float((leg["trade_cost"] * init_nav).sum())
    funding = float((leg["funding_cost"] * init_nav).sum()) if "funding_cost" in leg.columns else 0.0
    return gross, fees, funding, net


def extract_trade_log(
    bars: pd.DataFrame,
    *,
    min_exposure: float = 1e-8,
) -> pd.DataFrame:
    """
    Build a position-leg trade log from bar-level backtest output.

    Each row is one continuous long or short leg (exposure sign held constant
    and non-zero). Rebalances within a leg are aggregated.

    Expected columns on ``bars``: price, exposure, gross_pnl, net_pnl, trade_cost.
    """
    required = {"price", "exposure", "gross_pnl", "net_pnl", "trade_cost"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")

    init_nav = float(bars["nav"].iloc[0]) if "nav" in bars.columns else 1.0
    rows: list[dict] = []
    trade_id = 0

    leg_start_idx: int | None = None
    leg_direction = 0

    def _append_leg(start_idx: int, end_idx: int, direction: int) -> None:
        nonlocal trade_id
        if start_idx > end_idx or direction == 0:
            return
        leg = bars.iloc[start_idx : end_idx + 1]
        gross, fees, funding, net = _leg_pnl_components(leg, init_nav)
        trade_id += 1
        rows.append(
            {
                "trade_id": trade_id,
                "entry_time": leg.index[0],
                "exit_time": leg.index[-1],
                "direction": "long" if direction > 0 else "short",
                "entry_price": float(leg["price"].iloc[0]),
                "exit_price": float(leg["price"].iloc[-1]),
                "avg_exposure": float(leg["exposure"].mean()),
                "max_abs_exposure": float(leg["exposure"].abs().max()),
                "holding_bars": len(leg),
                "gross_pnl": gross,
                "fees": fees,
                "funding": funding,
                "net_pnl": net,
            }
        )

    for i in range(len(bars)):
        direction = _exposure_direction(float(bars["exposure"].iloc[i]), min_exposure)

        if leg_direction == 0:
            if direction != 0:
                leg_start_idx = i
                leg_direction = direction
            continue

        if direction == leg_direction:
            continue

        # Flat or flip: close existing leg through prior bar
        _append_leg(leg_start_idx, i - 1, leg_direction)
        leg_start_idx = None
        leg_direction = 0

        if direction != 0:
            leg_start_idx = i
            leg_direction = direction

    if leg_direction != 0 and leg_start_idx is not None:
        _append_leg(leg_start_idx, len(bars) - 1, leg_direction)

    columns = [
        "trade_id",
        "entry_time",
        "exit_time",
        "direction",
        "entry_price",
        "exit_price",
        "avg_exposure",
        "max_abs_exposure",
        "holding_bars",
        "gross_pnl",
        "fees",
        "funding",
        "net_pnl",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def extract_rebalance_log(
    bars: pd.DataFrame,
    *,
    min_turnover: float = 1e-8,
) -> pd.DataFrame:
    """
    One row per bar where exposure is adjusted (turnover > threshold).

    Useful for vol-targeted strategies that rebalance frequently.
    """
    if "turnover" not in bars.columns:
        raise ValueError("bars must include turnover column (from backtest_directional)")

    init_nav = float(bars["nav"].iloc[0]) if "nav" in bars.columns else 1.0
    mask = bars["turnover"] > min_turnover
    reb = bars.loc[mask]

    rows = []
    for ts, row in reb.iterrows():
        rows.append(
            {
                "time": ts,
                "price": float(row["price"]),
                "exposure": float(row["exposure"]),
                "turnover": float(row["turnover"]),
                "fee": float(row["trade_cost"] * init_nav),
                "funding": float(row.get("funding_cost", 0.0) * init_nav),
                "gross_pnl": float(row["gross_pnl"]),
                "net_pnl": float(row["net_pnl"]),
                "nav": float(row["nav"]) if "nav" in row.index else np.nan,
            }
        )

    return pd.DataFrame(rows)


def run_backtest(
    price: pd.Series,
    signal: pd.Series,
    *,
    symbol: str | None = None,
    trade_log_mode: Literal["legs", "rebalance"] = "legs",
    bars_per_year: int = 365 * 24 * 4,
    **backtest_kwargs,
) -> BacktestRunResult:
    """
    Run a directional backtest and return bar data, trade log, PnL, and metrics.

    Parameters
    ----------
    price : close price series indexed by time
    signal : strategy signal in [-1, 1] (same index as price)
    symbol : required for funding-aware simulation when apply_funding=True
    trade_log_mode :
        ``legs`` — aggregate into long/short holding periods (default)
        ``rebalance`` — one row per exposure change
    **backtest_kwargs : passed to backtest_directional
    """
    bars = backtest_directional(
        price,
        signal,
        symbol=symbol,
        bars_per_year=bars_per_year,
        **backtest_kwargs,
    )

    trades = (
        extract_rebalance_log(bars)
        if trade_log_mode == "rebalance"
        else extract_trade_log(bars)
    )
    metrics = compute_backtest_metrics(bars, bars_per_year=bars_per_year)

    return BacktestRunResult(
        bars=bars,
        trades=trades,
        pnl=bars["net_pnl"],
        nav=bars["nav"],
        metrics=metrics,
    )
