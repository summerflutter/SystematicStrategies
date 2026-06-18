"""Single-asset directional backtester with fees, slippage, and vol targeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.strategy.indicators import realized_vol
from src.trading.costs import (
    CostConfig,
    DEFAULT_COSTS,
    build_funding_series_for_symbol,
    compute_directional_funding_cost,
)


@dataclass
class BacktestMetrics:
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    turnover: float
    exposure_pct: float
    n_trades: int
    final_nav: float


def vol_target_position(
    signal: pd.Series,
    returns: pd.Series,
    target_vol: float = 0.15,
    vol_window: int = 20,
    max_leverage: float = 1.0,
    bars_per_year: int = 365 * 24 * 4,
) -> pd.Series:
    """Scale {-1,0,1} signal to vol-targeted exposure."""
    ann_factor = np.sqrt(bars_per_year)
    vol = realized_vol(returns, vol_window) * ann_factor
    weight = (target_vol / vol.replace(0, np.nan)).clip(upper=max_leverage)
    return (signal * weight).fillna(0.0)


def backtest_directional(
    price: pd.Series,
    signal: pd.Series,
    *,
    init_nav: float = 10_000.0,
    trading_fee_bps: float | None = None,
    slippage_bps: float | None = None,
    costs: CostConfig | None = None,
    symbol: str | None = None,
    apply_funding: bool = True,
    target_vol: float | None = 0.15,
    vol_window: int = 20,
    max_leverage: float = 1.0,
    allow_short: bool = True,
    bars_per_year: int = 365 * 24 * 4,
    position_mode: Literal["signal", "vol_target"] = "vol_target",
) -> pd.DataFrame:
    """
    Bar-by-bar backtest for a single asset.

    signal: desired direction in [-1, 0, 1] (or continuous exposure if vol_target).
    Trades execute at bar close; PnL accrues on next bar move.
    """
    df = pd.DataFrame({"price": price.astype(float), "signal": signal.astype(float)}).dropna()
    idx = df.index
    n = len(df)

    returns = df["price"].pct_change().fillna(0.0)
    raw_signal = df["signal"].clip(-1, 1)
    if not allow_short:
        raw_signal = raw_signal.clip(lower=0)

    if position_mode == "vol_target" and target_vol is not None:
        exposure = vol_target_position(
            raw_signal,
            returns,
            target_vol=target_vol,
            vol_window=vol_window,
            max_leverage=max_leverage,
            bars_per_year=bars_per_year,
        )
    else:
        exposure = raw_signal * max_leverage

    cfg = costs or DEFAULT_COSTS
    fee_bps = cfg.trading_fee_bps if trading_fee_bps is None else trading_fee_bps
    slip_bps = cfg.slippage_bps if slippage_bps is None else slippage_bps
    cost_bps = (fee_bps + slip_bps) * cfg.stress_multiplier

    exposure = exposure.shift(1).fillna(0.0)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    trade_cost = turnover * cost_bps / 10_000.0

    gross_pnl = exposure * returns * init_nav
    funding_cost = pd.Series(0.0, index=idx)
    if apply_funding and symbol:
        funding_rate = build_funding_series_for_symbol(symbol, idx)
        nav_for_funding = init_nav + (gross_pnl - trade_cost * init_nav).cumsum()
        funding_cost = compute_directional_funding_cost(
            exposure, df["price"], funding_rate, nav_for_funding
        ) / init_nav

    net_pnl = gross_pnl - trade_cost * init_nav - funding_cost * init_nav
    nav = init_nav + net_pnl.cumsum()

    out = pd.DataFrame(
        {
            "price": df["price"],
            "signal": raw_signal,
            "exposure": exposure,
            "returns": returns,
            "turnover": turnover,
            "trade_cost": trade_cost,
            "funding_cost": funding_cost,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "nav": nav,
        },
        index=idx,
    )
    out["position"] = np.sign(exposure)
    return out


def compute_backtest_metrics(
    bt: pd.DataFrame,
    bars_per_year: int = 365 * 24 * 4,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    """Compute summary statistics from backtest output."""
    nav = bt["nav"]
    rets = nav.pct_change().fillna(0.0)
    ann_factor = np.sqrt(bars_per_year)

    excess = rets - risk_free_rate / bars_per_year
    sharpe = 0.0
    if excess.std() > 0:
        sharpe = float(excess.mean() / excess.std() * ann_factor)

    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = float(drawdown.min())

    trade_mask = bt["turnover"] > 1e-8
    n_trades = int(trade_mask.sum())
    win_rate = 0.0
    if n_trades > 0:
        win_rate = float((bt.loc[trade_mask, "net_pnl"] > 0).mean())

    exposure_pct = float((bt["exposure"].abs() > 1e-8).mean())
    turnover = float(bt["turnover"].sum())

    return BacktestMetrics(
        total_return=float(nav.iloc[-1] / nav.iloc[0] - 1),
        sharpe=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        turnover=turnover,
        exposure_pct=exposure_pct,
        n_trades=n_trades,
        final_nav=float(nav.iloc[-1]),
    )
