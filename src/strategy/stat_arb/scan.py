"""Rolling cointegration pair scan for stat-arb."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from src.strategy.stat_arb.engine import (
    backtest_spread_signal,
    compute_beta_spread,
    estimate_half_life,
    generate_spread_signal,
)
from src.trading.costs import CostConfig, DEFAULT_COSTS, build_funding_series_for_symbol


@dataclass
class PairScanResult:
    pair: Tuple[str, str]
    interval: str
    window_bars: int
    coint_pvalue: float
    adf_pvalue: float
    hedge_ratio: float
    half_life: Optional[float]
    backtest_sharpe: float
    backtest_return: float
    backtest_max_dd: float
    n_bars: int


def rolling_cointegration_scan(
    prices: Dict[str, pd.Series],
    pair: Tuple[str, str],
    window_bars: int = 60 * 24 * 4,
    step_bars: int = 60 * 24,
    coint_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Rolling Engle-Granger cointegration test across a price pair.

    prices: dict of symbol -> close price series (aligned index).
    Returns DataFrame with rolling stats per window end timestamp.
    """
    y_sym, x_sym = pair
    y = prices[y_sym]
    x = prices[x_sym]
    aligned = pd.DataFrame({y_sym: y, x_sym: x}).dropna()

    rows = []
    for end in range(window_bars, len(aligned), step_bars):
        window = aligned.iloc[end - window_bars : end]
        y_w = window[y_sym]
        x_w = window[x_sym]
        if len(window) < 30:
            continue

        score, pvalue, _ = coint(y_w, x_w)
        beta, spread = compute_beta_spread(y_w, x_w)
        from statsmodels.tsa.stattools import adfuller

        adf_p = adfuller(spread.dropna(), autolag="AIC")[1]
        hl = estimate_half_life(spread)

        rows.append(
            {
                "end_time": window.index[-1],
                "coint_pvalue": pvalue,
                "adf_pvalue": adf_p,
                "hedge_ratio": beta,
                "half_life": hl,
                "is_cointegrated": pvalue < coint_threshold and adf_p < coint_threshold,
            }
        )

    return pd.DataFrame(rows)


def scan_all_pairs(
    prices: Dict[str, pd.Series],
    symbols: List[str],
    interval: str,
    window_bars: int = 60 * 24 * 4,
    entry_z: float = 2.0,
    exit_z_tp: float = 0.5,
    exit_z_sl: float = 3.0,
    trading_fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    costs: CostConfig | None = None,
    bars_per_year: int = 365 * 24 * 4,
) -> List[PairScanResult]:
    """Scan all pair combinations and backtest cointegrated windows."""
    cfg = costs or CostConfig(
        trading_fee_bps=trading_fee_bps,
        slippage_bps=slippage_bps,
    )
    results: List[PairScanResult] = []

    for y_sym, x_sym in combinations(symbols, 2):
        y = prices[y_sym]
        x = prices[x_sym]
        aligned = pd.DataFrame({y_sym: y, x_sym: x}).dropna()
        if len(aligned) < window_bars:
            continue

        y_clean = aligned[y_sym]
        x_clean = aligned[x_sym]

        score, pvalue, _ = coint(y_clean.iloc[-window_bars:], x_clean.iloc[-window_bars:])
        beta, spread = compute_beta_spread(y_clean.iloc[-window_bars:], x_clean.iloc[-window_bars:])
        from statsmodels.tsa.stattools import adfuller

        adf_p = adfuller(spread.dropna(), autolag="AIC")[1]
        hl = estimate_half_life(spread)

        signal = generate_spread_signal(
            y_clean, x_clean, entry_z=entry_z, exit_z_tp=exit_z_tp, exit_z_sl=exit_z_sl
        )
        funding_X = build_funding_series_for_symbol(x_sym, aligned.index)
        funding_Y = build_funding_series_for_symbol(y_sym, aligned.index)
        bt = backtest_spread_signal(
            y_clean,
            x_clean,
            signal,
            trading_fee_bps=cfg.stat_arb_fee_bps,
            slippage_bps=cfg.stat_arb_slippage_bps,
            funding_series_X=funding_X,
            funding_series_Y=funding_Y,
        )

        nav = bt["nav"]
        rets = nav.pct_change().fillna(0.0)
        sharpe = 0.0
        if rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year))

        running_max = nav.cummax()
        max_dd = float(((nav - running_max) / running_max).min())

        results.append(
            PairScanResult(
                pair=(y_sym, x_sym),
                interval=interval,
                window_bars=window_bars,
                coint_pvalue=float(pvalue),
                adf_pvalue=float(adf_p),
                hedge_ratio=float(beta),
                half_life=hl,
                backtest_sharpe=sharpe,
                backtest_return=float(nav.iloc[-1] / nav.iloc[0] - 1),
                backtest_max_dd=max_dd,
                n_bars=len(bt),
            )
        )

    return results


def pair_scan_to_dataframe(results: List[PairScanResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "pair": f"{r.pair[0]}/{r.pair[1]}",
                "interval": r.interval,
                "window_bars": r.window_bars,
                "coint_pvalue": r.coint_pvalue,
                "adf_pvalue": r.adf_pvalue,
                "hedge_ratio": r.hedge_ratio,
                "half_life": r.half_life,
                "backtest_sharpe": r.backtest_sharpe,
                "backtest_return": r.backtest_return,
                "backtest_max_dd": r.backtest_max_dd,
                "n_bars": r.n_bars,
                "passed_gate": r.backtest_sharpe > 0.3,
            }
        )
    return pd.DataFrame(rows)
