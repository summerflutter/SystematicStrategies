"""Diagnostics for whether a price pair forms a tradeable, mean-reverting spread.

These functions answer the prerequisite question before any threshold tuning:
is the BTC/ETH spread actually mean-reverting on this horizon, and is that
property stable through time? Full-sample stats use look-ahead and are for
*diagnosis only* - never for trading decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from src.strategy.stat_arb.engine import compute_beta_spread, estimate_half_life


@dataclass
class SpreadDiagnostics:
    n_obs: int
    beta: float
    coint_pvalue: float
    adf_pvalue: float
    half_life: Optional[float]
    hurst: Optional[float]
    spread_mean: float
    spread_std: float
    z_min: float
    z_max: float
    pct_beyond_2z: float

    def verdict(self) -> str:
        """Plain-language read on tradeable mean reversion."""
        reasons = []
        mean_reverting = (
            self.adf_pvalue < 0.05
            and self.half_life is not None
            and (self.hurst is None or self.hurst < 0.5)
        )
        if self.adf_pvalue >= 0.05:
            reasons.append(f"ADF p={self.adf_pvalue:.2f} (>=0.05, cannot reject unit root)")
        if self.half_life is None:
            reasons.append("half-life undefined (no mean reversion)")
        elif self.half_life > self.n_obs / 2:
            reasons.append(f"half-life {self.half_life:.0f} bars too slow vs sample")
        if self.hurst is not None and self.hurst >= 0.5:
            reasons.append(f"Hurst {self.hurst:.2f} (>=0.5, trending/random walk)")
        if mean_reverting and not reasons:
            return "Mean-reverting and tradeable on this sample."
        if mean_reverting:
            return "Borderline mean-reverting: " + "; ".join(reasons)
        return "NOT reliably mean-reverting: " + "; ".join(reasons)


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> Optional[float]:
    """Estimate the Hurst exponent via the rescaled-variance (lag) method.

    H < 0.5 => mean-reverting, H ~ 0.5 => random walk, H > 0.5 => trending.
    """
    s = pd.Series(series).dropna().values
    if len(s) < max_lag + 5:
        return None
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        std = np.std(diff)
        if std <= 0:
            return None
        tau.append(std)
    log_lags = np.log(list(lags))
    log_tau = np.log(tau)
    slope = np.polyfit(log_lags, log_tau, 1)[0]
    return float(slope)


def spread_diagnostics(y: pd.Series, x: pd.Series) -> SpreadDiagnostics:
    """Full-sample mean-reversion diagnostics for the spread y - beta*x."""
    aligned = pd.concat([y, x], axis=1).dropna()
    y_c = aligned.iloc[:, 0]
    x_c = aligned.iloc[:, 1]

    beta, spread = compute_beta_spread(y_c, x_c)
    spread = spread.dropna()

    _, coint_p, _ = coint(y_c, x_c)
    adf_p = adfuller(spread, autolag="AIC")[1]
    hl = estimate_half_life(spread)
    hurst = hurst_exponent(spread)

    s_mean = float(spread.mean())
    s_std = float(spread.std())
    z = (spread - s_mean) / s_std if s_std > 0 else spread * 0.0

    return SpreadDiagnostics(
        n_obs=int(len(spread)),
        beta=float(beta),
        coint_pvalue=float(coint_p),
        adf_pvalue=float(adf_p),
        half_life=hl,
        hurst=hurst,
        spread_mean=s_mean,
        spread_std=s_std,
        z_min=float(z.min()),
        z_max=float(z.max()),
        pct_beyond_2z=float((z.abs() >= 2.0).mean()),
    )


def reversion_profile(
    y: pd.Series,
    x: pd.Series,
    *,
    z_entry: float = 2.0,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
    window: Optional[int] = None,
) -> pd.DataFrame:
    """Conditional forward reversion given an extreme spread.

    For every bar where |z| >= z_entry, measure how far the z-score moves back
    toward zero over each forward horizon. ``mean_reversion`` > 0 means the
    spread reverted (the core property the strategy depends on); < 0 means it
    kept diverging (trending against the trade).

    When ``window`` is given, the z-score is normalised on a trailing rolling
    window (matching what the live strategy sees); otherwise full-sample
    mean/std are used.
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_c = aligned.iloc[:, 0]
    x_c = aligned.iloc[:, 1]
    _, spread = compute_beta_spread(y_c, x_c)
    spread = spread.dropna()
    if window is not None and window > 1:
        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std()
        z = ((spread - roll_mean) / roll_std).dropna()
    else:
        s_mean = spread.mean()
        s_std = spread.std()
        if s_std <= 0:
            return pd.DataFrame()
        z = (spread - s_mean) / s_std
    z_vals = z.values
    n = len(z_vals)

    extreme_idx = np.where(np.abs(z_vals) >= z_entry)[0]

    rows = []
    for h in horizons:
        moves = []
        for i in extreme_idx:
            j = i + h
            if j >= n:
                continue
            # reversion toward zero is a drop in |z|; sign so positive == reverts
            moves.append(np.abs(z_vals[i]) - np.abs(z_vals[j]))
        moves = np.asarray(moves, dtype=float)
        if moves.size == 0:
            continue
        rows.append(
            {
                "horizon_bars": h,
                "n_signals": int(moves.size),
                "mean_reversion": float(moves.mean()),
                "pct_reverted": float((moves > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def rolling_diagnostics(
    y: pd.Series,
    x: pd.Series,
    *,
    window_bars: int,
    step_bars: int = 1,
) -> pd.DataFrame:
    """Rolling beta, cointegration p-value, ADF p-value and half-life.

    Each row uses only the trailing ``window_bars`` ending at that timestamp,
    so it shows how stable the spread relationship is through time.
    """
    aligned = pd.concat([y, x], axis=1).dropna()
    y_c = aligned.iloc[:, 0]
    x_c = aligned.iloc[:, 1]
    idx = aligned.index
    n = len(idx)

    rows = []
    for end in range(window_bars, n + 1, step_bars):
        yw = y_c.iloc[end - window_bars : end]
        xw = x_c.iloc[end - window_bars : end]
        beta, spread = compute_beta_spread(yw, xw)
        spread = spread.dropna()
        try:
            _, coint_p, _ = coint(yw, xw)
        except Exception:
            coint_p = np.nan
        try:
            adf_p = adfuller(spread, autolag="AIC")[1]
        except Exception:
            adf_p = np.nan
        rows.append(
            {
                "end_time": idx[end - 1],
                "beta": float(beta),
                "coint_pvalue": float(coint_p),
                "adf_pvalue": float(adf_p),
                "half_life": estimate_half_life(spread),
            }
        )

    return pd.DataFrame(rows).set_index("end_time")
