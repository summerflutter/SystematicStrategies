"""Walk-forward out-of-sample evaluation for statistical arbitrage pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from src.backtest.directional import BacktestMetrics
from src.backtest.evaluation import bars_per_year_for_interval
from src.strategy.stat_arb.engine import compute_beta_spread
from src.trading.costs import (
    CostConfig,
    DEFAULT_COSTS,
    build_funding_series_for_symbol,
)


@dataclass
class StatArbWalkForwardResult:
    pair: Tuple[str, str]
    interval: str
    params: Dict[str, Any]
    train_metrics: BacktestMetrics
    val_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    stress_test_metrics: BacktestMetrics
    passed_gate: bool
    gate_reason: str
    train_bt: pd.DataFrame = field(repr=False)
    val_bt: pd.DataFrame = field(repr=False)
    test_bt: pd.DataFrame = field(repr=False)


def _spread_stats(y: pd.Series, x: pd.Series) -> tuple[float, float, float]:
    beta, spread = compute_beta_spread(y, x)
    s = spread.dropna()
    return float(beta), float(s.mean()), float(s.std())


def _is_cointegrated(y: pd.Series, x: pd.Series, threshold: float = 0.05) -> bool:
    if len(y) < 30:
        return False
    _, pvalue, _ = coint(y, x)
    beta, spread = compute_beta_spread(y, x)
    adf_p = adfuller(spread.dropna(), autolag="AIC")[1]
    return pvalue < threshold and adf_p < threshold


def backtest_spread_rolling(
    y: pd.Series,
    x: pd.Series,
    *,
    window_bars: int,
    refit_bars: int,
    entry_z: float,
    exit_z_tp: float,
    exit_z_sl: float,
    cooldown_bars: int = 20,
    coint_threshold: float = 0.05,
    require_cointegration: bool = True,
    costs: CostConfig | None = None,
    y_symbol: str | None = None,
    x_symbol: str | None = None,
    init_nav: float = 1000.0,
    leverage_long: float = 1.0,
    leverage_short: float = 1.0,
    max_margin_utilization: float = 1.0,
    freeze_beta_in_position: bool = True,
) -> pd.DataFrame:
    """
    Rolling walk-forward spread backtest with dynamic hedge ratio refits.

    When ``freeze_beta_in_position`` is True (default), scheduled refits are
    skipped while a position is open so the hedge ratio and entry/exit
    thresholds that sized the trade stay consistent until the position is
    flat. Refitting beta mid-trade would shift the spread definition under a
    position whose units were sized at the previous beta, producing phantom
    exits.
    """
    cfg = costs or DEFAULT_COSTS
    leg_cost_bps = cfg.stat_arb_leg_cost_bps
    aligned = pd.DataFrame({"y": y, "x": x}).dropna()
    y_clean = aligned["y"].values
    x_clean = aligned["x"].values
    idx = aligned.index
    n = len(idx)

    if n < window_bars + 2:
        return pd.DataFrame()

    funding_X = (
        build_funding_series_for_symbol(x_symbol, idx).values
        if x_symbol is not None
        else np.zeros(n)
    )
    funding_Y = (
        build_funding_series_for_symbol(y_symbol, idx).values
        if y_symbol is not None
        else np.zeros(n)
    )

    nav_arr = np.zeros(n)
    pnl_arr = np.zeros(n)
    fee_arr = np.zeros(n)
    funding_arr = np.zeros(n)
    units_Y_arr = np.zeros(n)
    units_X_arr = np.zeros(n)
    position_state_arr = np.zeros(n, dtype=np.int8)
    spread_arr = np.zeros(n)
    beta_arr = np.zeros(n)

    nav = init_nav
    units_Y = 0.0
    units_X = 0.0
    position_state = 0
    cooldown_count = 0
    beta = 1.0
    can_enter = True
    prev_y = y_clean[0]
    prev_x = x_clean[0]

    long_entry = short_entry = long_exit_tp = long_exit_sl = short_exit_tp = short_exit_sl = 0.0

    for i in range(n):
        Y_t = y_clean[i]
        X_t = x_clean[i]

        scheduled_refit = i >= window_bars and (
            i == window_bars or (i - window_bars) % refit_bars == 0
        )
        if freeze_beta_in_position and position_state != 0:
            scheduled_refit = False
        if scheduled_refit:
            est_y = pd.Series(y_clean[i - window_bars : i])
            est_x = pd.Series(x_clean[i - window_bars : i])
            beta, s_mean, s_std = _spread_stats(est_y, est_x)
            if s_std <= 0:
                s_std = 1.0
            can_enter = (not require_cointegration) or _is_cointegrated(est_y, est_x, coint_threshold)
            long_entry = (-entry_z) * s_std + s_mean
            short_entry = entry_z * s_std + s_mean
            long_exit_tp = (-exit_z_tp) * s_std + s_mean
            long_exit_sl = (-exit_z_sl) * s_std + s_mean
            short_exit_tp = exit_z_tp * s_std + s_mean
            short_exit_sl = exit_z_sl * s_std + s_mean

        spread_t = Y_t - beta * X_t
        spread_arr[i] = spread_t
        beta_arr[i] = beta

        pnl_t = 0.0
        fee_t = 0.0
        funding_t = 0.0

        if i > 0:
            pnl_price = units_Y * (Y_t - prev_y) + units_X * (X_t - prev_x)
            funding_t = funding_Y[i] * units_Y * Y_t + funding_X[i] * units_X * X_t
            pnl_t = pnl_price - funding_t
            nav += pnl_t

        cooldown_trigger = spread_t >= short_exit_sl or spread_t <= long_exit_sl
        if cooldown_trigger:
            cooldown_count = cooldown_bars

        if position_state != 0:
            if position_state == 1:
                exit_flag = spread_t >= long_exit_tp or spread_t <= long_exit_sl
            else:
                exit_flag = spread_t <= short_exit_tp or spread_t >= short_exit_sl

            if exit_flag:
                trade_notional = abs(units_X) * X_t + abs(units_Y) * Y_t
                fee_close = trade_notional * leg_cost_bps / 10000.0
                nav -= fee_close
                fee_t += fee_close
                pnl_t -= fee_close
                units_Y = 0.0
                units_X = 0.0
                position_state = 0

        elif position_state == 0 and cooldown_count == 0 and can_enter and i >= window_bars:
            long_entry_flag = spread_t < long_entry and spread_t > long_exit_sl
            short_entry_flag = spread_t > short_entry and spread_t < short_exit_sl

            if long_entry_flag or short_entry_flag:
                if long_entry_flag:
                    margin_per_unit = Y_t / leverage_long + beta * X_t / leverage_short
                else:
                    margin_per_unit = Y_t / leverage_short + beta * X_t / leverage_long

                max_margin = nav * max_margin_utilization
                units_spread = max_margin / margin_per_unit if margin_per_unit > 0 and max_margin > 0 else 0.0

                if units_spread > 0:
                    if long_entry_flag:
                        position_state = 1
                        units_Y = units_spread
                        units_X = -beta * units_spread
                    else:
                        position_state = -1
                        units_Y = -units_spread
                        units_X = beta * units_spread

                    trade_notional_open = abs(units_X) * X_t + abs(units_Y) * Y_t
                    fee_open = trade_notional_open * leg_cost_bps / 10000.0
                    nav -= fee_open
                    fee_t += fee_open
                    pnl_t -= fee_open

        elif position_state == 0 and cooldown_count > 0:
            cooldown_count -= 1

        nav_arr[i] = nav
        pnl_arr[i] = pnl_t
        fee_arr[i] = fee_t
        funding_arr[i] = funding_t
        units_Y_arr[i] = units_Y
        units_X_arr[i] = units_X
        position_state_arr[i] = position_state
        prev_y = Y_t
        prev_x = X_t

    return pd.DataFrame(
        {
            "y": y_clean,
            "x": x_clean,
            "spread": spread_arr,
            "beta": beta_arr,
            "units_Y": units_Y_arr,
            "units_X": units_X_arr,
            "position_state": position_state_arr,
            "pnl": pnl_arr,
            "fee": fee_arr,
            "funding": funding_arr,
            "nav": nav_arr,
        },
        index=idx,
    )


@dataclass
class RollingOOSResult:
    """Continuous stitched out-of-sample result for a single pair.

    With fixed parameters and rolling beta/spread recalibration, the entire
    post-warmup NAV is out-of-sample by construction, so it is treated as one
    stitched OOS equity curve. The per-month breakdown is diagnostic only.
    """

    pair: Tuple[str, str]
    interval: str
    params: Dict[str, Any]
    metrics: BacktestMetrics
    return_tstat: float
    gross_pnl: float
    total_fees: float
    total_funding: float
    pct_profitable_months: float
    worst_losing_streak_months: int
    bt: pd.DataFrame = field(repr=False)
    monthly: pd.DataFrame = field(repr=False)


def _monthly_breakdown(bt: pd.DataFrame, bars_per_year: int) -> pd.DataFrame:
    """Per-calendar-month diagnostics from a stitched OOS backtest."""
    if bt.empty or "nav" not in bt.columns:
        return pd.DataFrame()

    nav = bt["nav"]
    rets = nav.pct_change().fillna(0.0)
    log_rets = np.log1p(rets)
    idx = nav.index
    periods = (idx.tz_localize(None) if idx.tz is not None else idx).to_period("M")
    ann_factor = np.sqrt(bars_per_year)

    rows = []
    for period in periods.unique():
        mask = periods == period
        sub_rets = rets[mask]
        sub_state = bt["position_state"][mask] if "position_state" in bt.columns else None
        month_ret = float(np.expm1(log_rets[mask].sum()))
        sharpe = (
            float(sub_rets.mean() / sub_rets.std() * ann_factor)
            if sub_rets.std() > 0
            else 0.0
        )
        if sub_state is not None:
            n_trades = int((sub_state.diff().fillna(0).abs() > 0).sum())
        else:
            n_trades = 0
        rows.append(
            {
                "month": str(period),
                "return": month_ret,
                "sharpe": sharpe,
                "n_trades": n_trades,
                "fee": float(bt["fee"][mask].sum()) if "fee" in bt.columns else 0.0,
                "funding": float(bt["funding"][mask].sum()) if "funding" in bt.columns else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _worst_losing_streak(monthly_returns: pd.Series) -> int:
    """Longest run of consecutive negative-return months."""
    worst = 0
    current = 0
    for r in monthly_returns:
        if r < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return int(worst)


def evaluate_rolling_oos(
    y: pd.Series,
    x: pd.Series,
    *,
    pair: Tuple[str, str],
    interval: str,
    window_bars: int,
    refit_bars: int,
    entry_z: float = 2.0,
    exit_z_tp: float = 0.5,
    exit_z_sl: float = 3.0,
    cooldown_bars: int = 5,
    coint_threshold: float = 0.05,
    require_cointegration: bool = True,
    costs: CostConfig | None = None,
    init_nav: float = 1000.0,
    drop_warmup: bool = True,
) -> RollingOOSResult:
    """Run a rolling-beta spread backtest with fixed params and evaluate the
    full post-warmup NAV as one stitched out-of-sample equity curve.
    """
    cfg = costs or DEFAULT_COSTS
    y_symbol, x_symbol = pair
    bars_py = bars_per_year_for_interval(interval)

    bt = backtest_spread_rolling(
        y,
        x,
        window_bars=window_bars,
        refit_bars=refit_bars,
        entry_z=entry_z,
        exit_z_tp=exit_z_tp,
        exit_z_sl=exit_z_sl,
        cooldown_bars=cooldown_bars,
        coint_threshold=coint_threshold,
        require_cointegration=require_cointegration,
        costs=cfg,
        y_symbol=y_symbol,
        x_symbol=x_symbol,
        init_nav=init_nav,
    )

    if bt.empty:
        empty_metrics = BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        return RollingOOSResult(
            pair=pair,
            interval=interval,
            params={
                "entry_z": entry_z,
                "exit_z_tp": exit_z_tp,
                "exit_z_sl": exit_z_sl,
                "cooldown_bars": cooldown_bars,
                "window_bars": window_bars,
                "refit_bars": refit_bars,
            },
            metrics=empty_metrics,
            return_tstat=0.0,
            gross_pnl=0.0,
            total_fees=0.0,
            total_funding=0.0,
            pct_profitable_months=0.0,
            worst_losing_streak_months=0,
            bt=bt,
            monthly=pd.DataFrame(),
        )

    oos = bt.iloc[window_bars:] if drop_warmup and len(bt) > window_bars else bt

    metrics = _metrics_from_nav(oos, bars_py)

    rets = oos["nav"].pct_change().fillna(0.0)
    n = len(rets)
    return_tstat = (
        float(rets.mean() / rets.std() * np.sqrt(n)) if rets.std() > 0 and n > 1 else 0.0
    )

    gross_pnl = float(oos["nav"].iloc[-1] - oos["nav"].iloc[0])
    total_fees = float(oos["fee"].sum()) if "fee" in oos.columns else 0.0
    total_funding = float(oos["funding"].sum()) if "funding" in oos.columns else 0.0

    monthly = _monthly_breakdown(oos, bars_py)
    if monthly.empty:
        pct_profitable = 0.0
        worst_streak = 0
    else:
        pct_profitable = float((monthly["return"] > 0).mean())
        worst_streak = _worst_losing_streak(monthly["return"])

    return RollingOOSResult(
        pair=pair,
        interval=interval,
        params={
            "entry_z": entry_z,
            "exit_z_tp": exit_z_tp,
            "exit_z_sl": exit_z_sl,
            "cooldown_bars": cooldown_bars,
            "window_bars": window_bars,
            "refit_bars": refit_bars,
        },
        metrics=metrics,
        return_tstat=return_tstat,
        gross_pnl=gross_pnl,
        total_fees=total_fees,
        total_funding=total_funding,
        pct_profitable_months=pct_profitable,
        worst_losing_streak_months=worst_streak,
        bt=oos,
        monthly=monthly,
    )


@dataclass
class TrainTestResult:
    """Train / test split on a continuous rolling backtest curve.

    Parameters are fixed a priori; the split only separates metrics. The test
    period uses the same rolling recalibration as live trading (trailing window
    may include late-train data when test begins).
    """

    pair: Tuple[str, str]
    interval: str
    params: Dict[str, Any]
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    train_tstat: float
    test_tstat: float
    full: RollingOOSResult = field(repr=False)
    train_bt: pd.DataFrame = field(repr=False)
    test_bt: pd.DataFrame = field(repr=False)


def _return_tstat(bt: pd.DataFrame) -> float:
    if bt.empty or "nav" not in bt.columns:
        return 0.0
    rets = bt["nav"].pct_change().fillna(0.0)
    n = len(rets)
    if rets.std() <= 0 or n <= 1:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(n))


def _slice_bt_period(bt: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    mask = bt.index >= start_ts
    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC")
        mask &= bt.index <= end_ts
    return bt.loc[mask]


def evaluate_rolling_oos_train_test(
    y: pd.Series,
    x: pd.Series,
    *,
    pair: Tuple[str, str],
    interval: str,
    window_bars: int,
    refit_bars: int,
    train_start: str = "2024-01-01",
    train_end: str = "2024-12-31",
    test_start: str = "2025-01-01",
    test_end: str | None = None,
    entry_z: float = 2.0,
    exit_z_tp: float = 0.5,
    exit_z_sl: float = 3.0,
    cooldown_bars: int = 5,
    coint_threshold: float = 0.05,
    require_cointegration: bool = True,
    costs: CostConfig | None = None,
    init_nav: float = 1000.0,
) -> TrainTestResult:
    """Run rolling OOS backtest and report train vs test period metrics."""
    full = evaluate_rolling_oos(
        y,
        x,
        pair=pair,
        interval=interval,
        window_bars=window_bars,
        refit_bars=refit_bars,
        entry_z=entry_z,
        exit_z_tp=exit_z_tp,
        exit_z_sl=exit_z_sl,
        cooldown_bars=cooldown_bars,
        coint_threshold=coint_threshold,
        require_cointegration=require_cointegration,
        costs=costs,
        init_nav=init_nav,
        drop_warmup=True,
    )
    bars_py = bars_per_year_for_interval(interval)
    if test_end is None:
        test_end = str(full.bt.index[-1].date())

    train_bt = _slice_bt_period(full.bt, train_start, train_end)
    test_bt = _slice_bt_period(full.bt, test_start, test_end)

    return TrainTestResult(
        pair=pair,
        interval=interval,
        params=full.params,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        train_metrics=_metrics_from_nav(train_bt, bars_py),
        test_metrics=_metrics_from_nav(test_bt, bars_py),
        train_tstat=_return_tstat(train_bt),
        test_tstat=_return_tstat(test_bt),
        full=full,
        train_bt=train_bt,
        test_bt=test_bt,
    )


def train_test_to_dataframe(result: TrainTestResult) -> pd.DataFrame:
    tr, te = result.train_metrics, result.test_metrics
    return pd.DataFrame(
        [
            {
                "pair": f"{result.pair[0]}/{result.pair[1]}",
                "interval": result.interval,
                "params": str(result.params),
                "train_period": f"{result.train_start}..{result.train_end}",
                "test_period": f"{result.test_start}..{result.test_end}",
                "train_sharpe": tr.sharpe,
                "train_return": tr.total_return,
                "train_max_dd": tr.max_drawdown,
                "train_tstat": result.train_tstat,
                "train_trades": tr.n_trades,
                "test_sharpe": te.sharpe,
                "test_return": te.total_return,
                "test_max_dd": te.max_drawdown,
                "test_tstat": result.test_tstat,
                "test_trades": te.n_trades,
                "full_sharpe": result.full.metrics.sharpe,
                "full_return": result.full.metrics.total_return,
            }
        ]
    )


def rolling_oos_to_dataframe(result: RollingOOSResult) -> pd.DataFrame:
    """One-row summary of the stitched OOS metrics."""
    m = result.metrics
    return pd.DataFrame(
        [
            {
                "pair": f"{result.pair[0]}/{result.pair[1]}",
                "interval": result.interval,
                "params": str(result.params),
                "oos_sharpe": m.sharpe,
                "oos_return": m.total_return,
                "oos_max_dd": m.max_drawdown,
                "return_tstat": result.return_tstat,
                "n_trades": m.n_trades,
                "win_rate": m.win_rate,
                "gross_pnl": result.gross_pnl,
                "total_fees": result.total_fees,
                "total_funding": result.total_funding,
                "pct_profitable_months": result.pct_profitable_months,
                "worst_losing_streak_months": result.worst_losing_streak_months,
            }
        ]
    )


def _metrics_from_nav(bt: pd.DataFrame, bars_per_year: int) -> BacktestMetrics:
    if bt.empty or "nav" not in bt.columns:
        return BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    nav = bt["nav"]
    rets = nav.pct_change().fillna(0.0)
    ann_factor = np.sqrt(bars_per_year)
    sharpe = 0.0
    if rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * ann_factor)

    running_max = nav.cummax()
    max_dd = float(((nav - running_max) / running_max).min())

    if "turnover" in bt.columns:
        trade_mask = bt["turnover"] > 1e-8
        n_trades = int(trade_mask.sum())
        win_rate = float((bt.loc[trade_mask, "pnl"] > 0).mean()) if n_trades > 0 else 0.0
        turnover = float(bt["turnover"].sum())
    elif "position_state" in bt.columns:
        state_change = bt["position_state"].diff().fillna(0).abs() > 0
        n_trades = int(state_change.sum())
        win_rate = float((bt.loc[state_change, "pnl"] > 0).mean()) if n_trades > 0 else 0.0
        turnover = float(n_trades)
    else:
        n_trades = 0
        win_rate = 0.0
        turnover = 0.0

    exposure_pct = 0.0
    if "position_state" in bt.columns:
        exposure_pct = float((bt["position_state"] != 0).mean())
    elif "exposure" in bt.columns:
        exposure_pct = float((bt["exposure"].abs() > 1e-8).mean())

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


def _run_period_rolling(
    y: pd.Series,
    x: pd.Series,
    start: str,
    end: str,
    *,
    window_bars: int,
    refit_bars: int,
    params: Dict[str, Any],
    costs: CostConfig,
    y_symbol: str,
    x_symbol: str,
) -> pd.DataFrame:
    mask = (y.index >= pd.Timestamp(start, tz="UTC")) & (y.index <= pd.Timestamp(end, tz="UTC"))
    y_p = y.loc[mask]
    x_p = x.loc[mask]
    if len(y_p) < window_bars + 50:
        return pd.DataFrame()

    warmup_start = y.index[y.index < y_p.index[0]]
    if len(warmup_start) >= window_bars:
        warmup_y = y.loc[warmup_start[-window_bars:]]
        warmup_x = x.loc[warmup_start[-window_bars:]]
        y_run = pd.concat([warmup_y, y_p])
        x_run = pd.concat([warmup_x, x_p])
    else:
        y_run, x_run = y_p, x_p

    bt = backtest_spread_rolling(
        y_run,
        x_run,
        window_bars=window_bars,
        refit_bars=refit_bars,
        entry_z=params["entry_z"],
        exit_z_tp=params["exit_z_tp"],
        exit_z_sl=params["exit_z_sl"],
        cooldown_bars=params.get("cooldown_bars", 20),
        costs=costs,
        y_symbol=y_symbol,
        x_symbol=x_symbol,
    )
    if bt.empty:
        return bt
    return bt.loc[bt.index >= y_p.index[0]]


def _tune_on_val(
    y: pd.Series,
    x: pd.Series,
    val_start: str,
    val_end: str,
    *,
    window_bars: int,
    refit_bars: int,
    costs: CostConfig,
    y_symbol: str,
    x_symbol: str,
    interval: str,
) -> Dict[str, Any]:
    grid = {
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z_tp": [0.25, 0.5],
        "exit_z_sl": [3.0, 3.5],
    }
    bars_py = bars_per_year_for_interval(interval)
    best_params = {"entry_z": 2.0, "exit_z_tp": 0.5, "exit_z_sl": 3.0, "cooldown_bars": 20}
    best_sharpe = -np.inf

    for entry_z, exit_z_tp, exit_z_sl in product(
        grid["entry_z"], grid["exit_z_tp"], grid["exit_z_sl"]
    ):
        params = {
            "entry_z": entry_z,
            "exit_z_tp": exit_z_tp,
            "exit_z_sl": exit_z_sl,
            "cooldown_bars": 20,
        }
        bt = _run_period_rolling(
            y,
            x,
            val_start,
            val_end,
            window_bars=window_bars,
            refit_bars=refit_bars,
            params=params,
            costs=costs,
            y_symbol=y_symbol,
            x_symbol=x_symbol,
        )
        metrics = _metrics_from_nav(bt, bars_py)
        if metrics.sharpe > best_sharpe:
            best_sharpe = metrics.sharpe
            best_params = params

    return best_params


def walk_forward_stat_arb_evaluate(
    y: pd.Series,
    x: pd.Series,
    *,
    pair: Tuple[str, str],
    interval: str,
    window_bars: int,
    refit_bars: int | None = None,
    train_start: str = "2024-01-01",
    train_end: str = "2024-12-31",
    val_start: str = "2025-01-01",
    val_end: str = "2025-12-31",
    test_start: str = "2026-01-01",
    test_end: str = "2026-04-30",
    min_oos_sharpe: float = 0.5,
    min_stress_sharpe: float = 0.3,
    max_oos_drawdown: float = -0.25,
    costs: CostConfig | None = None,
) -> StatArbWalkForwardResult:
    """Walk-forward stat-arb evaluation with rolling beta and val param tuning."""
    cfg = costs or DEFAULT_COSTS
    y_symbol, x_symbol = pair
    bars_py = bars_per_year_for_interval(interval)
    if refit_bars is None:
        refit_bars = max(window_bars // 7, 24)

    best_params = _tune_on_val(
        y,
        x,
        val_start,
        val_end,
        window_bars=window_bars,
        refit_bars=refit_bars,
        costs=cfg,
        y_symbol=y_symbol,
        x_symbol=x_symbol,
        interval=interval,
    )

    train_bt = _run_period_rolling(
        y, x, train_start, train_end,
        window_bars=window_bars, refit_bars=refit_bars,
        params=best_params, costs=cfg, y_symbol=y_symbol, x_symbol=x_symbol,
    )
    val_bt = _run_period_rolling(
        y, x, val_start, val_end,
        window_bars=window_bars, refit_bars=refit_bars,
        params=best_params, costs=cfg, y_symbol=y_symbol, x_symbol=x_symbol,
    )
    test_bt = _run_period_rolling(
        y, x, test_start, test_end,
        window_bars=window_bars, refit_bars=refit_bars,
        params=best_params, costs=cfg, y_symbol=y_symbol, x_symbol=x_symbol,
    )
    stress_bt = _run_period_rolling(
        y, x, test_start, test_end,
        window_bars=window_bars, refit_bars=refit_bars,
        params=best_params, costs=cfg.with_stress(1.5),
        y_symbol=y_symbol, x_symbol=x_symbol,
    )

    train_m = _metrics_from_nav(train_bt, bars_py)
    val_m = _metrics_from_nav(val_bt, bars_py)
    test_m = _metrics_from_nav(test_bt, bars_py)
    stress_m = _metrics_from_nav(stress_bt, bars_py)

    passed = True
    reasons: List[str] = []
    if test_m.sharpe < min_oos_sharpe:
        passed = False
        reasons.append(f"test Sharpe {test_m.sharpe:.2f} < {min_oos_sharpe}")
    if test_m.max_drawdown < max_oos_drawdown:
        passed = False
        reasons.append(f"test max DD {test_m.max_drawdown:.2%} < {max_oos_drawdown:.0%}")
    if stress_m.sharpe < min_stress_sharpe:
        passed = False
        reasons.append(f"stress Sharpe {stress_m.sharpe:.2f} < {min_stress_sharpe}")

    return StatArbWalkForwardResult(
        pair=pair,
        interval=interval,
        params={**best_params, "window_bars": window_bars, "refit_bars": refit_bars},
        train_metrics=train_m,
        val_metrics=val_m,
        test_metrics=test_m,
        stress_test_metrics=stress_m,
        passed_gate=passed,
        gate_reason="PASS" if passed else "; ".join(reasons),
        train_bt=train_bt,
        val_bt=val_bt,
        test_bt=test_bt,
    )


def stat_arb_results_to_dataframe(results: List[StatArbWalkForwardResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "pair": f"{r.pair[0]}/{r.pair[1]}",
                "interval": r.interval,
                "params": str(r.params),
                "train_sharpe": r.train_metrics.sharpe,
                "val_sharpe": r.val_metrics.sharpe,
                "test_sharpe": r.test_metrics.sharpe,
                "stress_test_sharpe": r.stress_test_metrics.sharpe,
                "test_return": r.test_metrics.total_return,
                "test_max_dd": r.test_metrics.max_drawdown,
                "passed_gate": r.passed_gate,
                "gate_reason": r.gate_reason,
            }
        )
    return pd.DataFrame(rows)
