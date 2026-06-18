"""Walk-forward out-of-sample strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import pandas as pd

from src.backtest.directional import (
    BacktestMetrics,
    backtest_directional,
    compute_backtest_metrics,
)
from src.trading.costs import CostConfig, DEFAULT_COSTS


@dataclass
class WalkForwardResult:
    strategy: str
    symbol: str
    interval: str
    train_metrics: BacktestMetrics
    val_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    params: Dict[str, Any]
    passed_gate: bool
    gate_reason: str
    train_bt: pd.DataFrame = field(repr=False)
    val_bt: pd.DataFrame = field(repr=False)
    test_bt: pd.DataFrame = field(repr=False)


def bars_per_year_for_interval(interval: str) -> int:
    mapping = {
        "1m": 365 * 24 * 60,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "1h": 365 * 24,
        "4h": 365 * 6,
        "1d": 365,
    }
    return mapping.get(interval, 365 * 24 * 4)


def walk_forward_evaluate(
    price: pd.Series,
    signal_fn: Callable[..., pd.Series],
    signal_kwargs: Dict[str, Any],
    *,
    strategy_name: str,
    symbol: str,
    interval: str,
    train_start: str = "2024-01-01",
    train_end: str = "2024-12-31",
    val_start: str = "2025-01-01",
    val_end: str = "2025-12-31",
    test_start: str = "2026-01-01",
    test_end: str = "2026-04-30",
    min_oos_sharpe: float = 0.5,
    max_oos_drawdown: float = -0.25,
    trading_fee_bps: float | None = None,
    slippage_bps: float | None = None,
    costs: CostConfig | None = None,
    apply_funding: bool = True,
    target_vol: float = 0.15,
) -> WalkForwardResult:
    """Train/validate/test split with go/no-go gate on OOS test period."""
    bars_py = bars_per_year_for_interval(interval)
    cfg = costs or DEFAULT_COSTS
    full_signal = signal_fn(price, **signal_kwargs)

    def _run_period(start: str, end: str) -> tuple[pd.DataFrame, BacktestMetrics]:
        mask = (price.index >= pd.Timestamp(start, tz="UTC")) & (
            price.index <= pd.Timestamp(end, tz="UTC")
        )
        p = price.loc[mask]
        if len(p) < 50:
            empty = pd.DataFrame()
            metrics = BacktestMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            return empty, metrics

        sig = full_signal.reindex(p.index).fillna(0.0)
        bt_kwargs = {
            "target_vol": target_vol,
            "bars_per_year": bars_py,
            "costs": cfg,
            "symbol": symbol,
            "apply_funding": apply_funding,
        }
        if trading_fee_bps is not None:
            bt_kwargs["trading_fee_bps"] = trading_fee_bps
        if slippage_bps is not None:
            bt_kwargs["slippage_bps"] = slippage_bps

        bt = backtest_directional(p, sig.reindex(p.index).fillna(0.0), **bt_kwargs)
        metrics = compute_backtest_metrics(bt, bars_per_year=bars_py)
        return bt, metrics

    train_bt, train_m = _run_period(train_start, train_end)
    val_bt, val_m = _run_period(val_start, val_end)
    test_bt, test_m = _run_period(test_start, test_end)

    passed = True
    reasons: List[str] = []
    if test_m.sharpe < min_oos_sharpe:
        passed = False
        reasons.append(f"test Sharpe {test_m.sharpe:.2f} < {min_oos_sharpe}")
    if test_m.max_drawdown < max_oos_drawdown:
        passed = False
        reasons.append(f"test max DD {test_m.max_drawdown:.2%} < {max_oos_drawdown:.0%}")

    gate_reason = "PASS" if passed else "; ".join(reasons)

    return WalkForwardResult(
        strategy=strategy_name,
        symbol=symbol,
        interval=interval,
        train_metrics=train_m,
        val_metrics=val_m,
        test_metrics=test_m,
        params=signal_kwargs,
        passed_gate=passed,
        gate_reason=gate_reason,
        train_bt=train_bt,
        val_bt=val_bt,
        test_bt=test_bt,
    )


def results_to_dataframe(results: List[WalkForwardResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "strategy": r.strategy,
                "symbol": r.symbol,
                "interval": r.interval,
                "params": str(r.params),
                "train_sharpe": r.train_metrics.sharpe,
                "val_sharpe": r.val_metrics.sharpe,
                "test_sharpe": r.test_metrics.sharpe,
                "test_return": r.test_metrics.total_return,
                "test_max_dd": r.test_metrics.max_drawdown,
                "test_trades": r.test_metrics.n_trades,
                "passed_gate": r.passed_gate,
                "gate_reason": r.gate_reason,
            }
        )
    return pd.DataFrame(rows)
