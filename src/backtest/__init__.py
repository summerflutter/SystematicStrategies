"""Shared backtesting and walk-forward evaluation."""

from src.backtest.directional import (
    BacktestMetrics,
    backtest_directional,
    compute_backtest_metrics,
    vol_target_position,
)
from src.backtest.evaluation import (
    WalkForwardResult,
    bars_per_year_for_interval,
    results_to_dataframe,
    walk_forward_evaluate,
)
from src.backtest.trades import (
    BacktestRunResult,
    extract_rebalance_log,
    extract_trade_log,
    run_backtest,
)

__all__ = [
    "BacktestMetrics",
    "BacktestRunResult",
    "backtest_directional",
    "compute_backtest_metrics",
    "vol_target_position",
    "WalkForwardResult",
    "bars_per_year_for_interval",
    "results_to_dataframe",
    "walk_forward_evaluate",
    "extract_trade_log",
    "extract_rebalance_log",
    "run_backtest",
]
