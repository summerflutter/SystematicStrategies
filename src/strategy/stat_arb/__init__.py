"""Pairs statistical arbitrage: spread engine, scan, and walk-forward evaluation."""

from src.strategy.stat_arb.engine import (
    StatArbSignal,
    backtest_spread_signal,
    build_spread_signal_from_stats,
    compute_beta_spread,
    estimate_half_life,
    generate_spread_signal,
)

__all__ = [
    "StatArbSignal",
    "backtest_spread_signal",
    "build_spread_signal_from_stats",
    "compute_beta_spread",
    "estimate_half_life",
    "generate_spread_signal",
]
