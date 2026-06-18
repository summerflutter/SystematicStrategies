"""Classical directional signals: momentum, mean-reversion, regime switching."""

from src.strategy.classical.mean_reversion import bollinger_mr_signal, rsi_mr_signal
from src.strategy.classical.momentum import dual_ma_crossover_signal, ts_momentum_signal
from src.strategy.classical.regime import regime_switch_signal

__all__ = [
    "ts_momentum_signal",
    "dual_ma_crossover_signal",
    "bollinger_mr_signal",
    "rsi_mr_signal",
    "regime_switch_signal",
]
