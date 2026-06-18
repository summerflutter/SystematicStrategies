# Strategy Status

Updated from walk-forward backtests (train 2024 / val 2025 / test 2026).  
Run commands: [ARCHITECTURE.md](ARCHITECTURE.md#scripts).

## Active

| Strategy | Symbol | Timeframe | Notes |
|----------|--------|-----------|-------|
| `dual_ma` | BTCUSDT | 1h | Primary paper-trading candidate |
| `stat_arb` | BTC/ETH | 15m | Rolling-beta walk-forward; OOS gate currently failing |

## Frozen

| Area | Location | Reason |
|------|----------|--------|
| Predictive ML | `src/strategy/predictive/` | No edge after costs at 1m horizon |
| Standalone MR | `bollinger_mr`, `rsi_mr` | Failed OOS vs `dual_ma` |
| 15m classical | — | High turnover; poor 2026 test |
| Stat arb ETH/BNB | — | Validate BTC/ETH first |

## Latest gates

| Strategy | Test Sharpe | Stress Sharpe (1.5× costs) | Gate |
|----------|-------------|----------------------------|------|
| dual_ma BTC 1h | 1.07 | 0.41 | **PASS** |
| regime_switch BTC 1h | 0.54 | −1.26 | PASS base / FAIL stress |
| stat_arb BTC/ETH 15m | −0.00 | −0.39 | **FAIL** |

Stress = fees, slippage, and funding at 1.5× base assumptions.

## Promotion criteria

1. OOS test Sharpe ≥ 0.5 and max drawdown ≥ −25%
2. Stress Sharpe ≥ 0.3
3. Walk-forward with no look-ahead (rolling beta for stat arb)
4. Paper trading matches backtest for ≥ 4 weeks before live size
