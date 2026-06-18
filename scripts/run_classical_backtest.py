#!/usr/bin/env python3
"""Compare classical strategies with walk-forward evaluation and realistic costs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest.evaluation import results_to_dataframe, walk_forward_evaluate
from src.data.data_load import load_data
from src.strategy.classical.mean_reversion import bollinger_mr_signal, rsi_mr_signal
from src.strategy.classical.momentum import dual_ma_crossover_signal, ts_momentum_signal
from src.strategy.classical.regime import regime_switch_signal
from src.strategy.indicators import prepare_ohlcv, resample_ohlcv
from src.trading.costs import DEFAULT_COSTS, STRESS_COSTS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

STRATEGIES = {
    "ts_momentum": (ts_momentum_signal, {"lookback": 24}),
    "dual_ma": (dual_ma_crossover_signal, {"fast": 20, "slow": 60}),
    "bollinger_mr": (bollinger_mr_signal, {"period": 20}),
    "rsi_mr": (rsi_mr_signal, {"period": 14}),
}


def load_close(symbol: str, interval: str, start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    df = load_data(
        symbols=symbol,
        start_date=start,
        end_date=end,
        interval="1m" if interval in ("15m", "1h") else interval,
    )
    ohlcv = prepare_ohlcv(df, symbol=symbol)
    if interval in ("15m", "1h"):
        ohlcv = resample_ohlcv(ohlcv, interval)
    return ohlcv, ohlcv["close"]


def run_for_symbol_interval(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    costs=DEFAULT_COSTS,
) -> list:
    ohlcv, close = load_close(symbol, interval, start, end)
    results = []

    mom = ts_momentum_signal(close, lookback=24)
    mr = bollinger_mr_signal(close, period=20)
    regime_sig = regime_switch_signal(ohlcv, mom, mr)["combined_signal"]

    all_strategies = {
        **STRATEGIES,
        "regime_switch": (lambda p, **_: regime_sig, {}),
    }

    for name, (fn, kwargs) in all_strategies.items():
        results.append(
            walk_forward_evaluate(
                close,
                fn,
                kwargs,
                strategy_name=name,
                symbol=symbol,
                interval=interval,
                costs=costs,
                apply_funding=True,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run classical strategy backtests")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--intervals", nargs="+", default=["15m", "1h"])
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument(
        "--gate-passing-only",
        action="store_true",
        help="Run only dual_ma and regime_switch on BTCUSDT 1h with stress test",
    )
    args = parser.parse_args()

    if args.gate_passing_only:
        configs = [("BTCUSDT", "1h")]
        strategy_filter = {"dual_ma", "regime_switch"}
    else:
        configs = [(sym, iv) for sym in args.symbols for iv in args.intervals]
        strategy_filter = None

    all_results = []
    for sym, interval in configs:
        print(f"Running {sym} @ {interval}...")
        all_results.extend(run_for_symbol_interval(sym, interval, args.start, args.end))

    if strategy_filter:
        summary = results_to_dataframe(all_results)
        summary = summary[summary["strategy"].isin(strategy_filter)]
        for _, row in summary.iterrows():
            stress = run_for_symbol_interval(
                row["symbol"], row["interval"], args.start, args.end, costs=STRESS_COSTS
            )
            stress_row = results_to_dataframe(stress)
            match = stress_row[stress_row["strategy"] == row["strategy"]]
            if not match.empty:
                print(
                    f"  {row['strategy']}: test Sharpe={row['test_sharpe']:.2f}, "
                    f"stress Sharpe={match.iloc[0]['test_sharpe']:.2f}"
                )
        out_path = RESULTS_DIR / "classical_backtest_with_funding.csv"
    else:
        summary = results_to_dataframe(all_results)
        out_path = RESULTS_DIR / "classical_backtest_summary.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print("\n=== Walk-Forward Results ===")
    print(summary.to_string(index=False))
    print(f"\nSaved to {out_path}")

    passed = summary[summary["passed_gate"]]
    if not passed.empty:
        print("\nStrategies passing OOS gate:")
        print(passed[["strategy", "symbol", "interval", "test_sharpe", "test_max_dd"]].to_string(index=False))
    else:
        print("\nNo strategies passed the OOS gate on 2026 test data.")


if __name__ == "__main__":
    main()
