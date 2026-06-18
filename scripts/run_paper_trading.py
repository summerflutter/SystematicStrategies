#!/usr/bin/env python3
"""
Log paper-trading runs for comparison against backtest expectations.

Append one row per run to results/paper_trading_log.csv.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.data_load import load_data
from src.strategy.indicators import prepare_ohlcv, resample_ohlcv
from src.trading.env import load_dotenv
from src.trading.paper_trader import BinanceTestnetClient, PaperTrader, PaperTradingConfig

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LOG_PATH = RESULTS_DIR / "paper_trading_log.csv"


def pick_best_dual_ma() -> dict:
    summary_path = RESULTS_DIR / "classical_backtest_with_funding.csv"
    if not summary_path.exists():
        summary_path = RESULTS_DIR / "classical_backtest_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        passed = df[(df["passed_gate"]) & (df["strategy"] == "dual_ma")]
        if not passed.empty:
            row = passed.sort_values("test_sharpe", ascending=False).iloc[0]
            return {
                "strategy": row["strategy"],
                "symbol": row["symbol"],
                "interval": row["interval"],
            }
    return {"strategy": "dual_ma", "symbol": "BTCUSDT", "interval": "1h"}


def main():
    parser = argparse.ArgumentParser(description="Paper trade dual_ma with vol-target sizing")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--interval", default=None)
    parser.add_argument("--init-nav", type=float, default=10_000.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    best = pick_best_dual_ma()
    symbol = args.symbol or best["symbol"]
    interval = args.interval or best["interval"]

    df = load_data(
        symbols=symbol,
        start_date=(pd.Timestamp.now("UTC") - pd.Timedelta(days=120)).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
        interval="1m",
    )
    ohlcv = resample_ohlcv(prepare_ohlcv(df, symbol=symbol), interval)
    close = ohlcv["close"]

    config = PaperTradingConfig(
        symbol=symbol,
        strategy="dual_ma",
        interval=interval,
        init_nav=args.init_nav,
    )
    trader = PaperTrader(config)

    if args.dry_run:
        signal = trader.compute_signal(close)
        exposure = trader.compute_target_exposure(close)
        price = 100_000.0
        qty = trader.compute_order_quantity(close, price)
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "strategy": "dual_ma",
            "interval": interval,
            "signal": signal,
            "target_exposure": exposure,
            "order_quantity": qty,
            "dry_run": True,
        }
        print(json.dumps(result, indent=2))
    else:
        client = BinanceTestnetClient()
        client.ping()
        trader.client = client
        result = trader.run_once(close)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["interval"] = interval
        print(json.dumps({k: str(v) for k, v in result.items()}, indent=2, default=str))

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "strategy": "dual_ma",
        "interval": interval,
        "signal": trader.compute_signal(close),
        "target_exposure": trader.compute_target_exposure(close),
        "dry_run": args.dry_run,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame([row])
    if LOG_PATH.exists():
        log_df = pd.concat([pd.read_csv(LOG_PATH), log_df], ignore_index=True)
    log_df.to_csv(LOG_PATH, index=False)
    print(f"\nLogged to {LOG_PATH}")


if __name__ == "__main__":
    main()
