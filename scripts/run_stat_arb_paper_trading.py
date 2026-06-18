#!/usr/bin/env python3
"""Paper trade rolling stat-arb strategies on Binance testnet futures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.data_load import load_data
from src.strategy.indicators import prepare_ohlcv, resample_ohlcv
from src.trading.env import load_dotenv
from src.trading.paper_trader import BinanceTestnetClient
from src.trading.stat_arb_trader import STAT_ARB_STRATEGIES, StatArbPaperTrader

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LOG_PATH = RESULTS_DIR / "stat_arb_paper_trading_log.csv"

# Need ~60d of 4h bars for window=360, plus buffer.
LOOKBACK_DAYS = {"4h": 120, "1d": 400}


def load_pair_close(symbol: str, interval: str, start: str, end: str) -> pd.Series:
    if interval in ("1d", "4h"):
        df = load_data(
            symbols=symbol, start_date=start, end_date=end, interval="15m", product="perp"
        )
        ohlcv = prepare_ohlcv(df, symbol=symbol)
        ohlcv = resample_ohlcv(ohlcv, interval)
        return ohlcv["close"]
    raise ValueError(f"Unsupported interval for stat-arb paper trading: {interval}")


def run_strategy(name: str, *, dry_run: bool, init_nav: float | None) -> dict:
    cfg = STAT_ARB_STRATEGIES[name]
    if init_nav is not None:
        cfg.init_nav = init_nav

    end = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    lookback = LOOKBACK_DAYS.get(cfg.interval, 120)
    start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")

    y = load_pair_close(cfg.y_symbol, cfg.interval, start, end)
    x = load_pair_close(cfg.x_symbol, cfg.interval, start, end)

    client = None if dry_run else BinanceTestnetClient(market=cfg.market)
    if not dry_run:
        client.ping()

    trader = StatArbPaperTrader(cfg, client=client)
    result = trader.run_once(y, x, dry_run=dry_run)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


def _require_testnet_keys() -> None:
    if not os.environ.get("BINANCE_TESTNET_API_KEY") or not os.environ.get(
        "BINANCE_TESTNET_API_SECRET"
    ):
        print(
            "Missing testnet API credentials.\n"
            "  1. cp .env.example .env\n"
            "  2. Add BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET\n"
            "  3. www.binance.com → Futures → profile → Demo/Mock Trading → API Management\n"
            "Or use --dry-run to test signals without keys.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Paper trade stat-arb pairs on testnet")
    parser.add_argument(
        "--strategy",
        choices=["all", "btc_sol_4h", "xrp_ada_4h"],
        default="all",
        help="Which strategy to run (default: both)",
    )
    parser.add_argument("--init-nav", type=float, default=5_000.0, help="NAV per strategy")
    parser.add_argument("--dry-run", action="store_true", help="Log signals without placing orders")
    args = parser.parse_args()

    if not args.dry_run:
        _require_testnet_keys()
        BinanceTestnetClient().ping()
        print("Testnet connection OK")

    names = list(STAT_ARB_STRATEGIES) if args.strategy == "all" else [args.strategy]
    results = []
    for name in names:
        print(f"\n--- {name} ---")
        try:
            result = run_strategy(name, dry_run=args.dry_run, init_nav=args.init_nav)
            print(json.dumps(result, indent=2, default=str))
            results.append(result)
        except Exception as exc:
            err = {"strategy": name, "error": str(exc), "timestamp": datetime.now(timezone.utc).isoformat()}
            print(json.dumps(err, indent=2))
            results.append(err)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(results)
    if LOG_PATH.exists():
        log_df = pd.concat([pd.read_csv(LOG_PATH), log_df], ignore_index=True)
    log_df.to_csv(LOG_PATH, index=False)
    print(f"\nLogged to {LOG_PATH}")


if __name__ == "__main__":
    main()
