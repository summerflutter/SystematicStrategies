#!/usr/bin/env python3
"""Download perp klines and funding rates for BTC/ETH."""

from __future__ import annotations

import argparse

import pandas as pd

from src.data.data_download import (
    download_funding_rates_monthly,
    download_perp_klines_monthly,
)


def get_last_month_end() -> pd.Timestamp:
    now = pd.Timestamp.now("UTC")
    return (now.replace(day=1) - pd.Timedelta(days=1)).normalize()


def main():
    parser = argparse.ArgumentParser(description="Download perp klines and funding rates")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--intervals", nargs="+", default=["5m", "15m"])
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    end = args.end or str(get_last_month_end().date())

    for interval in args.intervals:
        print(f"Downloading perp klines {args.symbols} @ {interval}...")
        download_perp_klines_monthly(
            symbols=args.symbols,
            interval=interval,
            start=args.start,
            end=end,
            overwrite=args.overwrite,
        )

    print(f"Downloading funding rates {args.symbols}...")
    download_funding_rates_monthly(
        symbols=args.symbols,
        start=args.start,
        end=end,
        overwrite=args.overwrite,
    )
    print("Done.")


if __name__ == "__main__":
    main()
