#!/usr/bin/env python3
"""Stat-arb pair scan and walk-forward evaluation."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.data.data_load import load_data
from src.strategy.indicators import prepare_ohlcv, resample_ohlcv
from src.strategy.stat_arb.diagnostics import spread_diagnostics
from src.strategy.stat_arb.evaluation import (
    evaluate_rolling_oos,
    evaluate_rolling_oos_train_test,
    rolling_oos_to_dataframe,
    stat_arb_results_to_dataframe,
    train_test_to_dataframe,
    walk_forward_stat_arb_evaluate,
)
from src.strategy.stat_arb.scan import pair_scan_to_dataframe, scan_all_pairs
from src.trading.costs import DEFAULT_COSTS, STRESS_COSTS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PANEL_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
]

BARS_PER_YEAR = {"5m": 365 * 24 * 12, "15m": 365 * 24 * 4, "4h": 365 * 6, "1d": 365}
# Calendar-equivalent windows: N daily bars * 6 four-hour bars per day.
_BARS_PER_DAY_4H = 6
WINDOW_BARS = {
    "5m": 60 * 24 * 12,
    "15m": 60 * 24 * 4,
    "4h": 60 * _BARS_PER_DAY_4H,  # ~60 calendar days
    "1d": 60,
}
REFIT_BARS = {
    "5m": 60 * 24,
    "15m": 60 * 24,
    "4h": 7 * _BARS_PER_DAY_4H,  # ~7 calendar days
    "1d": 7,
}
COOLDOWN_BARS = {
    "4h": 5 * _BARS_PER_DAY_4H,  # ~5 calendar days
    "1d": 5,
}


def load_close(symbol: str, interval: str, start: str, end: str) -> pd.Series:
    if interval in ("1d", "4h"):
        # Resample perp 15m klines so the price series matches the perp funding
        # model used by the stat-arb backtest.
        df = load_data(
            symbols=symbol, start_date=start, end_date=end, interval="15m", product="perp"
        )
        ohlcv = prepare_ohlcv(df, symbol=symbol)
        ohlcv = resample_ohlcv(ohlcv, interval)
        return ohlcv["close"]

    source = "1m" if interval == "15m" else interval
    df = load_data(symbols=symbol, start_date=start, end_date=end, interval=source)
    ohlcv = prepare_ohlcv(df, symbol=symbol)
    if interval == "15m":
        ohlcv = resample_ohlcv(ohlcv, interval)
    return ohlcv["close"]


def cmd_scan(args: argparse.Namespace) -> None:
    all_results = []
    for interval in args.intervals:
        print(f"Scanning pairs @ {interval}...")
        prices = {sym: load_close(sym, interval, args.start, args.end) for sym in PANEL_SYMBOLS}
        all_results.extend(
            scan_all_pairs(
                prices,
                PANEL_SYMBOLS,
                interval=interval,
                window_bars=WINDOW_BARS.get(interval, 60 * 24 * 4),
                bars_per_year=BARS_PER_YEAR.get(interval, 365 * 24 * 4),
            )
        )

    summary = pair_scan_to_dataframe(all_results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "stat_arb_pair_scan.csv"
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved to {out_path}")


def cmd_walkforward(args: argparse.Namespace) -> None:
    y_sym, x_sym = args.pair.split("/")
    print(f"Walk-forward {y_sym}/{x_sym} @ {args.interval}...")
    y = load_close(y_sym, args.interval, args.start, args.end)
    x = load_close(x_sym, args.interval, args.start, args.end)

    window = WINDOW_BARS.get(args.interval, 60 * 24 * 4)
    refit = REFIT_BARS.get(args.interval, 60 * 24)

    result = walk_forward_stat_arb_evaluate(
        y, x,
        pair=(y_sym, x_sym),
        interval=args.interval,
        window_bars=window,
        refit_bars=refit,
        costs=DEFAULT_COSTS,
    )

    summary = stat_arb_results_to_dataframe([result])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "stat_arb_walkforward_summary.csv"
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nGate: {result.gate_reason}")
    print(f"Saved to {out_path}")

    if args.stress:
        stress = walk_forward_stat_arb_evaluate(
            y, x,
            pair=(y_sym, x_sym),
            interval=args.interval,
            window_bars=window,
            refit_bars=refit,
            costs=STRESS_COSTS,
        )
        print(f"Stress gate: {stress.gate_reason}")


def cmd_panel(args: argparse.Namespace) -> None:
    """Scan all pair combinations: diagnostics + rolling OOS on 1d and 4h."""
    intervals = args.intervals
    print(f"Pair panel: {len(PANEL_SYMBOLS)} symbols -> "
          f"{len(list(combinations(PANEL_SYMBOLS, 2)))} pairs")
    print(f"Symbols: {', '.join(PANEL_SYMBOLS)}")
    print(f"Intervals: {', '.join(intervals)}\n")

    prices_by_interval = {
        iv: {sym: load_close(sym, iv, args.start, args.end) for sym in PANEL_SYMBOLS}
        for iv in intervals
    }

    rows = []
    for y_sym, x_sym in combinations(PANEL_SYMBOLS, 2):
        for interval in intervals:
            y = prices_by_interval[interval][y_sym]
            x = prices_by_interval[interval][x_sym]
            if len(y) < 100 or len(x) < 100:
                continue

            diag = spread_diagnostics(y, x)
            window = WINDOW_BARS.get(interval, 60)
            refit = REFIT_BARS.get(interval, 7)
            cooldown = COOLDOWN_BARS.get(interval, 5)
            costs = STRESS_COSTS if args.stress else DEFAULT_COSTS

            res = evaluate_rolling_oos(
                y, x,
                pair=(y_sym, x_sym),
                interval=interval,
                window_bars=window,
                refit_bars=refit,
                cooldown_bars=cooldown,
                require_cointegration=False,
                costs=costs,
            )
            m = res.metrics
            rows.append({
                "pair": f"{y_sym}/{x_sym}",
                "interval": interval,
                "window_bars": window,
                "refit_bars": refit,
                "coint_p": round(diag.coint_pvalue, 4),
                "adf_p": round(diag.adf_pvalue, 4),
                "half_life": round(diag.half_life, 1) if diag.half_life else None,
                "hurst": round(diag.hurst, 3) if diag.hurst else None,
                "oos_sharpe": m.sharpe,
                "oos_return": m.total_return,
                "oos_max_dd": m.max_drawdown,
                "return_tstat": res.return_tstat,
                "n_trades": m.n_trades,
                "total_fees": res.total_fees,
                "total_funding": res.total_funding,
                "pct_profitable_months": res.pct_profitable_months,
            })

    summary = pd.DataFrame(rows).sort_values(
        ["interval", "oos_sharpe"], ascending=[True, False]
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "stat_arb_pair_panel.csv"
    summary.to_csv(out_path, index=False)

    for interval in intervals:
        sub = summary[summary["interval"] == interval]
        print(f"=== {interval} (top 8 by Sharpe) ===")
        cols = ["pair", "oos_sharpe", "oos_return", "return_tstat", "n_trades", "coint_p", "adf_p"]
        print(sub[cols].head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()

    print(f"Saved full panel ({len(summary)} rows) to {out_path}")


def cmd_split(args: argparse.Namespace) -> None:
    """Train/test split on a continuous rolling backtest (fixed params)."""
    y_sym, x_sym = args.pair.split("/")
    print(f"Train/test split {y_sym}/{x_sym} @ {args.interval}")
    print(f"  Train: {args.train_start} .. {args.train_end}")
    print(f"  Test:  {args.test_start} .. {args.end}")
    y = load_close(y_sym, args.interval, args.start, args.end)
    x = load_close(x_sym, args.interval, args.start, args.end)

    window = args.window_bars or WINDOW_BARS.get(args.interval, 60)
    refit = args.refit_bars or REFIT_BARS.get(args.interval, 7)
    costs = STRESS_COSTS if args.stress else DEFAULT_COSTS

    result = evaluate_rolling_oos_train_test(
        y, x,
        pair=(y_sym, x_sym),
        interval=args.interval,
        window_bars=window,
        refit_bars=refit,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.end,
        entry_z=args.entry_z,
        exit_z_tp=args.exit_z_tp,
        exit_z_sl=args.exit_z_sl,
        cooldown_bars=args.cooldown_bars
        if args.cooldown_bars is not None
        else COOLDOWN_BARS.get(args.interval, 5),
        require_cointegration=args.require_coint,
        costs=costs,
    )

    summary = train_test_to_dataframe(result)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "stat_arb_train_test.csv"
    summary.to_csv(out_path, index=False)
    result.full.monthly.to_csv(RESULTS_DIR / "stat_arb_train_test_months.csv", index=False)

    tr, te = result.train_metrics, result.test_metrics
    print(summary.to_string(index=False))
    print(f"\nTrain: Sharpe {tr.sharpe:.2f}, return {tr.total_return:.2%}, t-stat {result.train_tstat:.2f}, trades {tr.n_trades}")
    print(f"Test:  Sharpe {te.sharpe:.2f}, return {te.total_return:.2%}, t-stat {result.test_tstat:.2f}, trades {te.n_trades}")
    passed = te.sharpe >= 0.3 and result.test_tstat > 0
    print(f"\nHeuristic test verdict: {'PASS (positive test Sharpe & t-stat)' if passed else 'FAIL'}")
    print(f"Saved to {out_path}")


def cmd_rolling(args: argparse.Namespace) -> None:
    y_sym, x_sym = args.pair.split("/")
    print(f"Rolling OOS {y_sym}/{x_sym} @ {args.interval} (fixed params)...")
    y = load_close(y_sym, args.interval, args.start, args.end)
    x = load_close(x_sym, args.interval, args.start, args.end)

    window = args.window_bars or WINDOW_BARS.get(args.interval, 60)
    refit = args.refit_bars or REFIT_BARS.get(args.interval, 7)
    costs = STRESS_COSTS if args.stress else DEFAULT_COSTS

    result = evaluate_rolling_oos(
        y, x,
        pair=(y_sym, x_sym),
        interval=args.interval,
        window_bars=window,
        refit_bars=refit,
        entry_z=args.entry_z,
        exit_z_tp=args.exit_z_tp,
        exit_z_sl=args.exit_z_sl,
        cooldown_bars=args.cooldown_bars
        if args.cooldown_bars is not None
        else COOLDOWN_BARS.get(args.interval, 5),
        require_cointegration=args.require_coint,
        costs=costs,
    )

    summary = rolling_oos_to_dataframe(result)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oos_path = RESULTS_DIR / "stat_arb_rolling_oos.csv"
    months_path = RESULTS_DIR / "stat_arb_rolling_months.csv"
    summary.to_csv(oos_path, index=False)
    result.monthly.to_csv(months_path, index=False)

    print(summary.to_string(index=False))
    print(f"\nReturn t-stat: {result.return_tstat:.2f}")
    print(f"Profitable months: {result.pct_profitable_months:.0%}")
    print(f"Worst losing streak (months): {result.worst_losing_streak_months}")
    print(f"Total fees: {result.total_fees:.2f} | Total funding: {result.total_funding:.2f}")
    print(f"\nSaved stitched OOS to {oos_path}")
    print(f"Saved per-month diagnostics to {months_path}")


def main():
    parser = argparse.ArgumentParser(description="Stat-arb research runners")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-04-30")
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser(
        "split",
        help="Train/test split on rolling backtest (default: train 2024, test 2025+)",
    )
    split.add_argument("--pair", default="XRPUSDT/ADAUSDT")
    split.add_argument("--interval", default="4h")
    split.add_argument("--train-start", default="2024-01-01", dest="train_start")
    split.add_argument("--train-end", default="2024-12-31", dest="train_end")
    split.add_argument("--test-start", default="2025-01-01", dest="test_start")
    split.add_argument("--window-bars", type=int, default=None, dest="window_bars")
    split.add_argument("--refit-bars", type=int, default=None, dest="refit_bars")
    split.add_argument("--entry-z", type=float, default=2.0, dest="entry_z")
    split.add_argument("--exit-z-tp", type=float, default=0.5, dest="exit_z_tp")
    split.add_argument("--exit-z-sl", type=float, default=3.0, dest="exit_z_sl")
    split.add_argument("--cooldown-bars", type=int, default=None, dest="cooldown_bars")
    split.add_argument("--require-coint", action="store_true", dest="require_coint")
    split.add_argument("--stress", action="store_true")
    split.set_defaults(func=cmd_split)

    panel = sub.add_parser(
        "panel",
        help="Compare all pair combinations: diagnostics + rolling OOS (1d and/or 4h)",
    )
    panel.add_argument("--intervals", nargs="+", default=["1d", "4h"])
    panel.add_argument("--stress", action="store_true")
    panel.set_defaults(func=cmd_panel)

    scan = sub.add_parser("scan", help="Exploratory pair scan (optimistic; use walkforward for OOS)")
    scan.add_argument("--intervals", nargs="+", default=["5m", "15m"])
    scan.set_defaults(func=cmd_scan)

    wf = sub.add_parser("walkforward", help="Walk-forward OOS evaluation with rolling beta")
    wf.add_argument("--pair", default="BTCUSDT/ETHUSDT")
    wf.add_argument("--interval", default="15m")
    wf.add_argument("--stress", action="store_true")
    wf.set_defaults(func=cmd_walkforward)

    roll = sub.add_parser(
        "rolling",
        help="Continuous stitched OOS with fixed params and rolling beta (multi-day)",
    )
    roll.add_argument(
        "--pair",
        default="BTCUSDT/SOLUSDT",
        help="Y/X pair (best OOS in scan: BTC/SOL @ 4h; BNB/SOL @ 1d)",
    )
    roll.add_argument("--interval", default="4h")
    roll.add_argument(
        "--window-bars",
        type=int,
        default=None,
        dest="window_bars",
        help="Trailing estimation window in bars (default: 360 for 4h ~60d, 60 for 1d)",
    )
    roll.add_argument(
        "--refit-bars",
        type=int,
        default=None,
        dest="refit_bars",
        help="Recalibrate beta + spread stats every N bars (default: 42 for 4h ~7d, 7 for 1d)",
    )
    roll.add_argument("--entry-z", type=float, default=2.0, dest="entry_z")
    roll.add_argument("--exit-z-tp", type=float, default=0.5, dest="exit_z_tp")
    roll.add_argument("--exit-z-sl", type=float, default=3.0, dest="exit_z_sl")
    roll.add_argument(
        "--cooldown-bars",
        type=int,
        default=None,
        dest="cooldown_bars",
        help="Cooldown after stop-loss in bars (default: 30 for 4h ~5d, 5 for 1d)",
    )
    roll.add_argument(
        "--require-coint",
        action="store_true",
        dest="require_coint",
        help="Gate entries on a per-window cointegration test (off by default: "
        "BTC/ETH rarely pass the formal test on daily windows, so this would "
        "block all trades)",
    )
    roll.add_argument("--stress", action="store_true")
    roll.set_defaults(func=cmd_rolling)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
