"""Paper trading for rolling stat-arb spread strategies (two-leg perp)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from src.strategy.stat_arb.evaluation import _is_cointegrated, _spread_stats
from src.trading.paper_trader import BinanceTestnetClient, OrderResult

DEFAULT_STATE_DIR = Path(__file__).resolve().parents[2] / "results" / "paper_stat_arb"


@dataclass
class StatArbPaperConfig:
    name: str
    y_symbol: str
    x_symbol: str
    interval: str = "4h"
    window_bars: int = 360
    refit_bars: int = 42
    entry_z: float = 2.0
    exit_z_tp: float = 0.5
    exit_z_sl: float = 3.0
    cooldown_bars: int = 30
    require_cointegration: bool = False
    init_nav: float = 5_000.0
    leverage_long: float = 1.0
    leverage_short: float = 1.0
    max_margin_utilization: float = 1.0
    market: Literal["spot", "futures"] = "futures"
    leverage: int = 1


@dataclass
class StatArbPaperState:
    position_state: int = 0  # +1 long spread, -1 short, 0 flat
    units_Y: float = 0.0
    units_X: float = 0.0
    beta: float = 1.0
    nav: float = 5_000.0
    cooldown_count: int = 0
    can_enter: bool = True
    bars_seen: int = 0
    long_entry: float = 0.0
    short_entry: float = 0.0
    long_exit_tp: float = 0.0
    long_exit_sl: float = 0.0
    short_exit_tp: float = 0.0
    short_exit_sl: float = 0.0
    last_bar_time: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_nav: float) -> "StatArbPaperState":
        state = cls()
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)
        if state.nav <= 0:
            state.nav = default_nav
        return state


# Best OOS configs from research (fixed params, 4h).
STAT_ARB_STRATEGIES: Dict[str, StatArbPaperConfig] = {
    "btc_sol_4h": StatArbPaperConfig(
        name="btc_sol_4h",
        y_symbol="BTCUSDT",
        x_symbol="SOLUSDT",
    ),
    "xrp_ada_4h": StatArbPaperConfig(
        name="xrp_ada_4h",
        y_symbol="XRPUSDT",
        x_symbol="ADAUSDT",
    ),
}


class StatArbPaperTrader:
    """Execute one stat-arb strategy on Binance testnet futures."""

    def __init__(
        self,
        config: StatArbPaperConfig,
        client: BinanceTestnetClient | None = None,
        state_dir: Path = DEFAULT_STATE_DIR,
    ):
        self.config = config
        self.client = client or BinanceTestnetClient(market=config.market)
        self.state_dir = state_dir
        self.state_path = state_dir / f"{config.name}_state.json"

    def load_state(self) -> StatArbPaperState:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            return StatArbPaperState.from_dict(data, self.config.init_nav)
        return StatArbPaperState(nav=self.config.init_nav)

    def save_state(self, state: StatArbPaperState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state.to_dict(), indent=2))

    def _maybe_refit(
        self,
        state: StatArbPaperState,
        y_hist: pd.Series,
        x_hist: pd.Series,
        *,
        bar_index: int,
    ) -> None:
        cfg = self.config
        if bar_index < cfg.window_bars:
            return
        scheduled = bar_index == cfg.window_bars or (bar_index - cfg.window_bars) % cfg.refit_bars == 0
        if scheduled and state.position_state == 0:
            est_y = y_hist.iloc[-cfg.window_bars :]
            est_x = x_hist.iloc[-cfg.window_bars :]
            beta, s_mean, s_std = _spread_stats(est_y, est_x)
            if s_std <= 0:
                s_std = 1.0
            state.beta = float(beta)
            state.can_enter = (not cfg.require_cointegration) or _is_cointegrated(
                est_y, est_x
            )
            state.long_entry = (-cfg.entry_z) * s_std + s_mean
            state.short_entry = cfg.entry_z * s_std + s_mean
            state.long_exit_tp = (-cfg.exit_z_tp) * s_std + s_mean
            state.long_exit_sl = (-cfg.exit_z_sl) * s_std + s_mean
            state.short_exit_tp = cfg.exit_z_tp * s_std + s_mean
            state.short_exit_sl = cfg.exit_z_sl * s_std + s_mean

    def step_on_bar(
        self,
        state: StatArbPaperState,
        y_hist: pd.Series,
        x_hist: pd.Series,
        *,
        bar_time: pd.Timestamp,
        bar_index: int,
    ) -> tuple[StatArbPaperState, str]:
        """Advance state machine one bar; return updated state and action label."""
        if state.last_bar_time is not None and str(bar_time) <= state.last_bar_time:
            return state, "skip_duplicate_bar"

        cfg = self.config
        state.bars_seen = bar_index + 1
        self._maybe_refit(state, y_hist, x_hist, bar_index=bar_index)

        Y_t = float(y_hist.iloc[-1])
        X_t = float(x_hist.iloc[-1])
        spread_t = Y_t - state.beta * X_t

        action = "hold"
        if spread_t >= state.short_exit_sl or spread_t <= state.long_exit_sl:
            state.cooldown_count = cfg.cooldown_bars

        if state.position_state != 0:
            if state.position_state == 1:
                exit_flag = spread_t >= state.long_exit_tp or spread_t <= state.long_exit_sl
            else:
                exit_flag = spread_t <= state.short_exit_tp or spread_t >= state.short_exit_sl
            if exit_flag:
                state.position_state = 0
                state.units_Y = 0.0
                state.units_X = 0.0
                action = "exit"
        elif state.cooldown_count == 0 and state.can_enter and bar_index >= cfg.window_bars:
            long_flag = spread_t < state.long_entry and spread_t > state.long_exit_sl
            short_flag = spread_t > state.short_entry and spread_t < state.short_exit_sl
            if long_flag or short_flag:
                if long_flag:
                    margin_per_unit = Y_t / cfg.leverage_long + state.beta * X_t / cfg.leverage_short
                else:
                    margin_per_unit = Y_t / cfg.leverage_short + state.beta * X_t / cfg.leverage_long
                max_margin = state.nav * cfg.max_margin_utilization
                units_spread = (
                    max_margin / margin_per_unit if margin_per_unit > 0 and max_margin > 0 else 0.0
                )
                if units_spread > 0:
                    if long_flag:
                        state.position_state = 1
                        state.units_Y = units_spread
                        state.units_X = -state.beta * units_spread
                        action = "enter_long_spread"
                    else:
                        state.position_state = -1
                        state.units_Y = -units_spread
                        state.units_X = state.beta * units_spread
                        action = "enter_short_spread"
        elif state.cooldown_count > 0:
            state.cooldown_count -= 1

        state.last_bar_time = str(bar_time)
        return state, action

    def _round_qty(self, symbol: str, qty: float) -> float:
        if qty <= 0:
            return 0.0
        # Conservative defaults; testnet accepts coarse sizes for majors/alts.
        if symbol == "BTCUSDT":
            return round(qty, 3)
        if symbol in ("ETHUSDT", "BNBUSDT", "SOLUSDT"):
            return round(qty, 2)
        return round(qty, 1)

    def _place_leg_delta(
        self,
        symbol: str,
        current_qty: float,
        target_qty: float,
    ) -> List[OrderResult]:
        delta = target_qty - current_qty
        qty = self._round_qty(symbol, abs(delta))
        if qty <= 0:
            return []
        side = "BUY" if delta > 0 else "SELL"
        return [self.client.place_market_order(symbol, side, qty)]

    def execute_targets(
        self,
        state: StatArbPaperState,
        *,
        current_units_y: float | None = None,
        current_units_x: float | None = None,
    ) -> List[OrderResult]:
        """Move testnet positions toward state.units_Y / state.units_X."""
        cfg = self.config
        cur_y = 0.0 if current_units_y is None else current_units_y
        cur_x = 0.0 if current_units_x is None else current_units_x
        orders: List[OrderResult] = []
        orders.extend(self._place_leg_delta(cfg.y_symbol, cur_y, state.units_Y))
        orders.extend(self._place_leg_delta(cfg.x_symbol, cur_x, state.units_X))
        return orders

    def run_once(
        self,
        y: pd.Series,
        x: pd.Series,
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        aligned = pd.DataFrame({"y": y, "x": x}).dropna()
        if len(aligned) < self.config.window_bars + 1:
            raise ValueError(
                f"Need at least {self.config.window_bars + 1} bars; got {len(aligned)}"
            )

        state = self.load_state()
        was_fresh = state.last_bar_time is None
        cfg = self.config
        start_idx = cfg.window_bars
        if state.last_bar_time is not None:
            last_ts = pd.Timestamp(state.last_bar_time)
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            start_idx = max(start_idx, int(aligned.index.searchsorted(last_ts, side="right")))

        if start_idx >= len(aligned):
            bar_time = aligned.index[-1]
            spread = float(aligned["y"].iloc[-1] - state.beta * aligned["x"].iloc[-1])
            return {
                "strategy": cfg.name,
                "pair": f"{cfg.y_symbol}/{cfg.x_symbol}",
                "interval": cfg.interval,
                "bar_time": str(bar_time),
                "action": "skip_already_current",
                "position_state": state.position_state,
                "beta": state.beta,
                "spread": spread,
                "target_units_Y": state.units_Y,
                "target_units_X": state.units_X,
                "dry_run": dry_run,
                "orders": [],
            }

        action = "hold"
        for i in range(start_idx, len(aligned)):
            sub = aligned.iloc[: i + 1]
            state, bar_action = self.step_on_bar(
                state,
                sub["y"],
                sub["x"],
                bar_time=aligned.index[i],
                bar_index=i,
            )
            action = bar_action

        bar_time = aligned.index[-1]

        spread = float(aligned["y"].iloc[-1] - state.beta * aligned["x"].iloc[-1])
        result: Dict[str, Any] = {
            "strategy": self.config.name,
            "pair": f"{self.config.y_symbol}/{self.config.x_symbol}",
            "interval": self.config.interval,
            "bar_time": str(bar_time),
            "action": action,
            "position_state": state.position_state,
            "beta": state.beta,
            "spread": spread,
            "target_units_Y": state.units_Y,
            "target_units_X": state.units_X,
            "long_entry": state.long_entry,
            "short_entry": state.short_entry,
            "cooldown_count": state.cooldown_count,
            "dry_run": dry_run,
            "orders": [],
        }

        if not dry_run:
            if self.config.market == "futures" and self.config.leverage > 1:
                self.client.set_leverage(self.config.y_symbol, self.config.leverage)
                self.client.set_leverage(self.config.x_symbol, self.config.leverage)
            trade_now = action in ("enter_long_spread", "enter_short_spread", "exit") or (
                was_fresh and state.position_state != 0
            )
            if trade_now:
                orders = self.execute_targets(state)
                result["orders"] = [
                    {
                        "symbol": o.symbol,
                        "side": o.side,
                        "quantity": o.quantity,
                        "status": o.status,
                        "order_id": o.order_id,
                    }
                    for o in orders
                ]

        self.save_state(state)
        return result
