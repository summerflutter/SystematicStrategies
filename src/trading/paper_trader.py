"""Binance testnet paper trading execution layer."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlencode

import requests

TESTNET_SPOT = "https://testnet.binance.vision"
# Binance deprecated the testnet.binancefuture.com web UI; futures demo keys use demo-fapi.
DEMO_FUTURES = "https://demo-fapi.binance.com"


@dataclass
class OrderResult:
    symbol: str
    side: str
    quantity: float
    order_id: Optional[int]
    status: str
    raw: Dict[str, Any]


class BinanceTestnetClient:
    """
    Minimal Binance demo/testnet REST client for paper trading.

    Futures uses Binance Demo Trading (https://demo.binance.com) via demo-fapi.
    Spot still uses the legacy spot testnet (testnet.binance.vision).

    Set environment variables:
      BINANCE_TESTNET_API_KEY
      BINANCE_TESTNET_API_SECRET
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        market: Literal["spot", "futures"] = "futures",
    ):
        self.api_key = api_key or os.environ.get("BINANCE_TESTNET_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("BINANCE_TESTNET_API_SECRET", "")
        self.market = market
        self.base_url = DEMO_FUTURES if market == "futures" else TESTNET_SPOT

    def _sign(self, params: Dict[str, Any]) -> str:
        query = urlencode(params)
        return hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        params: Dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        params = dict(params or {})
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign(params)

        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def ping(self) -> bool:
        path = "/fapi/v1/ping" if self.market == "futures" else "/api/v3/ping"
        self._request("GET", path)
        return True

    def get_account(self) -> Dict[str, Any]:
        path = "/fapi/v2/account" if self.market == "futures" else "/api/v3/account"
        return self._request("GET", path, signed=True)

    def get_price(self, symbol: str) -> float:
        path = "/fapi/v1/ticker/price" if self.market == "futures" else "/api/v3/ticker/price"
        data = self._request("GET", path, params={"symbol": symbol})
        return float(data["price"])

    def place_market_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
    ) -> OrderResult:
        path = "/fapi/v1/order" if self.market == "futures" else "/api/v3/order"
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        }
        data = self._request("POST", path, params=params, signed=True)
        return OrderResult(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_id=data.get("orderId"),
            status=data.get("status", "UNKNOWN"),
            raw=data,
        )

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        if self.market != "futures":
            raise ValueError("Leverage only applies to futures")
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": leverage},
            signed=True,
        )


@dataclass
class PaperTradingConfig:
    symbol: str = "BTCUSDT"
    strategy: str = "dual_ma"
    lookback: int = 24
    interval: str = "1h"
    target_vol: float = 0.15
    vol_window: int = 20
    max_leverage: float = 1.0
    target_exposure: float = 0.5
    quantity: float | None = None
    market: Literal["spot", "futures"] = "futures"
    leverage: int = 1
    init_nav: float = 10_000.0


class PaperTrader:
    """
    Execute the best OOS classical strategy signal on Binance testnet.

    Reads the latest signal from price data and places market orders to
    align testnet position with target exposure.
    """

    def __init__(self, config: PaperTradingConfig, client: BinanceTestnetClient | None = None):
        self.config = config
        self.client = client or BinanceTestnetClient(market=config.market)

    def compute_signal(self, close) -> float:
        from src.strategy.classical.momentum import ts_momentum_signal

        if self.config.strategy == "ts_momentum":
            sig = ts_momentum_signal(close, lookback=self.config.lookback)
            return float(sig.iloc[-1])
        if self.config.strategy == "dual_ma":
            from src.strategy.classical.momentum import dual_ma_crossover_signal

            sig = dual_ma_crossover_signal(close, fast=20, slow=60)
            return float(sig.iloc[-1])
        raise ValueError(f"Unknown strategy: {self.config.strategy}")

    def compute_target_exposure(self, close) -> float:
        """Vol-targeted exposure aligned with classical backtest."""
        from src.backtest.directional import vol_target_position
        from src.backtest.evaluation import bars_per_year_for_interval

        if self.config.strategy == "ts_momentum":
            from src.strategy.classical.momentum import ts_momentum_signal
            raw_signal = ts_momentum_signal(close, lookback=self.config.lookback)
        elif self.config.strategy == "dual_ma":
            from src.strategy.classical.momentum import dual_ma_crossover_signal
            raw_signal = dual_ma_crossover_signal(close, fast=20, slow=60)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

        returns = close.pct_change().fillna(0.0)
        exposure = vol_target_position(
            raw_signal,
            returns,
            target_vol=self.config.target_vol,
            vol_window=self.config.vol_window,
            max_leverage=self.config.max_leverage,
            bars_per_year=bars_per_year_for_interval(self.config.interval),
        )
        return float(exposure.iloc[-1])

    def compute_order_quantity(self, close, price: float) -> float:
        """Size order to match backtest vol-target notional."""
        if self.config.quantity is not None:
            return self.config.quantity

        exposure = abs(self.compute_target_exposure(close))
        if exposure < 1e-6:
            return 0.0

        notional = self.config.init_nav * exposure
        qty = notional / price
        return round(qty, 3)

    def rebalance(self, close, signal: float) -> Optional[OrderResult]:
        """Place order to move toward target exposure. signal in [-1, 1]."""
        price = self.client.get_price(self.config.symbol)
        quantity = self.compute_order_quantity(close, price)
        if quantity <= 0:
            return None

        if signal > 0:
            return self.client.place_market_order(
                self.config.symbol, "BUY", quantity
            )
        if signal < 0:
            return self.client.place_market_order(
                self.config.symbol, "SELL", quantity
            )
        return None

    def run_once(self, close) -> Dict[str, Any]:
        if self.config.market == "futures" and self.config.leverage > 1:
            self.client.set_leverage(self.config.symbol, self.config.leverage)

        signal = self.compute_signal(close)
        target_exposure = self.compute_target_exposure(close)
        order = self.rebalance(close, signal)
        price = self.client.get_price(self.config.symbol)

        return {
            "symbol": self.config.symbol,
            "strategy": self.config.strategy,
            "signal": signal,
            "target_exposure": target_exposure,
            "order_quantity": self.compute_order_quantity(close, price),
            "price": price,
            "order": order,
        }
