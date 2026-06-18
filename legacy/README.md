# Legacy forecast projects

Code migrated from standalone repos (2026-06-18). See [docs/MIGRATION.md](../docs/MIGRATION.md).

| Directory | Source repo | Description |
|-----------|-------------|-------------|
| `price_forecast/` | summerflutter/PriceForecast | ML price forecasting pipeline |
| `orderbook_forecast/` | summerflutter/orderbook_forecast | FX ECN multi-venue order book pipeline |
| `flow_rate_estimation/` | summerflutter/flow_rate_estimation | Flow rate estimation notebook + pipeline |
| `intraday_volume_forecast/` | summerflutter/intradayVolumeForecast | Intraday volume Kalman model |

These are reference implementations, not part of the active trading stack. Active strategies live in `src/strategy/`.
