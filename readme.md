# SystematicStrategies

> **Status:** Active | Last updated: 2026-06 | Supersedes Crypto_Quant_Trading (deleted)

Crypto strategy research and paper trading on Binance spot/perp data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[research]"
```

Optional ML stack (frozen module): `pip install -e ".[prediction]"`

## Data

Market data lives in `data/` (gitignored, ~500 MB+). Download after clone:

```bash
# Spot klines
.venv/bin/python scripts/run_download_and_save_data.py

# Perp klines + funding rates
.venv/bin/python scripts/run_download_perp_data.py
```

Load in code: `from src.data.data_load import load_data`

## Quick start

```bash
# Classical walk-forward (gate-passing strategies only)
.venv/bin/python scripts/run_classical_backtest.py --gate-passing-only

# Stat-arb OOS validation
.venv/bin/python scripts/run_stat_arb.py walkforward --pair BTCUSDT/ETHUSDT --interval 15m

# Paper trade signal (no orders)
.venv/bin/python scripts/run_paper_trading.py --dry-run

# Stat-arb paper trade (BTC/SOL + XRP/ADA @ 4h) — see .env.example for demo API keys
cp .env.example .env   # keys via www.binance.com → Futures → Demo/Mock Trading → API Management
.venv/bin/python scripts/run_stat_arb_paper_trading.py --dry-run
.venv/bin/python scripts/run_stat_arb_paper_trading.py
```

Outputs go to `results/*.csv`.

## Documentation

| File | Purpose |
|------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Folder layout, data flow, scripts, cost model |
| [STRATEGY_STATUS.md](STRATEGY_STATUS.md) | Active vs frozen strategies, latest backtest gates |
| [docs/PREDICTIVE_ML.md](docs/PREDICTIVE_ML.md) | Frozen ML direction-forecast module (reference only) |
| [docs/MIGRATION.md](docs/MIGRATION.md) | Consolidated legacy repos map |
| [docs/REPO_INVENTORY.md](docs/REPO_INVENTORY.md) | Local/GitHub repo audit and tags |
| [docs/DATA_PATHS.md](docs/DATA_PATHS.md) | External dataset locations (`~/Data/`) |
| [legacy/](legacy/) | Migrated forecast code (PriceForecast, orderbook, etc.) |

## Current focus

**Live candidate:** `dual_ma` on BTCUSDT 1h (OOS Sharpe 1.07, stress 0.41). See [STRATEGY_STATUS.md](STRATEGY_STATUS.md).
