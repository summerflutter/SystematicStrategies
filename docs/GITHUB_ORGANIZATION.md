# GitHub organization

Run after `gh auth login`:

```bash
~/Documents/GitHub/QuantTrading/scripts/github_remote_ops.sh
```

## Recommended pinned repos

1. [QuantTrading](https://github.com/summerflutter/QuantTrading) — main research repo
2. [deepLOB](https://github.com/summerflutter/deepLOB) — order book deep learning
3. [Time-LLM](https://github.com/summerflutter/Time-LLM) — time-series LLM experiments

Pin at: https://github.com/summerflutter → Customize pins

## Topics (applied by script)

| Repo | Topics |
|------|--------|
| QuantTrading | `quantitative-finance`, `crypto`, `backtesting`, `stat-arb`, `paper-trading` |
| deepLOB | `deep-learning`, `orderbook`, `quantitative-finance` |
| Time-LLM | `time-series`, `llm`, `forecasting` |
| llmtime | `time-series`, `llm`, `forecasting` |

## Repos to archive (superseded)

- Crypto_Quant_Trading
- PriceForecast
- orderbook_forecast
- flow_rate_estimation
- intradayVolumeForecast

## Repos to delete (empty/stale)

- Miscellaneous (no commits)
- rust_tutorial (no commits)
- LSTM-For-Stock-Market-Prediction (2019 fork, superseded by Stock-Price-Prediction-Using-LSTM)

## Push QuantTrading (first time)

```bash
cd ~/Documents/GitHub/QuantTrading
git push -u origin main
# Or let the script create the repo: scripts/github_remote_ops.sh
```
