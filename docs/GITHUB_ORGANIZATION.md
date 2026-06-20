# GitHub organization

## Priority repos (push after `gh auth login`)

```bash
~/Documents/GitHub/SystematicStrategies/scripts/setup_priority_repos.sh
```

This script will:

1. Rename `aSystematicStrategies` → `SystematicStrategies` (and the other three `a*` folders)
2. Delete GitHub repos: `Crypto_Quant_Trading`, `BTC_price_prediction`
3. Rename `QuantTrading` → `SystematicStrategies` on GitHub (if it exists)
4. Create/push: `SystematicStrategies`, `ElectronicMarketMaking`, `HighFreqPriceForecast`, `ReadingNotes`

## Recommended pinned repos

1. [SystematicStrategies](https://github.com/summerflutter/SystematicStrategies)
2. [HighFreqPriceForecast](https://github.com/summerflutter/HighFreqPriceForecast)
3. [ElectronicMarketMaking](https://github.com/summerflutter/ElectronicMarketMaking)

Pin at: https://github.com/summerflutter → Customize pins

## Local layout (after script)

```
~/Documents/GitHub/
├── SystematicStrategies/
├── ElectronicMarketMaking/
├── HighFreqPriceForecast/
├── ReadingNotes/
├── z_old/                    # local archive only (not on GitHub)
└── … forks (Time-LLM, etc.)
```

## Obsolete (deleted from GitHub)

- `Crypto_Quant_Trading` → superseded by SystematicStrategies
- `BTC_price_prediction` → moved to z_old locally
