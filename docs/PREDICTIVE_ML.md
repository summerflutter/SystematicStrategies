# Predictive ML (frozen)

Next-candle **direction** classification on 1m klines. Kept for reference and possible use as a **regime filter** — not a standalone trading strategy.

**Status:** Frozen — accuracy does not survive realistic fees at 1m horizons.

## Install

```bash
pip install -e ".[prediction]"
```

## Layout

```
src/strategy/predictive/
├── features.py   # 80+ technical features from OHLCV
├── models.py     # Logistic, RF, XGBoost, GBM, MLP, CNN, LSTM, Transformer
└── train.py      # ModelTrainer, TimeSeriesSplitter, FeatureSelector
```

## Minimal example

```python
from src.data.data_load import load_data
from src.strategy.predictive import (
    prepare_data_for_modeling,
    TimeSeriesSplitter,
    RandomForestModel,
)

df = load_data(symbols="BTCUSDT", start_date="2025-01-01", end_date="2025-06-30", interval="1m")
df_features, feature_cols, target_col = prepare_data_for_modeling(df)

X = df_features[feature_cols].values
y = df_features[target_col].values
X_train, X_val, X_test, y_train, y_val, y_test = TimeSeriesSplitter.train_val_test_split(X, y)

model = RandomForestModel(n_estimators=100)
model.train(X_train, y_train)
print(f"Test accuracy: {(model.predict(X_test) == y_test).mean():.2%}")
```

## Runnable examples

- Script: `scripts/examples_price_prediction.py`
- Notebook: `notebooks/price_forecast_complete.ipynb`

## Models

| Model | Use case |
|-------|----------|
| LogisticRegression | Fast baseline |
| RandomForest / XGBoost / GBM | Best tabular accuracy |
| MLP / CNN / LSTM / Transformer | Sequence experiments (slow, needs GPU for scale) |

Always use `TimeSeriesSplitter` — never shuffle time series.

## If revisiting

- Predict at **15m–1h** horizons, not 1m
- Use model output to **filter** classical signals (meta-labeling), not direct orders
- Compare against `dual_ma` benchmark in [STRATEGY_STATUS.md](../STRATEGY_STATUS.md)
