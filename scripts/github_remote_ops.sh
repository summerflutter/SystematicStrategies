#!/usr/bin/env bash
# GitHub remote operations — requires: gh auth login
set -euo pipefail

GH="${GH:-/usr/local/bin/gh}"
OWNER="${GITHUB_OWNER:-summerflutter}"

if ! "$GH" auth status &>/dev/null; then
  echo "Run: gh auth login"
  exit 1
fi

echo "=== Create QuantTrading repo (if missing) ==="
if ! "$GH" repo view "$OWNER/QuantTrading" &>/dev/null; then
  "$GH" repo create "$OWNER/QuantTrading" --public \
    --description "Crypto strategy research and paper trading" \
    --source /Users/qinli/Documents/GitHub/QuantTrading --remote origin --push
else
  echo "QuantTrading repo exists; push manually if needed:"
  echo "  cd ~/Documents/GitHub/QuantTrading && git push -u origin main"
fi

echo "=== Set topics ==="
"$GH" repo edit "$OWNER/QuantTrading" \
  --add-topic quantitative-finance --add-topic crypto --add-topic backtesting \
  --add-topic stat-arb --add-topic paper-trading

"$GH" repo edit "$OWNER/deepLOB" \
  --add-topic deep-learning --add-topic orderbook --add-topic quantitative-finance

"$GH" repo edit "$OWNER/Time-LLM" \
  --add-topic time-series --add-topic llm --add-topic forecasting

"$GH" repo edit "$OWNER/llmtime" \
  --add-topic time-series --add-topic llm --add-topic forecasting

echo "=== Archive superseded repos ==="
for repo in Crypto_Quant_Trading PriceForecast orderbook_forecast flow_rate_estimation intradayVolumeForecast; do
  if "$GH" repo view "$OWNER/$repo" &>/dev/null; then
    "$GH" repo archive "$OWNER/$repo" --yes
    echo "Archived $repo"
  fi
done

echo "=== Delete empty / stale repos ==="
for repo in Miscellaneous rust_tutorial LSTM-For-Stock-Market-Prediction; do
  if "$GH" repo view "$OWNER/$repo" &>/dev/null; then
    "$GH" repo delete "$OWNER/$repo" --yes
    echo "Deleted $repo"
  fi
done

echo "=== Pin repos on profile ==="
"$GH" api -X PUT "user/starred/$OWNER/QuantTrading" 2>/dev/null || true
echo "Pin manually at https://github.com/$OWNER — recommended: QuantTrading, deepLOB, Time-LLM"

echo "Done."
