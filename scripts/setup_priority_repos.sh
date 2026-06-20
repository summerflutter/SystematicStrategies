#!/usr/bin/env bash
# Rename a-prefixed folders, push 4 priority repos to GitHub, delete obsolete repos.
# Requires: gh auth login
set -euo pipefail

GH="${GH:-/usr/local/bin/gh}"
OWNER="${GITHUB_OWNER:-summerflutter}"
ROOT="${GITHUB_ROOT:-$HOME/Documents/GitHub}"

PRIORITY=(
  SystematicStrategies
  ElectronicMarketMaking
  HighFreqPriceForecast
  ReadingNotes
)

DELETE_REPOS=(
  Crypto_Quant_Trading
  BTC_price_prediction
)

if ! "$GH" auth status &>/dev/null; then
  echo "Run: gh auth login"
  exit 1
fi

cd "$ROOT"

echo "=== Step 1: Remove 'a' prefix from priority folders ==="
for name in "${PRIORITY[@]}"; do
  src="$ROOT/a$name"
  dst="$ROOT/$name"
  if [[ -d "$src" ]]; then
    if [[ -d "$dst" && "$src" -ef "$dst" ]]; then
      echo "  $name: already at final path"
    elif [[ -d "$dst" ]]; then
      echo "  ERROR: both a$name and $name exist — merge manually, then re-run"
      exit 1
    else
      mv "$src" "$dst"
      echo "  renamed a$name -> $name"
    fi
  elif [[ -d "$dst" ]]; then
    echo "  $name: ok"
  else
    echo "  WARNING: missing $name (skipped for rename)"
  fi
done

echo ""
echo "=== Step 2: Delete obsolete GitHub repos ==="
for repo in "${DELETE_REPOS[@]}"; do
  if "$GH" repo view "$OWNER/$repo" &>/dev/null; then
    "$GH" repo delete "$OWNER/$repo" --yes
    echo "  deleted $OWNER/$repo"
  else
    echo "  $repo: not on GitHub (skip)"
  fi
done

echo ""
echo "=== Step 3: Migrate QuantTrading -> SystematicStrategies on GitHub ==="
if "$GH" repo view "$OWNER/QuantTrading" &>/dev/null; then
  if "$GH" repo view "$OWNER/SystematicStrategies" &>/dev/null; then
    echo "  both QuantTrading and SystematicStrategies exist — keeping SystematicStrategies"
  else
    "$GH" repo rename SystematicStrategies --repo "$OWNER/QuantTrading"
    echo "  renamed GitHub repo QuantTrading -> SystematicStrategies"
  fi
fi

desc_for() {
  case "$1" in
    SystematicStrategies) echo "Systematic crypto strategies: backtest, stat-arb, paper trading" ;;
    ElectronicMarketMaking) echo "Electronic market making: volume, toxic flow, hedging" ;;
    HighFreqPriceForecast) echo "High-frequency price forecast from order book / tick data" ;;
    ReadingNotes) echo "Reading notes: quant finance, Python, papers" ;;
    *) echo "$1 research project" ;;
  esac
}

echo ""
echo "=== Step 4: Push priority repos ==="
for name in "${PRIORITY[@]}"; do
  dir="$ROOT/$name"
  if [[ ! -d "$dir" ]]; then
    echo "  SKIP $name — folder not found at $dir"
    continue
  fi

  echo "--- $name ---"
  cd "$dir"

  if [[ ! -d .git ]]; then
    git init
    git branch -M main
    [[ -f .gitignore ]] || cat > .gitignore <<'EOF'
.DS_Store
.idea/
.vscode/
__pycache__/
*.py[cod]
.venv/
venv/
.env
.env.*
!.env.example
.ipynb_checkpoints/
EOF
    git add -A
    git -c commit.template= commit -m "Initial commit: $name" || true
  fi

  url="https://github.com/$OWNER/$name.git"
  if git remote get-url origin &>/dev/null; then
    git remote set-url origin "$url"
  else
    git remote add origin "$url"
  fi

  if ! "$GH" repo view "$OWNER/$name" &>/dev/null; then
    "$GH" repo create "$OWNER/$name" --public \
      --description "$(desc_for "$name")" \
      --source "$dir" --remote origin --push
    echo "  created and pushed $name"
  else
    git add -A
    if ! git diff --staged --quiet 2>/dev/null; then
      git -c commit.template= commit -m "Sync local $name"
    fi
    git push -u origin main
    echo "  pushed $name"
  fi
done

echo ""
echo "=== Step 5: Set topics ==="
"$GH" repo edit "$OWNER/SystematicStrategies" \
  --add-topic quantitative-finance --add-topic crypto --add-topic backtesting \
  --add-topic stat-arb --add-topic paper-trading 2>/dev/null || true
"$GH" repo edit "$OWNER/ElectronicMarketMaking" \
  --add-topic market-making --add-topic quantitative-finance --add-topic microstructure 2>/dev/null || true
"$GH" repo edit "$OWNER/HighFreqPriceForecast" \
  --add-topic orderbook --add-topic high-frequency --add-topic forecasting 2>/dev/null || true
"$GH" repo edit "$OWNER/ReadingNotes" \
  --add-topic notes --add-topic quantitative-finance 2>/dev/null || true

echo ""
echo "Done. Pin repos at: https://github.com/$OWNER"
echo "Local paths (no 'a' prefix):"
for name in "${PRIORITY[@]}"; do
  echo "  $ROOT/$name"
done
