#!/usr/bin/env bash
# Fix XGBoost import on macOS when libomp.dylib is missing or wrong architecture.
# Run from repo root: ./scripts/fix_xgboost_macos.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS only."
  exit 0
fi

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $PYTHON. Set PYTHON=... or create .venv first."
  exit 1
fi

if "$PYTHON" -c "import xgboost" 2>/dev/null; then
  echo "XGBoost already imports successfully."
  exit 0
fi

XGB_LIB="$("$PYTHON" -c "import xgboost, pathlib; print(pathlib.Path(xgboost.__file__).parent / 'lib')" 2>/dev/null || true)"
if [[ -z "$XGB_LIB" || ! -f "$XGB_LIB/libxgboost.dylib" ]]; then
  echo "Installing xgboost into the active environment..."
  "$PYTHON" -m pip install -q 'xgboost>=1.7'
  XGB_LIB="$("$PYTHON" -c "import xgboost, pathlib; print(pathlib.Path(xgboost.__file__).parent / 'lib')")"
fi

pick_libomp() {
  local candidate
  for candidate in \
    /opt/homebrew/opt/libomp/lib/libomp.dylib \
    /opt/anaconda3/lib/libomp.dylib \
    /opt/anaconda3/pkgs/llvm-openmp-*/lib/libomp.dylib; do
    if [[ -f "$candidate" ]] && file "$candidate" | grep -q 'arm64'; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

LIBOMP="$(pick_libomp || true)"
if [[ -z "$LIBOMP" ]]; then
  cat <<'EOF'
Could not find an arm64 libomp.dylib.

Install OpenMP for Apple Silicon, then re-run this script:
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  /opt/homebrew/bin/brew install libomp

Or use a conda environment where xgboost already works.
EOF
  exit 1
fi

echo "Using libomp from: $LIBOMP"
cp "$LIBOMP" "$XGB_LIB/libomp.dylib"
install_name_tool -change @rpath/libomp.dylib @loader_path/libomp.dylib "$XGB_LIB/libxgboost.dylib"

"$PYTHON" -c "import xgboost; print('XGBoost OK:', xgboost.__version__)"
