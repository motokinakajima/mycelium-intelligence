#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EPOCHS="${1:-250}"
MODEL="${2:-node_nn_model.nn}"
TEST_RATIO="${3:-0.2}"
CSV="${4:-heuristics/training_data.csv}"

if [[ ! -f "$CSV" ]]; then
  echo "[ERROR] CSV not found: $CSV"
  echo "Usage: bat/train_only.sh [epochs] [model] [test_ratio] [csv]"
  exit 1
fi

echo "[INFO] Building train_nn target..."
cmake --build cmake-build-debug --target train_nn

echo "[INFO] Training model..."
echo "       CSV=$CSV"
echo "       EPOCHS=$EPOCHS"
echo "       MODEL=$MODEL"
echo "       TEST_RATIO=$TEST_RATIO"

./cmake-build-debug/train_nn "$CSV" "$EPOCHS" "$MODEL" "$TEST_RATIO"

echo "[OK] Training finished. Model saved to $MODEL"
