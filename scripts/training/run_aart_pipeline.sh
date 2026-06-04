#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "$ROOT"

"$PYTHON" scripts/data_processing/stage_data_aart.py
"$PYTHON" scripts/training/train_aart.py --config configs/aart/base.yaml
