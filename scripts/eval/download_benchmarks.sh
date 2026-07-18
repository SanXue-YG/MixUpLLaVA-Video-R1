#!/usr/bin/env bash
# Optional: prepare dirs for TinyLLaVA-Video-R1 full benchmarks.
# Data volume is LARGE (~600GB for all four). Prefer training-period eval.
#
# Upstream guide:
#   https://github.com/ZhangXJ199/TinyLLaVA-Video-R1
#
# Usage:
#   export EVAL_ROOT=/content/drive/MyDrive/MixUpLLaVA-video-r1/data/eval
#   bash scripts/eval/download_benchmarks.sh --init-only
#   # then manually download each dataset into the printed folders
#   bash scripts/eval/download_benchmarks.sh --confirm-large

set -euo pipefail

INIT_ONLY=0
CONFIRM=0
for arg in "$@"; do
  case "$arg" in
    --init-only) INIT_ONLY=1 ;;
    --confirm-large) CONFIRM=1 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
  esac
done

EVAL_ROOT="${EVAL_ROOT:-${PROJECT_DIR:-.}/data/eval}"

echo "============================================================"
echo " WARNING: Video-MME + MVBench + MLVU + MMVU ≈ 600GB total."
echo " Phase 5 default path is TRAINING-PERIOD metrics only."
echo " See docs/PHASE5_BENCHMARKS.md"
echo "============================================================"
echo "EVAL_ROOT=$EVAL_ROOT"

mkdir -p \
  "$EVAL_ROOT/Video-MME" \
  "$EVAL_ROOT/MVBench" \
  "$EVAL_ROOT/MLVU" \
  "$EVAL_ROOT/MMVU"

echo "Created layout:"
echo "  $EVAL_ROOT/Video-MME   # put Video-MME here"
echo "  $EVAL_ROOT/MVBench     # put MVBench here"
echo "  $EVAL_ROOT/MLVU"
echo "  $EVAL_ROOT/MMVU"

if [[ "$INIT_ONLY" -eq 1 ]]; then
  echo "(--init-only) Done. Download datasets yourself from their official releases."
  exit 0
fi

if [[ "$CONFIRM" -ne 1 ]]; then
  echo "Refusing to proceed without --confirm-large (no automatic multi-hundred-GB download)."
  echo "This script does NOT auto-fetch datasets (licenses / mirrors vary)."
  echo "After manual download, run: bash scripts/eval/run_videomme.sh  (etc.)"
  exit 1
fi

echo "Confirmed. Please ensure each subdirectory already contains the dataset."
echo "Then use scripts/eval/run_{videomme,mvbench,mlvu,mmvu}.sh with MODEL_PATH set."
