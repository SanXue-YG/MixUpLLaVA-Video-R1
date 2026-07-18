#!/usr/bin/env bash
# Wrapper for upstream MLVU eval.
# Ref: https://github.com/ZhangXJ199/TinyLLaVA-Video-R1/blob/main/scripts/eval/mlvu.sh
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH}"
: "${EVAL_DIR:?set EVAL_DIR}"
TINYLLAVA_REPO="${TINYLLAVA_REPO:?set TINYLLAVA_REPO}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
NUM_FRAME="${NUM_FRAME:-16}"

cd "$TINYLLAVA_REPO"
mkdir -p "$EVAL_DIR/answers"

python -m tinyllava.eval.eval_mlvu \
  --model-path "$MODEL_PATH" \
  --video-folder "$EVAL_DIR/video" \
  --question-file "$EVAL_DIR/json" \
  --answers-file "$EVAL_DIR/answers/${MODEL_NAME}.jsonl" \
  --temperature 0 \
  --conv-mode qwen2_base \
  --num_frame "$NUM_FRAME" \
  --max_frame "$NUM_FRAME"
