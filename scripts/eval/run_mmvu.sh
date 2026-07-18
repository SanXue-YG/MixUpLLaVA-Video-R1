#!/usr/bin/env bash
# Wrapper for upstream MMVU eval.
# Ref: https://github.com/ZhangXJ199/TinyLLaVA-Video-R1/blob/main/scripts/eval/mmvu.sh
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH}"
: "${EVAL_DIR:?set EVAL_DIR}"
TINYLLAVA_REPO="${TINYLLAVA_REPO:?set TINYLLAVA_REPO}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
NUM_FRAME="${NUM_FRAME:-16}"

cd "$TINYLLAVA_REPO"
mkdir -p "$EVAL_DIR/answers"

python -m tinyllava.eval.eval_mmvu \
  --model_path "$MODEL_PATH" \
  --image_folder "$EVAL_DIR" \
  --question_file "$EVAL_DIR/validation.json" \
  --answers_file "$EVAL_DIR/answers/${MODEL_NAME}.jsonl" \
  --temperature 0 \
  --conv_mode qwen2_base \
  --num_frame "$NUM_FRAME" \
  --max_frame "$NUM_FRAME"
