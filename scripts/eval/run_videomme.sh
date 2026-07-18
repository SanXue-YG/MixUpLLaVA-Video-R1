#!/usr/bin/env bash
# Wrapper for upstream Video-MME eval (TinyLLaVA-Video-R1).
# Requires dataset under $EVAL_DIR (see download_benchmarks.sh --init-only).
#
# Env:
#   MODEL_PATH, MODEL_NAME, EVAL_DIR, TINYLLAVA_REPO
#   DURATION=short|medium|long (default long)
#   NUM_FRAME=16

set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH}"
: "${EVAL_DIR:?set EVAL_DIR}"
TINYLLAVA_REPO="${TINYLLAVA_REPO:?set TINYLLAVA_REPO}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
DURATION="${DURATION:-long}"
NUM_FRAME="${NUM_FRAME:-16}"

cd "$TINYLLAVA_REPO"
mkdir -p "$EVAL_DIR/answers"

python -m tinyllava.eval.eval_videomme \
  --model-path "$MODEL_PATH" \
  --question-file "$EVAL_DIR/videomme/test-00000-of-00001.parquet" \
  --image-folder "$EVAL_DIR/data" \
  --answers-file "$EVAL_DIR/answers/${MODEL_NAME}.jsonl" \
  --temperature 0 \
  --conv-mode qwen2_base \
  --duration "$DURATION" \
  --num_frame "$NUM_FRAME" \
  --max_frame "$NUM_FRAME"
