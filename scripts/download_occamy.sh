#!/usr/bin/env bash
set -euo pipefail
# Adapt to your layout. Defaults to the parent of this script.
ROOT="${OCCAMY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_ROOT="$ROOT/model_save"
MODEL_DIR="$MODEL_ROOT/occamy-1.0-NVFP4"
REVISION="c1aa9ae71c9d5d7f906b1575557e273cbdedf6f1"
mkdir -p "$MODEL_DIR" "$MODEL_ROOT/.cache/huggingface" "$MODEL_ROOT/.cache/huggingface/hub"
export HF_HOME="$MODEL_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$MODEL_ROOT/.cache/huggingface/hub"
exec hf download Accio-Lab/occamy-1.0-NVFP4 \
  --revision "$REVISION" \
  --local-dir "$MODEL_DIR"
