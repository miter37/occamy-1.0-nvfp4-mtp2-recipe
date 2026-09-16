#!/usr/bin/env bash
set -euo pipefail
# Adapt to your layout. Defaults to the parent of this script.
ROOT="${OCCAMY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPARK="$ROOT/spark_vllm"
MODEL_ROOT="$ROOT/model_save"
ASSEMBLED_DIR="$MODEL_ROOT/occamy-1.0-NVFP4-with-MTP"
HF_HOME="$MODEL_ROOT/.cache/huggingface"
export HF_HOME
LAUNCH_SCRIPT="$ROOT/configs/occamy-1.0-nvfp4-mtp1-smoke-vllm.sh"
CONTAINER_NAME="self_vllm_occamy"

[[ -x "$SPARK/launch-cluster.sh" ]] || { echo "spark_vllm launcher is missing" >&2; exit 1; }
[[ -f "$ASSEMBLED_DIR/config.json" ]] || { echo "assembled MTP model is missing" >&2; exit 1; }
[[ -f "$LAUNCH_SCRIPT" ]] || { echo "MTP1 launch script is missing" >&2; exit 1; }
if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "Container already exists: $CONTAINER_NAME" >&2
  exit 2
fi
exec "$SPARK/launch-cluster.sh" \
  --solo \
  --no-cache-dirs \
  -t vllm-node \
  --name "$CONTAINER_NAME" \
  -p 8000:8000 \
  -v "$MODEL_ROOT:/models:ro" \
  --launch-script "$LAUNCH_SCRIPT" \
  -d
