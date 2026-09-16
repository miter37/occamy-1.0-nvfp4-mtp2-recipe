#!/usr/bin/env bash
set -euo pipefail
# Adapt to your layout. Defaults to the parent of this script.
ROOT="${OCCAMY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPARK="$ROOT/spark_vllm"
MODEL_DIR="$ROOT/model_save/occamy-1.0-NVFP4"
HF_HOME="$ROOT/model_save/.cache/huggingface"
export HF_HOME
LAUNCH_SCRIPT="$ROOT/configs/occamy-1.0-nvfp4-vllm.sh"
CONTAINER_NAME="self_vllm_occamy"

[[ -x "$SPARK/launch-cluster.sh" ]] || { echo "spark_vllm launcher is missing" >&2; exit 1; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "Occamy model is not downloaded" >&2; exit 1; }
[[ -f "$LAUNCH_SCRIPT" ]] || { echo "Occamy launch script is missing" >&2; exit 1; }
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
  -v "$MODEL_DIR:/models/occamy-1.0-NVFP4:ro" \
  --launch-script "$LAUNCH_SCRIPT" \
  -d
