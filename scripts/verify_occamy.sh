#!/usr/bin/env bash
set -euo pipefail
# Adapt to your layout. Defaults to the parent of this script.
ROOT="${OCCAMY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="$ROOT/model_save/occamy-1.0-NVFP4"
LAUNCH="$ROOT/configs/occamy-1.0-nvfp4-vllm.sh"
[[ -f "$MODEL_DIR/config.json" ]] || { echo 'FAIL model config missing'; exit 1; }
count=$(find "$MODEL_DIR" -maxdepth 1 -name 'model-*-of-*.safetensors' -type f | wc -l)
[[ "$count" -eq 5 ]] || { echo "FAIL safetensors shards=$count expected=5"; exit 1; }
# Hugging Face local-dir metadata, if present, must remain under model_save.
for required in '--tensor-parallel-size 1' '--gpu-memory-utilization 0.70' '--max-model-len 262144'; do
  grep -Fq -- "$required" "$LAUNCH" || { echo "FAIL missing $required"; exit 1; }
done
if grep -Ev '^[[:space:]]*#' "$LAUNCH" | grep -Eiq 'mtp|speculative|draft|num_speculative'; then
  echo 'FAIL MTP/speculative option found'; exit 1
fi
if find "$ROOT" -path "$ROOT/model_save" -prune -o -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.gguf' \) -print | grep -q .; then
  echo 'FAIL model weight outside model_save'; exit 1
fi
echo "PASS static configuration; shards=$count"
