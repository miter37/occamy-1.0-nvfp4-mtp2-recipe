#!/usr/bin/env bash
set -euo pipefail

# Occamy has no donor MTP head: no speculative/MTP options are used.
export VLLM_MARLIN_USE_ATOMIC_ADD=1

exec vllm serve /models/occamy-1.0-NVFP4 \
  --served-model-name occamy-1.0-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --gpu-memory-utilization 0.70 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
