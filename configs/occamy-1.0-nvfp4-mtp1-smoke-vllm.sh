#!/usr/bin/env bash
set -euo pipefail

# Minimal vLLM compatibility smoke test for the Occamy BF16 MTP head.
# Keep the NVFP4 target model and FP8 KV cache; remove optional runtime features.
export VLLM_MARLIN_USE_ATOMIC_ADD=1

exec vllm serve /models/occamy-1.0-NVFP4-with-MTP \
  --served-model-name occamy-1.0-nvfp4-mtp1-smoke \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --gpu-memory-utilization 0.70 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching \
  --enforce-eager \
  --load-format fastsafetensors \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"moe_backend":"triton"}'
