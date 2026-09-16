#!/usr/bin/env bash
set -euo pipefail

# User-requested experiment: vLLM MTP with 3 speculative tokens.
# The released Occamy head was officially validated only for MTP1/2 draft tokens;
# this MTP3 setting is an experimental compatibility/performance test.
export VLLM_MARLIN_USE_ATOMIC_ADD=1

exec vllm serve /models/occamy-1.0-NVFP4-with-MTP \
  --served-model-name occamy-1.0-nvfp4-mtp3 \
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
  --load-format fastsafetensors \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
