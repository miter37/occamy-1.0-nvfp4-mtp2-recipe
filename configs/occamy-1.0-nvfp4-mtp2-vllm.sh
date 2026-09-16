#!/usr/bin/env bash
set -euo pipefail

# vLLM Occamy MTP with 2 speculative tokens.
# Target model uses NVFP4/Marlin; BF16 MTP draft layer uses Triton.
export VLLM_MARLIN_USE_ATOMIC_ADD=1

exec vllm serve /models/occamy-1.0-NVFP4-with-MTP \
  --served-model-name occamy-1.0-nvfp4-mtp2 \
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
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}'
