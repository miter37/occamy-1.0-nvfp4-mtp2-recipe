# vLLM startup log — MTP2, verbatim excerpts

Captured from `docker logs` on the DGX Spark (GB10) host described in README section 13,
launching `configs/occamy-1.0-nvfp4-mtp2-vllm.sh`. Reproduced so you can match these
strings against your own log. Analysis is in README section 9.

## Architecture resolution — confirms the MTP path was built

You should see **two** of these. The first is the target model, the second is the draft.
Only one line means the MTP head was not picked up at all.

```text
INFO 09-16 05:21:27 [model.py:692] Resolved architecture: Qwen3_5MoeForConditionalGeneration
INFO 09-16 05:21:34 [model.py:692] Resolved architecture: Qwen3_5MoeMTP
```

## Warning 1 — speculative depth > 1 reuses a single MTP layer

```text
WARNING 09-16 05:21:34 [speculative.py:1361] Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer,which may result in lower acceptance rate
```

The released head is a single layer (`mtp_num_hidden_layers: 1`). Depth 2 and 3
re-invoke it. Primary cause of the MTP3 acceptance drop.

## Warning 2 — no draft KV cache group identified

```text
WARNING 09-16 05:22:58 [kv_cache_utils.py:2211] Speculative decoding (method=mtp) is enabled but no KV cache group could be identified as the draft model's, so every group -- including Mamba groups [0, 1, 2] -- will be treated as a draft group. A Mamba group cannot satisfy the widened lookup window that implies, so prefix-cache reuse across requests will be disabled and any external KV offload tier will store without ever serving a hit.
```

Contradicted in practice by the measured prefix cache hit rate below. Unresolved.

## Info — FlashInfer lacks fused multi-step draft decode

```text
INFO 09-16 05:22:58 [speculator.py:120] Fused multi-step draft decode is not supported by attention backend(s) FLASHINFER; falling back to rebuilding attention metadata between draft steps.
```

Attention metadata is rebuilt between draft steps, so per-step cost grows with depth.

## Speculative decoding metrics — the spread is the point

Same server, same session. Acceptance is dominated by workload predictability, not by
configuration:

```text
INFO 09-16 05:26:57 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 3.00, Accepted throughput: 9.00 tokens/s, Drafted throughput: 9.00 tokens/s, Accepted: 90 tokens, Drafted: 90 tokens, Per-position acceptance rate: 1.000, 1.000, Avg Draft acceptance rate: 100.0%
INFO 09-16 05:27:27 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 3.00, Accepted throughput: 3.20 tokens/s, Drafted throughput: 3.20 tokens/s, Accepted: 32 tokens, Drafted: 32 tokens, Per-position acceptance rate: 1.000, 1.000, Avg Draft acceptance rate: 100.0%
```

Low acceptance was open-ended prose; near-total acceptance was structured code.

## Prefix cache hit rate actually observed

```text
INFO 09-16 11:59:18 [loggers.py:311] Engine 000: Avg prompt throughput: 471.1 tokens/s, Avg generation throughput: 34.8 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.4%, Prefix cache hit rate: 93.4%
```
