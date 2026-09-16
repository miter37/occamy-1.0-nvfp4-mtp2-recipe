# Occamy-1.0-NVFP4 + MTP2 on vLLM

The **SGLang-only** Occamy MTP head, run under **stock vLLM** with MTP2 speculative decoding.
Measured on an **NVIDIA DGX Spark (GB10)** — aarch64, 121 GB unified memory, TP 1.

## Results — DGX Spark (GB10)

| | Context | Decode | Acceptance |
|---|---:|---:|---:|
| no-MTP baseline | 64K | 30.3 tok/s | — |
| MTP2, prose | 56K | **51.4 tok/s** | 70.6% |
| MTP2, code | 56K | **58.0 tok/s** | 92.5% |

Not a controlled A/B: completion length, cache state and sampling differ. MTP3 was worse
(54.3%).

## The model

[**Occamy-1.0-NVFP4**](https://huggingface.co/Accio-Lab/occamy-1.0-NVFP4) — a Qwen3.6-35B-A3B
MoE, 35B total / 3B active, 262K ctx, mixed-precision NVFP4: routed experts NVFP4, attention
and shared experts BF16. It ships **without** an MTP head — that is a separate BF16 release,
[**occamy-1.0-MTP**](https://huggingface.co/Accio-Lab/occamy-1.0-MTP), validated for
**SGLang + MTP1 only**.

## What we changed to run MTP2

**1. Exclude the MTP head from NVFP4 quantization.** The critical one. `assemble_head.py`
symlinks `hf_quant_config.json` from the base, so the grafted BF16 `mtp.*` tensors inherit
the NVFP4 recipe. Break the symlink into a real file, then add `"mtp.layers.0*"` and `"mtp*"`
to `exclude_modules` **and** to `config.json` → `quantization_config.ignore`. Skip it and
NVFP4 packing (2 values/byte) halves an axis: `shared_expert.gate_proj [512,2048]` loads as
256 → `tensor a (256) vs tensor b (512)`.

**2. Split the MoE backend between target and draft.**

```bash
--moe-backend marlin                                 # target: NVFP4
--speculative-config '{...,"moe_backend":"triton"}'  # draft: BF16
```

`SpeculativeConfig.moe_backend` is a stock vLLM 0.29 field for exactly this (quantized
generator + unquantized drafter). No patch, no fork.

**3. Drop the vendor's required SGLang hooks.** `canonical_attention`, `canonical_gdn`,
`mtp_state_guard`, `state_commit_audit` all monkey-patch `sglang.srt.*` internals — no attach
point in vLLM, so they go unused.

**4. Let vLLM handle `mtp.fc`.** Upstream forces that tensor unquantized on NVFP4 checkpoints
([PR #38650](https://github.com/vllm-project/vllm/pull/38650)) — but only that one. Every
other `mtp.*` tensor is step 1's job.

**5. Climb, don't leap.** MTP1 at 2048 ctx, `--enforce-eager`, no prefix cache → MTP2 → FP8
KV → prefix cache + CUDA graph → 262K. One variable per rung, so failures name their cause.

## Caveats

- **Outside the validated envelope**: they tested SGLang, MTP1, 2048 ctx, greedy, no FP8.
- Step 3 removes a guard meant to fail closed on state mismatch; correctness now rests on
  vLLM alone. **Output parity against no-MTP is unverified.**
- Acceptance is workload-dependent: 40%–100% observed — code and JSON best, prose worst.
- Single-request only (`--max-num-seqs 1`); concurrency untested.
- With sampling on, long outputs drifted into mixed scripts; `temperature=0` was clean.

More detail: **[recipe_book.md](recipe_book.md)**
