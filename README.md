# Occamy-1.0-NVFP4 + MTP2 on vLLM

The Occamy MTP head ships as an **SGLang-only** artifact, validated for MTP1 at 2048 context
with greedy decoding. This is what it took to run it under **stock vLLM** at MTP2 — no fork, no
patched engine — on an **NVIDIA DGX Spark (GB10)**, aarch64, 121 GB unified memory, TP 1.

One checkpoint edit does most of the work. The rest is knowing which flag exists, and what you
are giving up.

## Results — DGX Spark (GB10), single request

| | Context | Decode | Draft acceptance |
|---|---:|---:|---:|
| no-MTP baseline | 64K | 30.3 tok/s | — |
| MTP2, prose | 56K | **51.4 tok/s** | 70.6% |
| MTP2, code | 56K | **58.0 tok/s** | 92.5% |

**Not a controlled A/B** — the rows differ in context length, completion length (64 vs 4096
tokens), cache state and sampling. Read it as "this clearly helps," not as a defensible ratio.
Acceptance swings with workload: per-position rates ran `0.66 / 0.38` on prose to `1.00 / 1.00`
on code.

## The model

[**Occamy-1.0-NVFP4**](https://huggingface.co/Accio-Lab/occamy-1.0-NVFP4) — a Qwen3.6-35B-A3B
MoE, 35B total / 3B active, 262K context, mixed-precision NVFP4: routed experts NVFP4,
attention / router / shared experts BF16. It ships **without** an MTP head.

That is a separate BF16 release,
[**occamy-1.0-MTP**](https://huggingface.co/Accio-Lab/occamy-1.0-MTP) — 844M params of which
only 8.4M were trained. Its card validates **SGLang + MTP1 only** and explicitly lists FP8,
sampling, concurrency and MTP2 as *not validated*.

## What we changed to run MTP2

**1. Exclude the MTP head from NVFP4 quantization.** The one that matters.

`assemble_head.py` symlinks every base file it does not rewrite — and `hf_quant_config.json`
is not in its skip list. So the assembled checkpoint still carries the base NVFP4 recipe, which
says *quantize every Linear*, while now holding 19 grafted **BF16** `mtp.*` tensors. Nothing
marks them as an exception.

```
hf_quant_config.json  ->  .quantization.exclude_modules   += "mtp.layers.0*", "mtp*"
config.json           ->  .quantization_config.ignore     += "mtp.layers.0*", "mtp*"
```

Break the symlink into a real file first — writing through it edits the **base** checkpoint and
breaks your no-MTP setup too. `patch_mtp_quant_config.py` does this idempotently.

Skip this and you get a shape mismatch. **The tell is that one axis is exactly half.** NVFP4
packs two 4-bit values per byte, so a BF16 tensor wrongly read as NVFP4 comes back half-width:
`shared_expert.gate_proj [512,2048]` loads as 256 → `tensor a (256) vs tensor b (512)`. Any
clean halving in a mixed-precision graft means suspect the recipe, not the weights.

**2. Split the MoE backend between target and draft.** They are now different precisions, so
they should not share a kernel.

```bash
--moe-backend marlin                                 # target: NVFP4
--speculative-config '{...,"moe_backend":"triton"}'  # draft only: BF16
```

`SpeculativeConfig.moe_backend` is a stock vLLM 0.29 field for exactly this — "quantized
generator with unquantized drafter." Do **not** put `--moe-backend triton` on the target; that
applies Triton to the NVFP4 experts.

**3. Drop the vendor's required SGLang hooks.** `canonical_attention`, `canonical_gdn`,
`mtp_state_guard`, `state_commit_audit` monkey-patch `sglang.srt.*` internals — no attach point
in vLLM, so they go unused. See Caveats; this is the real cost.

**4. Let vLLM handle `mtp.fc` itself.** Upstream already forces that one tensor unquantized on
NVFP4 checkpoints ([PR #38650](https://github.com/vllm-project/vllm/pull/38650)) — and only
that one. Every other `mtp.*` tensor is step 1's job, so don't assume upstream covers you.

**5. Climb, don't leap.** MTP1 at 2048 ctx, `--enforce-eager`, prefix caching off (closest to
the validated setup) → MTP2 → FP8 KV → prefix caching + CUDA graph → 262K. One variable per
rung, so a failure names its own cause.

**Bonus gotcha — mounting.** The assembled directory is *relative* symlinks into its siblings
(`../occamy-1.0-NVFP4/...`). Mount only that directory into a container and every weight file
breaks — mount the parent holding all three.

## Verifying it worked

The log resolves **two** architectures — target, then draft. One line means the head was never
picked up. Then check that drafting is live, not silently disabled:

```text
Resolved architecture: Qwen3_5MoeForConditionalGeneration   <- target
Resolved architecture: Qwen3_5MoeMTP                        <- draft, the one you need
SpecDecoding metrics: Mean acceptance length: 2.85, ... Avg Draft acceptance rate: 92.5%
```

## Warnings you will also see

Expected, not errors — and they explain the ceiling.

- `...will run multiple times of forward on same MTP layer` — this head is one layer; depth 2
  and 3 re-invoke it.
- `Fused multi-step draft decode is not supported by FLASHINFER` — metadata is rebuilt between
  draft steps, so per-step cost grows with depth.
- `no KV cache group could be identified as the draft model's ... prefix-cache reuse will be
  disabled` — yet we measured a **91.7%** hit rate. Unreconciled, so we claim nothing about
  cross-request reuse.

The first two are why **MTP3 was worse**: prose acceptance fell 70.6% → 54.3%, and the marginal
code gain did not pay for it. That is this one-layer head plus FlashInfer, not MTP3 in general.

## What transfers to your setup

| | |
|---|---|
| **Steps 1, 2, the ladder** | Portable — checkpoint surgery and a stock flag. |
| `configs/*.sh` | Adapt. Tuned for 121 GB unified memory; recompute the sizing flags. |
| `scripts/*.sh` | Reference only — hardcoded to `eugr/spark-vllm-docker`. |
| Our numbers | Re-measure. One box, one request, our prompts. |

## Caveats

- **Outside the validated envelope**: the vendor tested SGLang, MTP1, 2048 ctx, greedy, no FP8.
- Step 3 removed hooks designed to **fail closed** on state mismatch — they raise `RuntimeError`
  rather than let a bad verify through. We deleted that assertion without replacing it, so
  verify-path correctness rests entirely on vLLM. **Output parity against no-MTP is unverified**,
  and starting fast is not evidence of being correct.
- Single-request only (`--max-num-seqs 1`). Concurrency untested.
- With sampling on, long outputs drifted into mixed scripts; `temperature=0` was clean. We did
  not isolate whether MTP2, NVFP4 or FP8 KV caused it.

One test before trusting this: same prompts at `temperature=0`, MTP off vs MTP2, compare **exact
output token sequences**. The head's repo ships `evaluation_cases.json` and output fingerprints,
so it is straightforward.

`patch_mtp_quant_config.py` applies step 1; `configs/` has every rung; `results/` the raw
measurements and verbatim logs. **[recipe_book.md](recipe_book.md)** has the long version.
