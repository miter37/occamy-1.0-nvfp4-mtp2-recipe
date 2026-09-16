# Recipe Book — Occamy-1.0-NVFP4 + MTP2 on vLLM

*Full reasoning behind [README.md](README.md). Measured on an NVIDIA DGX Spark (GB10),
aarch64, 121 GB unified memory, single GPU, TP 1.*

A field report on grafting the separately-released **BF16 Occamy MTP head** onto the
**Occamy-1.0-NVFP4** checkpoint and running it under **stock vLLM** with MTP2
speculative decoding.

The head ships as an **SGLang-only** artifact. Getting it to load under vLLM at all
required one non-obvious checkpoint edit; getting it to run fast required one vLLM
flag most people don't know exists. Both are documented here, with the reasoning,
so you can reproduce the *fix* rather than our *setup*.

> [!WARNING]
> **Scope and validation status.** This runs the head **outside** its published
> validation envelope (SGLang-only, MTP1, ctx 2048, greedy decoding, no FP8 KV cache).
> The SGLang runtime hooks the head's model card lists as *required* have no vLLM
> equivalent and are **not applied**. The speedup is measured; **output parity against
> the no-MTP baseline has not been verified.** Treat this as an experimental
> deployment recipe, not a support statement from Accio-Lab, SGLang, or vLLM.

---

## What this repo is for

You are probably here because you tried to serve Occamy NVFP4 with its MTP head on
vLLM and hit a tensor shape mismatch, or because you got it loaded and want to know
which knobs matter. This repo is a **diagnosis you can port**, not an installer.

Read it, take the two portable pieces, and adapt the rest to your own stack.

### Portability map

| Piece | Transfers to your setup? | Why |
|---|---|---|
| **The quantization exclusion fix** (§3) | **Yes — this is the point** | Pure checkpoint surgery. No dependency on our hardware, container, or paths. |
| **Target/draft MoE backend split** (§5) | **Yes** | A stock `vllm serve` flag. Applies to any NVFP4-target + BF16-draft combination. |
| **The bring-up ladder** (§7) | **Yes, as method** | How to isolate which of FP8 / CUDA graph / prefix cache / long context broke you. |
| `configs/*.sh` flag sets | Mostly — retune sizing | Flag *combinations* are portable. But they are tuned for a **DGX Spark (GB10), 121 GB unified memory, TP 1**. `--max-model-len 262144`, `--gpu-memory-utilization 0.70` and `--max-num-batched-tokens` are sizing decisions for that box — recompute them for yours. |
| `scripts/start_*.sh` | **No — reference only** | Hardcoded to `eugr/spark-vllm-docker` and our absolute paths. Shown so you can see exactly how the model directory is mounted (§8 has the one subtle part). |
| Our throughput numbers | **No** | Single DGX Spark (GB10), single request, specific prompts. Unified memory and aarch64 both matter here. Re-measure on your own box. |

---

## 1. The failure

Assembling the head with the vendor's `assemble_head.py` and pointing `vllm serve`
at the result fails during weight loading:

```text
The size of tensor a (256) must match the size of tensor b (512)
```

The numbers vary by which tensor trips first, but the signature is constant: **one
axis is exactly half its expected size.**

## 2. Root cause

Half is the tell. NVFP4 packs two 4-bit values per byte, so a BF16 weight that the
ModelOpt loader mistakenly treats as NVFP4 reads back at **half the expected width**.
Something is telling vLLM to quantize tensors that are BF16 on disk.

That something is the assembled checkpoint's quantization recipe. `assemble_head.py`
copies the base checkpoint forward like this:

```python
skip = {'config.json', 'model.safetensors.index.json', 'README.md',
        'CANDIDATE_STATUS.json', 'VALIDATION.json', 'SHA256SUMS'}
for f in base.iterdir():
    if f.is_file() and f.name not in skip:
        (out / f.name).symlink_to(f)
```

`hf_quant_config.json` **is not in that skip set**, so it is symlinked straight from
the NVFP4 base. And `config.json`, which the script *does* rewrite, carries
`quantization_config` over verbatim.

So both quantization recipes still say *"quantize every Linear"* — while the checkpoint
now contains 19 freshly grafted **BF16** `mtp.*` tensors that were never part of the
NVFP4 export. Nothing in the assembly step tells the loader they are an exception.

Concretely, `mtp.layers.0.mlp.shared_expert.gate_proj.weight` is `[512, 2048]` BF16 on
disk. Quantized as NVFP4 it is read as `256`-wide. Hence the mismatch.

## 3. The fix — exclude the MTP namespace from quantization

**This is the portable core of this repo.** After assembly, before first launch:

1. `hf_quant_config.json` is a symlink into the base checkpoint. **Break the symlink**
   and make it a real file — otherwise you are editing the base model in place, which
   will corrupt your no-MTP setup too.
2. Add the MTP namespace to the exclusion list in **both** recipes.

```diff
  // hf_quant_config.json  ->  .quantization.exclude_modules
       "model.visual*",
+      "mtp.layers.0*",
+      "mtp*"
   ]

  // config.json  ->  .quantization_config.ignore
       "model.visual*",
+      "mtp.layers.0*",
+      "mtp*"
   ]
```

That is the entire delta. `patch_mtp_quant_config.py` in this repo applies it
idempotently and refuses to touch the base checkpoint.

> **We edited both files and did not test which one alone is sufficient.** vLLM
> resolves this checkpoint as `quantization=modelopt_fp4`, which reads
> `hf_quant_config.json`, so that one is very likely the load-bearing edit. Both are
> cheap; we left both in.

### Verifying the fix

```bash
python - <<'PY'
import json, pathlib
d = pathlib.Path("<assembled-dir>")
assert not (d/"hf_quant_config.json").is_symlink(), "still a symlink - you are editing the base model"
q = json.loads((d/"hf_quant_config.json").read_text())
c = json.loads((d/"config.json").read_text())
print("hf_quant:", [p for p in q["quantization"]["exclude_modules"] if p.startswith("mtp")])
print("config  :", [p for p in c["quantization_config"]["ignore"] if p.startswith("mtp")])
PY
```

Then confirm the server actually built the MTP path. The log resolves **two**
architectures — target first, draft second. Only one line means the head was never
picked up:

```text
Resolved architecture: Qwen3_5MoeForConditionalGeneration     <- target
Resolved architecture: Qwen3_5MoeMTP                          <- draft, this is the one you need
... speculative_config=SpeculativeConfig(method='mtp', ..., num_spec_tokens=2)
```

and that drafting is live (not silently disabled):

```text
SpecDecoding metrics: Mean acceptance length: 2.85, ... Avg Draft acceptance rate: 92.5%
```

If acceptance is reported but near zero, the head loaded but is not predicting —
a different problem from this one.

## 4. What vLLM already handles for you

Don't over-patch. Current vLLM already special-cases one MTP tensor
(`vllm/model_executor/models/qwen3_5_mtp.py`):

```python
# Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is
# missing from hf_quant_config.json exclude_modules. Force unquantized.
# Ref: https://github.com/vllm-project/vllm/pull/38650
fc_quant = None if (quant_config and quant_config.get_name() == "modelopt_fp4") else quant_config
```

Upstream fixed `mtp.fc` specifically. It does **not** generalize to the MTP attention
and MoE tensors. The two fixes are complementary:

```text
mtp.fc          -> handled by vLLM upstream
all other mtp.* -> you must exclude them in the checkpoint (§3)
```

If a future vLLM widens that workaround to the whole `mtp.` namespace, §3 becomes
redundant. Check your version before assuming you need it.

## 5. Different MoE backends for target and draft

After §3 the model loads, but target and draft now have **different precision**:

```text
Target : Occamy NVFP4 routed experts
Draft  : BF16 MTP layer
```

Forcing one MoE kernel across both is wrong. vLLM has a field for exactly this case —
`SpeculativeConfig.moe_backend`, documented upstream as *"Useful when the drafter and
generator require different MoE kernels (e.g. quantized generator with unquantized
drafter)"*:

```bash
--moe-backend marlin \
--speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}'
```

```text
Target NVFP4 MoE -> Marlin
Draft  BF16  MoE -> Triton
```

**Do not put `--moe-backend triton` on the target.** That applies Triton to the NVFP4
experts, which is not what you want.

We did not patch or fork vLLM. Our tree was clean; this is a stock flag.

## 6. The SGLang runtime hooks are not used

The head's repository ships four hooks its model card lists as required:

```text
canonical_attention    canonical_gdn    mtp_state_guard    state_commit_audit
```

They monkey-patch SGLang internals (`sglang.srt.*`) to prevent stale prefill-state
initialization during verification, reuse decode kernels across verify nodes, and
**fail closed** on unsupported tree/batch shapes.

They have no vLLM equivalent, and we ran without them (`PYTHONPATH` empty in the
serving container). Correctness of the verify path therefore rests entirely on vLLM's
own MTP implementation.

This is the most important limitation in this repo. The vendor considered those
guarantees worth enforcing with a hard `RuntimeError`; we removed the assertion
without replacing it. **A server that starts and generates quickly is not evidence
that the verify path is correct** across long-context, concurrent, and prefix-cached
requests.

## 7. Bring-up ladder — how to isolate your own failures

We did not enable everything at once, and you shouldn't either. Each rung changes
exactly one thing, so a failure names its own cause:

```text
1. MTP1, ctx 2048, enforce_eager, prefix cache OFF   <- closest to the vendor's validated config
2. -> MTP2                                            (does the head survive depth 2?)
3. -> FP8 KV cache
4. -> prefix caching, CUDA graph, async scheduling
5. -> max_model_len 262144
```

`configs/occamy-1.0-nvfp4-mtp1-smoke-vllm.sh` is rung 1 and
`configs/occamy-1.0-nvfp4-mtp2-vllm.sh` is rung 5. Diffing those two files shows every
production flag we added, in order. Start at rung 1 on your hardware even if you intend
to land on rung 5.

## 8. One mounting gotcha (if you containerize)

`assemble_head.py` builds the assembled directory out of **relative symlinks** into its
siblings:

```text
occamy-1.0-NVFP4-with-MTP/model-00001-of-00005.safetensors -> ../occamy-1.0-NVFP4/...
occamy-1.0-NVFP4-with-MTP/mtp-trained.safetensors          -> ../occamy-1.0-MTP/...
```

Mounting only the assembled directory into a container **breaks every one of them**.
Mount the parent that holds all three directories. Ours:

```bash
-v "$MODEL_ROOT:/models:ro"        # not -v "$ASSEMBLED_DIR:/models/assembled:ro"
```

## 9. Startup warnings you will also see

These are expected, not errors. They explain the performance envelope.

**Repeated forward over a single MTP layer**
```text
Enabling num_speculative_tokens > 1 will run multiple times of forward on same
MTP layer, which may result in lower acceptance rate
```
This head is one layer (`mtp_num_hidden_layers: 1`, vendor config: `mtp_steps: 1`,
`draft_tokens: 2`). Depth >1 re-invokes the same layer. This is the primary reason
MTP3 degrades — see §11.

**FlashInfer multi-step fallback**
```text
Fused multi-step draft decode is not supported by attention backend(s) FLASHINFER;
falling back to rebuilding attention metadata between draft steps
```
Attention metadata is rebuilt between draft steps, so per-step overhead grows with
depth. Compounds the above.

**Draft KV cache group not identified**
```text
Speculative decoding (method=mtp) is enabled but no KV cache group could be identified
as the draft model's, so every group -- including Mamba groups [0, 1, 2] -- will be
treated as a draft group ... prefix-cache reuse across requests will be disabled
```
Our measured prefix cache hit rate was **91.7%**, which contradicts the warning.
We do not claim cross-request prefix reuse is fully correct here — the discrepancy is
unresolved and is on the verification list (§12).

## 10. Measurements — and why you should not quote the ratio

On one **DGX Spark (GB10)**, single request, `--max-num-seqs 1`:

| Run | Context | Completion | Decode |
|---|---:|---:|---:|
| no-MTP baseline | 65,567 | 64 tok | 30.26 tok/s |
| MTP2 essay | 57,344 | 4,096 tok | **51.43 tok/s** (70.6% acceptance) |
| MTP2 code | 57,344 | 2,048 tok | **58.02 tok/s** (92.5% acceptance) |

> [!IMPORTANT]
> **This is not a controlled A/B.** The baseline and the MTP2 runs differ in context
> length (65,567 vs 57,344), completion length (64 vs 4,096 tokens), prefix-cache state
> (deliberately bypassed vs enabled), and sampling (unspecified vs `temperature=0.2`).
> The often-quoted "~1.7x" is **indicative only**. A same-length, same-sampling,
> same-cache-state baseline was never measured. If you need a number you can defend,
> measure it yourself with `scripts/run_occamy_mtp_56k_suite.py` against both servers.

Acceptance is strongly workload-dependent — per-position rates in our logs ranged from
`0.66 / 0.38` on open-ended prose to `1.00 / 1.00` on structured code. Code, JSON, and
other high-predictability output benefit most.

## 11. MTP2 vs MTP3

MTP3 starts and generates. It is not better here.

| | Essay decode | Essay acceptance | Code decode | Code acceptance |
|---|---:|---:|---:|---:|
| MTP2 | **51.43 tok/s** | **70.6%** | 58.02 tok/s | **92.5%** |
| MTP3 | 48.84 tok/s | 54.3% | **60.23 tok/s** | 72.9% |

Acceptance drops materially at depth 3 for both workloads, for the reasons in §9:
one layer re-invoked three times, plus metadata rebuild per draft step. The marginal
code-decode win does not offset it.

**MTP2 was the better operating point in our configuration.** This is not a structural
claim about MTP3 — with a deeper head, or an attention backend supporting fused
multi-step draft decode, the result could differ.

## 12. What we did not verify

Listed so you know what to test before trusting this in production:

```text
output parity vs the no-MTP baseline at temperature 0   <- start here
tool-call argument exactness under MTP
long-context needle retrieval at 262K
multi-request concurrency (we ran --max-num-seqs 1 only)
prefix-cache correctness (see the §9 contradiction)
FP8 KV cache interaction with speculative verify
multimodal paths with MTP active
```

Suggested minimum, same prompt set, `temperature=0`, comparing exact output token
sequences:

```text
A. MTP off, prefix cache off      C. MTP2, prefix cache off
B. MTP off, prefix cache on       D. MTP2, prefix cache on
```

A vs C is the parity test that matters. The head's repository ships
`evaluation_cases.json` and per-request input/output SHA256 fingerprints, which makes
this test straightforward to set up.

One related observation: with sampling enabled we saw multilingual character collapse
in long generations, which went away at `temperature=0`. The vendor also validates
greedy only. We did not isolate whether MTP2, NVFP4, or FP8 KV contributed — so if you
enable sampling, verify output quality deliberately.

## 13. Our environment (reference — yours will differ)

Every number and flag set above came from this box. Pinned because all of it is
fast-moving — and note the platform, because `aarch64` + unified memory constrains
which prebuilt wheels and container images are even available to you:

```text
Host            NVIDIA DGX Spark (GB10), aarch64, 121 GB unified memory, 1 GPU
Engine          vLLM 0.29.1rc1.dev17+gd2d649e67.d20260913
Container       eugr/spark-vllm-docker @ 3e1578b3c898255e7c79532569fd8d194261c3fd
Base model      Accio-Lab/occamy-1.0-NVFP4      @ c1aa9ae71c9d5d7f906b1575557e273cbdedf6f1
MTP head        Accio-Lab/occamy-1.0-MTP        @ a3bf37704b264e320a3ac8bc901a5787a5497f36
Attention       FlashInfer
KV cache        FP8
TP              1
```

Serving flags we settled on are in `configs/occamy-1.0-nvfp4-mtp2-vllm.sh`. Note
`--max-num-seqs 1`: we never validated concurrency, so the config does not pretend to.

### Validated envelope vs what we ran

| | Vendor-validated | This repo |
|---|---|---|
| Engine | SGLang 0.5.13.post1 | vLLM 0.29.1rc1.dev17 |
| Speculative depth | MTP1 | **MTP2** (MTP3 starts) |
| Context | 2,048 | **262,144** |
| KV cache | not FP8 | **FP8** |
| Sampling | greedy, deterministic | 0.0–0.2 |
| CUDA graph / prefix cache / overlap | off | **on** |
| Runtime hooks | required | **not applied** |

## 14. Repo layout

```text
README.md                   short version — start there
recipe_book.md              this file — full reasoning, logs, methodology
patch_mtp_quant_config.py   the §3 fix, idempotent, refuses to touch the base checkpoint
configs/                    vllm serve flag sets: no-MTP, MTP1 smoke, MTP2, MTP3 (the bring-up ladder)
scripts/                    our container orchestration — reference only, hardcoded to our stack
                            run_occamy_mtp_56k_suite.py is reusable: plain OpenAI-compatible client
results/                    raw measurements, startup_warnings.md (verbatim log excerpts),
                            generated outputs kept as evidence
```

## Attribution

This repository contains original configuration, scripts, and analysis. It redistributes
no model weights and no vendor code — download those from the sources below.

- [`Accio-Lab/occamy-1.0-NVFP4`](https://huggingface.co/Accio-Lab/occamy-1.0-NVFP4) — Apache-2.0
- [`Accio-Lab/occamy-1.0-MTP`](https://huggingface.co/Accio-Lab/occamy-1.0-MTP) — Apache-2.0
  (`assemble_head.py` and `runtime/` belong to this repository; get them there)
- [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
- [vLLM](https://github.com/vllm-project/vllm) — incl. [PR #38650](https://github.com/vllm-project/vllm/pull/38650)

Licensed Apache-2.0. See `NOTICE`.

## Disclaimer

An independent field report. Not an official deployment guide from Accio-Lab, the
Qwen team, SGLang, or vLLM, and not endorsed by them. It combines MTP2, 262K context,
FP8 KV cache, prefix caching, and a non-target inference engine — none of which the
head's publisher validated together. Run your own correctness and stability testing
before production use.
