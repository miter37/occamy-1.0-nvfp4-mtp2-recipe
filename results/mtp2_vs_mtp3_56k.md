# Occamy vLLM MTP2 vs MTP3 — 56K long-context generation

## Fixed conditions

- Prompt tokens: 57,344 (`56 × 1024`)
- Tensor parallel: 1
- Max model length: 262,144
- GPU memory utilization: 0.70
- KV cache dtype: FP8
- Target model: Occamy NVFP4 with Marlin
- MTP draft layer: BF16 with Triton
- Temperature: 0.2
- Prefix caching: enabled
- Chunked prefill: enabled
- Async scheduling: enabled

## MTP3

### 4K essay

- Prompt: 57,344 tokens
- Completion: 4,096 tokens
- TTFT: 15.3683 s
- Estimated prefill: 3,731.31 tok/s
- Estimated decode: 48.84 tok/s
- Total: 99.2148 s
- Finish reason: length
- Average draft acceptance: 54.3%
- Output: `occamy_mtp3_56k_20260916_131529/essay.txt`

### 2K Python code

- Prompt: 57,344 tokens
- Completion: 2,048 tokens
- TTFT: 1.2965 s
- Estimated decode: 60.23 tok/s
- Total: 35.2854 s
- Finish reason: length
- Average draft acceptance: 72.9%
- Syntax: incomplete because generation was cut exactly at 2,048 tokens
- Output: `occamy_mtp3_56k_20260916_131529/code.txt`

## MTP2

### 4K essay

- Prompt: 57,344 tokens
- Completion: 4,096 tokens
- TTFT: 15.7164 s
- Estimated prefill: 3,648.67 tok/s
- Estimated decode: 51.43 tok/s
- Total: 95.3365 s
- Finish reason: length
- Average draft acceptance: 70.6%
- Output: `occamy_mtp2_56k_20260916_132043/essay.txt`

### 2K Python code

- Prompt: 57,344 tokens
- Completion: 2,048 tokens
- TTFT: 1.5716 s
- Estimated decode: 58.02 tok/s
- Total: 36.8500 s
- Finish reason: length
- Average draft acceptance: 92.5%
- Syntax: incomplete because generation was cut exactly at 2,048 tokens
- Output: `occamy_mtp2_56k_20260916_132043/code.txt`

## Comparison

- Essay decode: MTP2 51.43 vs MTP3 48.84 tok/s — MTP2 is 5.30% faster.
- Essay total time: MTP2 95.3365 vs MTP3 99.2148 s — MTP2 is 3.91% shorter.
- Code decode: MTP3 60.23 vs MTP2 58.02 tok/s — MTP3 is 3.67% faster.
- Code total time: MTP3 35.2854 vs MTP2 36.8500 s — MTP3 is 4.43% shorter.
- MTP2 acceptance was materially higher for both workloads.

## Caveat

The code request followed the essay request with a largely identical 56K prefix, so prefix-cache reuse reduced its TTFT. Code prefill rates should not be compared as cold-prefill measurements. Decode rates and completion totals remain directly useful.
