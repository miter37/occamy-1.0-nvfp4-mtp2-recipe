#!/usr/bin/env python3
import argparse
import ast
import datetime as dt
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path("${OCCAMY_ROOT}")
MODEL_DIR = ROOT / "model_save" / "occamy-1.0-NVFP4-with-MTP"
URL = "http://127.0.0.1:8000/v1/chat/completions"
TARGET_PROMPT_TOKENS = 56 * 1024

REFERENCE = """
The reference material describes how reliable software systems are designed, observed, tested, and improved. Engineers separate correctness, performance, reliability, maintainability, and security because optimizing one dimension can affect the others. Measurements should state hardware, software versions, workload shape, concurrency, cache state, input length, output length, and sampling parameters. A benchmark is useful only when another person can reproduce its conditions and understand its limitations. Long-context workloads stress memory movement, attention kernels, cache allocation, scheduling, and prompt ingestion differently from short interactive requests. Mixture-of-experts models route each token through a subset of experts, reducing active computation while introducing routing and communication costs. Speculative decoding proposes several future tokens and verifies them with the target model, which can reduce sequential decoding steps when acceptance is high. Acceptance rates depend on the draft model, prompt domain, sampling settings, and prediction depth. Operational testing should begin with a minimal configuration, verify one feature at a time, and preserve artifacts for later comparison. Logs and metrics are evidence, but they must be interpreted together with user-visible outputs. A successful server startup does not prove that long generation, structured output, or tool calling works correctly. Recovery procedures should avoid changing unrelated services and should preserve the original model files. Clear reports distinguish measured results from hypotheses and record both successful and failed experiments.
""".strip() + "\n"

TASKS = {
    "essay": {
        "max_tokens": 4096,
        "instruction": (
            "Using the reference material only as background context, write a polished, continuous English essay "
            "of approximately 4,000 output tokens about designing trustworthy high-performance AI inference systems. "
            "Use a title and well-developed sections, but do not discuss these instructions or the repeated context. "
            "Do not show chain-of-thought. Write the essay itself and continue until it is complete."
        ),
    },
    "code": {
        "max_tokens": 2048,
        "instruction": (
            "Produce approximately 2,000 output tokens of complete, executable Python code implementing a small "
            "benchmark-results analyzer. It must load JSON records, validate required numeric fields, compute summary "
            "statistics, compare two configurations, and expose a command-line interface using argparse. Include type "
            "hints, dataclasses, docstrings, and a main guard. Output only one Python code block and no prose outside it. "
            "Do not show chain-of-thought. Ensure the code is syntactically complete before ending."
        ),
    },
}


def prompt_token_count(tokenizer, content: str) -> int:
    messages = [{"role": "user", "content": content}]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(ids)


def make_content(tokenizer, instruction: str) -> tuple[str, int]:
    suffix = "\n\nFINAL TASK:\n" + instruction
    material = REFERENCE * 4000
    low, high = 0, len(material)
    best = ""
    best_count = 0
    while low <= high:
        mid = (low + high) // 2
        candidate = material[:mid] + suffix
        count = prompt_token_count(tokenizer, candidate)
        if count <= TARGET_PROMPT_TOKENS:
            best, best_count = candidate, count
            low = mid + 1
        else:
            high = mid - 1
    return best, best_count


def latest_spec_metrics(since_epoch: float) -> str | None:
    proc = subprocess.run(
        ["docker", "logs", "--since", str(int(since_epoch)), "self_vllm_occamy"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in (proc.stdout + proc.stderr).splitlines() if "SpecDecoding metrics:" in line]
    return lines[-1] if lines else None


def extract_python(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text.strip()


def run_request(model: str, mode: str, content: str, local_prompt_tokens: int, output_dir: Path) -> dict:
    spec = TASKS[mode]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": spec["max_tokens"],
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started_epoch = time.time()
    start = time.perf_counter()
    first_content = None
    last_content = None
    usage = {}
    finish_reason = None
    pieces: list[str] = []
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            item = json.loads(data)
            if item.get("usage"):
                usage = item["usage"]
            choices = item.get("choices") or []
            if not choices:
                continue
            finish_reason = choices[0].get("finish_reason") or finish_reason
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                now = time.perf_counter()
                first_content = first_content or now
                last_content = now
                pieces.append(piece)
    end = time.perf_counter()
    text = "".join(pieces)
    output_path = output_dir / f"{mode}.txt"
    output_path.write_text(text, encoding="utf-8")
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    ttft = (first_content or end) - start
    decode_elapsed = max((last_content or end) - (first_content or end), 1e-9)
    result = {
        "mode": mode,
        "local_prompt_tokens": local_prompt_tokens,
        "api_prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": round(ttft, 4),
        "estimated_prefill_tok_s": round(prompt_tokens / ttft, 2) if ttft else None,
        "decode_seconds_after_first": round(decode_elapsed, 4),
        "estimated_decode_tok_s": round(max(completion_tokens - 1, 0) / decode_elapsed, 2),
        "total_seconds": round(end - start, 4),
        "finish_reason": finish_reason,
        "output_path": str(output_path),
        "output_chars": len(text),
        "spec_metrics_log": latest_spec_metrics(started_epoch),
    }
    if mode == "code":
        code = extract_python(text)
        code_path = output_dir / "code_extracted.py"
        code_path.write_text(code, encoding="utf-8")
        try:
            ast.parse(code)
            result["python_syntax_valid"] = True
            result["python_syntax_error"] = None
        except SyntaxError as exc:
            result["python_syntax_valid"] = False
            result["python_syntax_error"] = f"{exc.msg} at line {exc.lineno}:{exc.offset}"
        result["extracted_code_path"] = str(code_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "logs" / f"occamy_{args.label}_56k_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
    results = []
    for mode in ("essay", "code"):
        content, local_count = make_content(tokenizer, TASKS[mode]["instruction"])
        result = run_request(args.model, mode, content, local_count, output_dir)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {
        "label": args.label,
        "model": args.model,
        "target_prompt_tokens": TARGET_PROMPT_TOKENS,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
