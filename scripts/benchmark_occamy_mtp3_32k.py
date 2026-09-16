import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
TARGET_MODEL = "occamy-1.0-nvfp4-mtp3"
content = ("x " * 32700) + "\nSummarize the preceding input in one sentence."
payload = {
    "model": TARGET_MODEL,
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 256,
    "temperature": 0.2,
    "stream": True,
    "stream_options": {"include_usage": True},
}
request = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
start = time.perf_counter()
first_content = None
last_content = None
usage = {}
finish_reason = None
chunks = 0
with urllib.request.urlopen(request, timeout=600) as response:
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
            chunks += 1
end = time.perf_counter()
prompt_tokens = usage.get("prompt_tokens", 0)
completion_tokens = usage.get("completion_tokens", 0)
ttft = (first_content or end) - start
decode_elapsed = max((last_content or end) - (first_content or end), 1e-9)
result = {
    "prompt_tokens": prompt_tokens,
    "completion_tokens": completion_tokens,
    "ttft_seconds": round(ttft, 4),
    "estimated_prefill_tok_s": round(prompt_tokens / ttft, 2) if ttft else None,
    "decode_seconds_after_first": round(decode_elapsed, 4),
    "estimated_decode_tok_s": round(max(completion_tokens - 1, 0) / decode_elapsed, 2),
    "total_seconds": round(end - start, 4),
    "stream_chunks": chunks,
    "finish_reason": finish_reason,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
