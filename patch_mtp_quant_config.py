#!/usr/bin/env python3
"""Exclude the BF16 MTP head from the base checkpoint's NVFP4 quantization recipe.

Run once against an assembled checkpoint, after `assemble_head.py` and before the
first `vllm serve`. Without this, loading fails with a shape mismatch where one axis
is exactly half its expected size -- NVFP4 packs two 4-bit values per byte, so BF16
tensors wrongly treated as NVFP4 read back at half width.

Why it is needed: assemble_head.py symlinks every base file it does not rewrite, and
`hf_quant_config.json` is not in its skip set. The rewritten `config.json` also carries
`quantization_config` over verbatim. Both therefore still instruct the ModelOpt loader
to quantize every Linear -- including the 19 BF16 `mtp.*` tensors just grafted in.

This script is idempotent and refuses to modify the base checkpoint through a symlink.

    python patch_mtp_quant_config.py /path/to/occamy-1.0-NVFP4-with-MTP
    python patch_mtp_quant_config.py /path/to/assembled --check   # report only

See README section 3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Both patterns as used in production. "mtp*" alone would very likely suffice;
# "mtp.layers.0*" is kept explicit so the intent survives a future head with more layers.
MTP_PATTERNS = ["mtp.layers.0*", "mtp*"]

# (filename, dotted path to the exclusion list)
TARGETS = [
    ("hf_quant_config.json", ("quantization", "exclude_modules")),
    ("config.json", ("quantization_config", "ignore")),
]


def resolve_list(doc: dict, path: tuple[str, ...]) -> list | None:
    node = doc
    for key in path[:-1]:
        node = node.get(key)
        if not isinstance(node, dict):
            return None
    value = node.get(path[-1])
    return value if isinstance(value, list) else None


def patch_file(directory: Path, name: str, path: tuple[str, ...], check: bool) -> bool:
    target = directory / name
    if not target.exists():
        print(f"  {name}: MISSING -- is this an assembled checkpoint?", file=sys.stderr)
        return False

    was_symlink = target.is_symlink()
    doc = json.loads(target.read_text())
    entries = resolve_list(doc, path)
    if entries is None:
        print(f"  {name}: no .{'.'.join(path)} list -- not an NVFP4 checkpoint?", file=sys.stderr)
        return False

    missing = [p for p in MTP_PATTERNS if p not in entries]

    if check:
        state = "symlink -> base checkpoint" if was_symlink else "regular file"
        status = "OK" if not missing else f"MISSING {missing}"
        print(f"  {name}: {state}, .{'.'.join(path)} {status}")
        return not missing and not was_symlink

    if not missing and not was_symlink:
        print(f"  {name}: already patched, unchanged")
        return True

    entries.extend(missing)

    # Critical: break the symlink before writing. Writing through it would edit the
    # base NVFP4 checkpoint in place and corrupt the no-MTP configuration as well.
    if was_symlink:
        target.unlink()
        print(f"  {name}: symlink broken, now an independent file")

    target.write_text(json.dumps(doc, indent=4) + "\n")
    print(f"  {name}: added {missing or '(nothing)'} to .{'.'.join(path)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("assembled_dir", type=Path, help="output directory produced by assemble_head.py")
    ap.add_argument("--check", action="store_true", help="report state without writing")
    args = ap.parse_args()

    directory = args.assembled_dir.resolve()
    if not (directory / "config.json").exists():
        print(f"error: {directory} does not look like a model directory", file=sys.stderr)
        return 2

    index = directory / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        mtp_tensors = [k for k in weight_map if k.startswith("mtp.")]
        if not mtp_tensors:
            print("error: no mtp.* tensors in the weight index -- run assemble_head.py first", file=sys.stderr)
            return 2
        print(f"{directory}\n  weight index: {len(mtp_tensors)} mtp.* tensors present")

    ok = all(patch_file(directory, name, path, args.check) for name, path in TARGETS)

    if args.check:
        print("\nresult:", "patched" if ok else "NOT patched -- run without --check")
        return 0 if ok else 1

    print("\nDone. Start vLLM and confirm the log shows:")
    print("  Resolved architecture: Qwen3_5MoeMTP")
    print("  speculative_config=SpeculativeConfig(method='mtp', ..., num_spec_tokens=2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
