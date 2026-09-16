```python
"""
benchmark_analyzer.py

A small, self-contained command-line tool for loading, validating, summarizing,
and comparing JSON benchmark records.

The reference material emphasizes that a benchmark is only useful when its
conditions can be reproduced and its limitations understood. This tool enforces
that discipline by:

1. Requiring explicit metadata (hardware, software, workload shape, concurrency,
   cache state, input/output lengths, sampling parameters) in every record.
2. Validating that all numeric fields are present and finite.
3. Computing summary statistics (mean, median, std, min, max, count) for key
   performance dimensions.
4. Allowing side-by-side comparison of two configurations (e.g., baseline vs.
   speculative decoding, or MoE vs. dense).
5. Producing a clear report that distinguishes measured results from hypotheses.

The tool is designed to be deterministic and reproducible. It does not perform
any network calls or modify files outside the explicit output path.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRecord:
    """A single benchmark observation with full provenance metadata.

    Attributes:
        config_id:
            Identifier for the configuration under test (e.g. "moe-v1",
            "speculative-decoding").
        hardware:
            Description of the compute hardware (e.g. "A100 80GB").
        software_versions:
            Dict mapping package/library names to their version strings.
        workload_shape:
            Human-readable description of the workload (e.g. "long-context",
            "short-interactive").
        concurrency:
            Number of concurrent requests during the benchmark.
        cache_state:
            Description of cache state (e.g. "warm", "cold", "eviction-policy").
        input_length:
            Number of input tokens.
        output_length:
            Number of output tokens generated.
        sampling_params:
            Dict describing sampling settings (temperature, top_p, etc.).
        time_to_first_token_s:
            Latency from request start to first output token (seconds).
        throughput_tokens_per_s:
            End-to-end throughput in tokens per second.
        wall_clock_s:
            Total wall-clock time for the request (seconds).
        acceptance_rate:
            Fraction of speculative tokens accepted (optional, None if not used).
        notes:
            Free-form text for additional context or limitations.
    """

    config_id: str
    hardware: str
    software_versions: Dict[str, str]
    workload_shape: str
    concurrency: int
    cache_state: str
    input_length: int
    output_length: int
    sampling_params: Dict[str, Any]
    time_to_first_token_s: float
    throughput_tokens_per_s: float
    wall_clock_s: float
    acceptance_rate: Optional[float]
    notes: str = ""

    # Internal flag used to track whether required numeric fields validated.
    _validated: bool = field(default=False, repr=False)

    def validate(self) -> List[str]:
        """Validate required numeric fields.

        Returns:
            A list of error message strings. Empty list means the record is valid.
        """
        errors: List[str] = []

        # Concurrency must be a positive integer.
        if not isinstance(self.concurrency, int) or self.concurrency <= 0:
            errors.append(
                f"concurrency must be a positive integer, got {self.concurrency!r}"
            )

        # Input and output lengths must be non-negative integers.
        if not isinstance(self.input_length, int) or self.input_length < 0:
            errors.append(
                f"input_length must be a non-negative integer, got {self.input_length!r}"
            )
        if not isinstance(self.output_length, int) or self.output_length < 0:
            errors.append(
                f"output_length must be a non-negative integer, got {self.output_length!r}"
            )

        # Time-based metrics must be finite, non-negative floats.
        for attr_name in (
            "time_to_first_token_s",
            "throughput_tokens_per_s",
            "wall_clock_s",
        ):
            value = getattr(self, attr_name)
            if not isinstance(value, (int, float)):
                errors.append(
                    f"{attr_name} must be a number, got {value!r}"
                )
            elif math.isnan(value) or math.isinf(value):
                errors.append(
                    f"{attr_name} must be finite, got {value}"
                )
            elif value < 0:
                errors.append(
                    f"{attr_name} must be non-negative, got {value}"
                )

        # Acceptance rate, if present, must be in [0, 1].
        if self.acceptance_rate is not None:
            if not isinstance(self.acceptance_rate, (int, float)):
                errors.append(
                    f"acceptance_rate must be a number or None, got {self.acceptance_rate!r}"
                )
            elif math.isnan(self.acceptance_rate) or math.isinf(self.acceptance_rate):
                errors.append(
                    f"acceptance_rate must be finite, got {self.acceptance_rate}"
                )
            elif not (0.0 <= self.acceptance_rate <= 1.0):
                errors.append(
                    f"acceptance_rate must be in [0, 1], got {self.acceptance_rate}"
                )

        self._validated = len(errors) == 0
        return errors


@dataclass
class ConfigSummary:
    """Aggregated statistics for a single configuration.

    Attributes:
        config_id:
            The configuration identifier.
        count:
            Number of valid records in this configuration.
        ttft:
            Summary statistics for time-to-first-token.
        throughput:
            Summary statistics for throughput.
        wall_clock:
            Summary statistics for wall-clock time.
        acceptance_rate:
            Summary statistics for acceptance rate (if available).
        input_length:
            Summary statistics for input length (for workload characterization).
        output_length:
            Summary statistics for output length.
        records:
            The original validated records (preserved for inspection).
    """

    config_id: str
    count: int
    ttft: Dict[str, float]
    throughput: Dict[str, float]
    wall_clock: Dict[str, float]
    acceptance_rate: Optional[Dict[str, float]]
    input_length: Dict[str, float]
    output_length: Dict[str, float]
    records: List[BenchmarkRecord] = field(default_factory=list, repr=False)


@dataclass
class ComparisonReport:
    """A side-by-side comparison of two configurations.

    Attributes:
        baseline:
            Summary for the baseline configuration.
        comparison:
            Summary for the configuration being compared against the baseline.
        differences:
            Dict mapping metric names to percentage differences
            (comparison - baseline) / baseline * 100.
        notes:
            Human-readable observations.
    """

    baseline: ConfigSummary
    comparison: ConfigSummary
    differences: Dict[str, float]
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_records(path: Path) -> Tuple[List[BenchmarkRecord], List[str]]:
    """Load and validate benchmark records from a JSON file.

    The JSON file should be a list of dictionaries, each representing one
    BenchmarkRecord. Missing fields are reported as errors rather than
    silently defaulting, because incomplete provenance makes a benchmark
    unreproducible.

    Args:
        path: Path to the JSON file.

    Returns:
        A tuple of (valid_records, error_messages). error_messages may contain
        file-level errors (e.g. JSON decode errors) and per-record validation
        errors.
    """
    if not path.exists():
        return [], [f"File not found: {path}"]

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        return [], [f"Invalid JSON in {path}: {exc}"]

    if isinstance(data, dict):
        # Allow a single record wrapped in an object.
        data = [data]
    elif not isinstance(data, list):
        return [], [f"Expected a list or object in {path}, got {type(data).__name__}"]

    records: List[BenchmarkRecord] = []
    errors: List[str] = []

    for idx, raw in enumerate(data):
        if not isinstance(raw, dict):
            errors.append(f"Record {idx}: expected a dict, got {type(raw).__name__}")
            continue

        # Extract required fields with clear error messages for missing keys.
        try:
            record = BenchmarkRecord(
                config_id=raw["config_id"],
                hardware=raw["hardware"],