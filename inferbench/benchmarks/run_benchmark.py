"""Benchmark runner CLI.

Usage:
    python -m inferbench.benchmarks.run_benchmark --scenario A
    python -m inferbench.benchmarks.run_benchmark --scenario A --duration 30 --warmup 5

W1 implements Scenario A only (single-request closed-loop baseline).
Later weeks add B-F via inferbench.benchmarks.workloads.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from inferbench.engine.metrics import latency_stats, throughput_stats

DEFAULT_TEXTS = [
    "this movie was surprisingly good",
    "absolute waste of time",
    "the plot was clever and the acting was solid",
    "i would not recommend this to anyone",
    "an unexpected gem with great cinematography",
    "boring, predictable, and far too long",
    "a delightful watch from start to finish",
    "the dialogue felt forced and the pacing was off",
]


def _next_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in output_root.glob("run_*") if p.is_dir())
    next_n = 1
    if existing:
        last = existing[-1].split("_")[-1]
        next_n = int(last) + 1 if last.isdigit() else len(existing) + 1
    run_dir = output_root / f"run_{next_n:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _system_info() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def _run_scenario_a(base_url: str, duration_sec: float, warmup_sec: float) -> dict:
    infer_url = f"{base_url.rstrip('/')}/infer"
    latencies_ms: list[float] = []
    inference_ms: list[float] = []
    queue_wait_ms: list[float] = []
    errors = 0
    backend = ""
    precision = ""

    with httpx.Client(timeout=10.0) as client:
        # Warmup — discarded
        warmup_end = time.perf_counter() + warmup_sec
        idx = 0
        while time.perf_counter() < warmup_end:
            try:
                client.post(infer_url, json={"inputs": [DEFAULT_TEXTS[idx % len(DEFAULT_TEXTS)]]})
            except httpx.HTTPError:
                pass
            idx += 1

        # Measured window
        start = time.perf_counter()
        end = start + duration_sec
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            try:
                resp = client.post(
                    infer_url,
                    json={"inputs": [DEFAULT_TEXTS[idx % len(DEFAULT_TEXTS)]]},
                )
                resp.raise_for_status()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                body = resp.json()
                inference_ms.append(body["inference_ms"])
                queue_wait_ms.append(body["queue_wait_ms"])
                backend = body["model_backend"]
                precision = body["model_precision"]
            except httpx.HTTPError:
                errors += 1
            idx += 1
        duration_actual = time.perf_counter() - start

    return {
        "latency_e2e": latency_stats(latencies_ms).to_dict(),
        "latency_inference": latency_stats(inference_ms).to_dict() if inference_ms else None,
        "latency_queue_wait": latency_stats(queue_wait_ms).to_dict() if queue_wait_ms else None,
        "throughput": throughput_stats(len(latencies_ms), errors, duration_actual).to_dict(),
        "model_backend": backend,
        "model_precision": precision,
    }


def _format_summary_md(scenario_id: str, scenario_cfg: dict, results: dict, run_meta: dict) -> str:
    lat = results["latency_e2e"]
    inf = results.get("latency_inference") or {}
    tput = results["throughput"]

    return f"""# Benchmark Run: scenario {scenario_id}

**Scenario**: {scenario_cfg["name"]}
**Started**: {run_meta["started_at"]}
**Duration**: {tput["duration_sec"]:.1f}s
**Backend**: {results["model_backend"]} | **Precision**: {results["model_precision"]}

## End-to-end latency (ms)

| count | mean | p50 | p90 | p95 | p99 | min | max |
|------:|-----:|----:|----:|----:|----:|----:|----:|
| {lat["count"]} | {lat["mean_ms"]:.2f} | {lat["p50_ms"]:.2f} | {lat["p90_ms"]:.2f} | {lat["p95_ms"]:.2f} | {lat["p99_ms"]:.2f} | {lat["min_ms"]:.2f} | {lat["max_ms"]:.2f} |

## Inference-only latency (ms)

| p50 | p95 | p99 |
|----:|----:|----:|
| {inf.get("p50_ms", 0):.2f} | {inf.get("p95_ms", 0):.2f} | {inf.get("p99_ms", 0):.2f} |

## Throughput

- **Requests**: {tput["requests"]}
- **Throughput**: {tput["requests_per_sec"]:.1f} req/s
- **Errors**: {tput["errors"]} ({tput["error_rate"] * 100:.2f}%)

## Environment

- Host: `{run_meta["system"]["hostname"]}`
- Platform: `{run_meta["system"]["platform"]}`
- Python: `{run_meta["system"]["python_version"]}`
- CPU count: {run_meta["system"]["cpu_count"]}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=["A"])
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--base-url", default=None, help="Override target.base_url from config")
    parser.add_argument("--duration", type=float, default=None, help="Override duration_sec")
    parser.add_argument("--warmup", type=float, default=None, help="Override warmup_sec")
    parser.add_argument("--output-root", type=Path, default=None, help="Override results dir")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    scenario_cfg = config["scenarios"][args.scenario]
    base_url = args.base_url or config["target"]["base_url"]
    duration_sec = args.duration if args.duration is not None else float(scenario_cfg["duration_sec"])
    warmup_sec = args.warmup if args.warmup is not None else float(scenario_cfg.get("warmup_sec", 0))

    output_root = args.output_root or Path(config["run"]["output_dir"])
    run_dir = _next_run_dir(output_root)

    started_at = datetime.now(timezone.utc).isoformat()
    if args.scenario == "A":
        results = _run_scenario_a(base_url, duration_sec, warmup_sec)
    else:
        raise NotImplementedError(f"Scenario {args.scenario} lands in a later week.")

    run_meta = {
        "scenario": args.scenario,
        "started_at": started_at,
        "duration_sec": duration_sec,
        "warmup_sec": warmup_sec,
        "base_url": base_url,
        "system": _system_info(),
    }

    (run_dir / "results.json").write_text(json.dumps({"meta": run_meta, "results": results}, indent=2))
    shutil.copy2(args.config, run_dir / "config.snapshot.yaml")
    (run_dir / "summary.md").write_text(_format_summary_md(args.scenario, scenario_cfg, results, run_meta))

    lat = results["latency_e2e"]
    tput = results["throughput"]
    print(
        f"\nbench({args.scenario}): n={lat['count']} "
        f"p50={lat['p50_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms p99={lat['p99_ms']:.2f}ms "
        f"throughput={tput['requests_per_sec']:.1f} req/s err={tput['error_rate'] * 100:.2f}%"
    )
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
