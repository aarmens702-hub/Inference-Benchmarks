"""Benchmark runner CLI.

Usage:
    python -m inferbench.benchmarks.run_benchmark --scenario A
    python -m inferbench.benchmarks.run_benchmark --scenario B
    python -m inferbench.benchmarks.run_benchmark --scenario A --duration 30 --warmup 5

Scenarios A and B are wired in W2; C-F land in later weeks.
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

import yaml

from inferbench.benchmarks.workloads import WorkloadResult, closed_loop, poisson
from inferbench.engine.metrics import latency_stats, throughput_stats
from inferbench.reports.generate_report import write_run_report, write_sweep_report


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


def _summarize(workload: WorkloadResult) -> dict:
    return {
        "latency_e2e": latency_stats(workload.latencies_ms).to_dict() if workload.latencies_ms else None,
        "latency_inference": latency_stats(workload.inference_ms).to_dict() if workload.inference_ms else None,
        "latency_queue_wait": latency_stats(workload.queue_wait_ms).to_dict() if workload.queue_wait_ms else None,
        "throughput": throughput_stats(
            len(workload.latencies_ms), workload.errors, workload.duration_sec
        ).to_dict(),
        "model_backend": workload.backend,
        "model_precision": workload.precision,
    }


def _print_run(scenario_id: str, summary: dict, suffix: str = "") -> None:
    if not summary.get("latency_e2e"):
        print(f"bench({scenario_id}){suffix}: no measurements")
        return
    lat = summary["latency_e2e"]
    tput = summary["throughput"]
    print(
        f"bench({scenario_id}){suffix}: n={lat['count']} "
        f"p50={lat['p50_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms p99={lat['p99_ms']:.2f}ms "
        f"throughput={tput['requests_per_sec']:.1f} req/s err={tput['error_rate'] * 100:.2f}%"
    )


def _scenario_a(args, scenario_cfg: dict, base_url: str, run_dir: Path, run_meta: dict, config_path: Path) -> None:
    duration = args.duration if args.duration is not None else float(scenario_cfg["duration_sec"])
    warmup = args.warmup if args.warmup is not None else float(scenario_cfg.get("warmup_sec", 0))

    wl = closed_loop(base_url, concurrency=1, duration_sec=duration, warmup_sec=warmup)
    summary = _summarize(wl)
    run_meta["workload"] = "closed_loop(c=1)"
    run_meta["duration_sec"] = duration
    run_meta["warmup_sec"] = warmup
    write_run_report(run_dir, "A", scenario_cfg, summary, run_meta, config_path)
    _print_run("A", summary)
    print(f"Results: {run_dir}")


def _scenario_b(args, scenario_cfg: dict, base_url: str, run_dir: Path, run_meta: dict, config_path: Path) -> None:
    duration = args.duration if args.duration is not None else float(scenario_cfg["duration_sec"])
    warmup = args.warmup if args.warmup is not None else float(scenario_cfg.get("warmup_sec", 0))
    concurrencies = scenario_cfg["concurrency"]

    sweep: list[dict] = []
    backend = precision = ""
    for c in concurrencies:
        wl = closed_loop(base_url, concurrency=int(c), duration_sec=duration, warmup_sec=warmup)
        summary = _summarize(wl)
        backend = backend or summary["model_backend"]
        precision = precision or summary["model_precision"]
        sweep.append({"concurrency": int(c), "results": summary})
        _print_run("B", summary, suffix=f" c={c}")

    run_meta["workload"] = "closed_loop"
    run_meta["duration_sec"] = duration
    run_meta["warmup_sec"] = warmup
    run_meta["backend"] = backend
    run_meta["precision"] = precision
    write_sweep_report(run_dir, "B", scenario_cfg, sweep, run_meta, config_path)
    print(f"Results: {run_dir}")


def _scenario_c(args, scenario_cfg: dict, base_url: str, run_dir: Path, run_meta: dict, config_path: Path) -> None:
    duration = args.duration if args.duration is not None else float(scenario_cfg["duration_sec"])
    warmup = args.warmup if args.warmup is not None else float(scenario_cfg.get("warmup_sec", 0))
    rates = scenario_cfg["rate_rps"]

    sweep: list[dict] = []
    backend = precision = ""
    for r in rates:
        wl = poisson(base_url, rate_rps=float(r), duration_sec=duration, warmup_sec=warmup)
        summary = _summarize(wl)
        backend = backend or summary["model_backend"]
        precision = precision or summary["model_precision"]
        sweep.append({"concurrency": f"poisson@{r}rps", "results": summary})
        _print_run("C", summary, suffix=f" rate={r}rps")

    run_meta["workload"] = "open_loop_poisson"
    run_meta["duration_sec"] = duration
    run_meta["warmup_sec"] = warmup
    run_meta["backend"] = backend
    run_meta["precision"] = precision
    write_sweep_report(run_dir, "C", scenario_cfg, sweep, run_meta, config_path)
    print(f"Results: {run_dir}")


_SCENARIOS = {"A": _scenario_a, "B": _scenario_b, "C": _scenario_c}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=sorted(_SCENARIOS.keys()))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--warmup", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    scenario_cfg = config["scenarios"][args.scenario]
    base_url = args.base_url or config["target"]["base_url"]
    output_root = args.output_root or Path(config["run"]["output_dir"])
    run_dir = _next_run_dir(output_root)

    run_meta = {
        "scenario": args.scenario,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "system": _system_info(),
    }

    handler = _SCENARIOS[args.scenario]
    handler(args, scenario_cfg, base_url, run_dir, run_meta, args.config)


if __name__ == "__main__":
    main()
