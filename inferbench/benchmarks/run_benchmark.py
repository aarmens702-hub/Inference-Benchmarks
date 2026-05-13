"""Benchmark runner CLI.

Usage:
    python -m inferbench.benchmarks.run_benchmark --scenario A
    python -m inferbench.benchmarks.run_benchmark --scenario B
    python -m inferbench.benchmarks.run_benchmark --scenario A --duration 30 --warmup 5

Scenarios A and B are wired in W2; C-F land in later weeks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

import httpx

from inferbench.benchmarks.workloads import (
    WorkloadResult,
    cache_repeat,
    closed_loop,
    poisson,
    spike,
)
from inferbench.engine.metrics import latency_stats, throughput_stats
from inferbench.reports.generate_report import write_run_report, write_sweep_report
from inferbench.reports.plot_results import write_run_charts, write_sweep_charts


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


def _gpu_info() -> dict | None:
    """Best-effort GPU detection via nvidia-smi. None if unavailable."""
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if not out:
        return None
    name, driver, mem_mb = [x.strip() for x in out.splitlines()[0].split(",")]
    return {"name": name, "driver_version": driver, "memory_mib": int(mem_mb)}


def _system_info() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu_info(),
    }


def _save_raw_samples(run_dir: Path, scenario_id: str, workloads: list[tuple[str, WorkloadResult]]) -> None:
    """Persist raw sample arrays so plot_results can build CDFs.

    For sweeps, concatenates all sub-runs (good enough for an aggregate
    CDF — per-config CDFs would need separate dump files; deferred until
    we want them).
    """
    e2e: list[float] = []
    inf: list[float] = []
    qw: list[float] = []
    for _label, w in workloads:
        e2e.extend(w.latencies_ms)
        inf.extend(w.inference_ms)
        qw.extend(w.queue_wait_ms)
    payload = {
        "scenario": scenario_id,
        "e2e_ms": e2e,
        "inference_ms": inf,
        "queue_wait_ms": qw,
    }
    (run_dir / "raw_latencies.json").write_text(json.dumps(payload))


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
    _save_raw_samples(run_dir, "A", [("c=1", wl)])
    write_run_charts(run_dir)
    _print_run("A", summary)
    print(f"Results: {run_dir}")


def _scenario_b(args, scenario_cfg: dict, base_url: str, run_dir: Path, run_meta: dict, config_path: Path) -> None:
    duration = args.duration if args.duration is not None else float(scenario_cfg["duration_sec"])
    warmup = args.warmup if args.warmup is not None else float(scenario_cfg.get("warmup_sec", 0))
    concurrencies = scenario_cfg["concurrency"]

    sweep: list[dict] = []
    raw: list[tuple[str, WorkloadResult]] = []
    backend = precision = ""
    for c in concurrencies:
        wl = closed_loop(base_url, concurrency=int(c), duration_sec=duration, warmup_sec=warmup)
        summary = _summarize(wl)
        backend = backend or summary["model_backend"]
        precision = precision or summary["model_precision"]
        sweep.append({"concurrency": int(c), "results": summary})
        raw.append((f"c={c}", wl))
        _print_run("B", summary, suffix=f" c={c}")

    run_meta["workload"] = "closed_loop"
    run_meta["duration_sec"] = duration
    run_meta["warmup_sec"] = warmup
    run_meta["backend"] = backend
    run_meta["precision"] = precision
    write_sweep_report(run_dir, "B", scenario_cfg, sweep, run_meta, config_path)
    _save_raw_samples(run_dir, "B", raw)
    write_sweep_charts(run_dir)
    print(f"Results: {run_dir}")


def _scenario_c(args, scenario_cfg: dict, base_url: str, run_dir: Path, run_meta: dict, config_path: Path) -> None:
    duration = args.duration if args.duration is not None else float(scenario_cfg["duration_sec"])
    warmup = args.warmup if args.warmup is not None else float(scenario_cfg.get("warmup_sec", 0))
    rates = scenario_cfg["rate_rps"]

    sweep: list[dict] = []
    raw: list[tuple[str, WorkloadResult]] = []
    backend = precision = ""
    for r in rates:
        wl = poisson(base_url, rate_rps=float(r), duration_sec=duration, warmup_sec=warmup)
        summary = _summarize(wl)
        backend = backend or summary["model_backend"]
        precision = precision or summary["model_precision"]
        sweep.append({"concurrency": f"poisson@{r}rps", "results": summary})
        raw.append((f"rate={r}rps", wl))
        _print_run("C", summary, suffix=f" rate={r}rps")

    run_meta["workload"] = "open_loop_poisson"
    run_meta["duration_sec"] = duration
    run_meta["warmup_sec"] = warmup
    run_meta["backend"] = backend
    run_meta["precision"] = precision
    write_sweep_report(run_dir, "C", scenario_cfg, sweep, run_meta, config_path)
    _save_raw_samples(run_dir, "C", raw)
    write_sweep_charts(run_dir)
    print(f"Results: {run_dir}")


def _get_metrics(base_url: str) -> dict | None:
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{base_url.rstrip('/')}/metrics")
            r.raise_for_status()
            return r.json()
    except (httpx.HTTPError, ValueError):
        return None


def _reset_cache(base_url: str) -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.post(f"{base_url.rstrip('/')}/admin/cache/reset")
            return r.status_code == 200
    except httpx.HTTPError:
        return False


def _scenario_d(args, scenario_cfg: dict, base_url: str, run_dir: Path, run_meta: dict, config_path: Path) -> None:
    warmup = args.warmup if args.warmup is not None else float(scenario_cfg.get("warmup_sec", 0))
    phases = scenario_cfg["phases"]

    pre = _get_metrics(base_url)
    wl = spike(base_url, phases=phases, warmup_sec=warmup)
    post = _get_metrics(base_url)

    summary = _summarize(wl)
    run_meta["workload"] = "spike"
    run_meta["phases"] = phases
    run_meta["warmup_sec"] = warmup
    run_meta["metrics_pre"] = pre
    run_meta["metrics_post"] = post
    write_run_report(run_dir, "D", scenario_cfg, summary, run_meta, config_path)
    _save_raw_samples(run_dir, "D", [("spike", wl)])
    write_run_charts(run_dir)
    _print_run("D", summary)
    print(f"Results: {run_dir}")


def _scenario_e(args, scenario_cfg: dict, base_url: str, run_dir: Path, run_meta: dict, config_path: Path) -> None:
    duration = args.duration if args.duration is not None else float(scenario_cfg["duration_sec"])
    warmup = args.warmup if args.warmup is not None else float(scenario_cfg.get("warmup_sec", 0))
    rate = float(scenario_cfg["rate_rps"])
    ratios = scenario_cfg["repeat_ratios"]

    sweep: list[dict] = []
    raw: list[tuple[str, WorkloadResult]] = []
    backend = precision = ""
    for ratio in ratios:
        # W6: reset cache before each sub-run so the observed hit ratio
        # reflects only this ratio's traffic, not stale entries.
        _reset_cache(base_url)
        pre = _get_metrics(base_url)
        wl = cache_repeat(
            base_url,
            rate_rps=rate,
            duration_sec=duration,
            repeat_ratio=float(ratio),
            warmup_sec=warmup,
        )
        post = _get_metrics(base_url)

        summary = _summarize(wl)
        backend = backend or summary["model_backend"]
        precision = precision or summary["model_precision"]

        observed_ratio = None
        if pre and post and "cache" in pre and "cache" in post:
            d_hits = post["cache"]["hits"] - pre["cache"]["hits"]
            d_total = d_hits + (post["cache"]["misses"] - pre["cache"]["misses"])
            observed_ratio = d_hits / d_total if d_total else None

        sweep.append({
            "concurrency": f"repeat={ratio:.2f}",
            "results": summary,
            "observed_cache_hit_ratio": observed_ratio,
        })
        raw.append((f"repeat={ratio:.2f}", wl))
        suffix = f" repeat={ratio:.2f}"
        if observed_ratio is not None:
            suffix += f" hits={observed_ratio * 100:.1f}%"
        _print_run("E", summary, suffix=suffix)

    run_meta["workload"] = "cache_repeat"
    run_meta["duration_sec"] = duration
    run_meta["warmup_sec"] = warmup
    run_meta["rate_rps"] = rate
    run_meta["backend"] = backend
    run_meta["precision"] = precision
    write_sweep_report(run_dir, "E", scenario_cfg, sweep, run_meta, config_path)
    _save_raw_samples(run_dir, "E", raw)
    write_sweep_charts(run_dir)
    print(f"Results: {run_dir}")


def _scenario_f(args, scenario_cfg: dict, base_url: str, run_dir: Path, run_meta: dict, config_path: Path) -> None:
    """Cold-start scenario: spawn a fresh server with warmup disabled,
    measure first /infer (cold) and the next N (warmed)."""
    import subprocess
    import sys

    n_warm = int(scenario_cfg.get("repeat_warm", 100))
    bind_host = "127.0.0.1"
    bind_port = 8765  # avoid collision with the user's already-running 8000

    env = {**os.environ, "INFERBENCH_SKIP_WARMUP": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "inferbench.server.app:app",
         "--host", bind_host, "--port", str(bind_port), "--log-level", "warning"],
        env=env,
    )
    cold_url = f"http://{bind_host}:{bind_port}"

    try:
        # Wait for /health to come up.
        deadline = time.perf_counter() + 30.0
        ready = False
        with httpx.Client(timeout=2.0) as client:
            while time.perf_counter() < deadline:
                try:
                    r = client.get(f"{cold_url}/health")
                    if r.status_code == 200:
                        ready = True
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
        if not ready:
            raise RuntimeError("cold-start server never became healthy")

        cold_ms = None
        warm_ms: list[float] = []
        with httpx.Client(timeout=30.0) as client:
            t0 = time.perf_counter()
            r = client.post(f"{cold_url}/infer", json={"inputs": ["cold start request"]})
            r.raise_for_status()
            cold_ms = (time.perf_counter() - t0) * 1000.0

            for i in range(n_warm):
                t = time.perf_counter()
                rr = client.post(f"{cold_url}/infer", json={"inputs": [f"warm {i}"]})
                rr.raise_for_status()
                warm_ms.append((time.perf_counter() - t) * 1000.0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    warm_stats = latency_stats(warm_ms)
    speedup = cold_ms / warm_stats.p50_ms if warm_stats.p50_ms else float("inf")
    summary = {
        "cold_ms": cold_ms,
        "warm_count": len(warm_ms),
        "warm_p50_ms": warm_stats.p50_ms,
        "warm_p95_ms": warm_stats.p95_ms,
        "warm_p99_ms": warm_stats.p99_ms,
        "warm_mean_ms": warm_stats.mean_ms,
        "cold_to_warm_p50_ratio": speedup,
    }
    run_meta["workload"] = "cold_start"
    run_meta["bind"] = f"{bind_host}:{bind_port}"
    run_meta["repeat_warm"] = n_warm

    payload = {"meta": run_meta, "results": summary, "raw_warm_ms": warm_ms}
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))
    config_text = config_path.read_text()
    (run_dir / "config.snapshot.yaml").write_text(config_text)
    md = (
        f"# Benchmark Run: scenario F\n\n"
        f"**Scenario**: cold-start  \n"
        f"**Started**: {run_meta['started_at']}  \n"
        f"\n"
        f"## Cold vs warm\n\n"
        f"| metric           | value     |\n"
        f"|------------------|----------:|\n"
        f"| cold (1 request) | {cold_ms:>7.2f} ms |\n"
        f"| warm p50         | {warm_stats.p50_ms:>7.2f} ms |\n"
        f"| warm p95         | {warm_stats.p95_ms:>7.2f} ms |\n"
        f"| warm p99         | {warm_stats.p99_ms:>7.2f} ms |\n"
        f"| warm mean        | {warm_stats.mean_ms:>7.2f} ms |\n"
        f"| cold / warm-p50  | {speedup:>7.2f} x |\n"
        f"| warm samples     | {len(warm_ms):>9} |\n"
    )
    (run_dir / "summary.md").write_text(md)
    (run_dir / "raw_latencies.json").write_text(json.dumps(
        {"scenario": "F", "e2e_ms": warm_ms, "cold_ms": cold_ms}
    ))
    write_run_charts(run_dir)
    print(
        f"bench(F): cold={cold_ms:.2f}ms warm_p50={warm_stats.p50_ms:.2f}ms "
        f"warm_p99={warm_stats.p99_ms:.2f}ms ratio={speedup:.1f}x"
    )
    print(f"Results: {run_dir}")


_SCENARIOS = {
    "A": _scenario_a,
    "B": _scenario_b,
    "C": _scenario_c,
    "D": _scenario_d,
    "E": _scenario_e,
    "F": _scenario_f,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=sorted(_SCENARIOS.keys()))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--warmup", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--hardware",
        default=None,
        help="Free-text tag stored in run_meta (e.g. 'a2000-ada', 'cpu-m1', 'a10g-lambda')",
    )
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
        "hardware_tag": args.hardware,
        "system": _system_info(),
    }

    handler = _SCENARIOS[args.scenario]
    handler(args, scenario_cfg, base_url, run_dir, run_meta, args.config)


if __name__ == "__main__":
    main()
