"""Markdown + JSON report generation for benchmark runs.

Single-run reports are written by run_benchmark.py via write_run_report().
Multi-run sweep reports (e.g. Scenario B's concurrency sweep) are
written by write_sweep_report().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(d: dict, key: str, fmt: str = ".2f", default: float = 0.0) -> str:
    val = d.get(key, default) if d else default
    return f"{val:{fmt}}"


def format_run_summary(scenario_id: str, scenario_cfg: dict, results: dict, run_meta: dict) -> str:
    lat = results["latency_e2e"]
    inf = results.get("latency_inference") or {}
    qw = results.get("latency_queue_wait") or {}
    tput = results["throughput"]

    return f"""# Benchmark Run: scenario {scenario_id}

**Scenario**: {scenario_cfg["name"]}
**Started**: {run_meta["started_at"]}
**Duration**: {tput["duration_sec"]:.1f}s
**Backend**: {results["model_backend"]} | **Precision**: {results["model_precision"]}
**Workload**: {run_meta.get("workload", "n/a")}

## End-to-end latency (ms)

| count | mean | p50 | p90 | p95 | p99 | min | max |
|------:|-----:|----:|----:|----:|----:|----:|----:|
| {lat["count"]} | {lat["mean_ms"]:.2f} | {lat["p50_ms"]:.2f} | {lat["p90_ms"]:.2f} | {lat["p95_ms"]:.2f} | {lat["p99_ms"]:.2f} | {lat["min_ms"]:.2f} | {lat["max_ms"]:.2f} |

## Inference-only latency (ms)

| p50 | p95 | p99 |
|----:|----:|----:|
| {_fmt(inf, "p50_ms")} | {_fmt(inf, "p95_ms")} | {_fmt(inf, "p99_ms")} |

## Queue wait (ms)

| p50 | p95 | p99 |
|----:|----:|----:|
| {_fmt(qw, "p50_ms")} | {_fmt(qw, "p95_ms")} | {_fmt(qw, "p99_ms")} |

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


def format_sweep_summary(scenario_id: str, scenario_cfg: dict, sweep: list[dict], run_meta: dict) -> str:
    lines: list[str] = [
        f"# Benchmark Sweep: scenario {scenario_id}",
        "",
        f"**Scenario**: {scenario_cfg['name']}",
        f"**Started**: {run_meta['started_at']}",
        f"**Backend**: {run_meta.get('backend', '?')} | **Precision**: {run_meta.get('precision', '?')}",
        f"**Workload**: {run_meta.get('workload', 'closed_loop')}",
        "",
        "## Latency vs concurrency",
        "",
        "| concurrency | n | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | rps | err % |",
        "|------------:|--:|-------:|-------:|-------:|-------:|--------:|----:|------:|",
    ]
    for entry in sweep:
        lat = entry["results"].get("latency_e2e")
        tput = entry["results"]["throughput"]
        if lat is None:
            # No successful samples (every request errored — common when a
            # backend is unstable under concurrent load). Render a placeholder
            # row instead of crashing the report.
            lines.append(
                f"| {entry['concurrency']} | 0 | n/a | n/a | n/a | n/a | n/a "
                f"| {tput['requests_per_sec']:.1f} | {tput['error_rate'] * 100:.2f} |"
            )
            continue
        lines.append(
            f"| {entry['concurrency']} | {lat['count']} "
            f"| {lat['p50_ms']:.2f} | {lat['p90_ms']:.2f} | {lat['p95_ms']:.2f} | {lat['p99_ms']:.2f} "
            f"| {lat['mean_ms']:.2f} | {tput['requests_per_sec']:.1f} "
            f"| {tput['error_rate'] * 100:.2f} |"
        )
    lines += [
        "",
        "## Environment",
        f"- Host: `{run_meta['system']['hostname']}`",
        f"- Platform: `{run_meta['system']['platform']}`",
        f"- Python: `{run_meta['system']['python_version']}`",
        f"- CPU count: {run_meta['system']['cpu_count']}",
        "",
    ]
    return "\n".join(lines)


def write_run_report(
    run_dir: Path,
    scenario_id: str,
    scenario_cfg: dict,
    results: dict,
    run_meta: dict,
    config_snapshot_path: Path | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"meta": run_meta, "results": results}
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "summary.md").write_text(format_run_summary(scenario_id, scenario_cfg, results, run_meta))
    if config_snapshot_path is not None:
        (run_dir / "config.snapshot.yaml").write_text(Path(config_snapshot_path).read_text())


def write_sweep_report(
    run_dir: Path,
    scenario_id: str,
    scenario_cfg: dict,
    sweep: list[dict],
    run_meta: dict,
    config_snapshot_path: Path | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": run_meta, "sweep": sweep}
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (run_dir / "summary.md").write_text(format_sweep_summary(scenario_id, scenario_cfg, sweep, run_meta))
    if config_snapshot_path is not None:
        (run_dir / "config.snapshot.yaml").write_text(Path(config_snapshot_path).read_text())
