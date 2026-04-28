"""Matplotlib chart generation for benchmark runs.

Two entry points:

- write_run_charts(run_dir): single-run CDF + breakdown chart.
- write_sweep_charts(run_dir): per-concurrency p50/p95/p99 curves +
  throughput vs concurrency curve.

Looks for raw_latencies.json (written by run_benchmark when present)
to draw the latency CDF; falls back to percentile-only visuals when
raw samples are unavailable.

CLI:
    python -m inferbench.reports.plot_results --run-dir results/run_007
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Use a non-interactive backend for headless / CI use.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _ensure_charts_dir(run_dir: Path) -> Path:
    out = run_dir / "charts"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plot_cdf(latencies: list[float], title: str, out_path: Path) -> None:
    if not latencies:
        return
    arr = np.sort(np.asarray(latencies, dtype=np.float64))
    cdf = np.arange(1, len(arr) + 1) / len(arr)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(arr, cdf, linewidth=1.5)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    for p, label in [(p50, "p50"), (p95, "p95"), (p99, "p99")]:
        ax.axvline(p, linestyle="--", alpha=0.4)
        ax.annotate(f"{label} = {p:.1f} ms", (p, 0.05),
                    rotation=90, fontsize=8, alpha=0.7)
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_sweep_latency(sweep: list[dict], x_label: str, title: str, out_path: Path) -> None:
    xs: list = []
    p50s: list[float] = []
    p95s: list[float] = []
    p99s: list[float] = []
    for entry in sweep:
        c = entry["concurrency"]
        lat = entry["results"]["latency_e2e"]
        xs.append(c)
        p50s.append(lat["p50_ms"])
        p95s.append(lat["p95_ms"])
        p99s.append(lat["p99_ms"])

    x_pos = list(range(len(xs)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_pos, p50s, marker="o", label="p50")
    ax.plot(x_pos, p95s, marker="s", label="p95")
    ax.plot(x_pos, p99s, marker="^", label="p99")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(x_label)
    ax.set_ylabel("latency (ms)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_sweep_throughput(sweep: list[dict], x_label: str, title: str, out_path: Path) -> None:
    xs: list = []
    rps: list[float] = []
    for entry in sweep:
        xs.append(entry["concurrency"])
        rps.append(entry["results"]["throughput"]["requests_per_sec"])

    x_pos = list(range(len(xs)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_pos, rps, marker="o", color="tab:green")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(x_label)
    ax.set_ylabel("throughput (req/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_run_charts(run_dir: Path) -> list[Path]:
    """Charts for a single-config run (Scenarios A, D)."""
    charts: list[Path] = []
    out_dir = _ensure_charts_dir(run_dir)
    raw_path = run_dir / "raw_latencies.json"
    if not raw_path.exists():
        return charts
    raw = _load(raw_path)
    title = f"Latency CDF: scenario {raw.get('scenario', '?')}"
    cdf_path = out_dir / "latency_cdf.png"
    _plot_cdf(raw.get("e2e_ms", []), title, cdf_path)
    charts.append(cdf_path)
    return charts


def write_sweep_charts(run_dir: Path) -> list[Path]:
    """Charts for sweep runs (Scenarios B, C, E)."""
    charts: list[Path] = []
    results = _load(run_dir / "results.json")
    sweep = results.get("sweep")
    if not sweep:
        return charts
    out_dir = _ensure_charts_dir(run_dir)
    scenario = results["meta"].get("scenario", "?")
    workload = results["meta"].get("workload", "")
    x_label = "concurrency" if workload != "open_loop_poisson" else "rate (rps)"
    if not isinstance(sweep[0]["concurrency"], int):
        x_label = "config"

    lat_path = out_dir / "latency_curves.png"
    _plot_sweep_latency(sweep, x_label, f"Latency vs {x_label}: scenario {scenario}", lat_path)
    charts.append(lat_path)

    tput_path = out_dir / "throughput_curve.png"
    _plot_sweep_throughput(sweep, x_label, f"Throughput vs {x_label}: scenario {scenario}", tput_path)
    charts.append(tput_path)
    return charts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    results = _load(args.run_dir / "results.json")
    if "sweep" in results:
        produced = write_sweep_charts(args.run_dir)
    else:
        produced = write_run_charts(args.run_dir)
    if not produced:
        print(f"No charts produced for {args.run_dir} (no raw_latencies.json or sweep data?)")
        return
    for p in produced:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
