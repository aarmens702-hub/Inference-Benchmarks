"""Build the bench(B) headline charts under results/headline/.

Produces three figures:

- concurrency_sweep.png: two-panel throughput + p99 across CPU and CUDA runs.
- cpu_vs_cuda_peak.png: peak throughput bar chart.
- fp32_vs_fp16_gpu.png: GPU-only fp32 vs fp16 throughput and latency.

CLI:
    python -m inferbench.reports.headline_concurrency_sweep
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RUNS: list[tuple[str, str, str, str]] = [
    ("CPU fp32, no batch (run_002)",     "results/run_002", "#999999", "--"),
    ("CPU fp32, static-batch (run_003)", "results/run_003", "#4c72b0", "-"),
    ("CPU int8, static-batch (run_007)", "results/run_007", "#55a868", "-"),
    ("CUDA fp32, dyn-batch (run_014)",   "results/run_014", "#c44e52", "-"),
    ("CUDA fp16, dyn-batch (run_015)",   "results/run_015", "#8172b2", "-"),
]


def _load_sweep(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "results.json").read_text())["sweep"]


def _series(run_dir: Path) -> tuple[list[int], list[float], list[float]]:
    sweep = _load_sweep(run_dir)
    cs = [e["concurrency"] for e in sweep]
    rps = [e["results"]["throughput"]["requests_per_sec"] for e in sweep]
    p99 = [e["results"]["latency_e2e"]["p99_ms"] for e in sweep]
    return cs, rps, p99


def write_concurrency_sweep(out_dir: Path) -> Path:
    fig, (ax_t, ax_l) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for label, rel, color, ls in RUNS:
        run_dir = ROOT / rel
        if not (run_dir / "results.json").exists():
            print(f"skip (missing): {rel}")
            continue
        cs, rps, p99 = _series(run_dir)
        ax_t.plot(cs, rps, marker="o", label=label, color=color, linestyle=ls, linewidth=1.8)
        ax_l.plot(cs, p99, marker="o", label=label, color=color, linestyle=ls, linewidth=1.8)

    concurrencies = [1, 4, 8, 16, 32, 64]
    for ax in (ax_t, ax_l):
        ax.set_xscale("log")
        ax.set_xticks(concurrencies)
        ax.set_xticklabels([str(c) for c in concurrencies])
        ax.grid(True, which="both", alpha=0.3)

    ax_t.set_ylabel("throughput (req/s)")
    ax_t.set_title("bench(B): throughput vs concurrency")
    ax_t.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax_l.set_ylabel("p99 latency (ms)")
    ax_l.set_xlabel("concurrent clients")
    ax_l.set_yscale("log")
    ax_l.set_title("bench(B): p99 latency vs concurrency")

    fig.suptitle(
        "DistilBERT-SST2, closed-loop concurrency sweep\n"
        "CPU (M-series, ORT 1.25) vs CUDA EP (RTX A2000 12GB, ORT 1.26)",
        fontsize=11,
    )
    fig.tight_layout()
    out = out_dir / "concurrency_sweep.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def write_cpu_vs_cuda_peak(out_dir: Path) -> Path:
    labels: list[str] = []
    peaks: list[float] = []
    colors: list[str] = []
    for label, rel, color, _ls in RUNS:
        run_dir = ROOT / rel
        if not (run_dir / "results.json").exists():
            continue
        _cs, rps, _p99 = _series(run_dir)
        short = label.split(" (")[0]
        labels.append(short)
        peaks.append(max(rps))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, peaks, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, peaks):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("peak throughput (req/s)")
    ax.set_title("Peak bench(B) throughput by backend + precision")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(peaks) * 1.18)
    fig.tight_layout()
    out = out_dir / "cpu_vs_cuda_peak.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def write_fp32_vs_fp16_gpu(out_dir: Path) -> Path:
    fp32 = ROOT / "results/run_014"
    fp16 = ROOT / "results/run_015"
    cs32, rps32, p99_32 = _series(fp32)
    cs16, rps16, p99_16 = _series(fp16)
    assert cs32 == cs16, "sweep concurrencies differ"

    fig, (ax_t, ax_l) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax_t.plot(cs32, rps32, marker="o", color="#c44e52", label="CUDA fp32", linewidth=1.8)
    ax_t.plot(cs16, rps16, marker="s", color="#8172b2", label="CUDA fp16", linewidth=1.8)
    ax_t.set_xscale("log")
    ax_t.set_xticks(cs32)
    ax_t.set_xticklabels([str(c) for c in cs32])
    ax_t.set_xlabel("concurrent clients")
    ax_t.set_ylabel("throughput (req/s)")
    ax_t.set_title("throughput vs concurrency")
    ax_t.legend(loc="upper left")
    ax_t.grid(True, which="both", alpha=0.3)

    pct = [(b - a) / a * 100 for a, b in zip(rps32, rps16)]
    bar_colors = ["#8172b2" if v >= 0 else "#c44e52" for v in pct]
    x = np.arange(len(cs32))
    bars = ax_l.bar(x, pct, color=bar_colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, pct):
        offset = 0.5 if v >= 0 else -0.5
        va = "bottom" if v >= 0 else "top"
        ax_l.text(b.get_x() + b.get_width() / 2, v + offset, f"{v:+.1f}%",
                  ha="center", va=va, fontsize=9)
    ax_l.axhline(0, color="black", linewidth=0.8)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([str(c) for c in cs32])
    ax_l.set_xlabel("concurrent clients")
    ax_l.set_ylabel("fp16 vs fp32 throughput (%)")
    ax_l.set_title("fp16 delta over fp32, by concurrency")
    ax_l.grid(True, axis="y", alpha=0.3)

    fig.suptitle("CUDA EP, RTX A2000 12GB: fp32 vs fp16", fontsize=11)
    fig.tight_layout()
    out = out_dir / "fp32_vs_fp16_gpu.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    out_dir = ROOT / "results" / "headline"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in (write_concurrency_sweep(out_dir),
                 write_cpu_vs_cuda_peak(out_dir),
                 write_fp32_vs_fp16_gpu(out_dir)):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
