"""Regenerate the README cover chart at results/headline/concurrency_sweep.png.

Reads a hand-picked set of bench(B) runs and produces a two-panel figure:

- Top: throughput (req/s) vs concurrency
- Bottom: p99 latency (ms) vs concurrency, log y

CLI:
    python -m inferbench.reports.headline_concurrency_sweep
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

# Each tuple: (label, run_dir relative to repo root, color, linestyle)
RUNS: list[tuple[str, str, str, str]] = [
    ("CPU fp32, no batch (run_002)",       "results/run_002", "#999999", "--"),
    ("CPU fp32, static-batch (run_003)",   "results/run_003", "#4c72b0", "-"),
    ("CPU int8, static-batch (run_007)",   "results/run_007", "#55a868", "-"),
    ("CUDA fp32, dyn-batch (run_014)",     "results/run_014", "#c44e52", "-"),
    ("CUDA fp16, dyn-batch (run_015)",     "results/run_015", "#8172b2", "-"),
]


def _load_sweep(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "results.json").read_text())["sweep"]


def main() -> None:
    fig, (ax_t, ax_l) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for label, rel, color, ls in RUNS:
        run_dir = ROOT / rel
        if not (run_dir / "results.json").exists():
            print(f"skip (missing): {rel}")
            continue
        sweep = _load_sweep(run_dir)
        cs = [e["concurrency"] for e in sweep]
        rps = [e["results"]["throughput"]["requests_per_sec"] for e in sweep]
        p99 = [e["results"]["latency_e2e"]["p99_ms"] for e in sweep]
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
        "DistilBERT-SST2 — closed-loop concurrency sweep\n"
        "CPU (M-series, ORT 1.25) vs CUDA EP (RTX A2000 12GB, ORT 1.26)",
        fontsize=11,
    )
    fig.tight_layout()

    out = ROOT / "results" / "headline" / "concurrency_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
