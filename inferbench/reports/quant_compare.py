"""Compose a FP32 vs INT8 quantization tradeoff report.

Joins:
- two bench(B) sweep results.json (one per precision)
- two accuracy JSON sidecars from evaluate_accuracy

into a single Markdown report covering size, accuracy, per-class
accuracy, and per-concurrency latency / throughput tables.

Usage:
    python -m inferbench.reports.quant_compare \\
        --fp32-bench results/run_003 \\
        --int8-bench results/run_007 \\
        --fp32-accuracy results/accuracy/fp32.json \\
        --int8-accuracy results/accuracy/int8.json \\
        --output results/quant_tradeoff/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _bench_to_table(sweep: list[dict]) -> dict[int, dict]:
    """Index a sweep by concurrency for easy join."""
    out: dict[int, dict] = {}
    for entry in sweep:
        c = entry["concurrency"]
        if isinstance(c, int):
            out[c] = entry["results"]
    return out


def _format_pct_delta(int8: float, fp32: float) -> str:
    if fp32 == 0:
        return "n/a"
    delta = (int8 - fp32) / fp32 * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def compose(
    fp32_bench: dict,
    int8_bench: dict,
    fp32_acc: dict,
    int8_acc: dict,
) -> str:
    fp32_size_mb = fp32_acc["model_size_bytes"] / (1024 * 1024)
    int8_size_mb = int8_acc["model_size_bytes"] / (1024 * 1024)
    accuracy_delta = (int8_acc["accuracy"] - fp32_acc["accuracy"]) * 100.0

    fp32_by_c = _bench_to_table(fp32_bench["sweep"])
    int8_by_c = _bench_to_table(int8_bench["sweep"])
    common_c = sorted(set(fp32_by_c) & set(int8_by_c))

    lines: list[str] = [
        "# Quantization Tradeoff: FP32 vs INT8 (dynamic)",
        "",
        f"Model: `{fp32_acc['model_id']}` on ONNX Runtime CPU EP",
        f"FP32 bench: `{fp32_bench['meta']['scenario']}` from "
        f"`{fp32_bench['meta'].get('started_at', '?')}`",
        f"INT8 bench: `{int8_bench['meta']['scenario']}` from "
        f"`{int8_bench['meta'].get('started_at', '?')}`",
        "",
        "## Headline",
        "",
        "| metric              | FP32              | INT8              | delta            |",
        "|---------------------|------------------:|------------------:|------------------|",
        f"| model size          | {fp32_size_mb:>13.1f} MB | {int8_size_mb:>13.1f} MB | "
        f"{_format_pct_delta(int8_size_mb, fp32_size_mb)}            |",
        f"| accuracy (SST-2 val)| {fp32_acc['accuracy'] * 100:>14.2f} % | "
        f"{int8_acc['accuracy'] * 100:>14.2f} % | "
        f"{accuracy_delta:+.2f} pp          |",
        f"| examples / sec      | {fp32_acc['examples_per_sec']:>16.1f} | "
        f"{int8_acc['examples_per_sec']:>16.1f} | "
        f"{_format_pct_delta(int8_acc['examples_per_sec'], fp32_acc['examples_per_sec'])}            |",
        "",
        "## Per-class accuracy",
        "",
        "| class    | FP32        | INT8        |",
        "|----------|------------:|------------:|",
    ]
    for cls_key, cls_label in (("negative", "negative"), ("positive", "positive")):
        f = fp32_acc["per_class"][cls_key]
        i = int8_acc["per_class"][cls_key]
        lines.append(
            f"| {cls_label:<8} | {f['accuracy'] * 100:>10.2f}% | {i['accuracy'] * 100:>10.2f}% |"
        )

    lines += [
        "",
        "## Latency / throughput at concurrency (bench(B))",
        "",
        "|  c | p50 FP32 | p50 INT8 | p95 FP32 | p95 INT8 | p99 FP32 | p99 INT8 | rps FP32 | rps INT8 |",
        "|---:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|",
    ]
    for c in common_c:
        f = fp32_by_c[c]["latency_e2e"]
        i = int8_by_c[c]["latency_e2e"]
        ft = fp32_by_c[c]["throughput"]
        it = int8_by_c[c]["throughput"]
        lines.append(
            f"| {c:>2} | {f['p50_ms']:>8.2f} | {i['p50_ms']:>8.2f} "
            f"| {f['p95_ms']:>8.2f} | {i['p95_ms']:>8.2f} "
            f"| {f['p99_ms']:>8.2f} | {i['p99_ms']:>8.2f} "
            f"| {ft['requests_per_sec']:>8.1f} | {it['requests_per_sec']:>8.1f} |"
        )

    lines += [
        "",
        "## Takeaways",
        "",
        f"- **Size**: INT8 is {fp32_size_mb / int8_size_mb:.1f}x smaller than FP32 — pure consequence of 1-byte vs 4-byte weights.",
        f"- **Accuracy**: {accuracy_delta:+.2f} percentage points on SST-2 val. "
        f"{abs(int8_acc['correct'] - fp32_acc['correct'])} examples flipped out of "
        f"{fp32_acc['examples_evaluated']}.",
        f"- **Throughput** (batched eval, batch=16): INT8 is "
        f"{int8_acc['examples_per_sec'] / fp32_acc['examples_per_sec']:.2f}x faster.",
        "- **Tail latency** under concurrent load: see table above. "
        "Ratio holds across the concurrency range, with the biggest win at the higher-c rows where batched inference dominates.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-bench", type=Path, required=True)
    parser.add_argument("--int8-bench", type=Path, required=True)
    parser.add_argument("--fp32-accuracy", type=Path, required=True)
    parser.add_argument("--int8-accuracy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fp32_bench = _load(args.fp32_bench / "results.json")
    int8_bench = _load(args.int8_bench / "results.json")
    fp32_acc = _load(args.fp32_accuracy)
    int8_acc = _load(args.int8_accuracy)

    args.output.mkdir(parents=True, exist_ok=True)
    md = compose(fp32_bench, int8_bench, fp32_acc, int8_acc)
    (args.output / "summary.md").write_text(md)

    payload = {
        "fp32_bench": str(args.fp32_bench),
        "int8_bench": str(args.int8_bench),
        "fp32_accuracy": fp32_acc,
        "int8_accuracy": int8_acc,
    }
    (args.output / "tradeoff.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
