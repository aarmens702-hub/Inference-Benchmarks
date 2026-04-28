"""Evaluate sentiment-classification accuracy against the SST-2 validation split.

Usage:
    python -m inferbench.models.evaluate_accuracy --model-dir models/distilbert-sst2-fp32
    python -m inferbench.models.evaluate_accuracy --model-dir models/distilbert-sst2-int8 --output results/accuracy/int8.json

Loads GLUE/SST-2 validation (872 examples) via HuggingFace `datasets`,
runs the local ModelRunner over batches, and reports overall accuracy
plus a per-class breakdown. Writes a JSON sidecar so a later
quant-tradeoff report can join it against bench(*) numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from inferbench.engine.model_runner import ModelRunner

DEFAULT_MODEL_DIR = Path("models/distilbert-sst2-fp32")
DEFAULT_OUTPUT = Path("results/accuracy/fp32.json")


# SST-2 label space:
#   GLUE/SST-2: 0 = negative, 1 = positive
#   DistilBERT-SST2: id2label = {0: NEGATIVE, 1: POSITIVE}
# So a model prediction with idx == ground-truth means correct.
_PRED_LABEL_TO_IDX = {"NEGATIVE": 0, "POSITIVE": 1}


def _load_sst2_validation(limit: int | None) -> list[tuple[str, int]]:
    from datasets import load_dataset

    ds = load_dataset("glue", "sst2", split="validation")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return [(row["sentence"], int(row["label"])) for row in ds]


def evaluate(model_dir: Path, batch_size: int = 16, limit: int | None = None) -> dict:
    runner = ModelRunner(model_dir)
    runner.warmup(n=2)
    examples = _load_sst2_validation(limit)

    correct = 0
    total = 0
    per_class = {0: {"correct": 0, "total": 0}, 1: {"correct": 0, "total": 0}}

    t0 = time.perf_counter()
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        texts = [t for t, _ in batch]
        labels = [y for _, y in batch]
        out = runner.run(texts)
        for pred, y in zip(out.predictions, labels):
            pred_idx = _PRED_LABEL_TO_IDX[pred.label]
            per_class[y]["total"] += 1
            if pred_idx == y:
                per_class[y]["correct"] += 1
                correct += 1
            total += 1
    elapsed = time.perf_counter() - t0

    return {
        "model_dir": str(model_dir),
        "model_id": runner.metadata.model_id,
        "precision": runner.metadata.precision,
        "model_size_bytes": runner.metadata.model_size_bytes,
        "examples_evaluated": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "per_class": {
            "negative": {
                "correct": per_class[0]["correct"],
                "total": per_class[0]["total"],
                "accuracy": per_class[0]["correct"] / per_class[0]["total"]
                if per_class[0]["total"]
                else 0.0,
            },
            "positive": {
                "correct": per_class[1]["correct"],
                "total": per_class[1]["total"],
                "accuracy": per_class[1]["correct"] / per_class[1]["total"]
                if per_class[1]["total"]
                else 0.0,
            },
        },
        "batch_size": batch_size,
        "elapsed_sec": elapsed,
        "examples_per_sec": total / elapsed if elapsed else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="evaluate first N examples only")
    args = parser.parse_args()

    result = evaluate(args.model_dir, batch_size=args.batch_size, limit=args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print(
        f"{result['precision']:>14}  acc={result['accuracy'] * 100:.2f}%  "
        f"({result['correct']}/{result['examples_evaluated']})  "
        f"size={result['model_size_bytes'] / (1024 * 1024):.1f} MB  "
        f"{result['examples_per_sec']:.1f} ex/s"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
