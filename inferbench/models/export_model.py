"""Export a HuggingFace sequence-classification model to ONNX FP32.

Default target: distilbert-base-uncased-finetuned-sst-2-english (SST-2 sentiment).

Usage:
    python -m inferbench.models.export_model \\
        --model-id distilbert-base-uncased-finetuned-sst-2-english \\
        --output-dir models/distilbert-sst2-fp32

The script also writes a small metadata.json sidecar so the runner can
verify the model_id, input shapes, and label mapping at load time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_OUTPUT = Path("models/distilbert-sst2-fp32")


def export(model_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    onnx_path = output_dir / "model.onnx"
    metadata = {
        "model_id": model_id,
        "precision": "fp32",
        "labels": model.config.id2label,
        "max_position_embeddings": getattr(model.config, "max_position_embeddings", 512),
        "model_size_bytes": onnx_path.stat().st_size if onnx_path.exists() else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    onnx_path = export(args.model_id, args.output_dir)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported: {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
