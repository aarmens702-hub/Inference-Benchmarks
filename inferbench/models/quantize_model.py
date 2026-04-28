"""ONNX Runtime dynamic INT8 quantization.

Reads a previously-exported FP32 model directory, writes a sibling
INT8-quantized directory:

    models/distilbert-sst2-fp32/  ->  models/distilbert-sst2-int8/

The output dir contains:
- model.onnx          (quantized weights, INT8)
- tokenizer.json, vocab.txt, ... (copied from input — no change)
- metadata.json       (precision="int8-dynamic", new model_size_bytes)

Dynamic quantization quantizes weights to INT8 at export time and
activations on the fly at runtime; no calibration data required. Static
quantization (calibration-based) is a stretch goal for W6+.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_INPUT = Path("models/distilbert-sst2-fp32")
DEFAULT_OUTPUT = Path("models/distilbert-sst2-int8")

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "config.json",
)


def quantize(input_dir: Path, output_dir: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if not (input_dir / "model.onnx").exists():
        raise FileNotFoundError(f"{input_dir / 'model.onnx'} not found — run export_model first")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = output_dir / "model.onnx"

    quantize_dynamic(
        model_input=str(input_dir / "model.onnx"),
        model_output=str(out_onnx),
        weight_type=QuantType.QInt8,
    )

    for fname in _TOKENIZER_FILES:
        src = input_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    src_meta = json.loads((input_dir / "metadata.json").read_text())
    metadata = {
        "model_id": src_meta["model_id"],
        "precision": "int8-dynamic",
        "labels": src_meta["labels"],
        "max_position_embeddings": src_meta.get("max_position_embeddings", 512),
        "model_size_bytes": out_onnx.stat().st_size,
        "source_model_size_bytes": src_meta.get("model_size_bytes"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    out = quantize(args.input_dir, args.output_dir)
    src_size = (args.input_dir / "model.onnx").stat().st_size / (1024 * 1024)
    out_size = out.stat().st_size / (1024 * 1024)
    print(f"FP32 size: {src_size:.1f} MB")
    print(f"INT8 size: {out_size:.1f} MB ({out_size / src_size * 100:.1f}% of FP32)")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
