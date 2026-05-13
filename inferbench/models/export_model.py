"""Export a HuggingFace sequence-classification model to ONNX.

Default target: distilbert-base-uncased-finetuned-sst-2-english (SST-2 sentiment).

Usage:
    # fp32 (default)
    python -m inferbench.models.export_model \\
        --model-id distilbert-base-uncased-finetuned-sst-2-english \\
        --output-dir models/distilbert-sst2-fp32

    # fp16 (for GPU/CUDA EP) — converts the fp32 graph in-place
    python -m inferbench.models.export_model \\
        --precision fp16 \\
        --output-dir models/distilbert-sst2-fp16

The script also writes a small metadata.json sidecar so the runner can
verify the model_id, input shapes, and label mapping at load time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_OUTPUT_FP32 = Path("models/distilbert-sst2-fp32")
DEFAULT_OUTPUT_FP16 = Path("models/distilbert-sst2-fp16")


def export_fp32(model_id: str, output_dir: Path) -> Path:
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


def _convert_fp16_graph(fp32_onnx: Path, fp16_onnx: Path) -> None:
    """Convert an fp32 ONNX graph to fp16.

    Prefers `onnxruntime.transformers.float16.convert_float_to_float16` which
    properly rewrites Cast `to=FLOAT` attributes to `to=FLOAT16` on transformer
    graphs. Falls back to `onnxconverter_common.float16.convert_float_to_float16`
    if the ORT helper isn't available (older onnxruntime wheels).

    The fallback path can produce graphs where a Cast node still declares its
    output as fp32 even though its input has become fp16, which makes ORT fail
    to load the model with a tensor-type mismatch. The ORT helper is the only
    reliable converter for the DistilBERT family.
    """
    import onnx

    model = onnx.load(str(fp32_onnx))
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
        model_fp16 = convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_shape_infer=False,
        )
    except ImportError:
        from onnxconverter_common import float16  # pragma: no cover
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(fp16_onnx))


def export_fp16(model_id: str, output_dir: Path, fp32_dir: Path) -> Path:
    """Convert an existing fp32 ONNX export to fp16 in-place.

    Pure graph transform via ONNX Runtime's transformer-aware float16 helper.
    Reuses the fp32 export's tokenizer and metadata so the runner sees the
    same label mapping and input names.
    """
    import shutil

    if not fp32_dir.exists():
        raise FileNotFoundError(
            f"fp32 source dir {fp32_dir} not found — run --precision fp32 first"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer + config files (everything except the fp32 .onnx).
    for src in fp32_dir.iterdir():
        if src.name == "model.onnx" or src.name == "metadata.json":
            continue
        dst = output_dir / src.name
        if src.is_file():
            shutil.copy2(src, dst)

    fp32_onnx = fp32_dir / "model.onnx"
    fp16_onnx = output_dir / "model.onnx"
    _convert_fp16_graph(fp32_onnx, fp16_onnx)

    fp32_meta = json.loads((fp32_dir / "metadata.json").read_text())
    metadata = {
        "model_id": model_id,
        "precision": "fp16",
        "labels": fp32_meta["labels"],
        "max_position_embeddings": fp32_meta.get("max_position_embeddings", 512),
        "model_size_bytes": fp16_onnx.stat().st_size,
        "source_fp32_size_bytes": fp32_meta.get("model_size_bytes"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return fp16_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--fp32-source-dir",
        type=Path,
        default=DEFAULT_OUTPUT_FP32,
        help="Source fp32 export to convert (only used when --precision fp16)",
    )
    args = parser.parse_args()

    if args.precision == "fp32":
        out = args.output_dir or DEFAULT_OUTPUT_FP32
        onnx_path = export_fp32(args.model_id, out)
    else:
        out = args.output_dir or DEFAULT_OUTPUT_FP16
        onnx_path = export_fp16(args.model_id, out, args.fp32_source_dir)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported [{args.precision}]: {onnx_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
