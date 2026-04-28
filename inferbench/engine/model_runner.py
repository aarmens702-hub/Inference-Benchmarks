"""ONNX Runtime ModelRunner for sequence-classification models.

Owns:
- The InferenceSession (chosen execution provider)
- The HF tokenizer (preprocessing)
- The label mapping (postprocessing)

Exposes a synchronous `run(texts)` that takes a batch of strings and
returns predictions plus a timing breakdown. Async wrapping happens
upstream in the server layer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


@dataclass
class RunResult:
    predictions: list[Prediction]
    inference_ms: float
    preprocess_ms: float
    postprocess_ms: float
    batch_size: int


@dataclass
class ModelMetadata:
    model_id: str
    precision: str
    backend: str
    labels: dict[int, str]
    model_size_bytes: int
    max_position_embeddings: int = 512
    extra: dict = field(default_factory=dict)


_PROVIDERS_BY_BACKEND = {
    "onnxruntime-cpu": ["CPUExecutionProvider"],
    "onnxruntime-cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


class ModelRunner:
    """Single-process ONNX Runtime model runner."""

    def __init__(
        self,
        model_dir: str | Path,
        backend: str = "onnxruntime-cpu",
        intra_op_num_threads: int | None = None,
        max_seq_length: int = 128,
    ) -> None:
        self.model_dir = Path(model_dir)
        if backend not in _PROVIDERS_BY_BACKEND:
            raise ValueError(f"Unsupported backend: {backend}")
        self.backend = backend
        self.max_seq_length = max_seq_length

        metadata_path = self.model_dir / "metadata.json"
        meta_raw = json.loads(metadata_path.read_text())
        self.metadata = ModelMetadata(
            model_id=meta_raw["model_id"],
            precision=meta_raw["precision"],
            backend=backend,
            labels={int(k): v for k, v in meta_raw["labels"].items()},
            model_size_bytes=meta_raw["model_size_bytes"],
            max_position_embeddings=meta_raw.get("max_position_embeddings", 512),
        )

        sess_options = ort.SessionOptions()
        if intra_op_num_threads is not None:
            sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        onnx_path = self.model_dir / "model.onnx"
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=_PROVIDERS_BY_BACKEND[backend],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._input_names = {inp.name for inp in self.session.get_inputs()}

    def warmup(self, n: int = 3, sample_text: str = "warmup input text") -> None:
        for _ in range(n):
            self.run([sample_text])

    def run(self, texts: Sequence[str]) -> RunResult:
        if len(texts) == 0:
            raise ValueError("Cannot run inference on empty batch")

        t0 = time.perf_counter()
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )
        feed = {name: encoded[name].astype(np.int64) for name in encoded if name in self._input_names}
        t1 = time.perf_counter()

        outputs = self.session.run(None, feed)
        logits = outputs[0]
        t2 = time.perf_counter()

        # softmax over class axis
        max_logits = logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits - max_logits)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        pred_idx = probs.argmax(axis=-1)

        predictions = [
            Prediction(
                label=self.metadata.labels[int(idx)],
                score=float(probs[i, idx]),
            )
            for i, idx in enumerate(pred_idx)
        ]
        t3 = time.perf_counter()

        return RunResult(
            predictions=predictions,
            preprocess_ms=(t1 - t0) * 1000.0,
            inference_ms=(t2 - t1) * 1000.0,
            postprocess_ms=(t3 - t2) * 1000.0,
            batch_size=len(texts),
        )
