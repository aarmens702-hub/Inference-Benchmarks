"""GPU smoke test — verifies onnxruntime-gpu can actually reach CUDA.

A silent EP fallback to CPU is the most expensive bug in this project:
benchmarks run, numbers look plausible, but every claim is wrong. This
script makes the fallback loud. Exits 0 on success, 1 on any failure.

Run: python -m inferbench.tools.gpu_smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnxruntime as ort

from inferbench.engine.model_runner import ModelRunner, resolve_providers


def main() -> int:
    print(f"ort version           : {ort.__version__}")
    available = ort.get_available_providers()
    print(f"available providers   : {available}")

    if "CUDAExecutionProvider" not in available:
        print(
            "FAIL: CUDAExecutionProvider not in available providers.\n"
            "      onnxruntime-gpu is not seeing CUDA. Common causes:\n"
            "      - the CPU 'onnxruntime' wheel is still installed (uninstall it)\n"
            "      - LD_LIBRARY_PATH doesn't include /usr/local/cuda-*/lib64\n"
            "      - cuDNN version mismatch (need cuDNN 9.x for ort 1.20)",
            file=sys.stderr,
        )
        return 1

    providers, effective = resolve_providers("onnxruntime-cuda")
    print(f"resolved providers    : {providers}")
    print(f"effective backend tag : {effective}")
    if effective != "onnxruntime-cuda":
        print(f"FAIL: resolver picked {effective}, expected onnxruntime-cuda", file=sys.stderr)
        return 1

    fp32_dir = Path("models/distilbert-sst2-fp32")
    if not (fp32_dir / "model.onnx").exists():
        print(
            f"SKIP inference check: {fp32_dir}/model.onnx not present. "
            "Run `make export-model` then re-run this smoke test.",
            file=sys.stderr,
        )
        print("GPU SMOKE TEST: PROVIDERS OK (model missing)")
        return 0

    runner = ModelRunner(model_dir=fp32_dir, backend="onnxruntime-cuda")
    session_providers = runner.session.get_providers()
    print(f"session providers     : {session_providers}")
    if session_providers[0] != "CUDAExecutionProvider":
        print(
            f"FAIL: session opened with {session_providers[0]} as head, "
            "expected CUDAExecutionProvider. Silent EP fallback.",
            file=sys.stderr,
        )
        return 1

    result = runner.run(["this movie was surprisingly good"])
    pred = result.predictions[0]
    print(f"sample inference ok   : {pred.label} (score {pred.score:.4f})")
    print(f"inference_ms          : {result.inference_ms:.2f}")
    print("GPU SMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
