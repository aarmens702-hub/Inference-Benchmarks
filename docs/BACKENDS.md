# Execution Backends

InferBench picks an ONNX Runtime execution-provider chain from
`configs/server.yaml > model.backend`. Each named backend is a
priority-ordered list; the first provider actually available at
runtime wins, the rest fall through.

| backend (config name)   | provider chain                                                        | targeted hardware     |
|-------------------------|-----------------------------------------------------------------------|-----------------------|
| `onnxruntime-cpu`       | `CPUExecutionProvider`                                                | any CPU (default)     |
| `onnxruntime-cuda`      | `CUDAExecutionProvider`, `CPUExecutionProvider`                       | NVIDIA GPU            |
| `onnxruntime-coreml`    | `CoreMLExecutionProvider`, `CPUExecutionProvider`                     | Apple Silicon / macOS |
| `onnxruntime-tensorrt`  | `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider` | NVIDIA GPU + TRT |

`model_runner.resolve_providers` filters the chain against
`onnxruntime.get_available_providers()`. If none of the requested
providers are present, startup fails with an explicit message — we
never silently downgrade unless the chain itself permits a fallback.

`ModelMetadata.extra.providers_used` and
`ModelMetadata.extra.requested_backend` record what was asked for
versus what ran, so benchmark reports can detect a fallback after
the fact.

## CPU (default)

```yaml
model:
  backend: onnxruntime-cpu
```

ONNX Runtime ships with the CPU EP by default. On Apple Silicon
this is unusually fast — for DistilBERT-SST2 the M-series CPU
mean inference time is ~3 ms (FP32) and ~1.6 ms (INT8).

`intra_op_num_threads` can be set in the runner constructor for
manual thread tuning; default is ORT's auto-picked count.

## NVIDIA CUDA

Install `onnxruntime-gpu` instead of `onnxruntime`:

```bash
pip install onnxruntime-gpu
```

Then switch the config:

```yaml
model:
  backend: onnxruntime-cuda
```

`CUDAExecutionProvider` requires a CUDA Toolkit + cuDNN matching
the ONNX Runtime build (consult ORT release notes — versions are
strict). Verify with:

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Expect 'CUDAExecutionProvider' to appear
```

If CUDA isn't found at runtime, the chain falls through to CPU
and `model_backend` in `/health` reports `onnxruntime-cpu`. Watch
for that — a CUDA-intended deployment that silently runs on CPU
will look "working" but be 10–50× slower.

## Apple CoreML

```yaml
model:
  backend: onnxruntime-coreml
```

Available out-of-the-box with the standard `onnxruntime` package
on macOS. CoreML targets the Apple Neural Engine + GPU. **It is
not always a speedup.**

For DistilBERT-SST2 on M-series:
- CPU EP:    mean 3.31 ms
- CoreML EP: mean 17.00 ms

The reason: ORT logs

> CoreMLExecutionProvider::GetCapability, number of partitions
> supported by CoreML: 57, number of nodes in the graph: 417,
> number of nodes supported by CoreML: 287

Only 287 of 417 graph nodes have a CoreML kernel. The graph gets
partitioned between CoreML and CPU; the resulting cross-EP copies
dominate the small per-request compute on a small model.

CoreML is more competitive for:
- larger transformer models (compute outweighs copy)
- vision models (CoreML has very strong CNN coverage)
- batch=1 latency-critical inference where ANE is significantly
  more power-efficient than CPU

## TensorRT

```yaml
model:
  backend: onnxruntime-tensorrt
```

TensorRT is the NVIDIA-only "compile a TRT engine for this exact
model + shape + precision and reuse it" path. Setup is more
involved than the CUDA EP:

1. Install TensorRT and a matching `onnxruntime-gpu` build.
2. First inference triggers engine compilation (slow — 30 s+ for
   transformers); ORT caches the engine in `~/.onnxruntime`.
3. Engines are tied to the GPU model, driver, and TRT version.

A common alternate workflow is offline conversion via `trtexec`:

```bash
trtexec --onnx=models/distilbert-sst2-fp32/model.onnx \
        --saveEngine=models/distilbert-sst2.trt \
        --fp16 \
        --workspace=4096
```

…then load the cached engine through TensorRT EP. TensorRT typically
yields a further 1.5–3× speedup over the plain CUDA EP on transformer
models, with the tradeoff that the engine is non-portable.

## Picking a backend

| situation                                              | recommended backend       |
|--------------------------------------------------------|---------------------------|
| Local development / CI / Apple Silicon dev            | `onnxruntime-cpu`         |
| Local Mac latency experiment, larger models            | `onnxruntime-coreml`      |
| NVIDIA GPU server, simple deploy                      | `onnxruntime-cuda`        |
| NVIDIA GPU + maximum throughput, willing to compile    | `onnxruntime-tensorrt`    |
