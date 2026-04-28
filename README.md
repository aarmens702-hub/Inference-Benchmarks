# InferBench

Local inference-serving benchmark and optimizer for ONNX models.

## What it is

InferBench exposes an ONNX model behind a FastAPI server and benchmarks how
**dynamic batching, caching, quantization, and request scheduling** affect
p50/p95/p99 latency and throughput under realistic traffic.

## Why it matters

Inference serving is a latency/throughput tradeoff problem. Larger batches
improve throughput but hurt tail latency. Quantization is faster but trades
accuracy. A real serving system has to make these calls under bursty load —
this project measures and tunes those tradeoffs end to end.

## Features

- FastAPI inference server with async request handling
- Dynamic batching with configurable `max_batch_size` / `max_wait_ms`
- Request queue with backpressure (HTTP 429 on overload)
- ONNX Runtime backends (CPU default, optional CUDA)
- INT8 quantization experiments with accuracy delta reporting
- LRU cache with hit-rate metrics
- p50 / p95 / p99 / throughput / queue-time / inference-time reporting
- Adaptive batching controller (SLO-driven, stretch goal)
- Reproducible Markdown + JSON benchmark reports

## Architecture

```
client / load generator
        │
        ▼
FastAPI /infer endpoint
        │
        ▼
RequestQueue ──▶ DynamicBatcher ──▶ ModelRunner ──▶ ONNX Runtime Session
        │                                                     │
        └───────────── ResponseFuture ◀───────────────────────┘
```

## Quickstart

```bash
make serve              # start FastAPI on :8000
make bench SCENARIO=A   # run benchmark scenario A
```

*(Quickstart will be filled in as components land — see commit log.)*

## Benchmark Scenarios

| ID | Name                      | Purpose                                    |
|----|---------------------------|--------------------------------------------|
| A  | Single-request baseline   | Raw model latency, no queueing             |
| B  | Fixed concurrency sweep   | Closed-loop at 1, 4, 8, 16, 32, 64 clients |
| C  | Poisson arrivals          | Realistic open-loop traffic                |
| D  | Spike test                | 10 → 100 → 10 req/s burst                  |
| E  | Cache-heavy workload      | 0% / 20% / 50% / 80% repeat ratio          |
| F  | Cold start                | First-request latency vs warmed-up         |

## Example Results

*Populated after first reference run lands in `results/reference/`.*

## Design Decisions

*Section grows as decisions are made — batching strategy, SLO targets,
quantization choice, etc.*

## Limitations

- Single-node only.
- No streaming inference.
- Accuracy evaluation limited to SST-2 validation split for the default model.

## Future Work

- gRPC transport
- TensorRT execution provider
- Comparison against Triton Inference Server
- Multi-model serving

## License

MIT
