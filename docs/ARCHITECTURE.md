# InferBench Architecture

## Request flow

```
┌──────────────────┐
│  load generator  │  closed-loop / Poisson / spike / cache-repeat / k6
│  (workloads.py)  │
└────────┬─────────┘
         │ HTTP
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI / uvicorn                                              │
│                                                                 │
│  POST /infer                                                    │
│    │                                                            │
│    ├─→ PredictionCache.get(key)  ──── HIT ────► 200 (≈0 ms)     │
│    │                                                            │
│    └─→ DynamicBatcher.submit(text)                              │
│           │                                                     │
│           ▼                                                     │
│       RequestQueue (asyncio.Queue, bounded; 429 on full)        │
│           │                                                     │
│           ▼                                                     │
│       _collect_batch():                                         │
│         - block on first request                                │
│         - drain up to max_batch_size or max_wait_ms             │
│           │                                                     │
│           ▼                                                     │
│       asyncio.to_thread(ModelRunner.run, texts)                 │
│           │                                                     │
│           ▼                                                     │
│       ONNX Runtime InferenceSession                             │
│         (CPU / CUDA / CoreML / TensorRT EP — picked at boot)    │
│           │                                                     │
│           ▼                                                     │
│       resolve futures, populate cache, return InferResponse     │
│                                                                 │
│  AdaptiveBatchController (optional, separate asyncio task)      │
│    every adjust_interval_sec:                                   │
│      reads batcher.recent_latencies(),                          │
│      computes p95, mutates max_batch_size / max_wait_ms.        │
│                                                                 │
│  GET /metrics       cache hit/miss, queue size, batch hist,     │
│                     last 10 controller decisions                │
│  POST /admin/cache/reset                                        │
└─────────────────────────────────────────────────────────────────┘
```

Latency decomposition reported on every response:

```
┌──────────────── latency_ms (e2e, on the wire) ────────────────┐
│  ┌── queue_wait_ms ──┐  ┌── inference_ms ──┐  ┌── (server) ──┐│
│  │ time from submit  │  │ ORT session.run │  │ post-process │ │
│  │ to batch flush    │  │ for the whole   │  │ + serialize  │ │
│  └───────────────────┘  └─ batch (shared)─┘  └──────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

## Module map

```
inferbench/
├── server/
│   ├── app.py        FastAPI factory + lifespan: builds runner /
│   │                 batcher / cache / controller from server.yaml
│   ├── routes.py     /infer (cache → batcher → runner), /health,
│   │                 /metrics, /admin/cache/reset
│   └── schemas.py    Pydantic InferRequest / InferResponse
│
├── engine/
│   ├── model_runner.py   ONNX Runtime + tokenizer wrapper.
│   │                     resolve_providers() picks the EP chain.
│   ├── batcher.py        DynamicBatcher: an asyncio task that pulls
│   │                     from a queue, forms batches by size or
│   │                     wait, calls runner via asyncio.to_thread.
│   ├── request_queue.py  Bounded asyncio.Queue wrapper +
│   │                     QueueOverflowError → HTTP 429.
│   ├── cache.py          Thread-safe LRU keyed by
│   │                     sha1(model_id || precision || text).
│   ├── controller.py     AdaptiveBatchController: SLO-driven,
│   │                     saturation-aware batch tuner.
│   └── metrics.py        Single source of truth for percentiles
│                         + throughput math.
│
├── models/
│   ├── export_model.py       HF → ONNX FP32 (optimum)
│   ├── quantize_model.py     ORT dynamic INT8
│   └── evaluate_accuracy.py  SST-2 val accuracy
│
├── benchmarks/
│   ├── run_benchmark.py  CLI dispatcher for scenarios A..F
│   ├── workloads.py      closed_loop / poisson / spike /
│   │                     cache_repeat workload generators
│   └── k6/               infer_load_test.js (HTTP-level)
│
└── reports/
    ├── generate_report.py  results.json + summary.md per run
    ├── plot_results.py     latency CDFs + sweep curves (matplotlib)
    └── quant_compare.py    FP32 vs INT8 tradeoff report
```

## Async model

The whole hot path lives on a single asyncio event loop:

- `/infer` is async. Cache lookup is in-memory; if hit, returns
  immediately without touching the queue.
- On miss, `batcher.submit(text)` puts a `BatchedRequest` (with a
  per-request future) on the queue and `await`s the future.
- A single `DynamicBatcher._run` task pulls from the queue. ORT
  inference is CPU-bound and would block the loop, so it runs in
  a thread via `asyncio.to_thread`. Only one inference runs at a
  time (one ORT session, one worker thread).
- `AdaptiveBatchController._run` is a separate task that wakes
  every `adjust_interval_sec`. It only reads observability state
  and writes `BatcherConfig` fields — it doesn't intercept any
  request path.
- Lifespan is the source of truth: it constructs runner →
  batcher → cache → controller in dependency order at startup
  and stops them in reverse on shutdown.
