# Benchmark Run: scenario A

**Scenario**: single-request-baseline
**Started**: 2026-04-28T22:57:48.274074+00:00
**Duration**: 30.0s
**Backend**: onnxruntime-cpu | **Precision**: int8-dynamic
**Workload**: closed_loop(c=1)

## End-to-end latency (ms)

| count | mean | p50 | p90 | p95 | p99 | min | max |
|------:|-----:|----:|----:|----:|----:|----:|----:|
| 2070 | 14.49 | 14.59 | 15.34 | 15.57 | 16.07 | 12.08 | 29.41 |

## Inference-only latency (ms)

| p50 | p95 | p99 |
|----:|----:|----:|
| 1.72 | 2.52 | 2.88 |

## Queue wait (ms)

| p50 | p95 | p99 |
|----:|----:|----:|
| 12.11 | 12.14 | 12.18 |

## Throughput

- **Requests**: 2070
- **Throughput**: 69.0 req/s
- **Errors**: 0 (0.00%)

## Environment

- Host: `Mac-2.lan`
- Platform: `macOS-15.5-arm64-arm-64bit`
- Python: `3.11.14`
- CPU count: 8
