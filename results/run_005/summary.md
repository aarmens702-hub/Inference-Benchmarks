# Benchmark Run: scenario D

**Scenario**: spike-test
**Started**: 2026-04-28T22:23:48.701626+00:00
**Duration**: 89.9s
**Backend**: onnxruntime-cpu | **Precision**: fp32
**Workload**: spike

## End-to-end latency (ms)

| count | mean | p50 | p90 | p95 | p99 | min | max |
|------:|-----:|----:|----:|----:|----:|----:|----:|
| 3599 | 400.66 | 28.71 | 1602.43 | 2258.71 | 2432.12 | 9.76 | 2615.23 |

## Inference-only latency (ms)

| p50 | p95 | p99 |
|----:|----:|----:|
| 8.45 | 22.50 | 32.84 |

## Queue wait (ms)

| p50 | p95 | p99 |
|----:|----:|----:|
| 11.15 | 2184.50 | 2373.27 |

## Throughput

- **Requests**: 3599
- **Throughput**: 40.0 req/s
- **Errors**: 1 (0.03%)

## Environment

- Host: `Mac-2.lan`
- Platform: `macOS-15.5-arm64-arm-64bit`
- Python: `3.11.14`
- CPU count: 8
