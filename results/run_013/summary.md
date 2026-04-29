# Benchmark Sweep: scenario B

**Scenario**: fixed-concurrency-sweep
**Started**: 2026-04-29T00:52:18.485336+00:00
**Backend**: onnxruntime-coreml | **Precision**: fp32
**Workload**: closed_loop

## Latency vs concurrency

| concurrency | n | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | rps | err % |
|------------:|--:|-------:|-------:|-------:|-------:|--------:|----:|------:|
| 1 | 621 | 46.28 | 52.80 | 59.29 | 88.49 | 48.33 | 20.7 | 0.00 |
| 4 | 0 | n/a | n/a | n/a | n/a | n/a | 0.0 | 100.00 |
| 8 | 192 | 52.72 | 60.72 | 64.53 | 88.53 | 54.07 | 6.4 | 97.53 |
| 16 | 201 | 58.61 | 70.35 | 76.69 | 105.84 | 61.30 | 6.7 | 98.58 |
| 32 | 1 | 96.05 | 96.05 | 96.05 | 96.05 | 96.05 | 0.0 | 100.00 |
| 64 | 0 | n/a | n/a | n/a | n/a | n/a | 0.0 | 100.00 |

## Environment
- Host: `Mac-2.lan`
- Platform: `macOS-15.5-arm64-arm-64bit`
- Python: `3.11.14`
- CPU count: 8
