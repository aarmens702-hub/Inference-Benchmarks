# Benchmark Sweep: scenario B

**Scenario**: fixed-concurrency-sweep
**Started**: 2026-04-28T22:47:43.905885+00:00
**Backend**: onnxruntime-cpu | **Precision**: int8-dynamic
**Workload**: closed_loop

## Latency vs concurrency

| concurrency | n | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | rps | err % |
|------------:|--:|-------:|-------:|-------:|-------:|--------:|----:|------:|
| 1 | 2024 | 14.76 | 15.71 | 16.17 | 17.01 | 14.82 | 67.5 | 0.00 |
| 4 | 6584 | 18.24 | 19.17 | 19.50 | 20.35 | 18.22 | 219.4 | 0.00 |
| 8 | 10970 | 21.80 | 22.96 | 23.41 | 24.76 | 21.87 | 365.6 | 0.00 |
| 16 | 20145 | 24.64 | 26.50 | 27.31 | 35.61 | 23.83 | 670.9 | 0.00 |
| 32 | 18731 | 51.11 | 53.02 | 53.92 | 56.09 | 51.26 | 623.3 | 0.00 |
| 64 | 18458 | 103.90 | 106.55 | 107.57 | 110.74 | 104.01 | 613.1 | 0.00 |

## Environment
- Host: `Mac-2.lan`
- Platform: `macOS-15.5-arm64-arm-64bit`
- Python: `3.11.14`
- CPU count: 8
