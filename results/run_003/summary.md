# Benchmark Sweep: scenario B

**Scenario**: fixed-concurrency-sweep
**Started**: 2026-04-28T21:01:38.562572+00:00
**Backend**: onnxruntime-cpu | **Precision**: fp32
**Workload**: closed_loop

## Latency vs concurrency

| concurrency | n | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | rps | err % |
|------------:|--:|-------:|-------:|-------:|-------:|--------:|----:|------:|
| 1 | 1628 | 17.10 | 20.19 | 24.67 | 33.53 | 18.43 | 54.2 | 0.00 |
| 4 | 4056 | 25.88 | 29.65 | 38.68 | 66.25 | 29.59 | 135.1 | 0.00 |
| 8 | 6176 | 38.65 | 40.49 | 41.15 | 45.94 | 38.85 | 205.8 | 0.00 |
| 16 | 5672 | 91.80 | 97.43 | 99.17 | 105.20 | 84.72 | 188.5 | 0.00 |
| 32 | 4974 | 190.82 | 202.75 | 206.64 | 290.89 | 193.25 | 164.8 | 0.00 |
| 64 | 4688 | 337.00 | 421.44 | 804.72 | 2498.09 | 411.29 | 154.4 | 0.00 |

## Environment
- Host: `Mac-2.lan`
- Platform: `macOS-15.5-arm64-arm-64bit`
- Python: `3.11.14`
- CPU count: 8
