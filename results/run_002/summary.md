# Benchmark Sweep: scenario B

**Scenario**: fixed-concurrency-sweep
**Started**: 2026-04-28T20:47:42.120047+00:00
**Backend**: onnxruntime-cpu | **Precision**: fp32
**Workload**: closed_loop

## Latency vs concurrency

| concurrency | n | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | rps | err % |
|------------:|--:|-------:|-------:|-------:|-------:|--------:|----:|------:|
| 1 | 3912 | 6.66 | 9.71 | 11.60 | 20.98 | 7.66 | 130.4 | 0.00 |
| 4 | 7203 | 15.55 | 21.30 | 25.11 | 42.25 | 16.65 | 240.0 | 0.00 |
| 8 | 5866 | 34.70 | 66.56 | 81.66 | 130.60 | 40.92 | 195.3 | 0.00 |
| 16 | 5156 | 81.73 | 161.89 | 198.78 | 307.53 | 93.14 | 171.5 | 0.00 |
| 32 | 7025 | 129.22 | 219.01 | 252.86 | 372.12 | 136.47 | 233.7 | 0.00 |
| 64 | 7146 | 266.43 | 373.38 | 411.65 | 490.10 | 268.15 | 237.0 | 0.00 |

## Environment
- Host: `Mac-2.lan`
- Platform: `macOS-15.5-arm64-arm-64bit`
- Python: `3.11.14`
- CPU count: 8
