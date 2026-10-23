# Hardware-Aware CUDA MatMul Optimization

Standalone computer architecture course project, extracted from the larger `transformer-gpu-kernels` learning repository. The scope ends at FP32 matrix multiplication; future Softmax/Attention kernels remain in the parent project.

## Results (A100, final same-run comparison)

| N×N | Naive (ms) | Shared tiled (ms) | Register tiled (ms) | cuBLAS SGEMM (ms) |
|---:|---:|---:|---:|---:|
| 128 | 0.010107 | 0.008131 | 0.009400 | 0.009789 |
| 256 | 0.022293 | 0.021842 | 0.022436 | 0.015677 |
| 512 | 0.143124 | 0.105155 | 0.084316 | 0.035267 |
| 1024 | 0.826388 | 0.580086 | 0.449700 | 0.129638 |

For 1024×1024, the register-tiled version reaches **1.84× speedup** versus naive; cuBLAS is **3.47× faster** than our register-tiled kernel. Small shapes favor different variants. See [`docs/matmul_optimization_report.md`](docs/matmul_optimization_report.md) for methods, caveats and 7 figures.

## Method

1. CPU reference; naive CUDA, one output per thread.
2. Thread-mapping analysis on T4 (`Row-major` baseline vs deliberately poor `Column-first`). This is *diagnosis*, not an 8× optimization.
3. Shared-memory cooperative A/B tile loads with K-loop, zero padding and two block barriers.
4. Compile-time tile-size specializations (8/16/32), resource/occupancy trade-offs.
5. Register tiling R=2, one thread computes two output columns using the same A value.
6. cuBLAS SGEMM row-major interoperability, final CUDA-event comparison.

## Build and test

Requires NVIDIA CUDA Toolkit, cuBLAS, C++17 compiler and NVIDIA GPU. `CUDA_ARCH=sm_80` targets A100; change it for another device.

```bash
make CUDA_ARCH=sm_80
make test
make benchmark
```

Tests are self-contained and include their CPU reference where relevant. On Colab, run these commands in the repository root. See [`Makefile`](Makefile) for separate binaries. To reproduce benchmark figures from recorded CSVs, run `python scripts/plot_matmul_report.py` (Python packages as required by the script).

## Repository layout

- `src/` CPU reference.
- `cuda/` Naive, shared-memory tiled, register-tiled GPU implementations.
- `tests/` independent correctness checks.
- `benchmarks/` distinct experiments, culminating in `matmul_final.cu`.
- `results/` original benchmark summaries from the learning project.
- `docs/` analysis and figures.
- `scripts/` plotting source.

## Provenance and verification

This is a **standalone reconstruction**, not an identical GitHub-history export. The original standalone `matmul_naive.cu` and `matmul_cpu.cpp` were not attached, so the naive kernel was reconstructed from the validated index mapping and the CPU function from the uploaded tests. `matmul_final.cu` has two known launch/declaration typos corrected. The attached `benchmark_tile_size.cu` predates local typo corrections; minor edits have been made in this snapshot. **No GPU toolchain is available in this artifact-building environment**, so this assembled ZIP has not been recompiled end-to-end; rerun `make test` and `make benchmark` in A100 Colab before publishing performance claims against this exact commit. Numbers are from the user's prior A100 logs, not fresh results from this package.

This project is for educational performance analysis, not a production BLAS replacement. CUDA event timings are kernel-only (no malloc/copies); cuBLAS is configured to default math mode in the final benchmark. Differences in build flags or runs can change numbers.
