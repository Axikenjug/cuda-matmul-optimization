**GPU 矩阵乘法算子实现与优化｜CUDA C++、cuBLAS、NVIDIA A100**

- 基于 CPU Reference 实现 FP32 CUDA MatMul，并通过线程映射对照实验分析 Memory Coalescing 对 GPU 访存性能的影响。
- 独立实现 Shared-Memory Tiling、K 维分块累加与边界处理，以及单线程双输出的 Register Tiling；建立正确性测试与 CUDA Event 性能评估流程，并对比 TILE=8/16/32。
- 在 A100 的 1024×1024 MatMul 同批次对照中，Register-Tiled 相比 Naive 延迟从 0.826 ms 降至 0.450 ms，获得 **1.84× 加速**；对照 cuBLAS SGEMM 达到其约 **28.8% 的吞吐量**。
