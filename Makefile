NVCC ?= nvcc
CXX ?= g++
CUDA_ARCH ?= sm_80
CUDAFLAGS = -O2 -std=c++17 -arch=$(CUDA_ARCH) -rdc=true
OUT = build

.PHONY: all test benchmark clean
all: $(OUT)/test_matmul_cpu $(OUT)/test_matmul_naive $(OUT)/test_matmul_tiled $(OUT)/test_matmul_reg_tiled $(OUT)/test_matmul_cublas $(OUT)/matmul_final
$(OUT):
	mkdir -p $(OUT)
$(OUT)/test_matmul_cpu: src/matmul_cpu.cpp tests/test_matmul_cpu.cpp | $(OUT)
	$(CXX) -O2 -std=c++17 $^ -o $@
$(OUT)/test_matmul_naive: cuda/matmul_naive.cu tests/test_matmul_naive.cu | $(OUT)
	$(NVCC) $(CUDAFLAGS) $^ -o $@
$(OUT)/test_matmul_tiled: cuda/matmul_tiled.cu tests/test_matmul_tiled.cu | $(OUT)
	$(NVCC) $(CUDAFLAGS) $^ -o $@
$(OUT)/test_matmul_reg_tiled: cuda/matmul_reg_tiled.cu tests/test_matmul_reg_tiled.cu | $(OUT)
	$(NVCC) $(CUDAFLAGS) $^ -o $@
$(OUT)/test_matmul_cublas: tests/test_matmul_cublas.cu | $(OUT)
	$(NVCC) $(CUDAFLAGS) $^ -lcublas -o $@
$(OUT)/matmul_final: cuda/matmul_naive.cu cuda/matmul_tiled.cu cuda/matmul_reg_tiled.cu benchmarks/matmul_final.cu | $(OUT)
	$(NVCC) $(CUDAFLAGS) $^ -lcublas -o $@
test: all
	./$(OUT)/test_matmul_cpu
	./$(OUT)/test_matmul_naive
	./$(OUT)/test_matmul_tiled
	./$(OUT)/test_matmul_reg_tiled
	./$(OUT)/test_matmul_cublas
benchmark: $(OUT)/matmul_final
	./$(OUT)/matmul_final
clean:
	rm -rf $(OUT)
