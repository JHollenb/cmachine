CC ?= cc
NVCC ?= /usr/local/cuda/bin/nvcc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -Iruntime

# The runtime and CUDA bridge are copied from the authenticated query-response
# predecessor.  driver.c is the only experiment-specific execution source; it
# consumes the six frozen MLP/query images and the full native FP32 K/V cache.
build/operand-composition: driver.c runtime/qwen.c runtime/model.c runtime/qwen_cuda.cu
	mkdir -p build
	$(CC) $(CFLAGS) -c runtime/model.c -o build/model.o
	$(CC) $(CFLAGS) -c driver.c -o build/driver.o
	$(NVCC) -O3 -std=c++17 --fmad=false -arch=sm_89 -Iruntime -c runtime/qwen_cuda.cu -o build/cuda.o
	$(NVCC) build/model.o build/driver.o build/cuda.o -lcublas -lm -o $@

.PHONY: clean
clean:
	rm -rf build
