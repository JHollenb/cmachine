CC ?= cc
NVCC ?= /usr/local/cuda/bin/nvcc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -Iruntime
build/query-key-cut: driver.c runtime/qwen.c runtime/qwen_operands.inc runtime/prototype_attention.h runtime/model.c runtime/qwen_cuda.cu
	mkdir -p build
	$(CC) $(CFLAGS) -c runtime/model.c -o build/model.o
	$(CC) $(CFLAGS) -c runtime/qwen.c -o build/qwen.o
	$(CC) $(CFLAGS) -c driver.c -o build/driver.o
	$(NVCC) -O3 -std=c++17 --fmad=false -arch=sm_89 -Iruntime -c runtime/qwen_cuda.cu -o build/cuda.o
	$(NVCC) build/model.o build/qwen.o build/driver.o build/cuda.o -lcublas -lm -o $@
