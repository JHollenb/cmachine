CC ?= cc
NVCC ?= /usr/local/cuda/bin/nvcc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -Iruntime -I.

build/memory-response: runtime/model.c runtime/qwen.c runtime/qwen_cuda.cu driver.c cache.h audit.h cache_cuda.cu
	mkdir -p build
	$(CC) $(CFLAGS) -c runtime/model.c -o build/model.o
	$(CC) $(CFLAGS) -c runtime/qwen.c -o build/qwen.o
	$(CC) $(CFLAGS) -c driver.c -o build/driver.o
	$(NVCC) -O3 -std=c++17 --fmad=false -arch=sm_89 -Iruntime -I. -c cache_cuda.cu -o build/cache_cuda.o
	$(NVCC) build/model.o build/qwen.o build/driver.o build/cache_cuda.o -lcublas -lm -o $@

.PHONY: clean
clean:
	rm -rf build
