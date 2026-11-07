CC ?= cc
NVCC ?= /usr/local/cuda/bin/nvcc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -Iruntime/src -Iruntime/tests

build/operand-chat: runtime/build/tcm-cuda driver.c
	mkdir -p build
	$(CC) $(CFLAGS) -c driver.c -o build/driver.o
	$(NVCC) runtime/build/model.o runtime/build/qwen.o runtime/build/qwen_cuda.o build/driver.o -Xlinker --wrap=gpu_matvec -lcublas -lm -o $@

runtime/build/tcm-cuda:
	$(MAKE) -C runtime cuda stack-mechanics
