CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off
LDLIBS = -lm
SRC = src/model.c src/isa.c src/main.c
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

all: build/tcm

build/tcm: $(SRC) src/machine.h
	mkdir -p build
	$(CC) $(CFLAGS) -Isrc $(SRC) $(LDLIBS) -o $@

sanitize: build/tcm-sanitize

build/tcm-sanitize: $(SRC) src/machine.h
	mkdir -p build
	$(CC) -O1 -g -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -fsanitize=address,undefined -Isrc $(SRC) $(LDLIBS) -o $@

clean:
	rm -rf build

.PHONY: all sanitize clean

cuda: build/tcm-cuda

build/tcm-cuda: $(SRC) src/qwen.c src/qwen_serve.c src/qwen_cuda.cu src/qwen.h src/gpu.h src/machine.h
	mkdir -p build
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/model.c -o build/model.o
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/isa.c -o build/isa.o
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/main.c -o build/main.o
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/qwen.c -o build/qwen.o
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/qwen_serve.c -o build/qwen_serve.o
	$(NVCC) -O3 -std=c++17 --fmad=false -arch=sm_89 -Isrc -c src/qwen_cuda.cu -o build/qwen_cuda.o
	$(NVCC) build/model.o build/isa.o build/main.o build/qwen.o build/qwen_serve.o build/qwen_cuda.o -lcublas -lm -o $@

.PHONY: cuda

gpu-mechanics: build/gpu-mechanics

build/gpu-mechanics: build/tcm-cuda tests/gpu_mechanics.c
	$(CC) $(CFLAGS) -Isrc -c tests/gpu_mechanics.c -o build/gpu_mechanics.o
	$(NVCC) build/model.o build/qwen.o build/qwen_cuda.o build/gpu_mechanics.o -lcublas -lm -o $@

.PHONY: gpu-mechanics

state-mechanics: build/state-mechanics
	./build/state-mechanics

build/state-mechanics: src/model.c src/qwen.c src/qwen.h src/gpu.h tests/gpu_stub.c tests/state_mechanics.c tests/writer_mechanics.h tests/gate_mechanics.h
	mkdir -p build
	$(CC) -O1 -g -std=c11 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fsanitize=address,undefined -Isrc src/model.c src/qwen.c tests/gpu_stub.c tests/state_mechanics.c -lm -o $@

.PHONY: state-mechanics

stack-mechanics: build/stack-mechanics

build/stack-mechanics: build/tcm-cuda tests/state_mechanics.c tests/writer_mechanics.h tests/gate_mechanics.h
	$(CC) $(CFLAGS) -DTCM_REAL_MODEL -Isrc -c tests/state_mechanics.c -o build/stack_mechanics.o
	$(NVCC) build/model.o build/qwen.o build/qwen_cuda.o build/stack_mechanics.o -lcublas -lm -o $@

.PHONY: stack-mechanics

plane-experiment: build/plane-experiment

build/plane-experiment: build/tcm-cuda tests/plane_experiment.c
	$(CC) $(CFLAGS) -Isrc -c tests/plane_experiment.c -o build/plane_experiment.o
	$(NVCC) build/model.o build/qwen.o build/qwen_cuda.o build/plane_experiment.o -lcublas -lm -o $@

.PHONY: plane-experiment

reuse-experiment: build/reuse-experiment

build/reuse-experiment: build/tcm-cuda tests/reuse_experiment.c tests/assay_io.h tests/formation_io.h tests/writer_io.h
	$(CC) $(CFLAGS) -Isrc -c tests/reuse_experiment.c -o build/reuse_experiment.o
	$(NVCC) build/model.o build/qwen.o build/qwen_cuda.o build/reuse_experiment.o -lcublas -lm -o $@

.PHONY: reuse-experiment
