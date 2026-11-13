CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off
LDLIBS = -lm
SRC = src/model.c src/isa.c src/program.c src/main.c
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
BLAS_CPPFLAGS ?=
BLAS_LDLIBS ?= -lopenblas

all: build/tcm

build/tcm: $(SRC) src/machine.h src/program.h
	mkdir -p build
	$(CC) $(CFLAGS) -Isrc $(SRC) $(LDLIBS) -o $@

sanitize: build/tcm-sanitize

build/tcm-sanitize: $(SRC) src/machine.h src/program.h
	mkdir -p build
	$(CC) -O1 -g -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -fsanitize=address,undefined -Isrc $(SRC) $(LDLIBS) -o $@

clean:
	rm -rf build

.PHONY: all sanitize clean

cuda: build/tcm-cuda

build/tcm-cuda: $(SRC) src/program.h src/qwen.c src/qwen_operands.inc src/qwen_serve.c src/qwen_bounded_serve.c src/bounded_cache.h src/qwen_cuda.cu src/qwen_cache.inc src/qwen_weights.inc src/qwen.h src/gpu.h src/machine.h
	mkdir -p build
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/model.c -o build/model.o
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/isa.c -o build/isa.o
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/main.c -o build/main.o
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/program.c -o build/program.o
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/qwen.c -o build/qwen.o
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/qwen_serve.c -o build/qwen_serve.o
	$(CC) $(CFLAGS) -DTCM_WITH_CUDA -Isrc -c src/qwen_bounded_serve.c -o build/qwen_bounded_serve.o
	$(NVCC) -O3 -std=c++17 --fmad=false -arch=sm_89 -Isrc -c src/qwen_cuda.cu -o build/qwen_cuda.o
	$(NVCC) build/model.o build/isa.o build/program.o build/main.o build/qwen.o build/qwen_serve.o build/qwen_bounded_serve.o build/qwen_cuda.o -lcublas -lm -o $@

# The same graph executor with an explicit CPU BLAS reduction profile.
program-blas: build/tcm-blas
build/tcm-blas: $(SRC) src/machine.h src/program.h
	mkdir -p build
	$(CC) $(CFLAGS) -DTCM_USE_CBLAS $(BLAS_CPPFLAGS) -Isrc $(SRC) $(BLAS_LDLIBS) $(LDLIBS) -o $@
.PHONY: program-blas

.PHONY: cuda

gpu-mechanics: build/gpu-mechanics

build/gpu-mechanics: build/tcm-cuda tests/gpu_mechanics.c
	$(CC) $(CFLAGS) -Isrc -c tests/gpu_mechanics.c -o build/gpu_mechanics.o
	$(NVCC) build/model.o build/qwen.o build/qwen_cuda.o build/gpu_mechanics.o -lcublas -lm -o $@

.PHONY: gpu-mechanics

state-mechanics: build/state-mechanics
	./build/state-mechanics

build/state-mechanics: src/model.c src/qwen.c src/qwen_operands.inc src/qwen.h src/gpu.h tests/gpu_stub.c tests/state_mechanics.c tests/writer_mechanics.h tests/gate_mechanics.h
	mkdir -p build
	$(CC) -O1 -g -std=c11 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fsanitize=address,undefined -Isrc src/model.c src/qwen.c tests/gpu_stub.c tests/state_mechanics.c -lm -o $@

.PHONY: state-mechanics

operand-mechanics: build/operand-mechanics
	./build/operand-mechanics

build/operand-mechanics: src/model.c src/qwen.c src/qwen_operands.inc src/qwen.h src/prototype_attention.h src/gpu.h tests/gpu_stub.c tests/operand_mechanics.c tests/state_mechanics.c tests/writer_mechanics.h tests/gate_mechanics.h
	mkdir -p build
	$(CC) -O1 -g -std=c11 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fsanitize=address,undefined -Isrc src/model.c src/qwen.c tests/gpu_stub.c tests/operand_mechanics.c -lm -o $@

.PHONY: operand-mechanics

# Optional circuit inspection; no model download, CUDA, or learned dictionary.
circuit-mechanics: build/circuit-mechanics
	./build/circuit-mechanics

build/circuit-mechanics: src/circuit_math.c src/circuit_math.h src/circuit_expansions.c src/circuit_expansions.h src/qwen_circuit.c src/qwen_circuit.h src/model.c src/qwen.c src/qwen.h src/gpu.h src/machine.h src/qwen_operands.inc tests/gpu_stub.c tests/circuit_mechanics.c tests/state_mechanics.c tests/writer_mechanics.h tests/gate_mechanics.h
	mkdir -p build
	$(CC) -O1 -g -std=c11 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fsanitize=address,undefined -Isrc src/model.c src/qwen.c src/circuit_math.c src/circuit_expansions.c src/qwen_circuit.c tests/gpu_stub.c tests/circuit_mechanics.c -lm -o $@
.PHONY: circuit-mechanics

# Standalone opt-in native inspector. Original serving executable stays native.
circuit-cuda: build/tcm-circuit
build/tcm-circuit: build/tcm-cuda src/circuit_math.c src/circuit_math.h src/circuit_expansions.c src/circuit_expansions.h src/qwen_circuit.c src/qwen_circuit.h tools/circuit_inspect.c
	$(CC) $(CFLAGS) -Isrc -c src/circuit_math.c -o build/circuit_math.o
	$(CC) $(CFLAGS) -Isrc -c src/qwen_circuit.c -o build/qwen_circuit.o
	$(CC) $(CFLAGS) -Isrc -c src/circuit_expansions.c -o build/circuit_expansions.o
	$(CC) $(CFLAGS) -Isrc -c tools/circuit_inspect.c -o build/circuit_inspect.o
	$(NVCC) build/model.o build/qwen.o build/qwen_cuda.o build/circuit_math.o build/circuit_expansions.o build/qwen_circuit.o build/circuit_inspect.o -lcublas -lm -o $@
.PHONY: circuit-cuda

expansion-mechanics: build/expansion-mechanics
	./build/expansion-mechanics
build/expansion-mechanics: src/circuit_math.c src/circuit_math.h src/circuit_expansions.c src/circuit_expansions.h src/isa.c src/isa_circuit.c src/isa_circuit.h src/model.c src/machine.h tests/expansion_mechanics.c tests/isa_fixture.h
	mkdir -p build
	$(CC) -O1 -g -std=c11 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fsanitize=address,undefined -Isrc src/model.c src/isa.c src/circuit_math.c src/circuit_expansions.c src/isa_circuit.c tests/expansion_mechanics.c -lm -o $@
.PHONY: expansion-mechanics

isa-circuit: build/tcm-isa-circuit
build/tcm-isa-circuit: src/model.c src/isa.c src/isa_circuit.c src/isa_circuit.h src/machine.h src/circuit_math.c src/circuit_math.h src/circuit_expansions.c src/circuit_expansions.h tools/isa_circuit_inspect.c
	mkdir -p build
	$(CC) $(CFLAGS) -Isrc src/model.c src/isa.c src/circuit_math.c src/circuit_expansions.c src/isa_circuit.c tools/isa_circuit_inspect.c -lm -o $@
.PHONY: isa-circuit

bounded-mechanics: build/bounded-mechanics
	./build/bounded-mechanics

build/bounded-mechanics: src/model.c src/qwen.c src/qwen_operands.inc src/qwen.h src/bounded_cache.h src/byte_codec.h tests/bounded_mechanics.c tests/gpu_stub.c
	mkdir -p build
	$(CC) -O1 -g -std=c11 -Wall -Wextra -ffp-contract=off -fsanitize=address,undefined -Isrc src/model.c src/qwen.c tests/gpu_stub.c tests/bounded_mechanics.c -lm -o $@

.PHONY: bounded-mechanics

bounded-gpu-mechanics: build/bounded-gpu-mechanics

build/bounded-gpu-mechanics: build/tcm-cuda tests/bounded_gpu_mechanics.c src/bounded_cache.h
	$(CC) $(CFLAGS) -Isrc -c tests/bounded_gpu_mechanics.c -o build/bounded_gpu_mechanics.o
	$(NVCC) build/model.o build/qwen.o build/qwen_cuda.o build/bounded_gpu_mechanics.o -lcublas -lm -o $@

.PHONY: bounded-gpu-mechanics

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

# Analysis-overhead timing driver (research/overhead_timing).
overhead-run: build/overhead-run

build/overhead-run: build/tcm-cuda tests/overhead_run.c
	$(CC) $(CFLAGS) -Isrc -c tests/overhead_run.c -o build/overhead_run.o
	$(NVCC) build/model.o build/qwen.o build/qwen_cuda.o build/overhead_run.o -lcublas -lm -o $@

.PHONY: overhead-run
