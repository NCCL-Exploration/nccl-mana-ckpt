CC = nvcc
CXX = nvcc
CFLAGS = -O0 -g -Xcompiler -fPIC
CXXFLAGS = -O0 -g -Xcompiler -fPIC

# Set DMTCP_ROOT to your DMTCP installation.
ifndef DMTCP_ROOT
  DMTCP_ROOT=${CURDIR}/dmtcp
endif

LIBNAME = bin/libncclplugin
CUDA_PLUGIN = ${DMTCP_ROOT}/plugin/cuda/libdmtcp_cuda.so
NCCL_INCLUDE = ./include
DMTCP_INCLUDE = ${DMTCP_ROOT}/include
JALIB_INCLUDE = ${DMTCP_ROOT}/jalib
DMTCP_SRC = ${DMTCP_ROOT}/src
LINK = ${CC}

MPICH_BIN = /shared/EL9/explorer/MPICH/4.3.0b1/bin

DEMO_PORT=7781
DRAIN_NO_MSGS_TIMEOUT ?= 45

NCCL_HOME ?= $(NCCL_DIR)

# <nccl.h> includes <cuda_runtime.h>, so the host-only gcc test targets need the
# CUDA include path too (nvcc supplies it implicitly, plain gcc does not).
# Resolution order: CUDA_HOME from the environment, else derived from the real
# location of nvcc on PATH, else the conventional install prefix.
NVCC_PATH := $(shell command -v nvcc 2>/dev/null)
ifeq ($(origin CUDA_HOME),undefined)
  ifneq ($(NVCC_PATH),)
    CUDA_HOME := $(realpath $(dir $(realpath $(NVCC_PATH)))..)
  else
    CUDA_HOME := /usr/local/cuda
  endif
endif
HOST_TEST_FLAGS = -I src -I include -I${DMTCP_INCLUDE} -I$(CUDA_HOME)/include -lpthread

override CFLAGS += -I${DMTCP_INCLUDE} -I${NCCL_INCLUDE} -I./src -I/usr/lib/x86_64-linux-gnu/openmpi/include
ifneq ($(NCCL_HOME),)
  override CFLAGS += -I$(NCCL_HOME)/include -L$(NCCL_HOME)/lib
endif
override CFLAGS += -lnccl -lmpi

plugin: ${LIBNAME}.so

check: ${LIBNAME}.so ${DMTCP_ROOT}/test/dmtcp1
	# Kill an old coordinator on this port if present, just in case.
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Note that full path of plugin (using $$PWD in this case) is required.
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 5 \
	  --with-plugin $$PWD/${LIBNAME}.so ${DMTCP_ROOT}/test/dmtcp1

nccl_check: ${LIBNAME}.so tests/single/single_gpu_test
	# Run the actual NCCL test
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 10 \
		--with-plugin $$PWD/${LIBNAME}.so:${CUDA_PLUGIN} ./tests/single/single_gpu_test

drain_recv_check: ${LIBNAME}.so tests/multi/drain_recv
	# Kill an old coordinator on this port if present, just in case.
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Launch DMTCP coordinator
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT} 
	# Run the actual NCCL test
	${MPICH_BIN}/mpirun -np 2 ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} --interval 8\
	 	--with-plugin $$PWD/${LIBNAME}.so:${CUDA_PLUGIN} ./tests/multi/drain_recv

drain_send_check: ${LIBNAME}.so tests/multi/drain_send
	# Kill an old coordinator on this port if present, just in case.
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Launch DMTCP coordinator
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT} 
	# Run the actual NCCL test
	${MPICH_BIN}/mpirun -np 2 ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} --interval 8\
	 	--with-plugin $$PWD/${LIBNAME}.so:${CUDA_PLUGIN} ./tests/multi/drain_send

drain_no_msgs_check: ${LIBNAME}.so tests/multi/drain_no_msgs
	# Kill an old coordinator on this port if present, just in case.
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Launch DMTCP coordinator
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT}
	# Run the no-message NCCL drain test; timeout turns a hang into a failure.
	timeout ${DRAIN_NO_MSGS_TIMEOUT}s ${MPICH_BIN}/mpirun -np 2 ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} --interval 8\
	 	--with-plugin $$PWD/${LIBNAME}.so:${CUDA_PLUGIN} ./tests/multi/drain_no_msgs

cc_simple_check: ${LIBNAME}.so tests/multi/cc_simple
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT}
	${MPICH_BIN}/mpirun -np 2 ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} --interval 8\
		--with-plugin $$PWD/${LIBNAME}.so:${CUDA_PLUGIN} ./tests/multi/cc_simple

cc_complex_check: ${LIBNAME}.so tests/multi/cc_complex
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	 --coord-port ${DEMO_PORT} 2>/dev/null || true
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT}
	${MPICH_BIN}/mpirun -np 3 ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} --interval 8\
                --with-plugin $$PWD/${LIBNAME}.so:${CUDA_PLUGIN} ./tests/multi/cc_complex

# Host-only unit tests: no GPU, no DMTCP, no NCCL runtime. Fastest feedback loop.
tests/vcomm:
	gcc -Wall -Wextra -o test_vcomm tests/single/vcomm_test.c src/vcomm.c $(HOST_TEST_FLAGS)
	./test_vcomm

tests/cc: tests/vcomm
	gcc -o test_nccl_wrappers tests/single/seq_algorithm_test.c src/nccl_wrappers.c src/vcomm.c src/inflight.c tests/single/seq_stubs.c $(HOST_TEST_FLAGS)
	./test_nccl_wrappers

tests/single/single_gpu_test:
	$(MAKE) -C tests/single single_gpu_test

tests/multi/drain_recv:
	$(MAKE) -C tests/multi drain_recv

tests/multi/drain_send:
	$(MAKE) -C tests/multi drain_send

tests/multi/drain_no_msgs:
	$(MAKE) -C tests/multi drain_no_msgs

tests/multi/cc_simple:
	$(MAKE) -C tests/multi cc_simple

tests/multi/cc_complex:
	$(MAKE) -C tests/multi cc_complex

${DMTCP_ROOT}/test/dmtcp1: ${DMTCP_ROOT}/test
	cd ${DMTCP_ROOT}/test && make dmtcp1

${DMTCP_ROOT}/bin:
	cd ${DMTCP_ROOT} && ./configure --disable-prgname-prefix && make -j

OBJS = obj/libncclplugin.o obj/dmtcp_kvdb_c.o obj/nccl_wrappers.o obj/inflight.o obj/vcomm.o

${LIBNAME}.so: $(OBJS)
	@mkdir -p bin
	${CXX} -shared -o $@ $^ -L$(CUDA_HOME)/lib64 -lcudart -lpthread -ldl

obj/%.o: src/%.c
	@mkdir -p obj
	${CC} ${CFLAGS} -c -o $@ $<

obj/dmtcp_kvdb_c.o: src/dmtcp_kvdb_c.cpp
	@mkdir -p obj
	${CXX} ${CXXFLAGS} -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -I${DMTCP_SRC} -c -o $@ $<

CLANG_FORMAT ?= clang-format
CLANG_FORMAT_BIN := $(shell command -v $(CLANG_FORMAT) 2>/dev/null || echo "npx -y clang-format")
FORMAT_FILES := $(shell find src include tests \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.cu" \) -not -path "*/dmtcp/*")

.PHONY: format-clang lint-clang format-python lint-python clean

format-clang:
	@echo "Formatting codebase (ignoring submodules)..."
	$(CLANG_FORMAT_BIN) -i $(FORMAT_FILES)

lint-clang:
	@echo "Checking code format (ignoring submodules)..."
	@$(CLANG_FORMAT_BIN) --dry-run --Werror $(FORMAT_FILES)

format-python:
	@echo "Formatting Python files with uv, ruff, and black..."
	uv run ruff check --fix .
	uv run black .

lint-python:
	@echo "Checking Python formatting with uv, ruff, and black..."
	uv run ruff check .
	uv run black --check .

clean:
	cd tests && make clean
	rm -rf bin/ obj/ *.dmtcp *.temp dmtcp_coordinator*.json dmtcp_restart_script*.sh *.o ckpt_*_files src/*.o src/*.so test_nccl_wrappers

