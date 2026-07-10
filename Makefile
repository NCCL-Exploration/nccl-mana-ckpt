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

MPICH_BIN = /usr/bin

DEMO_PORT=7781

NCCL_HOME ?= $(NCCL_DIR)

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

tests/cc:
	gcc -o test_nccl_wrappers tests/single/seq_algorithm_test.c src/nccl_wrappers.c tests/single/seq_stubs.c -I include -lpthread
	./test_nccl_wrappers

tests/single/single_gpu_test:
	$(MAKE) -C tests/single single_gpu_test

tests/multi/drain_recv:
	$(MAKE) -C tests/multi drain_recv

tests/multi/drain_send:
	$(MAKE) -C tests/multi drain_send

tests/multi/cc_simple:
	$(MAKE) -C tests/multi cc_simple

tests/multi/cc_complex:
	$(MAKE) -C tests/multi cc_complex

${DMTCP_ROOT}/test/dmtcp1: ${DMTCP_ROOT}/test
	cd ${DMTCP_ROOT}/test && make dmtcp1

${DMTCP_ROOT}/bin:
	cd ${DMTCP_ROOT} && ./configure --disable-prgname-prefix && make -j

OBJS = obj/libncclplugin.o obj/dmtcp_kvdb_c.o obj/nccl_wrappers.o obj/inflight.o

${LIBNAME}.so: $(OBJS)
	@mkdir -p bin
	${CXX} -shared -o $@ $^

obj/%.o: src/%.c
	@mkdir -p obj
	${CC} ${CFLAGS} -c -o $@ $<

obj/dmtcp_kvdb_c.o: src/dmtcp_kvdb_c.cpp
	@mkdir -p obj
	${CXX} ${CXXFLAGS} -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -I${DMTCP_SRC} -c -o $@ $<

clean:
	cd tests && make clean
	rm -rf bin/ obj/ *.dmtcp *.temp dmtcp_coordinator*.json dmtcp_restart_script*.sh *.o ckpt_*_files src/*.o src/*.so test_nccl_wrappers
