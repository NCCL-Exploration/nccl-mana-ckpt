CC = nvcc
CXX = nvcc
CFLAGS = -O0 -g -Xcompiler -fPIC
CXXFLAGS = -O0 -g -Xcompiler -fPIC

# Set DMTCP_ROOT to your DMTCP installation.
ifndef DMTCP_ROOT
  DMTCP_ROOT=${CURDIR}/mana/dmtcp
endif

LIBNAME = src/libncclplugin
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

plugin: src/libncclplugin.so

check: ${LIBNAME}.so ${DMTCP_ROOT}/test/dmtcp1
	# Kill an old coordinator on this port if present, just in case.
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Note that full path of plugin (using $$PWD in this case) is required.
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 5 \
	  --with-plugin $$PWD/${LIBNAME}.so ${DMTCP_ROOT}/test/dmtcp1

nccl_check: ${LIBNAME}.so tests/single_gpu_test
	# Run the actual NCCL test
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 10 \
		--with-plugin $$PWD/${LIBNAME}.so ./tests/single_gpu_test

drain_recv_check: ${LIBNAME}.so tests/drain_recv
	# Kill an old coordinator on this port if present, just in case.
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Launch DMTCP coordinator
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT} 
	# Run the actual NCCL test
	${MPICH_BIN}/mpirun -np 2 ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} --interval 8\
	 	--with-plugin $$PWD/${LIBNAME}.so ./tests/drain_recv

drain_send_check: ${LIBNAME}.so tests/drain_send
	# Kill an old coordinator on this port if present, just in case.
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Launch DMTCP coordinator
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT} 
	# Run the actual NCCL test
	${MPICH_BIN}/mpirun -np 2 ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} --interval 8\
	 	--with-plugin $$PWD/${LIBNAME}.so ./tests/drain_send

cc_simple_check: ${LIBNAME}.so tests/cc_simple
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT}
	${MPICH_BIN}/mpirun -np 2 ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} --interval 8\
		--with-plugin $$PWD/${LIBNAME}.so ./tests/cc_simple

cc_complex_check: ${LIBNAME}.so tests/cc_complex
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	 --coord-port ${DEMO_PORT} 2>/dev/null || true
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT}
	${MPICH_BIN}/mpirun -np 3 ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} --interval 8\
                --with-plugin $$PWD/${LIBNAME}.so ./tests/cc_complex

tests/cc:
	gcc -o test_nccl_wrappers tests/seq_algorithm_test.c src/nccl_wrappers.c tests/seq_stubs.c -I include -lpthread
	./test_nccl_wrappers

tests/drain_recv: tests/drain_recv.c
	nvcc ${CFLAGS} -o $@ $<

tests/drain_send: tests/drain_send.c
	nvcc ${CFLAGS} -o $@ $<

${DMTCP_ROOT}/test/dmtcp1: ${DMTCP_ROOT}/test
	cd ${DMTCP_ROOT}/test && make dmtcp1

${DMTCP_ROOT}/bin:
	cd ${DMTCP_ROOT} && ./configure --disable-prgname-prefix && make -j

${LIBNAME}.so: ${LIBNAME}.o dmtcp_kvdb_c.o nccl_wrappers.o inflight.o
	${CXX} -shared -o $@ $^

${LIBNAME}.o: ${LIBNAME}.c
	${CC} ${CFLAGS} -c -o $@ $<

nccl_wrappers.o: src/nccl_wrappers.c
	${CC} ${CFLAGS} -c -o $@ $<

inflight.o: src/inflight.c
	${CC} ${CFLAGS} -c -o $@ $<

dmtcp_kvdb_c.o: src/dmtcp_kvdb_c.cpp
	${CXX} ${CXXFLAGS} -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -I${DMTCP_SRC} -c -o $@ $<

clean:
	cd tests && make clean
	rm -rf ${LIBNAME}.o ${LIBNAME}.so *.dmtcp *.temp dmtcp_coordinator*.json dmtcp_restart_script*.sh *.o ckpt_*_files src/*.o src/*.so test_nccl_wrappers
