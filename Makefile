CC = nvcc
CXX = nvcc
CFLAGS = -O0 -g -Xcompiler -fPIC
CXXFLAGS = -O0 -g -Xcompiler -fPIC

# Modify if your DMTCP_ROOT is located elsewhere.
ifndef DMTCP_ROOT
  DMTCP_ROOT=./dmtcp
endif

LIBNAME = src/libncclplugin
NCCL_INCLUDE = ./include
DMTCP_INCLUDE = ${DMTCP_ROOT}/include
LINK = ${CC}

DEMO_PORT=7781

override CFLAGS += -I${DMTCP_INCLUDE} -I${NCCL_INCLUDE} -lnccl -L/shared/EL9/explorer/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/REDIST/comm_libs/12.5/nccl/lib

check: ${LIBNAME}.so ${DMTCP_ROOT}/test/dmtcp1
	# Kill an old coordinator on this port if present, just in case.
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Note that full path of plugin (using $$PWD in this case) is required.
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 5 \
	  --with-plugin $$PWD/${LIBNAME}.so ${DMTCP_ROOT}/test/dmtcp1

nccl_check: ${LIBNAME}.so tests/single_gpu_test
	# Kill old coordinator
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Run the actual NCCL test
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 5 \
	  --with-plugin $$PWD/${LIBNAME}.so ./tests/single_gpu_test

tests/single_gpu_test: tests/single_gpu_test.cpp
	nvcc ${CFLAGS} -o $@ $<

${DMTCP_ROOT}/test/dmtcp1: ${DMTCP_ROOT}/test
	cd ${DMTCP_ROOT}/test; make dmtcp1

${LIBNAME}.so: ${LIBNAME}.o dmtcp_kvdb_c.o nccl_wrappers.o nccl_iprobe.o
	${CXX} -shared -o $@ $^

${LIBNAME}.o: ${LIBNAME}.c
	${CC} ${CFLAGS} -c -o $@ $<

nccl_wrappers.o: src/nccl_wrappers.c
	${CC} ${CFLAGS} -c -o $@ $<

nccl_iprobe.o: src/nccl_iprobe.c
	${CC} ${CFLAGS} -c -o $@ $<

dmtcp_kvdb_c.o: src/dmtcp_kvdb_c.cpp
	${CXX} ${CXXFLAGS} -I${DMTCP_INCLUDE} -c -o $@ $<

clean:
	rm -rf ${LIBNAME}.o ${LIBNAME}.so *.dmtcp *.temp dmtcp_restart_script*.sh *.o
