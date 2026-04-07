CC = nvcc
CXX = nvcc
CFLAGS = -O0 -g -Xcompiler -fPIC
CXXFLAGS = -O0 -g -Xcompiler -fPIC

# Set DMTCP_ROOT to your DMTCP installation.
ifndef DMTCP_ROOT
  DMTCP_ROOT=${CURDIR}/dmtcp
endif

# Set CRAC_ROOT to your CRAC installation.
ifndef CRAC_ROOT
  CRAC_ROOT=${CURDIR}/crac
endif

LIBNAME = src/libncclplugin
NCCL_INCLUDE = ./include
DMTCP_INCLUDE = ${DMTCP_ROOT}/include
JALIB_INCLUDE = ${DMTCP_ROOT}/jalib
DMTCP_SRC = ${DMTCP_ROOT}/src
LINK = ${CC}

DEMO_PORT=7781

override CFLAGS += -I${DMTCP_INCLUDE} -I${NCCL_INCLUDE} -lnccl -lmpi -L/shared/EL9/explorer/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/comm_libs/12.5/nccl/lib/ -L/shared/EL9/explorer/MPICH/4.3.0b1/lib

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

drain_check: ${LIBNAME}.so tests/drain
	# Kill an old coordinator on this port if present, just in case.
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Launch DMTCP coordinator
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT} 
	# Run the actual NCCL test
	mpirun -np 2  ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} --interval 8\
	 	--with-plugin $$PWD/${LIBNAME}.so ./tests/drain

cc_simple_check: ${LIBNAME}.so tests/cc_simple
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	@${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon -p ${DEMO_PORT}
	mpirun -np 2 ${DMTCP_ROOT}/bin/dmtcp_launch -j --coord-port ${DEMO_PORT} \
		--with-plugin $$PWD/${LIBNAME}.so ./tests/cc_simple

tests/cc:
	gcc -o test_nccl_wrappers tests/seq_algorithm_test.c src/nccl_wrappers.c tests/seq_stubs.c -I include -lpthread
	./test_nccl_wrappers

tests/drain: tests/drain.c
	nvcc ${CFLAGS} -o $@ $<

${DMTCP_ROOT}/test/dmtcp1: ${DMTCP_ROOT}/test
	cd ${DMTCP_ROOT}/test && make dmtcp1

${DMTCP_ROOT}/bin:
	cd ${DMTCP_ROOT} && ./configure --disable-prgname-prefix && make -j

${CRAC_ROOT}/libdmtcp_crac.so:
	cd ${CRAC_ROOT} && make -j DMTCP_ROOT=${DMTCP_ROOT}

${LIBNAME}.so: ${LIBNAME}.o dmtcp_kvdb_c.o nccl_wrappers.o
	${CXX} -shared -o $@ $^

${LIBNAME}.o: ${LIBNAME}.c
	${CC} ${CFLAGS} -c -o $@ $<

nccl_wrappers.o: src/nccl_wrappers.c
	${CC} ${CFLAGS} -c -o $@ $<

dmtcp_kvdb_c.o: src/dmtcp_kvdb_c.cpp
	${CXX} ${CXXFLAGS} -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -I${DMTCP_SRC} -c -o $@ $<

clean:
	cd ${CRAC_ROOT} && make clean
	cd tests && make clean
	rm -rf ${LIBNAME}.o ${LIBNAME}.so *.dmtcp *.temp dmtcp_coordinator*.json dmtcp_restart_script*.sh *.o ckpt_*_files
