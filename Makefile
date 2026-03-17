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

override CFLAGS += -I${DMTCP_INCLUDE} -I${NCCL_INCLUDE} -lnccl

plugin: src/libncclplugin.so

check: ${DMTCP_ROOT}/bin ${CRAC_ROOT}/libdmtcp_crac.so ${LIBNAME}.so ${DMTCP_ROOT}/test/dmtcp1
	# Kill an old coordinator on this port if present, just in case.
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Note that full path of plugin (using $$PWD in this case) is required.
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 5 \
	  --with-plugin $$PWD/${LIBNAME}.so:${CRAC_ROOT}/libdmtcp_crac.so ${DMTCP_ROOT}/test/dmtcp1

nccl_check: ${DMTCP_ROOT}/bin ${CRAC_ROOT}/libdmtcp_crac.so ${LIBNAME}.so tests/single_gpu_test
	# Kill old coordinator
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Run the actual NCCL test
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 10 \
		--with-plugin $$PWD/${LIBNAME}.so:${CRAC_ROOT}/libdmtcp_crac.so ./tests/single_gpu_test

drain_check: ${DMTCP_ROOT}/bin ${CRAC_ROOT}/libdmtcp_crac.so ${LIBNAME}.so tests/send_recv
	# Kill old coordinator
	@${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Run the actual NCCL test
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 10 \
		--with-plugin $$PWD/${LIBNAME}.so:${CRAC_ROOT}/libdmtcp_crac.so ./tests/send_recv

tests/single_gpu_test: tests/single_gpu_test.c
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
	rm -rf ${LIBNAME}.o ${LIBNAME}.so *.dmtcp *.temp dmtcp_restart_script*.sh *.o
