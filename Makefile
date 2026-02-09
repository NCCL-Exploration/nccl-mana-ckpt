CC = gcc
CFLAGS = -O0 -ggdb -fPIC

# Modify if your DMTCP_ROOT is located elsewhere.
ifndef DMTCP_ROOT
  DMTCP_ROOT=./dmtcp
endif

LIBNAME = src/libncclplugin
DMTCP_INCLUDE = ${DMTCP_ROOT}/include
LINK = ${CC}

DEMO_PORT=7781

override CFLAGS += -I${DMTCP_INCLUDE}

check: ${LIBNAME}.so ${DMTCP_ROOT}/test/dmtcp1
	# Kill an old coordinator on this port if present, just in case.
	@ ${DMTCP_ROOT}/bin/dmtcp_command --quit --quiet \
	  --coord-port ${DEMO_PORT} 2>/dev/null || true
	# Note that full path of plugin (using $$PWD in this case) is required.
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${DEMO_PORT} --interval 5 \
	  --with-plugin $$PWD/${LIBNAME}.so ${DMTCP_ROOT}/test/dmtcp1

${DMTCP_ROOT}/test/dmtcp1: ${DMTCP_ROOT}/test
	cd ${DMTCP_ROOT}/test; make dmtcp1

${LIBNAME}.so: ${LIBNAME}.o
	${LINK} -shared -o $@ $<

${LIBNAME}.o: ${LIBNAME}.c
	${CC} ${CFLAGS} -c -o $@ $<

clean:
	rm -rf ${LIBNAME}.o ${LIBNAME}.so
