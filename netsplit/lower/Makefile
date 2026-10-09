# netsplit-lh, netsplit's lower half (see netsplit-lh.cpp): MANA's kernel
# loader without MPI, plus NCCL's network back end (backend.cpp).
#
#   make -C netsplit/lower        # -> netsplit/bin/netsplit-lh

ifndef DMTCP_ROOT
  DMTCP_ROOT=../../dmtcp
endif
BIN = ../bin

CXX = g++
# No stack protector: the upper half's code that calls in here switches the FS
# register, and glibc's canary lives in the TLS that FS points to.
CXXFLAGS = -std=c++14 -g3 -O0 -fPIC -fno-stack-protector
INCLUDES = -I../include -I. -I$(DMTCP_ROOT)/include -I$(DMTCP_ROOT)/jalib \
           -I$(DMTCP_ROOT)/src -I$(DMTCP_ROOT)/src/mtcp

DMTCP_DEPS = $(DMTCP_ROOT)/src/libdmtcprestart.a \
             $(DMTCP_ROOT)/src/libdmtcpinternal.a \
             $(DMTCP_ROOT)/src/libjalib.a \
             $(DMTCP_ROOT)/src/libnohijack.a

# The lower half's text sits at 0x10000000, out of the upper half's way.
TEXTSEG_ADDR_FLAG = -Wl,-Ttext-segment=0x10000000

OBJS = netsplit-lh.o backend.o mem-wrapper.o switch-context.o \
       copy-stack.o get-symbol-offset.o patch-trampoline.o mmap-fixed-noreplace.o
HEADERS = lower-half-api.h backend.h mem-wrapper.h switch-context.h \
          patch-trampoline.h logging.h ../include/netsplit-api.h ../include/net_v6.h

default: $(BIN)/netsplit-lh

$(BIN):
	mkdir -p $@

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# These are C sources, but MANA built them as C++ (with its C++ compiler
# wrapper), and the rest of the loader declares them with C++ linkage.
%.o: %.c $(HEADERS)
	$(CXX) -x c++ $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BIN)/netsplit-lh: $(OBJS) $(DMTCP_DEPS) | $(BIN)
	$(CXX) -g3 $(TEXTSEG_ADDR_FLAG) $(CXXFLAGS) -o $@ $(OBJS) $(DMTCP_DEPS) \
	  -lpthread -ldl -latomic

clean:
	rm -f $(OBJS) $(BIN)/netsplit-lh

.PHONY: default clean
