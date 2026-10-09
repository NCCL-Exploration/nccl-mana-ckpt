# netsplit: NCCL's network back end in a lower half that is never checkpointed
# (see README.md).
#
#   module load cudatoolkit/13.2 nccl/2.29.2-cu13; make -C netsplit
#
# bin/netsplit-lh                 the lower half (lower/, from MANA's loader)
# lib/libnccl-net-netsplit.so     the upper half: net proxy + DMTCP plugin
# lib/libcudart_stub.so           the lower half's libcudart.so.13
# bin/nccl_launch, nccl_restart   the launchers
# bin/probe_v6, bin/gdr_ckpt      phase 0 probes

NCCL_HOME ?= $(NCCL_DIR)
NVCC_PATH := $(shell command -v nvcc 2>/dev/null)
ifeq ($(origin CUDA_HOME),undefined)
  CUDA_HOME := $(realpath $(dir $(realpath $(NVCC_PATH)))..)
endif
FABRIC_HOME ?= /opt/cray/libfabric/2.3.1
DMTCP_ROOT ?= $(abspath ../dmtcp)

BIN = bin
LIB = lib
WARN = -Wall -Wextra -Wno-unused-parameter
# The proxy switches the FS register around back-end calls: no stack
# protector (its canary lives in the TLS).
UH_FLAGS = -g -O2 -fPIC -fno-stack-protector $(WARN) -Iinclude -Iupper \
           -I$(DMTCP_ROOT)/include -I$(DMTCP_ROOT)/jalib

default: $(LIB)/libnccl-net-netsplit.so $(LIB)/libnccl-net.so $(LIB)/libcudart_stub.so \
         lower scripts $(BIN)/probe_v6 $(BIN)/gdr_ckpt

$(BIN) $(LIB) obj:
	mkdir -p $@

lower: | $(BIN)
	$(MAKE) -C lower DMTCP_ROOT=$(DMTCP_ROOT)

obj/netproxy.o: upper/netproxy.c upper/netsplit-uh.h include/netsplit-api.h include/net_v6.h | obj
	gcc -std=gnu11 $(UH_FLAGS) -c -o $@ $<

obj/netsplit_dmtcp.o: upper/netsplit_dmtcp.cpp upper/netsplit-uh.h include/netsplit-api.h | obj
	g++ -std=c++14 $(UH_FLAGS) -c -o $@ $<

# One library for NCCL (NCCL_NET_PLUGIN=netsplit) and DMTCP (--with-plugin):
# its SONAME makes NCCL's dlopen() find the instance DMTCP preloaded.
$(LIB)/libnccl-net-netsplit.so: obj/netproxy.o obj/netsplit_dmtcp.o | $(LIB)
	g++ -shared -o $@ $^ -Wl,-soname,libnccl-net-netsplit.so -ldl -lpthread

# NCCL unloads its net plugin when the last communicator is destroyed and,
# for the rebuild, opens the default name libnccl-net.so, not NCCL_NET_PLUGIN's
# again.  This link makes that the proxy: the loader finds the same file,
# which DMTCP has preloaded.
$(LIB)/libnccl-net.so: $(LIB)/libnccl-net-netsplit.so
	ln -sfn libnccl-net-netsplit.so $@

# SONAME libcudart.so.13 and its version node: aws-ofi-nccl's DT_NEEDED and
# symbol versions resolve to the stub once it is loaded.
$(LIB)/libcudart_stub.so: stub/cudart_stub.c stub/cudart_stub.map | $(LIB)
	gcc -O2 -g $(WARN) -fPIC -shared -o $@ $< \
	  -Wl,-soname,libcudart.so.13 -Wl,--version-script,stub/cudart_stub.map

scripts: | $(BIN)
	install -m 755 scripts/nccl_launch scripts/nccl_restart $(BIN)/

# No CUDA headers or libraries: the probe stands in for the lower half.
$(BIN)/probe_v6: tests/probe_v6.c include/net_v6.h | $(BIN)
	gcc -O2 -g $(WARN) -Iinclude -o $@ $< -ldl

$(BIN)/gdr_ckpt: tests/gdr_ckpt.c | $(BIN)
	gcc -O2 -g $(WARN) -I$(CUDA_HOME)/include -I$(FABRIC_HOME)/include -o $@ $< \
	  -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcudart -lcuda \
	  -L$(FABRIC_HOME)/lib64 -lfabric -Wl,-rpath,$(FABRIC_HOME)/lib64

clean:
	$(MAKE) -C lower clean
	rm -rf $(BIN) $(LIB) obj

.PHONY: default lower scripts clean
