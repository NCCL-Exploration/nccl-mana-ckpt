# netsplit: NCCL's network back end in a lower half that is never checkpointed
# (see README.md).
#
#   module load cudatoolkit/13.2 nccl/2.29.2-cu13; make -C netsplit
#
# bin/netsplit-lh                 the lower half (lower/, from MANA's loader)
# lib/libnccl-net-netsplit.so     the upper half: net proxy + DMTCP plugin
# lib/libcudart_stub.so           the lower half's libcudart.so.13
# lib/libns_cuda_fwd.so, libns_cudart_fwd.so, libns_nvml_fwd.so
#                                 libfabric's CUDA in the lower half (GDR)
# bin/nccl_launch, nccl_restart   the launchers
# bin/probe_v6, bin/gdr_ckpt      phase 0 probes
# bin/gdr_probe, bin/gdr_pin      GDR after a CUDA checkpoint: NIC registration, GDRCopy pin
# bin/ckpt_race                   a running thread vs. a CUDA checkpoint of its own process
# lib/fi-shim/libcudart.so        libfabric's CUDA runtime without the device-wide sync

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

FWD_LIBS = $(LIB)/libns_cuda_fwd.so $(LIB)/libns_cudart_fwd.so $(LIB)/libns_nvml_fwd.so

default: $(LIB)/libnccl-net-netsplit.so $(LIB)/libnccl-net.so $(LIB)/libcudart_stub.so $(FWD_LIBS) \
         lower scripts $(BIN)/probe_v6 $(BIN)/gdr_ckpt $(BIN)/gdr_probe $(BIN)/gdr_pin $(BIN)/ckpt_race \
         $(LIB)/fi-shim/libcudart.so

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
# The stand-ins switch the FS register around forwarded calls (cuda_fwd.h):
# no stack protector.  CUDA headers only; no CUDA library is linked.
STUB_FLAGS = -O2 -g $(WARN) -fPIC -shared -fno-stack-protector -Iinclude -I$(CUDA_HOME)/include
FWD_DEPS = stub/cuda_fwd.h include/netsplit-api.h

$(LIB)/libcudart_stub.so: stub/cudart_stub.c stub/cudart_stub.map $(FWD_DEPS) | $(LIB)
	gcc $(STUB_FLAGS) -o $@ $< \
	  -Wl,-soname,libcudart.so.13 -Wl,--version-script,stub/cudart_stub.map

# libfabric's dlopen() names, as SONAMEs; netsplit file names, so that the
# upper half (netsplit/lib first on its path) never loads them as CUDA.
$(LIB)/libns_cuda_fwd.so: stub/cuda_fwd.c $(FWD_DEPS) | $(LIB)
	gcc $(STUB_FLAGS) -DFWD_CUDA -o $@ $< -Wl,-soname,libcuda.so.1

$(LIB)/libns_cudart_fwd.so: stub/cuda_fwd.c $(FWD_DEPS) | $(LIB)
	gcc $(STUB_FLAGS) -DFWD_CUDART -o $@ $< -Wl,-soname,libcudart.so

$(LIB)/libns_nvml_fwd.so: stub/cuda_fwd.c $(FWD_DEPS) | $(LIB)
	gcc $(STUB_FLAGS) -DFWD_NVML -o $@ $< -Wl,-soname,libnvidia-ml.so.1

scripts: | $(BIN)
	install -m 755 scripts/nccl_launch scripts/nccl_restart $(BIN)/

# No CUDA headers or libraries: the probe stands in for the lower half.
$(BIN)/probe_v6: tests/probe_v6.c include/net_v6.h | $(BIN)
	gcc -O2 -g $(WARN) -Iinclude -o $@ $< -ldl

$(BIN)/gdr_ckpt: tests/gdr_ckpt.c | $(BIN)
	gcc -O2 -g $(WARN) -I$(CUDA_HOME)/include -I$(FABRIC_HOME)/include -o $@ $< \
	  -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcudart -lcuda \
	  -L$(FABRIC_HOME)/lib64 -lfabric -Wl,-rpath,$(FABRIC_HOME)/lib64

$(BIN)/gdr_probe: tests/gdr_probe.c | $(BIN)
	gcc -O2 -g $(WARN) -I$(CUDA_HOME)/include -I$(FABRIC_HOME)/include -o $@ $< \
	  -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcudart -lcuda \
	  -L$(FABRIC_HOME)/lib64 -lfabric -Wl,-rpath,$(FABRIC_HOME)/lib64

$(BIN)/gdr_pin: tests/gdr_pin.c | $(BIN)
	gcc -O2 -g $(WARN) -I$(CUDA_HOME)/include -o $@ $< \
	  -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcudart -lcuda -lgdrapi

$(BIN)/ckpt_race: tests/ckpt_race.c | $(BIN)
	gcc -O2 -g $(WARN) -I$(CUDA_HOME)/include -o $@ $< \
	  -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcudart -lcuda -lpthread

$(LIB)/fi-shim/libcudart.so: tests/fi_cudart_shim.c
	mkdir -p $(dir $@)
	gcc -O2 -g $(WARN) -fPIC -shared -I$(CUDA_HOME)/include -Wl,-soname,libcudart.so -o $@ $< -ldl

clean:
	$(MAKE) -C lower clean
	rm -rf $(BIN) $(LIB) obj

.PHONY: default lower scripts clean
