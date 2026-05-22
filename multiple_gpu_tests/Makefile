CC = nvcc
CFLAGS = -O3 -g -lnccl -L/shared/EL9/explorer/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/REDIST/comm_libs/12.5/nccl/lib -I../include -I../src
DMTCP_ROOT ?= ../dmtcp
CRAC_ROOT   ?= ../crac
DMTCP_LAUNCH  = $(DMTCP_ROOT)/bin/dmtcp_launch
DMTCP_CMD     = $(DMTCP_ROOT)/bin/dmtcp_command
DMTCP_RESTART = $(DMTCP_ROOT)/bin/dmtcp_restart
NCCL_PLUGIN = ../src/libncclplugin.so
CRAC_PLUGIN = $(CRAC_ROOT)/libdmtcp_crac.so
PLUGIN = $(NCCL_PLUGIN)
DRIVER = nccl_test_driver_multiple
P2P_DRIVER = nccl_p2p_tests
PORT     = 7782          # Different port from single-GPU Makefile to avoid conflicts
INTERVAL = 0             # Manual checkpointing for precise control
export NCCL_DEBUG=WARN

# --- Configurable Parameters (override on command line) ---
# MSGS  = Number of messages per test
# SIZE  = Message size in KB
# NGPUS = Number of GPUs to use
MSGS  ?= 5
SIZE  ?= 1024
NGPUS ?= $(shell nvidia-smi --list-gpus 2>/dev/null | wc -l)
GATHER_COUNT ?= 16
GATHER_ROOT  ?= 0

# Colors for output
GREEN = \033[0;32m
RED   = \033[0;31m
NC    = \033[0m

# Running test 1 from the make file does not use all gpus available

# --- Primary Entry Points ---
all: $(DRIVER)

test_all: test1 test2 test3 test4
	@echo -e "${GREEN}All multi-GPU test criteria completed.${NC}"

gather_p2p_test: $(P2P_DRIVER) $(NCCL_PLUGIN)
	@echo -e "${GREEN}==> Running ncclGather_p2p correctness test <==${NC}"
	@echo "  Params: count=$(GATHER_COUNT), root=$(GATHER_ROOT), gpus=$(NGPUS)"
	@$(MAKE) kill
	@$(DMTCP_LAUNCH) --coord-port $(PORT) --interval $(INTERVAL) \
		--with-plugin $(PLUGIN) ./$(P2P_DRIVER) \
		$(GATHER_COUNT) $(GATHER_ROOT) $(NGPUS)

# Test 1: Basic P2P ring — no checkpoint, just verify correctness
#   CKPT_DELAY=0 → no checkpoint triggered
test1: $(DRIVER)
	@$(call run_test,1,Test 1: Basic P2P ring,0,$(MSGS),$(SIZE),$(NGPUS))

# Test 2: Post-completion checkpoint — comms finish, then sleep 20s
#   CKPT_DELAY=5 → checkpoint ~5s after launch (comms should be done)
test2: $(DRIVER)
	@$(call run_test,2,Test 2: Post-completion checkpoint,5,$(MSGS),$(SIZE),$(NGPUS))

# Test 3: Pre-init checkpoint — sleep 10s before NCCL init
#   CKPT_DELAY=5 → checkpoint during the pre-init sleep window
test3: $(DRIVER)
	@$(call run_test,3,Test 3: Pre-init checkpoint,5,$(MSGS),$(SIZE),$(NGPUS))

# Test 4: In-transit checkpoint — async send+recv posted, no sync yet
#   CKPT_DELAY=10 → checkpoint while CUDA streams are still processing
test4: $(DRIVER)
	@$(call run_test,4,Test 4: In-transit checkpoint,10,$(MSGS),$(SIZE),$(NGPUS))

# --- Helper Macro ---
# Args: (1)=test_id (2)=description (3)=ckpt_delay (4)=msgs (5)=size_kb (6)=num_gpus
define run_test
	echo -e "${GREEN}==> Running $(2) <==${NC}"; \
	echo "  Params: msgs=$(4), size=$(5)KB, gpus=$(6), ckpt_delay=$(3)s"; \
	$(MAKE) kill; \
	rm -f ckpt_*.dmtcp; \
	$(DMTCP_LAUNCH) --coord-port $(PORT) --interval $(INTERVAL) --with-plugin $(PLUGIN) ./$(DRIVER) $(1) $(4) $(5) $(6) & \
	PID=$$!; \
	if [ "$(3)" != "0" ]; then \
		echo "Waiting $(3)s before triggering checkpoint..."; \
		sleep $(3); \
		echo "Triggering checkpoint..."; \
		$(DMTCP_CMD) --coord-port $(PORT) --checkpoint; \
		sleep 5; \
	fi; \
	wait $$PID; \
	if [ $$? -eq 0 ]; then \
		echo -e "${GREEN}PASS: Initial run successful${NC}"; \
		latest=$$(ls -t ckpt_*.dmtcp 2>/dev/null | head -n 1); \
		if [ -n "$$latest" ]; then \
			echo -e "${GREEN}--- Restarting from checkpoint: $$latest ---${NC}"; \
			$(DMTCP_RESTART) --coord-port $(PORT) $$latest; \
			if [ $$? -eq 0 ]; then \
				echo -e "${GREEN}PASS: Restart verified${NC}"; \
			else \
				echo -e "${RED}FAIL: Restart failed${NC}"; exit 1; \
			fi; \
		fi; \
	else \
		echo -e "${RED}FAIL: Initial run failed${NC}"; exit 1; \
	fi
endef

$(DRIVER): $(DRIVER).cpp
	$(CC) $(CFLAGS) -o $@ $<

$(P2P_DRIVER): $(P2P_DRIVER).cpp $(NCCL_PLUGIN)
	$(CC) $(CFLAGS) -o $@ $< $(NCCL_PLUGIN) -ldl

$(NCCL_PLUGIN): FORCE
	$(MAKE) -C .. plugin

FORCE:

kill:
	-$(DMTCP_CMD) --coord-port $(PORT) --quit --quiet 2>/dev/null || true
	-killall -9 $(DRIVER) 2>/dev/null || true
	-killall -9 $(P2P_DRIVER) 2>/dev/null || true

clean:
	rm -f $(DRIVER) $(P2P_DRIVER)
	rm -rf ckpt_*_files/ *.dmtcp *.temp dmtcp_restart_script*.sh

.PHONY: all test_all test1 test2 test3 test4 gather_p2p_test kill clean FORCE
