include ../Makefile.common

DRIVER = nccl_test_driver_multiple
P2P_DRIVER = nccl_p2p_tests
PORT = 7782
INTERVAL = 0

# Default parameters for tests
MSGS ?= 5
SIZE ?= 1024 # in KB
NGPUS ?= $(shell nvidia-smi --list-gpus 2>/dev/null | wc -l)
GATHER_COUNT ?= 16
GATHER_ROOT  ?= 0
SCATTER_COUNT ?= 16
SCATTER_ROOT  ?= 0

# --- Primary Entry Points ---
all: $(DRIVER)

test_all: test1 test2 test3 test4
	@echo -e "${GREEN}All multi-GPU test criteria completed.${NC}"

gather_p2p_test: $(P2P_DRIVER) $(NCCL_PLUGIN)
	@echo -e "${GREEN}==> Running ncclGather_p2p vs ncclAllGather correctness test <==${NC}"
	@echo "  Params: count=$(GATHER_COUNT), root=$(GATHER_ROOT), gpus=$(NGPUS)"
	@$(MAKE) kill
	@$(DMTCP_LAUNCH) --coord-port $(PORT) --interval $(INTERVAL) \
		--with-plugin $(PLUGIN) ./$(P2P_DRIVER) \
		$(GATHER_COUNT) $(GATHER_ROOT) $(NGPUS)

scatter_p2p_test: $(P2P_DRIVER) $(NCCL_PLUGIN)
	@echo -e "${GREEN}==> Running ncclScatter_p2p correctness test <==${NC}"
	@echo "  Params: count=$(SCATTER_COUNT), root=$(SCATTER_ROOT), gpus=$(NGPUS)"
	@$(MAKE) kill
	@$(DMTCP_LAUNCH) --coord-port $(PORT) --interval $(INTERVAL) \
		--with-plugin $(PLUGIN) ./$(P2P_DRIVER) \
		$(SCATTER_COUNT) $(SCATTER_ROOT) $(NGPUS) scatter

# Test 1: Basic P2P ring — no checkpoint, just verify correctness
test1: $(DRIVER) $(NCCL_PLUGIN)
	@$(call run_test,1,Test 1: Basic P2P ring,0,$(MSGS),$(SIZE),$(NGPUS))

# Test 2: Post-completion checkpoint — comms finish, then sleep 20s
test2: $(DRIVER) $(NCCL_PLUGIN)
	@$(call run_test,2,Test 2: Post-completion checkpoint,5,$(MSGS),$(SIZE),$(NGPUS))

# Test 3: Pre-init checkpoint — sleep 10s before NCCL init
test3: $(DRIVER) $(NCCL_PLUGIN)
	@$(call run_test,3,Test 3: Pre-init checkpoint,5,$(MSGS),$(SIZE),$(NGPUS))

# Test 4: In-transit checkpoint — async send+recv posted, no sync yet
test4: $(DRIVER) $(NCCL_PLUGIN)
	@$(call run_test,4,Test 4: In-transit checkpoint,10,$(MSGS),$(SIZE),$(NGPUS))

drain_send: drain_send.c
	nvcc ${CFLAGS} -o $@ $<

drain_recv: drain_recv.c
	nvcc ${CFLAGS} -o $@ $<

drain_no_msgs: drain_no_msgs.c
	nvcc ${CFLAGS} -o $@ $<

cc_simple: cc_simple.c
	nvcc ${CFLAGS} -o $@ $<

cc_complex: cc_complex.c
	nvcc ${CFLAGS} -o $@ $<

# --- Helper Macro ---
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

$(P2P_DRIVER): $(P2P_DRIVER).cpp
	$(CC) $(CFLAGS) -o $@ $< -ldl

clean:
	rm -f $(DRIVER) $(P2P_DRIVER) drain_send drain_recv drain_no_msgs cc_simple cc_complex
	rm -rf ckpt_*_files/ *.dmtcp *.temp dmtcp_restart_script*.sh

.PHONY: all test_all test1 test2 test3 test4 gather_p2p_test scatter_p2p_test clean
