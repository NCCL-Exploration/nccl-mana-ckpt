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
	nvcc ${CFLAGS} $(MPI_FLAGS) -o $@ $<

drain_recv: drain_recv.c
	nvcc ${CFLAGS} $(MPI_FLAGS) -o $@ $<

drain_no_msgs: drain_no_msgs.c
	nvcc ${CFLAGS} $(MPI_FLAGS) -o $@ $<

cc_simple: cc_simple.c
	nvcc ${CFLAGS} $(MPI_FLAGS) -o $@ $<

cc_complex: cc_complex.c
	nvcc ${CFLAGS} $(MPI_FLAGS) -o $@ $<

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

# --- Minimal PyTorch DDP checkpoint/restart ---
#
# Needs a venv with torch in this directory (PT_PY overrides the interpreter).
# PYTHONPATH picks up pyshim/sitecustomize.py, which stops `import torch` from
# deadlocking DMTCP in a fork; see that file. Both ranks share one GPU via
# NCCL_MULTI_RANK_GPU_ENABLE, so this runs on a single-GPU host.
PT_PY ?= $(CURDIR)/venv/bin/python
PT_PORT ?= 7783
# One rank per GPU by default, so this runs as-is on a laptop and on a node.
PT_GPUS := $(shell n=$$(nvidia-smi --list-gpus 2>/dev/null | wc -l); \
                   [ "$$n" -gt 0 ] && echo $$n || echo 1)
PT_RANKS ?= $(PT_GPUS)
PT_CKPT_DELAY ?= 25
PT_CKPT_TIMEOUT ?= 400
PT_RESTART_RUN ?= 45
PT_ENV = WORLD_SIZE=$(PT_RANKS) MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
         NCCL_MULTI_RANK_GPU_ENABLE=1 NCCL_DEBUG=WARN

pytorch_minimal: $(NCCL_PLUGIN)
	@echo -e "${GREEN}==> Minimal PyTorch DDP checkpoint/restart <==${NC}"; \
	if [ ! -x "$(PT_PY)" ]; then \
		echo -e "${RED}No interpreter at $(PT_PY); set PT_PY=/path/to/python${NC}"; exit 1; fi; \
	$(DMTCP_CMD) --coord-port $(PT_PORT) --quit --quiet 2>/dev/null || true; \
	rm -f ckpt_*.dmtcp dmtcp_restart_script*.sh pt_rank*.log; rm -rf ckpt_*_files; \
	$(DMTCP_ROOT)/bin/dmtcp_coordinator --daemon -p $(PT_PORT) >/dev/null 2>&1; sleep 1; \
	. $(CURDIR)/pytorch_dmtcp_env.sh; \
	PIDS=""; \
	for r in $$(seq 0 $$(($(PT_RANKS)-1))); do \
		setsid env $(PT_ENV) RANK=$$r LOCAL_RANK=$$((r % $(PT_GPUS))) \
		  $(DMTCP_LAUNCH) --coord-port $(PT_PORT) --interval 0 --no-gzip \
		  --with-plugin $(PLUGIN) $(PT_PY) pytorch_minimal_test.py --steps -1 \
		  > pt_rank$$r.log 2>&1 & \
		PIDS="$$PIDS $$!"; \
	done; \
	sleep $(PT_CKPT_DELAY); \
	echo "Triggering checkpoint mid-training..."; \
	timeout $(PT_CKPT_TIMEOUT) $(DMTCP_CMD) --coord-port $(PT_PORT) --checkpoint; \
	for i in $$(seq 1 40); do \
		[ $$(ls ckpt_*.dmtcp 2>/dev/null | wc -l) -ge $(PT_RANKS) ] && break; sleep 5; \
	done; \
	grep -aE "^\[pt\]|NCCL Plugin|NCCL PLUGIN" pt_rank0.log | tail -20; \
	nimg=$$(ls ckpt_*.dmtcp 2>/dev/null | wc -l); \
	$(DMTCP_CMD) --coord-port $(PT_PORT) --quit --quiet 2>/dev/null || true; \
	for q in $$PIDS; do kill -9 -$$q 2>/dev/null || kill -9 $$q 2>/dev/null || true; done; \
	if ! grep -aq "sequence consensus converged" pt_rank0.log; then \
		echo -e "${RED}FAIL: the plugin never reached a consistent cut${NC}"; exit 1; fi; \
	if [ "$$nimg" -lt $(PT_RANKS) ]; then \
		echo -e "${RED}FAIL: $$nimg checkpoint images, expected $(PT_RANKS)${NC}"; \
		echo -e "${RED}      (known DMTCP-side failure: ProcSelfMaps::getNextArea)${NC}"; exit 1; fi; \
	latest=$$(ls -t ckpt_*.dmtcp | head -n 1); \
	echo -e "${GREEN}--- Restarting from $$latest ---${NC}"; \
	$(DMTCP_RESTART) --coord-port $(PT_PORT) ckpt_*.dmtcp > pt_restart.log 2>&1 & \
	RPID=$$!; \
	sleep $(PT_RESTART_RUN); \
	kill -9 $$RPID 2>/dev/null || true; \
	$(DMTCP_CMD) --coord-port $(PT_PORT) --quit --quiet 2>/dev/null || true; \
	grep -aE "^\[pt\]|NCCL Plugin|rebuild" pt_restart.log | tail -12; \
	nreb=$$(grep -ac "rebuild: comm" pt_restart.log); \
	if [ "$$nreb" -lt $(PT_RANKS) ]; then \
		echo -e "${RED}FAIL: $$nreb communicators rebuilt, expected $(PT_RANKS)${NC}"; exit 1; fi; \
	if grep -aqE "^FAIL|Error\]|terminate called" pt_restart.log; then \
		echo -e "${RED}FAIL: the restarted run reported an error${NC}"; exit 1; fi; \
	steps=$$(grep -ac "all_reduce ok" pt_restart.log); \
	if [ "$$steps" -lt 3 ]; then \
		echo -e "${RED}FAIL: restarted run made no progress${NC}"; exit 1; fi; \
	echo -e "${GREEN}PASS: checkpointed, restarted, $$nreb comm(s) rebuilt, training resumed${NC}"

clean:
	rm -f $(DRIVER) $(P2P_DRIVER) drain_send drain_recv drain_no_msgs cc_simple cc_complex
	rm -f pt_rank*.log pt_restart.log
	rm -rf ckpt_*_files/ *.dmtcp *.temp dmtcp_restart_script*.sh

.PHONY: all test_all test1 test2 test3 test4 gather_p2p_test scatter_p2p_test clean pytorch_minimal
