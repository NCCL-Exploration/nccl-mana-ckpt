include ../Makefile.common

DRIVER = nccl_test_driver
PORT = 7781
INTERVAL = 0

# Default parameters for tests
MSGS ?= 5
SIZE ?= 1024 # in KB

# --- Primary Entry Points ---
all: $(DRIVER)

test_all: test1 test2 test3 test4
	@echo -e "${GREEN}All single-GPU test criteria completed.${NC}"

# Criteria 1: Basic P2P Complete
test1: $(DRIVER) $(NCCL_PLUGIN)
	@$(call run_test,1,"Criteria 1: Basic P2P",0,1,1024)

# Criteria 2: Post-completion Checkpoint
test2: $(DRIVER) $(NCCL_PLUGIN)
	@$(call run_test,2,"Criteria 2: Post-Completion",5,1,1024)

# Criteria 3: Pre-messaging Checkpoint
test3: $(DRIVER) $(NCCL_PLUGIN)
	@$(call run_test,3,"Criteria 3: Pre-Init",5,1,1024)

# Criteria 4: Single Msg In-Transit
test4: $(DRIVER) $(NCCL_PLUGIN)
	@$(call run_test,4,"Criteria 4: Single Msg In-Transit",10,1,$(SIZE))

send_recv: send_recv.c
	nvcc ${CFLAGS} -o $@ $<

simple_init_test: simple_init_test.cpp
	$(CC) $(CFLAGS) -o $@ $<

single_gpu_test: single_gpu_test.c
	nvcc ${CFLAGS} -o $@ $<

multiproc: multiproc.c
	nvcc ${CFLAGS} $(MPI_FLAGS) -o $@ $<

multicomm_test: multicomm_test.c
	nvcc ${CFLAGS} -o $@ $<

# Multi-communicator checkpoint/restart on a single GPU: three independent
# one-rank communicators with unequal per-comm collective counts. This is the
# single-GPU stand-in for tests/multi/cc_complex.c, which needs three GPUs.
MULTICOMM_ITERS ?= 12
MULTICOMM_CKPT_DELAY ?= 5
multicomm: multicomm_test $(NCCL_PLUGIN)
	@echo -e "${GREEN}==> Multi-communicator checkpoint/restart <==${NC}"; \
	$(MAKE) kill; \
	rm -f ckpt_*.dmtcp; \
	$(DMTCP_LAUNCH) --coord-port $(PORT) --interval 0 --with-plugin $(PLUGIN) \
	  ./multicomm_test $(MULTICOMM_ITERS) > multicomm_run1.log 2>&1 & \
	PID=$$!; \
	sleep $(MULTICOMM_CKPT_DELAY); \
	echo "Triggering checkpoint mid-run..."; \
	$(DMTCP_CMD) --coord-port $(PORT) --checkpoint; \
	wait $$PID; RC=$$?; \
	sed -n '1,200p' multicomm_run1.log; \
	if [ $$RC -ne 0 ]; then \
		echo -e "${RED}FAIL: initial run exited $$RC${NC}"; exit 1; fi; \
	if ! grep -q "^PASS multicomm" multicomm_run1.log; then \
		echo -e "${RED}FAIL: initial run did not report PASS${NC}"; exit 1; fi; \
	latest=$$(ls -t ckpt_*.dmtcp 2>/dev/null | head -n 1); \
	if [ -z "$$latest" ]; then \
		echo -e "${RED}FAIL: no checkpoint image was written${NC}"; exit 1; fi; \
	echo -e "${GREEN}--- Restarting from $$latest ---${NC}"; \
	$(DMTCP_RESTART) --coord-port $(PORT) $$latest > multicomm_run2.log 2>&1; \
	RC=$$?; \
	sed -n '1,200p' multicomm_run2.log; \
	if [ $$RC -ne 0 ]; then \
		echo -e "${RED}FAIL: restart exited $$RC${NC}"; exit 1; fi; \
	if ! grep -q "^PASS multicomm" multicomm_run2.log; then \
		echo -e "${RED}FAIL: restarted run did not report PASS${NC}"; exit 1; fi; \
	if ! grep -q "rebuild: comm .* rank 0/1" multicomm_run2.log; then \
		echo -e "${RED}FAIL: no per-communicator rebuild in the restart log${NC}"; exit 1; fi; \
	nrebuilt=$$(grep -c "rebuild: comm .* rank 0/1" multicomm_run2.log); \
	if [ "$$nrebuilt" -ne 3 ]; then \
		echo -e "${RED}FAIL: rebuilt $$nrebuilt communicators, expected 3${NC}"; exit 1; fi; \
	echo -e "${GREEN}PASS: 3 communicators rebuilt and verified after restart${NC}"

simple_test: simple_init_test $(NCCL_PLUGIN)
	@echo -e "${GREEN}==> Running Simple Init Test <==${NC}"; \
	$(MAKE) kill; \
	rm -f ckpt_*.dmtcp; \
	$(DMTCP_LAUNCH) --coord-port $(PORT) --interval 5 --with-plugin $(PLUGIN) ./simple_init_test & \
	PID=$$!; \
	echo "Waiting 5s for checkpoint trigger..."; \
	sleep 5; \
	echo "Triggering checkpoint..."; \
	$(DMTCP_CMD) --coord-port $(PORT) --checkpoint; \
	sleep 10; \
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

# --- Helper Macros ---

define run_test
	echo -e "${GREEN}==> Running $(2) (Test $(1)) <==${NC}"; \
	$(MAKE) kill; \
	rm -f ckpt_*.dmtcp; \
	$(DMTCP_LAUNCH) --coord-port $(PORT) --interval $(3) --with-plugin $(PLUGIN) ./$(DRIVER) $(1) $(4) $(5) & \
	PID=$$!; \
	if [ "$(3)" != "0" ]; then \
		echo "Waiting for checkpoint trigger..."; \
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

clean:
	rm -f $(DRIVER) single_gpu_test multiproc send_recv simple_init_test multicomm_test
	rm -f multicomm_run1.log multicomm_run2.log
	rm -rf ckpt_*_files/ *.dmtcp *.temp dmtcp_restart_script*.sh

.PHONY: all test_all test1 test2 test3 test4 clean simple_test multicomm
