CC = nvcc
NCCL_HOME ?= $(NCCL_DIR)
CFLAGS = -O3 -g -lnccl -I../include
ifneq ($(NCCL_HOME),)
  CFLAGS += -I$(NCCL_HOME)/include -L$(NCCL_HOME)/lib
endif

DMTCP_ROOT ?= ../dmtcp
CRAC_ROOT ?= ../crac

DMTCP_LAUNCH = $(DMTCP_ROOT)/bin/dmtcp_launch
DMTCP_CMD = $(DMTCP_ROOT)/bin/dmtcp_command
DMTCP_RESTART = $(DMTCP_ROOT)/bin/dmtcp_restart

NCCL_PLUGIN = ../src/libncclplugin.so
CUDA_PLUGIN = $(DMTCP_ROOT)/plugin/cuda/libdmtcp_cuda.so
PLUGIN = $(NCCL_PLUGIN):$(CUDA_PLUGIN)

DRIVER = nccl_test_driver
PORT = 7781
INTERVAL = 0 # Manual checkpointing for precise control

export NCCL_DEBUG=WARN

# Default parameters for tests
MSGS ?= 5
SIZE ?= 1024 # in KB

# Colors for output
GREEN = \033[0;32m
RED = \033[0;31m
NC = \033[0m

# --- Primary Entry Points ---

all: $(DRIVER)

test_all: test1 test2 test3 test4 test5
	@echo -e "${GREEN}All test criteria completed.${NC}"

# Criteria 1: Basic P2P Complete
test1: $(DRIVER)
	@$(call run_test,1,"Criteria 1: Basic P2P",0,1,1024)

# Criteria 2: Post-completion Checkpoint
test2: $(DRIVER)
	@$(call run_test,2,"Criteria 2: Post-Completion",5,1,1024)

# Criteria 3: Pre-messaging Checkpoint
test3: $(DRIVER)
	@$(call run_test,3,"Criteria 3: Pre-Init",5,1,1024)

# Criteria 4: Single Msg In-Transit
test4: $(DRIVER)
	@$(call run_test,4,"Criteria 4: Single Msg In-Transit",10,1,$(SIZE))

send_recv: send_recv.c
	nvcc ${CFLAGS} -o $@ $<

simple_init_test: simple_init_test.cpp
	$(CC) $(CFLAGS) -o $@ $<

simple_test: simple_init_test
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

kill:
	-$(DMTCP_CMD) --coord-port $(PORT) --quit --quiet 2>/dev/null || true
	-killall -9 $(DRIVER) 2>/dev/null || true

clean:
	rm -f $(DRIVER) single_gpu_test multiproc send_recv drain_send drain_recv cc_simple cc_complex simple_init_test
	rm -rf ckpt_*_files/ *.dmtcp *.temp dmtcp_restart_script*.sh
	$(MAKE) -C nas-benchmarks/IS clean


.PHONY: all test_all test1 test2 test3 test4 test5 kill clean
