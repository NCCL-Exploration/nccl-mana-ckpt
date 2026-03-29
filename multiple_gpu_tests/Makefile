CC = nvcc
CFLAGS = -O3 -g -lnccl -L/shared/EL9/explorer/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/REDIST/comm_libs/12.5/nccl/lib -I../include
DMTCP_ROOT ?= ../dmtcp
CRAC_ROOT   ?= ../crac
DMTCP_LAUNCH  = $(DMTCP_ROOT)/bin/dmtcp_launch
DMTCP_CMD     = $(DMTCP_ROOT)/bin/dmtcp_command
DMTCP_RESTART = $(DMTCP_ROOT)/bin/dmtcp_restart
NCCL_PLUGIN = ../src/libncclplugin.so
CRAC_PLUGIN = $(CRAC_ROOT)/libdmtcp_crac.so
PLUGIN = $(NCCL_PLUGIN):$(CRAC_PLUGIN)
DRIVER = nccl_test_driver_multiple
PORT     = 7782          # Different port from single-GPU Makefile to avoid conflicts
INTERVAL = 0             # Manual checkpointing for precise control
export NCCL_DEBUG=WARN
# Default parameters
MSGS ?= 5
SIZE ?= 1024             # in KB
# Colors for output
GREEN = \033[0;32m
RED   = \033[0;31m
NC    = \033[0m
# --- Primary Entry Points ---
all: $(DRIVER)
test_all: test1
	@echo -e "${GREEN}All multi-GPU test criteria completed.${NC}"
# Criteria 1: (your single test)
test1: $(DRIVER)
	@$(call run_test,1,"Criteria 1: Multi-GPU Test",10,$(MSGS),$(SIZE))
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
	rm -f $(DRIVER)
	rm -rf ckpt_*_files/ *.dmtcp *.temp dmtcp_restart_script*.sh
.PHONY: all test_all test1 kill clean