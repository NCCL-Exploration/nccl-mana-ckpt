CC = nvcc
CFLAGS = -O0 -g -lnccl -L/shared/EL9/explorer/nvidia-hpc-sdk/24.7/Linux_x86_64/24.7/REDIST/comm_libs/12.5/nccl/lib

PLUGIN_NAME = libncclplugin.so
DMTCP_ROOT = ../dmtcp

TEST_EXEC = single_gpu_test
PORT = 7781
WAIT = 10

GREEN = \033[0;32m
NC = \033[0m

checkpoint: ${TEST_EXEC}
	@echo -e "${GREEN}=> [MAKE:CHECKPOINT] Make plugin...${NC}"
	@echo "cd .. && make src/${PLUGIN_NAME}" | bash
	@echo -e "${GREEN}=> [MAKE:CHECKPOINT] Starting DMTCP Coordinator on port ${PORT}...${NC}"
	${DMTCP_ROOT}/bin/dmtcp_coordinator --daemon --port ${PORT}
	@echo -e "${GREEN}=> [MAKE:CHECKPOINT] Launching application with DMTCP...${NC}"
	${DMTCP_ROOT}/bin/dmtcp_launch --coord-port ${PORT} --with-plugin ../src/${PLUGIN_NAME} ./${TEST_EXEC} &
	@echo -e "${GREEN}=> [MAKE:CHECKPOINT] Application launched. Waiting for it to warm up (${WAIT}s)...${NC}"
	@sleep ${WAIT}
	@echo -e "${GREEN}=> [MAKE:CHECKPOINT] Before Triggering checkpoint...${NC}"
	@echo -e "${GREEN}=> [MAKE:CHECKPOINT] Triggering checkpoint...${NC}"
	${DMTCP_ROOT}/bin/dmtcp_command --coord-port ${PORT} --checkpoint
	@echo -e "${GREEN}=> [MAKE:CHECKPOINT] Waiting a bit after checkpoint (5s)...${NC}"
	@sleep 5
	@echo -e "${GREEN}=> [MAKE:CHECKPOINT] Exit.${NC}"

restart:
	@echo -e "${GREEN}=> [MAKE:RESTART] Restarting...${NC}"
	@latest=$$(ls -t ckpt_*.dmtcp 2>/dev/null | head -n1); \
	test -n "$$latest" || { echo "No checkpoint file found."; exit 1; }; \
	${DMTCP_ROOT}/bin/dmtcp_restart --coord-port ${PORT} "$$latest"
	@echo -e "${GREEN}=> [MAKE:RESTART] Exit.${NC}"

quit:
	@echo -e "${GREEN}=> [MAKE:QUIT] Quitting coordinator...${NC}"
	- ${DMTCP_ROOT}/bin/dmtcp_command -q --port ${PORT} 2>/dev/null
	@echo -e "${GREEN}=> [MAKE:QUIT] Killing process...${NC}"
	- kill -9 $$(pidof ${TEST_EXEC}) 2>/dev/null

${TEST_EXEC}: ${TEST_EXEC}.cpp
	${CC} ${CFLAGS} -o $@ $<

clean:
	rm -rf ckpt_*_files/ *.dmtcp *.temp dmtcp_restart_script*.sh ${TEST_EXEC}

