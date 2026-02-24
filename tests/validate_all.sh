#!/bin/bash
# Validation script for NCCL Checkpointing Suite

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

PORT=7781
DMTCP_CMD="../dmtcp/bin/dmtcp_command --coord-port $PORT"
DMTCP_LAUNCH="../dmtcp/bin/dmtcp_launch --coord-port $PORT --interval 0"
PLUGIN="../src/libncclplugin.so"
EXE="./nccl_test_driver"

function cleanup() {
    $DMTCP_CMD --quit --quiet 2>/dev/null || true
    rm -f ckpt_*.dmtcp
    rm -rf ckpt_*_files
}

function run_test() {
    local mode=$1
    local name=$2
    local ckpt_delay=$3
    local msgs=${4:-1}
    local size=${5:-4}

    echo -e "${GREEN}=== Running Test: $name (Mode $mode) ===${NC}"
    cleanup
    
    # Start in background
    $DMTCP_LAUNCH --with-plugin $PLUGIN $EXE $mode $msgs $size &
    local pid=$!

    if [ "$ckpt_delay" != "0" ]; then
        echo "Waiting $ckpt_delay seconds for checkpoint trigger..."
        sleep $ckpt_delay
        echo "Triggering checkpoint..."
        $DMTCP_CMD --checkpoint
        sleep 5
    fi

    wait $pid
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}PASS: Initial run successful${NC}"
    else
        echo -e "${RED}FAIL: Initial run failed${NC}"
        return 1
    fi

    # Restart Test
    echo -e "${GREEN}--- Restarting from checkpoint ---${NC}"
    latest=$(ls -t ckpt_*.dmtcp | head -n 1)
    if [ -n "$latest" ]; then
        ../dmtcp/bin/dmtcp_restart --coord-port $PORT "$latest"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}PASS: Restart verified${NC}"
        else
            echo -e "${RED}FAIL: Restart failed${NC}"
            return 1
        fi
    else
        echo "No checkpoint taken (expected for mode 1)"
    fi
}

echo "Building test driver..."
nvcc -O3 -I../include -lnccl -o nccl_test_driver nccl_test_driver.cpp

# Criteria 1: Basic
run_test 1 "Criteria 1: Basic P2P Complete" 0

# Criteria 2: Checkpoint Post-Completion
run_test 2 "Criteria 2: Post-Completion Checkpoint" 5

# Criteria 3: Checkpoint Pre-Messaging
run_test 3 "Criteria 3: Pre-Init Checkpoint" 5

# Criteria 4: Single Message In-Transit
run_test 4 "Criteria 4: Single Msg In-Transit" 10 1 1024

# Criteria 5: Multi Message In-Transit
run_test 5 "Criteria 5: Multi Msg In-Transit" 10 5 1024

cleanup
echo "All test logic generated and scripts prepared."
