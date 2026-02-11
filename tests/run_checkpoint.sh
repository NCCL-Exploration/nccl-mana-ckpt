#!/bin/bash
set -e

# Configuration
TEST_EXEC="./single_gpu_test"
PLUGIN_PATH="../src/libncclplugin.so"
DMTCP_ROOT="../dmtcp"
PORT=7781

# Check if plugin exists (warn only, as it might be built later)
if [ ! -f "$PLUGIN_PATH" ]; then
    echo "Warning: Plugin $PLUGIN_PATH not found. Make sure to build it first."
fi

# Clean up previous checkponts
rm -f ckpt_*.dmtcp
rm -f dmtcp_restart_script*.sh

echo "[RUN] Starting DMTCP Coordinator on port $PORT..."
$DMTCP_ROOT/bin/dmtcp_coordinator --daemon --port $PORT

echo "Launching application with DMTCP..."
$DMTCP_ROOT/bin/dmtcp_launch --coord-port $PORT --with-plugin $PLUGIN_PATH $TEST_EXEC &
APP_PID=$!

echo "Application launched. Waiting for it to warm up (10s)..."
sleep 35

echo "Before Triggering checkpoint..."
# sleep 10
echo "Triggering checkpoint..."
$DMTCP_ROOT/bin/dmtcp_command --coord-port $PORT --checkpoint

echo "Waiting a bit after checkpoint..."
sleep 5

# echo "Killing application..."
# $DMTCP_ROOT/bin/dmtcp_command --coord-port $PORT --quit || true
# wait $APP_PID 2>/dev/null || true

echo "Done. Check for ckpt_*.dmtcp files."
ls -l ckpt_*.dmtcp
