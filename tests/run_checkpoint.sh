#!/bin/bash
set -e

# Configuration
TEST_EXEC="./tests/single_gpu_test"
PLUGIN_PATH="./src/libncclplugin.so"
PORT=7781

# Check if plugin exists (warn only, as it might be built later)
if [ ! -f "$PLUGIN_PATH" ]; then
    echo "Warning: Plugin $PLUGIN_PATH not found. Make sure to build it first."
fi

# Clean up previous checkponts
rm -f ckpt_*.dmtcp
rm -f dmtcp_restart_script*.sh

echo "Starting DMTCP Coordinator on port $PORT..."
dmtcp_coordinator --daemon --port $PORT --exit-on-last --exit-after-ckpt

echo "Launching application with DMTCP..."
dmtcp_launch --coord-port $PORT --with-plugin $PLUGIN_PATH $TEST_EXEC &
APP_PID=$!

echo "Application launched. Waiting for it to warm up (10s)..."
sleep 10

echo "Triggering checkpoint..."
dmtcp_command --coord-port $PORT --checkpoint

echo "Waiting a bit after checkpoint..."
sleep 5

echo "Killing application..."
dmtcp_command --coord-port $PORT --quit || true
wait $APP_PID 2>/dev/null || true

echo "Done. Check for ckpt_*.dmtcp files."
ls -l ckpt_*.dmtcp
