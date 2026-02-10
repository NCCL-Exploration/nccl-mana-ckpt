#!/bin/bash
set -e

PORT=7781

# Find latest checkpoint
CKPT_FILE=$(ls -t ckpt_*.dmtcp 2>/dev/null | head -n 1)

if [ -z "$CKPT_FILE" ]; then
    echo "Error: No checkpoint file found."
    exit 1
fi

echo "Found checkpoint: $CKPT_FILE"
echo "Restarting..."

# Restart
dmtcp_restart --coord-port $PORT $CKPT_FILE
