# Environment required to run PyTorch under DMTCP.  Source this before
# dmtcp_launch.  Each setting works around a DMTCP-side limitation, not
# anything in the NCCL plugin; drop them as those get fixed upstream.
#
# POSIX sh: make runs recipes under /bin/sh, so no bashisms here.  A sourced
# script cannot portably locate itself, so callers pass PT_ENV_DIR; under bash
# it can be left unset.
#
#   PT_ENV_DIR=/path/to/tests/multi . /path/to/tests/multi/pytorch_dmtcp_env.sh

if [ -z "${PT_ENV_DIR:-}" ]; then
    PT_ENV_DIR=$(dirname "${BASH_SOURCE:-$0}")
fi
if [ ! -d "${PT_ENV_DIR}/pyshim" ]; then
    echo "pytorch_dmtcp_env.sh: no pyshim under '${PT_ENV_DIR}';" \
         "set PT_ENV_DIR to the directory holding this file" >&2
    return 1 2>/dev/null || exit 1
fi

# `import torch` reaches ctypes.util.find_library("dl"), which shells out to
# ldconfig; DMTCP deadlocks in that fork/exec and the import never returns.
# pyshim/sitecustomize.py answers from a table instead of spawning a child.
PYTHONPATH="${PT_ENV_DIR}/pyshim${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

# Note: compression must also be off -- DMTCP pipes the image through gzip, and
# forking that helper hits the same deadlock, leaving the checkpoint thread
# blocked in anon_pipe_write with a partial .dmtcp.temp.  That is passed as
# --no-gzip on the dmtcp_launch line rather than set here, so it is visible in
# the command.  Uncompressed images are several GB per rank.

# PyTorch's default TCPStore is libuv-backed and maps anon_inode:[io_uring].
# DMTCP cannot checkpoint those, and its socket drain does not complete for the
# libuv store either.  The pre-libuv TCPStore avoids both.
USE_LIBUV=0
export USE_LIBUV

PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED
