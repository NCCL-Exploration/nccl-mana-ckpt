# Environment required to run PyTorch under DMTCP.  Source this before
# dmtcp_launch.  Each setting works around a DMTCP-side limitation, not
# anything in the NCCL plugin; drop them as those get fixed upstream.
#
# Usage:  . "$(dirname "$0")/pytorch_dmtcp_env.sh"

_pt_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# `import torch` reaches ctypes.util.find_library("dl"), which shells out to
# ldconfig; DMTCP deadlocks in that fork/exec and the import never returns.
# pyshim/sitecustomize.py answers from a table instead of spawning a child.
export PYTHONPATH="${_pt_env_dir}/pyshim${PYTHONPATH:+:${PYTHONPATH}}"

# Note: compression must also be off -- DMTCP pipes the image through gzip, and
# forking that helper hits the same deadlock, leaving the checkpoint thread
# blocked in anon_pipe_write with a partial .dmtcp.temp.  That is passed as
# --no-gzip on the dmtcp_launch line rather than set here, so it is visible in
# the command.  Uncompressed images are several GB per rank.

# PyTorch's default TCPStore is libuv-backed and maps anon_inode:[io_uring].
# DMTCP cannot checkpoint those, and its socket drain does not complete for the
# libuv store either.  The pre-libuv TCPStore avoids both.
export USE_LIBUV=0

export PYTHONUNBUFFERED=1

unset _pt_env_dir
