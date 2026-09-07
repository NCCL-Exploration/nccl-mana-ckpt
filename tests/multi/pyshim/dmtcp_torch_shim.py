"""Stop `import torch` from deadlocking DMTCP.

cuda.pathfinder calls ctypes.util.find_library("dl"), which shells out to
ldconfig through subprocess; DMTCP deadlocks in that fork/exec and the import
never returns.  Answering from a table spawns no child.

Importing this module applies the patch, and is idempotent.  It must happen
before `import torch`.
"""

import ctypes.util

_KNOWN = {
    "dl": "libdl.so.2",
    "c": "libc.so.6",
    "m": "libm.so.6",
    "pthread": "libpthread.so.0",
    "rt": "librt.so.1",
}


def _find_library(name):
    return _KNOWN.get(name)


_find_library._dmtcp_shim = True

if not getattr(ctypes.util.find_library, "_dmtcp_shim", False):
    ctypes.util.find_library = _find_library
