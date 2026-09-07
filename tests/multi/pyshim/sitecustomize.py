"""Applied automatically when pyshim is on PYTHONPATH; see pytorch_dmtcp_env.sh.

Named sitecustomize so Python loads it at startup, before any user import.
The patch itself lives in dmtcp_torch_shim, which a script can also import
directly under a name that is not already taken by the system sitecustomize.
"""

import dmtcp_torch_shim  # noqa: F401
