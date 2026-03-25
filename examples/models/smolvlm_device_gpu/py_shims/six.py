import os
from pathlib import Path


_CANDIDATES = (
    "/usr/lib/python3/dist-packages/six.py",
    "/home/zzl/tools/conda/envs/py310/lib/python3.10/site-packages/six.py",
    "/home/zzl/tools/conda/envs/optimum-executorch/lib/python3.11/site-packages/six.py",
)


for _candidate in _CANDIDATES:
    if os.path.exists(_candidate):
        exec(Path(_candidate).read_text(), globals())
        break
else:
    raise ModuleNotFoundError(
        "Unable to locate a usable six.py on this machine for the local HF shim."
    )
