# noqa
# pylint: skip-file
import sys

try:
    import torch
except ImportError:
    raise ImportError("Install torch via `pip install torch`")
from packaging.version import Version as V

use_uv = "--uv" in sys.argv[1:]

v = V(torch.__version__)
cuda = str(torch.version.cuda)
try:
    is_ampere = torch.cuda.get_device_capability()[0] >= 8
except RuntimeError:
    is_ampere = False
# Accept a wider range of CUDA versions instead of a strict whitelist.
# Parse major/minor and allow any 12.x or 11.8+; if parsing fails, do not hard-fail.
supported = False
try:
    parts = cuda.split('.')
    major = int(parts[0]) if parts[0] else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    if major >= 12:
        supported = True
    elif major == 11 and minor >= 8:
        supported = True
except Exception:
    # If we can't parse the CUDA version string, avoid blocking installation.
    supported = True

if not supported:
    raise RuntimeError(f"CUDA = {cuda} not supported!")
if v <= V("2.1.0"):
    raise RuntimeError(f"Torch = {v} too old!")
elif v <= V("2.1.1"):
    x = "cu{}{}-torch211"
elif v <= V("2.1.2"):
    x = "cu{}{}-torch212"
elif v < V("2.3.0"):
    x = "cu{}{}-torch220"
elif v < V("2.4.0"):
    x = "cu{}{}-torch230"
elif v < V("2.5.0"):
    x = "cu{}{}-torch240"
elif v < V("2.6.0"):
    x = "cu{}{}-torch250"
else:
    raise RuntimeError(f"Torch = {v} too new!")
x = x.format(cuda.replace(".", ""), "-ampere" if is_ampere else "")
uv_prefix = "uv " if use_uv else ""
print(
    f'{uv_prefix}pip install unsloth-zoo==2024.12.1 && {uv_prefix}pip install --no-deps "unsloth[{x}]==2024.12.4"'
)
