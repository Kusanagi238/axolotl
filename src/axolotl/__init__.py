"""Axolotl - Train and fine-tune large language models"""

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # Make this a namespace package

if "__version__" not in globals():
    try:
        from importlib.metadata import version as _get_version
    except Exception:
        try:
            from importlib_metadata import version as _get_version
        except Exception:
            _get_version = None
    __version__ = _get_version("axolotl") if _get_version else None
