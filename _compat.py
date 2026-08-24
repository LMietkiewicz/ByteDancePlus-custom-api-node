"""Single place for the ComfyUI v3 API import shim.

Previously duplicated verbatim in __init__.py and nodes.py.
"""

try:
    import comfy_api.latest as _latest
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "This custom node needs a ComfyUI recent enough to expose `comfy_api.latest` "
        "(the v3 node API). Update ComfyUI."
    ) from exc

# Some builds export the module lowercase.
IO = getattr(_latest, "IO", None) or _latest.io
ComfyExtension = _latest.ComfyExtension

__all__ = ["IO", "ComfyExtension"]
