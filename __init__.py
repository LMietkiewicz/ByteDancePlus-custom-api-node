"""ByteDance custom nodes (Seedream image + Seedance video).

Registered through the ComfyUI v3 extension entrypoint. Requires a recent
ComfyUI that exposes `comfy_api.latest`.
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

try:
    from comfy_api.latest import IO, ComfyExtension
except ImportError:  # some builds export IO lowercase
    from comfy_api.latest import ComfyExtension
    from comfy_api.latest import io as IO

from .nodes import SeedanceVideoNode, SeedreamImageNode


class CustomByteDanceExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SeedreamImageNode, SeedanceVideoNode]


async def comfy_entrypoint() -> ComfyExtension:
    return CustomByteDanceExtension()