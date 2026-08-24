"""ByteDance custom nodes (Seedream image + Seedance video).

Credentials come from a `.env` file next to this one (see `.env.example`), or
from the ambient environment. There is deliberately no `api_key` widget:
widget values are serialized into the saved workflow JSON *and* into the
metadata of every PNG the workflow produces.

Registered through the ComfyUI v3 extension entrypoint.
"""

import logging
import os

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    # override=False -> an already-exported ARK_API_KEY wins over the file.
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
except ImportError:
    logger.warning(
        "python-dotenv is not installed, so .env will not be read. "
        "Export ARK_API_KEY before starting ComfyUI, or pip install python-dotenv."
    )

from ._compat import IO, ComfyExtension  # noqa: E402
from .nodes import SeedanceVideoNode, SeedreamImageNode  # noqa: E402


class CustomByteDanceExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SeedreamImageNode, SeedanceVideoNode]


async def comfy_entrypoint() -> ComfyExtension:
    return CustomByteDanceExtension()
