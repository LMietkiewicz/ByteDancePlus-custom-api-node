"""Thin wrapper around the BytePlus ModelArk (ByteDance) SDK.

Everything here is deliberately tiny: resolve a key + base URL and hand back a
plain `Ark` client. The SDK (`byteplussdkarkruntime`) is essentially the OpenAI
client, so callers just do `client.images.generate(...)` /
`client.content_generation.tasks.create(...)`.
"""

import os

DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
_API_KEY_ENV_VARS = ("ARK_API_KEY",)


def resolve_api_key(explicit: str = "") -> str:
    key = (explicit or "").strip()
    if key:
        return key
    for var in _API_KEY_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val:
            return val
    raise RuntimeError(
        "No ByteDance/BytePlus API key found. Put it in the node's `api_key` field "
        f"or set one of these env vars: {', '.join(_API_KEY_ENV_VARS)}."
    )


def resolve_base_url(explicit: str = "") -> str:
    return (explicit or "").strip() or os.environ.get("ARK_BASE_URL", "").strip() or DEFAULT_BASE_URL


def make_client(api_key: str = "", base_url: str = ""):
    """Sync Ark client. Node code runs its blocking calls via asyncio.to_thread,
    which keeps the ComfyUI event loop free without depending on AsyncArk's surface.
    Swap to `from byteplussdkarkruntime import AsyncArk` + `await` if you prefer.
    Imported lazily so a missing dependency surfaces at run time, not import time."""
    try:
        from byteplussdkarkruntime import Ark
    except ImportError as exc:
        raise RuntimeError(
            "byteplus-python-sdk-v2 is not installed. Run: pip install byteplus-python-sdk-v2"
        ) from exc
    return Ark(api_key=resolve_api_key(api_key), base_url=resolve_base_url(base_url))