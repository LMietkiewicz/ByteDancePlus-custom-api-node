"""Thin wrapper around the BytePlus ModelArk (ByteDance) SDK.

Credentials are environment-only. `__init__.py` loads `.env` from this package
at import time; nothing here reads a node widget, by design.

The SDK (`byteplussdkarkruntime`) is essentially the OpenAI client, so callers
just do `client.images.generate(...)` /
`client.content_generation.tasks.create(...)`.
"""

import os
import threading

DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"

API_KEY_ENV = "ARK_API_KEY"
BASE_URL_ENV = "ARK_BASE_URL"

_clients: dict[tuple[str, str], object] = {}
_lock = threading.Lock()


def resolve_api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set. Put it in a .env file next to this custom node "
            "(copy .env.example), or export it before starting ComfyUI."
        )
    return key


def resolve_base_url() -> str:
    return os.environ.get(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL


def make_client():
    """Sync Ark client, cached per (key, base_url) so the HTTP pool is reused.

    Node code runs its blocking calls via `asyncio.to_thread`, which keeps the
    ComfyUI event loop free without depending on AsyncArk's surface. The SDK is
    imported lazily so a missing dependency surfaces at run time with a useful
    message rather than breaking node registration at import time.
    """
    cache_key = (resolve_api_key(), resolve_base_url())
    with _lock:
        client = _clients.get(cache_key)
        if client is None:
            try:
                from byteplussdkarkruntime import Ark
            except ImportError as exc:
                raise RuntimeError(
                    "byteplus-python-sdk-v2 is not installed. "
                    "Run: pip install byteplus-python-sdk-v2"
                ) from exc
            client = Ark(api_key=cache_key[0], base_url=cache_key[1])
            _clients[cache_key] = client
    return client