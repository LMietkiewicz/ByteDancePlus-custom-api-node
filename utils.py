"""Small helpers for the ByteDance nodes.

These are the ~10-20% of Comfy's partner node worth keeping: tensor <-> image
conversion, downloads, and the model lists. Everything talks base64 data URIs so
there is no upload/hosting step (video references are the one exception -- the API
only accepts a URL for those).
"""

import base64
import io
import logging

import numpy as np
import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model lists. These are the BytePlus model IDs; edit to match what is actually
# activated on your ModelArk account (each model is activated + billed separately).
# Order = dropdown order; first entry is the default.
# ---------------------------------------------------------------------------
SEEDREAM_MODELS = [
    "dola-seedream-5-0-pro-260628",
    "seedream-5-0-260128",       # 5.0 lite
    "seedream-4-5-251128",
    "seedream-4-0-250828"
]

SEEDANCE_MODELS = [
    "dreamina-seedance-2-5-260628",
    "dreamina-seedance-2-0-260128",
    "dreamina-seedance-2-0-fast-260128",
    "dreamina-seedance-2-0-mini-260615",
    "seedance-1-5-pro-251215",
    "seedance-1-0-pro-250528",
    "seedance-1-0-pro-fast-251015"
]


# ---------------------------------------------------------------------------
# tensor -> base64 data URI  (ComfyUI IMAGE is (B, H, W, C) float 0..1)
# ---------------------------------------------------------------------------
def _to_pil(image: torch.Tensor) -> Image.Image:
    if image.dim() == 4:
        image = image[0]
    arr = (image.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    pil = Image.fromarray(arr, mode)
    return pil.convert("RGB") if mode == "RGBA" else pil


def tensor_to_data_uri(image: torch.Tensor, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    _to_pil(image).save(buf, format=fmt)
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def batch_to_data_uris(image) -> list[str]:
    """Every frame of an IMAGE batch as its own data URI (empty list if None)."""
    if image is None:
        return []
    if image.dim() == 4:
        return [tensor_to_data_uri(image[i : i + 1]) for i in range(image.shape[0])]
    return [tensor_to_data_uri(image)]


# ---------------------------------------------------------------------------
# download -> tensor
# ---------------------------------------------------------------------------
def _bytes_to_tensor(data: bytes) -> torch.Tensor:
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]  # (1, H, W, C)


def image_result_to_tensor(item) -> torch.Tensor:
    """One element of images.generate(...).data -> (1, H, W, C) tensor."""
    url = getattr(item, "url", None)
    if url:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return _bytes_to_tensor(r.content)
    b64 = getattr(item, "b64_json", None)
    if b64:
        return _bytes_to_tensor(base64.b64decode(b64))
    raise RuntimeError("Image response item had neither `url` nor `b64_json`.")


def stack_or_first(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    if len({t.shape for t in tensors}) == 1:
        return torch.cat(tensors, dim=0)
    # A batch can come back with mixed sizes; ComfyUI IMAGE batches must match.
    logger.warning("Seedream batch returned mixed sizes; returning the first image only.")
    return tensors[0]


def make_seq_options(max_images: int):
    """sequential_image_generation_options for batch mode."""
    try:
        from byteplussdkarkruntime.types.images.images import SequentialImageGenerationOptions

        return SequentialImageGenerationOptions(max_images=max_images)
    except Exception:  # SDK layout differs -> plain dict still serializes fine
        return {"max_images": max_images}


# ---------------------------------------------------------------------------
# video download -> ComfyUI VIDEO output
# ---------------------------------------------------------------------------
def _video_from_file_cls():
    """VideoFromFile has moved across ComfyUI versions; resolve it lazily."""
    last_err = None
    for module, attr in (
        ("comfy_api.latest._input_impl", "VideoFromFile"),
        ("comfy_api.input_impl", "VideoFromFile"),
        ("comfy_api.latest", "VideoFromFile"),
    ):
        try:
            return getattr(__import__(module, fromlist=[attr]), attr)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Could not import VideoFromFile from comfy_api: {last_err}")


def download_video(url: str):
    import os
    import uuid

    import folder_paths

    video_from_file = _video_from_file_cls()
    out_dir = folder_paths.get_temp_directory()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"formaai_seedance_{uuid.uuid4().hex}.mp4")
    r = requests.get(url, timeout=600, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
    return video_from_file(path)