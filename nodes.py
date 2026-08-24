"""ByteDance nodes: Seedream (image) and Seedance (video).

V3 IO schema, no `is_api_node` (so no Comfy Cloud auth/billing) -- these call
BytePlus ModelArk directly with your own key. Blocking SDK/HTTP calls run in a
thread so the ComfyUI event loop stays responsive.
"""

import asyncio
import logging

import comfy.model_management as mm

try:
    from comfy_api.latest import IO
except ImportError:  # some builds export it lowercase
    from comfy_api.latest import io as IO

try:
    from server import PromptServer
except Exception:  # pragma: no cover
    PromptServer = None

from . import utils
from .client import DEFAULT_BASE_URL, make_client

logger = logging.getLogger(__name__)


def _progress(node_id, text: str) -> None:
    if PromptServer is None or node_id is None:
        return
    try:
        PromptServer.instance.send_progress_text(text, node_id)
    except Exception:
        pass


class SeedreamImageNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomSeedreamImage",
            display_name="Custom Seedream (Image)",
            category="Custom/ByteDance",
            description=(
                "ByteDance Seedream text-to-image and image editing via BytePlus ModelArk, "
                "using your own API key. Reference/edit images are sent inline as base64."
            ),
            inputs=[
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Combo.Input("model", options=utils.SEEDREAM_MODELS),
                IO.String.Input(
                    "size",
                    default="2K",
                    tooltip="'2K' / '4K' (describe aspect in the prompt), or explicit 'WxH' "
                    "e.g. 2048x2048. Per-model pixel/aspect limits apply; Seedream 3.0 needs WxH.",
                ),
                IO.Int.Input(
                    "max_images",
                    default=1,
                    min=1,
                    max=15,
                    tooltip="1 = single image. >1 enables sequential/batch generation. "
                    "Total input + output images must stay <= 15.",
                ),
                IO.Boolean.Input("watermark", default=False),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional reference / edit image(s). A batch is sent as multiple refs.",
                ),
                IO.String.Input(
                    "api_key",
                    default="",
                    optional=True,
                    tooltip="Leave empty to use the ARK_API_KEY env var.",
                ),
                IO.String.Input("base_url", default=DEFAULT_BASE_URL, optional=True),
            ],
            outputs=[IO.Image.Output()],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        size: str,
        max_images: int,
        watermark: bool,
        seed: int,
        images=None,
        api_key: str = "",
        base_url: str = "",
    ) -> IO.NodeOutput:
        if not prompt.strip():
            raise ValueError("Prompt is required.")

        client = make_client(api_key, base_url)
        kwargs = dict(
            model=model,
            prompt=prompt,
            size=size,
            watermark=watermark,
            response_format="url",
            seed=seed,  # drop this line if your SDK build rejects `seed`
        )

        refs = utils.batch_to_data_uris(images)
        if refs:
            kwargs["image"] = refs[0] if len(refs) == 1 else refs

        if max_images > 1:
            kwargs["sequential_image_generation"] = "auto"
            kwargs["sequential_image_generation_options"] = utils.make_seq_options(max_images)

        resp = await asyncio.to_thread(lambda: client.images.generate(**kwargs))
        data = list(getattr(resp, "data", []) or [])
        if not data:
            raise RuntimeError("Seedream returned no images.")

        tensors = await asyncio.gather(
            *[asyncio.to_thread(utils.image_result_to_tensor, d) for d in data]
        )
        return IO.NodeOutput(utils.stack_or_first(list(tensors)))


class SeedanceVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomSeedanceVideo",
            display_name="Custom Seedance (Video)",
            category="Custom/ByteDance",
            description=(
                "ByteDance Seedance text/image-to-video via BytePlus ModelArk, using your own "
                "API key. first_frame / last_frame / reference images are sent inline as base64; "
                "reference videos must be a public URL."
            ),
            inputs=[
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Combo.Input("model", options=utils.SEEDANCE_MODELS),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p", "4k"],
                    default="720p",
                    tooltip="4k / 1080p availability depends on the model.",
                ),
                IO.Combo.Input(
                    "ratio",
                    options=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"],
                    default="16:9",
                ),
                IO.Int.Input(
                    "duration",
                    default=5,
                    min=3,
                    max=30,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Seconds. Range depends on model (1.x ~3-12, 2.x 4-30).",
                ),
                IO.Boolean.Input("generate_audio", default=True),
                IO.Boolean.Input("watermark", default=False),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="Re-run control only; not sent (Seedance is non-deterministic).",
                ),
                IO.Image.Input("first_frame", optional=True),
                IO.Image.Input("last_frame", optional=True, tooltip="Only valid together with first_frame."),
                IO.Image.Input(
                    "reference_images",
                    optional=True,
                    tooltip="Reference image(s) for subject/character consistency; sent inline as base64.",
                ),
                IO.String.Input(
                    "reference_video_url",
                    default="",
                    optional=True,
                    tooltip="Public URL of a reference video (the API accepts URLs only for video).",
                ),
                IO.String.Input("api_key", default="", optional=True),
                IO.String.Input("base_url", default=DEFAULT_BASE_URL, optional=True),
            ],
            outputs=[IO.Video.Output()],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        resolution: str,
        ratio: str,
        duration: int,
        generate_audio: bool,
        watermark: bool,
        seed: int,
        first_frame=None,
        last_frame=None,
        reference_images=None,
        reference_video_url: str = "",
        api_key: str = "",
        base_url: str = "",
    ) -> IO.NodeOutput:
        if not prompt.strip():
            raise ValueError("Prompt is required.")
        if last_frame is not None and first_frame is None:
            raise ValueError("last_frame can only be used together with first_frame.")

        client = make_client(api_key, base_url)

        content = [{"type": "text", "text": prompt}]
        if first_frame is not None:
            content.append(
                {"type": "image_url", "image_url": {"url": utils.tensor_to_data_uri(first_frame)}, "role": "first_frame"}
            )
        if last_frame is not None:
            content.append(
                {"type": "image_url", "image_url": {"url": utils.tensor_to_data_uri(last_frame)}, "role": "last_frame"}
            )
        for uri in utils.batch_to_data_uris(reference_images):
            content.append({"type": "image_url", "image_url": {"url": uri}, "role": "reference_image"})
        if reference_video_url.strip():
            content.append(
                {"type": "video_url", "video_url": {"url": reference_video_url.strip()}, "role": "reference_video"}
            )

        create = await asyncio.to_thread(
            lambda: client.content_generation.tasks.create(
                model=model,
                content=content,
                generate_audio=generate_audio,
                resolution=resolution,
                ratio=ratio,
                duration=duration,
                seed=seed,
                watermark=watermark,
            )
        )
        video_url = await cls._poll(client, create.id, cls.hidden.unique_id)
        video = await asyncio.to_thread(utils.download_video, video_url)
        return IO.NodeOutput(video)

    @classmethod
    async def _poll(cls, client, task_id: str, node_id, interval: int = 6):
        while True:
            mm.throw_exception_if_processing_interrupted()
            task = await asyncio.to_thread(lambda: client.content_generation.tasks.get(task_id=task_id))
            status = getattr(task, "status", None)
            if status == "succeeded":
                return task.content.video_url
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"Seedance task {status}: {getattr(task, 'error', '')}")
            _progress(node_id, f"Seedance [{task_id[:8]}]: {status}...")
            await asyncio.sleep(interval) 
