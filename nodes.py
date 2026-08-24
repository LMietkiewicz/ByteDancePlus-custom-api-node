"""ByteDance nodes: Seedream (image) and Seedance (video).

V3 IO schema, no `is_api_node` (so no Comfy Cloud auth/billing) -- these call
BytePlus ModelArk directly with the key from your `.env`. Blocking SDK/HTTP
calls run in a thread so the ComfyUI event loop stays responsive.
"""

import asyncio
import logging
import time

import comfy.model_management as mm

from . import utils
from ._compat import IO
from .client import make_client

try:
    from server import PromptServer
except Exception:  # pragma: no cover
    PromptServer = None

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 6
POLL_TIMEOUT_S = 30 * 60


def _node_id(cls):
    try:
        return cls.hidden.unique_id
    except Exception:
        return None


def _progress(node_id, text: str) -> None:
    if PromptServer is None or node_id is None:
        return
    try:
        PromptServer.instance.send_progress_text(text, node_id)
    except Exception:
        logger.debug("send_progress_text failed", exc_info=True)


class SeedreamImageNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomSeedreamImage",
            display_name="Custom Seedream (Image)",
            category="Custom/ByteDance",
            description=(
                "ByteDance Seedream text-to-image and image editing via BytePlus ModelArk. "
                "Credentials come from ARK_API_KEY in the .env next to this custom node. "
                "Reference/edit images are sent inline as base64."
            ),
            inputs=[
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Combo.Input(
                    "model",
                    options=utils.SEEDREAM_MODELS,
                    default=utils.SEEDREAM_MODELS[0],
                ),
                IO.String.Input(
                    "model_override",
                    default="",
                    optional=True,
                    tooltip="Non-empty wins over the dropdown. Use for models activated on your "
                    "ModelArk account that are not in the list.",
                ),
                IO.Combo.Input(
                    "size",
                    options=utils.SIZE_PRESETS,
                    default="2K",
                    tooltip="'1K'/'2K'/'4K' let the model pick the aspect from the prompt. "
                    "An explicit WxH pins it, and keeps batches a uniform size.",
                ),
                IO.String.Input(
                    "size_override",
                    default="",
                    optional=True,
                    tooltip="Non-empty wins over the dropdown, e.g. '3024x1296'. "
                    "Per-model pixel and aspect limits apply.",
                ),
                IO.Int.Input(
                    "max_images",
                    default=1,
                    min=1,
                    max=utils.SEEDREAM_TOTAL_IMAGE_LIMIT,
                    tooltip="1 = single image. >1 enables sequential/batch generation. "
                    f"Reference images + max_images must stay <= "
                    f"{utils.SEEDREAM_TOTAL_IMAGE_LIMIT}; this is checked before the request.",
                ),
                IO.Boolean.Input("watermark", default=False),
                IO.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="-1 lets the API pick. Sent with every request.",
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional reference / edit image(s). A batch is sent as multiple refs.",
                ),
            ],
            outputs=[IO.Image.Output(tooltip="Generated image(s) as a single IMAGE batch.")],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        model_override: str,
        size: str,
        size_override: str,
        max_images: int,
        watermark: bool,
        seed: int,
        images=None,
    ) -> IO.NodeOutput:
        if not prompt.strip():
            raise ValueError("Prompt is required.")

        model = (model_override or "").strip() or model
        size = (size_override or "").strip() or size

        refs = utils.batch_to_data_uris(images)
        total = len(refs) + max_images
        if total > utils.SEEDREAM_TOTAL_IMAGE_LIMIT:
            raise ValueError(
                f"{len(refs)} reference image(s) + max_images={max_images} = {total}, over the "
                f"{utils.SEEDREAM_TOTAL_IMAGE_LIMIT}-image per-request limit."
            )

        node_id = _node_id(cls)
        client = make_client()

        kwargs = dict(
            model=model,
            prompt=prompt,
            size=size,
            watermark=watermark,
            response_format="url",
            seed=seed,
        )
        if refs:
            kwargs["image"] = refs[0] if len(refs) == 1 else refs
        if max_images > 1:
            kwargs["sequential_image_generation"] = "auto"
            kwargs["sequential_image_generation_options"] = utils.make_seq_options(max_images)

        _progress(node_id, f"Seedream: generating with {model}...")
        resp = await asyncio.to_thread(lambda: client.images.generate(**kwargs))

        data = list(getattr(resp, "data", []) or [])
        if not data:
            raise RuntimeError("Seedream returned no images.")

        _progress(node_id, f"Seedream: downloading {len(data)} image(s)...")
        tensors = await asyncio.gather(
            *[asyncio.to_thread(utils.image_result_to_tensor, d) for d in data]
        )
        return IO.NodeOutput(utils.stack_images(list(tensors)))


class SeedanceVideoNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CustomSeedanceVideo",
            display_name="Custom Seedance (Video)",
            category="Custom/ByteDance",
            description=(
                "ByteDance Seedance text/image-to-video via BytePlus ModelArk. Credentials come "
                "from ARK_API_KEY in the .env next to this custom node. Images are sent inline as "
                "base64; reference videos must be public URLs. Note that first_frame/last_frame "
                "and reference_images are mutually exclusive modes."
            ),
            inputs=[
                IO.String.Input("prompt", multiline=True, default=""),
                IO.Combo.Input(
                    "model",
                    options=utils.SEEDANCE_MODELS,
                    default=utils.SEEDANCE_MODELS[0],
                ),
                IO.String.Input(
                    "model_override",
                    default="",
                    optional=True,
                    tooltip="Non-empty wins over the dropdown.",
                ),
                IO.Combo.Input(
                    "resolution",
                    options=["480p", "720p", "1080p", "4k"],
                    default="720p",
                    tooltip="4k / 1080p availability depends on the model; the API rejects "
                    "unsupported combinations.",
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
                    tooltip="Seconds. The valid range depends on the model family (1.x 3-12, "
                    "2.x 4-30) and is checked against the selected model before the request.",
                ),
                IO.Boolean.Input("generate_audio", default=True),
                IO.Boolean.Input("watermark", default=False),
                IO.Int.Input(
                    "seed",
                    default=-1,
                    min=-1,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="-1 lets the API pick. Sent with the request, but Seedance is not "
                    "fully deterministic, so a repeated seed will not reproduce a clip exactly.",
                ),
                IO.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip="Literal starting frame the model animates outward from. "
                    "Cannot be combined with reference_images.",
                ),
                IO.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="Literal ending frame. Only valid together with first_frame.",
                ),
                IO.Image.Input(
                    "reference_images",
                    optional=True,
                    tooltip="Unordered identity/style anchors for subject consistency, cited in "
                    "the prompt as 'image 1', 'image 2', .... NOT intermediate keyframes, and "
                    "cannot be combined with first_frame.",
                ),
                IO.String.Input(
                    "reference_video_urls",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip=f"Public URLs of reference videos, one per line (max "
                    f"{utils.MAX_REFERENCE_VIDEOS}). The API accepts URLs only for video, and "
                    "reference video shifts the request to a higher billing rate.",
                ),
            ],
            outputs=[IO.Video.Output(tooltip="Generated clip, downloaded to the temp directory.")],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        model_override: str,
        resolution: str,
        ratio: str,
        duration: int,
        generate_audio: bool,
        watermark: bool,
        seed: int,
        first_frame=None,
        last_frame=None,
        reference_images=None,
        reference_video_urls: str = "",
    ) -> IO.NodeOutput:
        if not prompt.strip():
            raise ValueError("Prompt is required.")

        model = (model_override or "").strip() or model

        lo, hi = utils.duration_limits(model)
        if not lo <= duration <= hi:
            raise ValueError(f"{model} accepts {lo}-{hi}s; got duration={duration}.")

        if last_frame is not None and first_frame is None:
            raise ValueError("last_frame can only be used together with first_frame.")

        # Mutually exclusive per the ByteDance docs: first/last frame pins literal start and
        # end frames, reference images are unordered identity anchors. Delete this if your
        # ModelArk account turns out to accept both.
        if reference_images is not None and first_frame is not None:
            raise ValueError(
                "first_frame/last_frame and reference_images are mutually exclusive modes. "
                "Use first/last frame to pin literal start and end frames, or reference_images "
                "to carry subject identity and style. Not both."
            )

        video_urls = [u.strip() for u in reference_video_urls.splitlines() if u.strip()]
        if len(video_urls) > utils.MAX_REFERENCE_VIDEOS:
            raise ValueError(
                f"At most {utils.MAX_REFERENCE_VIDEOS} reference videos; got {len(video_urls)}."
            )

        node_id = _node_id(cls)
        client = make_client()

        content = [{"type": "text", "text": prompt}]
        if first_frame is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": utils.single_image_data_uri(first_frame, "first_frame")},
                    "role": "first_frame",
                }
            )
        if last_frame is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": utils.single_image_data_uri(last_frame, "last_frame")},
                    "role": "last_frame",
                }
            )
        for uri in utils.batch_to_data_uris(reference_images):
            content.append(
                {"type": "image_url", "image_url": {"url": uri}, "role": "reference_image"}
            )
        for url in video_urls:
            content.append(
                {"type": "video_url", "video_url": {"url": url}, "role": "reference_video"}
            )

        _progress(node_id, f"Seedance: submitting to {model}...")
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

        video_url = await cls._poll(client, create.id, node_id)

        _progress(node_id, "Seedance: downloading clip...")
        video = await asyncio.to_thread(utils.download_video, video_url)
        return IO.NodeOutput(video)

    @classmethod
    async def _poll(cls, client, task_id: str, node_id):
        started = time.monotonic()
        while True:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                # A running task keeps billing after we walk away from it.
                await cls._try_cancel(client, task_id)
                raise

            task = await asyncio.to_thread(
                lambda: client.content_generation.tasks.get(task_id=task_id)
            )
            status = getattr(task, "status", None)
            if status == "succeeded":
                return task.content.video_url
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"Seedance task {status}: {getattr(task, 'error', '')}")

            elapsed = int(time.monotonic() - started)
            if elapsed > POLL_TIMEOUT_S:
                await cls._try_cancel(client, task_id)
                raise RuntimeError(
                    f"Seedance task {task_id} was still '{status}' after {elapsed}s. "
                    "Gave up and asked the API to cancel it."
                )

            _progress(node_id, f"Seedance [{task_id[:8]}]: {status} ({elapsed}s)")
            await asyncio.sleep(POLL_INTERVAL_S)

    @staticmethod
    async def _try_cancel(client, task_id: str) -> None:
        tasks = client.content_generation.tasks
        for attr in ("delete", "cancel"):
            fn = getattr(tasks, attr, None)
            if fn is None:
                continue
            try:
                await asyncio.to_thread(lambda: fn(task_id=task_id))
                logger.info("Cancelled Seedance task %s via tasks.%s", task_id, attr)
                return
            except Exception:
                logger.debug("tasks.%s failed for %s", attr, task_id, exc_info=True)
        logger.warning(
            "Could not cancel Seedance task %s -- it may keep running and billing. "
            "Check the ModelArk console.",
            task_id,
        )