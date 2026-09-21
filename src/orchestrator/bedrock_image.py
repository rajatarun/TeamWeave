"""One image-generation turn.

**This does not go through AgentCore, and that is deliberate.** The runtime
program calls `Converse`, and Bedrock's image models do not implement it --
they take `InvokeModel` with a provider-shaped body and answer with base64
bytes. Routing an image model through the agent runtime would fail at the
first call, and teaching that program to branch on model family would put
binary handling, S3 credentials and a second request shape inside the artifact
whose ARM64 packaging already has its own failure mode. An image step has no
ROLE and no STEP_GOAL to compose either; like `structured_transform`, it is a
transform, not an agent.

**The request body is provider-defined and botocore does not describe it.**
`InvokeModel`'s `body` is an opaque blob in the service model, so unlike the
`bedrock-agentcore` client -- which was written against the real shapes -- the
shape below cannot be verified offline. This is the exact trap that produced

    ValidationException: Malformed input request

when `structured_transform` kept an Anthropic-shaped body after moving to a
Nova model. The body builder is therefore per family and isolated, so the one
thing needing a real call to confirm is small and named.

The bytes never enter the pipeline. A base64 PNG is far past Step Functions'
256 KB state limit and every step output travels through it, so the image goes
to the artifact bucket and the step returns a reference.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config

from . import deadline
from .logger import get_logger

log = get_logger("bedrock_image")

DEFAULT_MODEL_ID = "amazon.nova-canvas-v1:0"
# Nova Canvas and Titan Image both cap the prompt; over it the call is
# rejected outright rather than truncated for you.
MAX_PROMPT_CHARS = 1024
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_CFG_SCALE = 6.5


def model_id(declared: str = "") -> str:
    """Most specific first: the agent's own, then the stack's, then a default.

    The declared id is what `team.json` says, and it has to win -- an image
    member that names a model and is answered by another is the same lie the
    text agents told when `AGENT_MODEL_ID` was read only from the runtime's
    environment. Editing the config then changes nothing and says nothing.
    """
    return (declared or "").strip() or (os.environ.get("IMAGE_MODEL_ID") or "").strip() \
        or DEFAULT_MODEL_ID


def _client():
    """Built per call, for the reason the AgentCore client is.

    The pipeline's budget shrinks as earlier steps spend it, and this step runs
    last -- a client made when the function was fresh would hand the image call
    a timeout computed for the director.
    """
    return boto3.client(
        "bedrock-runtime",
        config=Config(
            read_timeout=deadline.budget_for_call(),
            connect_timeout=60,
            retries={"max_attempts": 0},
        ),
    )


def truncate_prompt(text: str, limit: int = MAX_PROMPT_CHARS) -> str:
    """Cut on a word boundary where one is close, so the prompt still reads."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit - 80 else cut).rstrip()


def build_body(prompt: str, *, model: str, width: int, height: int,
               negative: str = "", seed: Optional[int] = None) -> Dict[str, Any]:
    """The provider-shaped request body.

    Nova Canvas inherited Titan Image's schema, so both take the same shape and
    are built together; anything else is refused here rather than sent in a
    body its provider will reject with a message that names nothing useful.
    """
    family = model.split(":")[0]
    if not (family.startswith("amazon.nova-canvas") or family.startswith("amazon.titan-image")):
        raise ValueError(
            f"no known InvokeModel body shape for image model {model!r}; "
            "Nova Canvas and Titan Image are supported, and a new family needs "
            "its body added here rather than the id changed alone"
        )
    config: Dict[str, Any] = {
        "numberOfImages": 1,
        "width": width,
        "height": height,
        "cfgScale": DEFAULT_CFG_SCALE,
    }
    if seed is not None:
        config["seed"] = int(seed)
    params: Dict[str, Any] = {"text": truncate_prompt(prompt)}
    if negative.strip():
        params["negativeText"] = truncate_prompt(negative)
    return {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": params,
        "imageGenerationConfig": config,
    }


def first_image_b64(payload: Dict[str, Any]) -> str:
    """The image out of a response, or a reason there is none.

    A body carrying `error` with no image is how these models report a content
    filter block, and returning empty bytes would land downstream as a corrupt
    object in the bucket rather than as the refusal it is.
    """
    images = payload.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    reason = payload.get("error") or payload.get("message") or "response carried no image"
    raise RuntimeError(f"image generation returned no image: {str(reason)[:300]}")


def generate(prompt: str, *, declared_model_id: str = "",
             width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
             negative: str = "", seed: Optional[int] = None,
             client=None) -> Dict[str, Any]:
    """Generate one image. Returns {bytes, model_id, prompt, width, height}."""
    model = model_id(declared_model_id)
    body = build_body(prompt, model=model, width=width, height=height,
                      negative=negative, seed=seed)
    response = (client or _client()).invoke_model(
        modelId=model,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    raw = response.get("body")
    payload = json.loads(raw.read() if hasattr(raw, "read") else raw)
    image_bytes = base64.b64decode(first_image_b64(payload))
    log.info("image_generated", extra={"model_id": model, "bytes": len(image_bytes),
                                       "width": width, "height": height})
    return {
        "bytes": image_bytes,
        "model_id": model,
        "prompt": body["textToImageParams"]["text"],
        "width": width,
        "height": height,
    }
