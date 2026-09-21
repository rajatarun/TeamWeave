"""Image generation through the Gemini API, as an alternative provider.

Bedrock refused the illustration twice for reasons that had nothing to do with
this code: `amazon.nova-canvas-v1:0` is a real id the account has no active
access to, and `amazon.nova-canvas-v2:0` does not exist. Model access there is
granted per model in a console. The Gemini key is already in Secrets Manager,
already read by `gemini.py`, and already reaching
`generativelanguage.googleapis.com` out of the VPC for the research brief --
so this path is blocked by nothing that is not already unblocked.

It is a *provider*, not a tool in `tool_registry`. Tools are pre/post
processors that shape a step's inputs or outputs around an agent turn; the
illustrator has no agent turn to wrap -- generating the image is the whole
step. Everything provider-independent (the prompt, the S3 write, the presign,
the degrade, the schema) stays in `worker_handler._run_image_step`, and this
answers only "how do I turn a prompt into bytes on this platform" -- the same
seam `agent_runtime.py` is for text.

**The request body is provider-defined and cannot be verified offline**, the
same caveat as `bedrock_image`. Gemini has two image paths with different
shapes -- `:generateContent` returning an `inlineData` part, and Imagen's
`:predict` returning `predictions[].bytesBase64Encoded` -- so the builder is
per family and **refuses** a family it does not know rather than sending one
shape to the other's endpoint.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from .gemini import _get_gemini_key
from .logger import get_logger

log = get_logger("gemini_image")

GENERATE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LIST_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

DEFAULT_MODEL = "gemini-3.1-flash-lite-image"
# The research brief uses 40 s; an image is slower but the worker's remaining
# budget is the real ceiling -- see deadline.py and the caller.
DEFAULT_TIMEOUT = 90

# Families whose `:generateContent` response carries an inlineData image part.
# Imagen answers on `:predict` with an entirely different body and is refused
# here rather than sent a shape it does not take.
KNOWN_FAMILIES = ("gemini-",)


def model_id(declared: str = "") -> str:
    """Most specific first: the member's, then the stack's, then a default."""
    return (declared or "").strip() or (os.environ.get("GEMINI_IMAGE_MODEL") or "").strip() \
        or DEFAULT_MODEL


def knows_shape(model: str) -> bool:
    return any(model.startswith(f) for f in KNOWN_FAMILIES)


def build_body(prompt: str) -> Dict[str, Any]:
    """The `:generateContent` request that asks for an image back.

    `responseModalities` is what separates an image answer from a description
    of one -- without it the model happily returns prose about the picture it
    would draw, which would reach the pipeline as a step that succeeded and
    produced no image.
    """
    return {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }


def first_inline_image(payload: Dict[str, Any]) -> Tuple[bytes, str]:
    """The image bytes and mime type, or a reason there are none.

    A candidate with only text parts is the model describing an image instead
    of drawing one, and a `promptFeedback.blockReason` is a safety refusal.
    Both must raise: returning empty bytes would store a corrupt object and
    report the step as succeeded.
    """
    block = (payload.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise RuntimeError(f"prompt was blocked: {block}")

    for candidate in payload.get("candidates") or []:
        for part in ((candidate or {}).get("content") or {}).get("parts") or []:
            inline = (part or {}).get("inlineData") or (part or {}).get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                mime = str(inline.get("mimeType") or inline.get("mime_type") or "image/png")
                return base64.b64decode(inline["data"]), mime

    finish = ""
    for candidate in payload.get("candidates") or []:
        finish = str((candidate or {}).get("finishReason") or "")
        if finish:
            break
    raise RuntimeError(
        "response carried no image part"
        + (f" (finishReason: {finish})" if finish else "")
        + " -- the model answered with text instead of an image"
    )


def _post(url: str, body: Dict[str, Any], api_key: str, timeout: int) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:300]
        # The body carries the reason -- a wrong model id, a disabled key, a
        # quota. The status alone sends whoever reads it to debug the wrong
        # thing, which is what "invalid model identifier" cost on Bedrock.
        raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc


def generate(prompt: str, *, declared_model_id: str = "", timeout: int = DEFAULT_TIMEOUT,
             api_key: str = "", post=None) -> Dict[str, Any]:
    """Generate one image. Returns {bytes, model_id, prompt, content_type}."""
    model = model_id(declared_model_id)
    if not knows_shape(model):
        raise ValueError(
            f"no known Gemini request shape for {model!r}; this builder speaks "
            "`:generateContent` with an inlineData response. Imagen answers on "
            "`:predict` with a different body and needs its own builder rather "
            "than a new id here"
        )

    key = api_key or _get_gemini_key()
    if not key:
        raise RuntimeError(
            "no Gemini API key: GEMINI_SECRET_ARN is unset or its secret is empty"
        )

    payload = (post or _post)(GENERATE_ENDPOINT.format(model=model), build_body(prompt),
                              key, timeout)
    image_bytes, mime = first_inline_image(payload)
    log.info("gemini_image_generated", extra={"model": model, "bytes": len(image_bytes),
                                              "content_type": mime})
    return {
        "bytes": image_bytes,
        "model_id": model,
        "prompt": prompt,
        "content_type": mime,
    }


# ── which models this key can actually call ─────────────────────────────────

def list_models(api_key: str = "", fetch=None) -> List[Dict[str, Any]]:
    """Models this key can see, with the generation methods each supports.

    The Gemini equivalent of `ListFoundationModels`, and here for the same
    reason: two deploys were spent guessing Bedrock ids, and guessing a Gemini
    one would repeat it. `supportedGenerationMethods` is what says whether a
    model answers `generateContent` at all.
    """
    key = api_key or _get_gemini_key()
    if not key:
        raise RuntimeError("no Gemini API key")

    if fetch is None:
        def fetch(url):  # noqa: E306
            request = urllib.request.Request(
                url, headers={"x-goog-api-key": key}, method="GET")
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))

    data = fetch(LIST_ENDPOINT)
    out = []
    for model in data.get("models") or []:
        name = str(model.get("name") or "").removeprefix("models/")
        methods = model.get("supportedGenerationMethods") or []
        out.append({
            "model": name,
            "generateContent": "generateContent" in methods,
            "shapeKnown": knows_shape(name),
        })
    return sorted(out, key=lambda m: m["model"])


def describe(api_key: str = "", fetch=None) -> str:
    """One line for a CI annotation. Never raises: runs inside error reporting."""
    try:
        models = list_models(api_key=api_key, fetch=fetch)
    except Exception as exc:  # noqa: BLE001 - see docstring
        return f"could not list Gemini models ({type(exc).__name__}: {str(exc)[:160]})"
    usable = [m["model"] for m in models if m["generateContent"] and m["shapeKnown"]]
    if usable:
        return "Gemini models this key can call: " + ", ".join(usable[:8])
    if models:
        return ("this key lists Gemini models but none both support "
                "generateContent and have a request shape here: "
                + ", ".join(m["model"] for m in models[:8]))
    return "this key lists no Gemini models"
