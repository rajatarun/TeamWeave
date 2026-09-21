"""Image generation through the Gemini API.

Bedrock refused the illustration twice for reasons unrelated to this code: a
real id with no account access, then an id that does not exist. Model access
there is granted per model in a console. The Gemini key is already in Secrets
Manager, already read by `gemini.py`, and already reaching
`generativelanguage.googleapis.com` from this VPC for the research brief.

The request body is provider-defined and unverifiable offline, exactly like
the Bedrock one, so these pin the shape the code intends to send and keep it
isolated to one builder that refuses families it does not know.
"""
from __future__ import annotations

import base64
import json
import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from src.orchestrator import gemini_image  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"fake"


def inline_response(data=None, mime="image/png"):
    return {"candidates": [{"content": {"parts": [
        {"inlineData": {"mimeType": mime,
                        "data": base64.b64encode(data if data is not None else PNG).decode()}}
    ]}}]}


def capturing_post(capture, response):
    def _post(url, body, api_key, timeout):
        capture.update(url=url, body=body, api_key=api_key, timeout=timeout)
        return response
    return _post


# ── the request ─────────────────────────────────────────────────────────────

def test_it_asks_for_an_image_back():
    """Without responseModalities the model returns prose describing the
    picture it would draw, which reaches the pipeline as a step that succeeded
    and produced no image."""
    body = gemini_image.build_body("a prompt")
    assert body["generationConfig"]["responseModalities"] == ["IMAGE"]
    assert body["contents"][0]["parts"][0]["text"] == "a prompt"


def test_the_model_goes_in_the_url_not_the_body():
    capture = {}
    gemini_image.generate("p", declared_model_id="gemini-2.5-flash-image",
                          api_key="k", post=capturing_post(capture, inline_response()))
    assert capture["url"].endswith("/models/gemini-2.5-flash-image:generateContent")


def test_an_imagen_model_is_refused_rather_than_sent():
    """Imagen answers on `:predict` with a different body entirely. Sending
    this shape there returns an error naming nothing useful -- the same trap
    the Bedrock body had."""
    with pytest.raises(ValueError, match="no known Gemini request shape"):
        gemini_image.generate("p", declared_model_id="imagen-3.0-generate-002", api_key="k")


def test_a_missing_key_is_named_not_guessed_at():
    with pytest.raises(RuntimeError, match="GEMINI_SECRET_ARN"):
        gemini_image.generate("p", declared_model_id="gemini-2.5-flash-image",
                              api_key="", post=lambda *a, **k: inline_response())


# ── which model ─────────────────────────────────────────────────────────────

def test_the_members_declared_model_wins(monkeypatch):
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", "gemini-from-env")
    assert gemini_image.model_id("gemini-from-config") == "gemini-from-config"


def test_the_stack_default_applies_when_none_is_declared(monkeypatch):
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", "gemini-from-env")
    assert gemini_image.model_id("") == "gemini-from-env"


def test_a_built_in_default_remains(monkeypatch):
    monkeypatch.delenv("GEMINI_IMAGE_MODEL", raising=False)
    assert gemini_image.model_id("").startswith("gemini-")


# ── the response ────────────────────────────────────────────────────────────

def test_the_image_comes_back_as_bytes():
    out = gemini_image.generate("p", api_key="k",
                                post=lambda *a, **k: inline_response())
    assert out["bytes"] == PNG
    assert out["content_type"] == "image/png"


def test_the_mime_type_is_taken_from_the_response():
    """Gemini may answer JPEG. Writing that to a .png key serves a file whose
    extension lies."""
    out = gemini_image.generate("p", api_key="k",
                                post=lambda *a, **k: inline_response(mime="image/jpeg"))
    assert out["content_type"] == "image/jpeg"


def test_snake_case_inline_data_is_also_read():
    """The REST API answers camelCase; some clients emit snake_case. Reading
    only one would drop a perfectly good image."""
    payload = {"candidates": [{"content": {"parts": [
        {"inline_data": {"mime_type": "image/png",
                         "data": base64.b64encode(PNG).decode()}}]}}]}
    assert gemini_image.first_inline_image(payload)[0] == PNG


def test_a_text_only_answer_raises():
    """The model describing an image instead of drawing one. Returning empty
    bytes would store a corrupt object and report the step as succeeded."""
    payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": "I would draw a blue circle."}]}}]}
    with pytest.raises(RuntimeError, match="no image part"):
        gemini_image.first_inline_image(payload)


def test_a_safety_block_is_reported_as_one():
    with pytest.raises(RuntimeError, match="blocked: SAFETY"):
        gemini_image.first_inline_image({"promptFeedback": {"blockReason": "SAFETY"}})


def test_an_http_error_carries_the_body_not_just_the_status(monkeypatch):
    """The reason is in the body -- a wrong model, a disabled key, a quota.
    The status alone sends whoever reads it to debug the wrong thing, which is
    exactly what "invalid model identifier" cost on Bedrock.

    This drives the real `_post`: passing `post=` would replace the function
    whose error handling is under test, which is how a test proves nothing.
    """
    import io
    import urllib.error

    def raise_404(request, timeout=None):
        raise urllib.error.HTTPError(
            "u", 404, "Not Found", {},
            io.BytesIO(b'{"error":{"message":"models/x is not found for API version v1beta"}}'))

    monkeypatch.setattr(gemini_image.urllib.request, "urlopen", raise_404)

    with pytest.raises(RuntimeError) as caught:
        gemini_image._post("https://example/x", {"a": 1}, "k", 5)
    assert "Gemini HTTP 404" in str(caught.value)
    assert "is not found" in str(caught.value)


def test_a_successful_post_returns_the_parsed_body(monkeypatch):
    import io

    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"ok": True}).encode()  # noqa: E704

    monkeypatch.setattr(gemini_image.urllib.request, "urlopen",
                        lambda request, timeout=None: Response())
    assert gemini_image._post("https://example/m:generateContent", {}, "k", 5) == {"ok": True}


def test_the_key_travels_in_the_header_not_the_url(monkeypatch):
    """A key in a query string lands in access logs and proxy caches."""
    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"{}"

    def capture(request, timeout=None):
        seen["url"] = request.full_url
        seen["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return Response()

    monkeypatch.setattr(gemini_image.urllib.request, "urlopen", capture)
    gemini_image._post("https://example/models/m:generateContent", {}, "SECRET", 5)
    assert seen["headers"].get("x-goog-api-key") == "SECRET"
    assert "SECRET" not in seen["url"]


# ── model discovery, so nobody guesses an id again ──────────────────────────

def test_list_models_reports_what_each_supports():
    models = gemini_image.list_models(api_key="k", fetch=lambda url: {"models": [
        {"name": "models/gemini-2.5-flash-image",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/imagen-3.0-generate-002",
         "supportedGenerationMethods": ["predict"]},
        {"name": "models/gemini-embedding-001",
         "supportedGenerationMethods": ["embedContent"]},
    ]})
    by_name = {m["model"]: m for m in models}
    assert by_name["gemini-2.5-flash-image"]["generateContent"] is True
    assert by_name["imagen-3.0-generate-002"]["generateContent"] is False
    # Imagen is a Gemini-API model whose shape this builder does not speak.
    assert by_name["imagen-3.0-generate-002"]["shapeKnown"] is False
    assert by_name["gemini-embedding-001"]["generateContent"] is False


def test_describe_names_usable_models():
    text = gemini_image.describe(api_key="k", fetch=lambda url: {"models": [
        {"name": "models/gemini-2.5-flash-image",
         "supportedGenerationMethods": ["generateContent"]}]})
    assert "gemini-2.5-flash-image" in text


def test_describe_never_raises():
    """It runs inside the reporting of another failure."""
    def boom(url):
        raise RuntimeError("network is down")

    assert "could not list Gemini models" in gemini_image.describe(api_key="k", fetch=boom)
