"""The image member: a team member that is not an agent turn.

Bedrock's image models do not implement `Converse`, which is all the AgentCore
runtime program speaks, and they answer with base64 bytes rather than
schema-shaped JSON. So the illustrator is a full member in `team.json` -- role,
department, a place in the workflow -- and the worker runs it through
`bedrock_image` instead of the agent runtime.

The body shape is the one part of this that cannot be verified offline:
`InvokeModel`'s body is an opaque blob in botocore's service model, so these
pin the shape the code *intends* to send and keep it isolated to one builder.
That is the trap that produced `ValidationException: Malformed input request`
when structured_transform kept an Anthropic body on a Nova model.
"""
from __future__ import annotations

import base64
import io
import json
import os

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("ARTIFACT_BUCKET", "test-artifacts")

from src.orchestrator import bedrock_image  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"fake"


class FakeBedrock:
    def __init__(self, payload=None, capture=None):
        self.payload = payload if payload is not None else {
            "images": [base64.b64encode(PNG).decode()]
        }
        self.capture = capture if capture is not None else {}

    def invoke_model(self, **kw):
        self.capture.update(kw)
        return {"body": io.BytesIO(json.dumps(self.payload).encode())}


# ── the request body ────────────────────────────────────────────────────────

def test_the_body_is_the_shape_nova_canvas_takes():
    body = bedrock_image.build_body("a prompt", model="amazon.nova-canvas-v1:0",
                                    width=1280, height=720)
    assert body["taskType"] == "TEXT_IMAGE"
    assert body["textToImageParams"]["text"] == "a prompt"
    assert body["imageGenerationConfig"]["width"] == 1280
    assert body["imageGenerationConfig"]["height"] == 720
    assert body["imageGenerationConfig"]["numberOfImages"] == 1


def test_titan_image_takes_the_same_shape():
    body = bedrock_image.build_body("p", model="amazon.titan-image-generator-v2:0",
                                    width=512, height=512)
    assert body["taskType"] == "TEXT_IMAGE"


def test_an_unknown_family_is_refused_rather_than_sent():
    """Changing the id is not enough when the body is provider-defined.

    A Stability or an OpenAI-shaped model takes a different body entirely, and
    sending this one would come back as ValidationException naming nothing
    useful -- the same failure structured_transform had moving to Nova.
    """
    with pytest.raises(ValueError, match="no known InvokeModel body shape"):
        bedrock_image.build_body("p", model="stability.sd3-large-v1:0",
                                 width=512, height=512)


def test_a_negative_prompt_is_included_only_when_given():
    plain = bedrock_image.build_body("p", model="amazon.nova-canvas-v1:0",
                                     width=8, height=8)
    assert "negativeText" not in plain["textToImageParams"]
    with_neg = bedrock_image.build_body("p", model="amazon.nova-canvas-v1:0",
                                        width=8, height=8, negative="blurry")
    assert with_neg["textToImageParams"]["negativeText"] == "blurry"


# ── the prompt cap ──────────────────────────────────────────────────────────

def test_an_over_long_prompt_is_truncated_not_rejected():
    """Over the cap the model refuses the call outright rather than trimming."""
    long = "word " * 1000
    out = bedrock_image.truncate_prompt(long)
    assert len(out) <= bedrock_image.MAX_PROMPT_CHARS


def test_truncation_lands_on_a_word_boundary():
    out = bedrock_image.truncate_prompt("alpha bravo charlie " * 200)
    assert not out.endswith(" ")
    assert out[-1].isalpha()


def test_a_short_prompt_is_left_alone():
    assert bedrock_image.truncate_prompt("  a   short prompt ") == "a short prompt"


def test_the_body_applies_the_cap():
    body = bedrock_image.build_body("x" * 5000, model="amazon.nova-canvas-v1:0",
                                    width=8, height=8)
    assert len(body["textToImageParams"]["text"]) <= bedrock_image.MAX_PROMPT_CHARS


# ── the response ────────────────────────────────────────────────────────────

def test_the_image_comes_back_as_bytes():
    result = bedrock_image.generate("p", client=FakeBedrock())
    assert result["bytes"] == PNG
    assert result["model_id"] == bedrock_image.DEFAULT_MODEL_ID


def test_a_filtered_response_raises_rather_than_writing_an_empty_object():
    """A body with no image is how these models report a content-filter block.

    Returning empty bytes would put a corrupt object in the artifact bucket
    and report the step as succeeded -- an empty success, which is the failure
    mode this platform has been bitten by three times.
    """
    fake = FakeBedrock(payload={"error": "blocked by content filter"})
    with pytest.raises(RuntimeError, match="content filter"):
        bedrock_image.generate("p", client=fake)


def test_an_empty_image_list_is_also_an_error():
    with pytest.raises(RuntimeError, match="no image"):
        bedrock_image.generate("p", client=FakeBedrock(payload={"images": []}))


def test_the_model_id_is_configurable(monkeypatch):
    monkeypatch.setenv("IMAGE_MODEL_ID", "amazon.titan-image-generator-v2:0")
    capture = {}
    bedrock_image.generate("p", client=FakeBedrock(capture=capture))
    assert capture["modelId"] == "amazon.titan-image-generator-v2:0"


def test_the_call_declares_json_content_types():
    capture = {}
    bedrock_image.generate("p", client=FakeBedrock(capture=capture))
    assert capture["contentType"] == "application/json"
    assert capture["accept"] == "application/json"
    json.loads(capture["body"])  # body must be serialised, not a dict


# ── the step, in the worker ─────────────────────────────────────────────────

def test_the_step_returns_a_reference_never_the_bytes(monkeypatch):
    """Every step output travels through Step Functions state, which caps at
    256 KB. A base64 PNG inline fails the run at the state transition -- after
    paying for the image."""
    from src.orchestrator import worker_handler

    monkeypatch.setattr(worker_handler.bedrock_image, "generate",
                        lambda prompt, **kw: {"bytes": PNG, "model_id": "m",
                                              "prompt": prompt, "width": 8, "height": 8})
    saved = {}
    monkeypatch.setattr(worker_handler, "save_bytes",
                        lambda *a, **kw: saved.setdefault("uri", "s3://b/runs/r/s.png"))

    class Bedrock:
        model_id = "amazon.nova-canvas-v1:0"

    class Agent:
        goal_template = "art direction"
        bedrock = Bedrock()

    out = worker_handler._run_image_step(Agent(), "step-1", "run-1", {"editor.output": {"post": "hi"}})
    assert out["image_uri"] == "s3://b/runs/r/s.png"
    assert "bytes" not in out
    assert PNG not in json.dumps(out, default=str).encode()


def test_the_art_direction_leads_the_prompt(monkeypatch):
    """The cap truncates the tail, so losing the style is worse than losing
    the last sentence of a post the image only has to evoke."""
    from src.orchestrator import worker_handler

    class Agent:
        goal_template = "ART DIRECTION HERE"

    prompt = worker_handler._image_prompt(Agent(), {"editor.output": {"post": "the copy"}})
    assert prompt.startswith("ART DIRECTION HERE")
    assert "the copy" in prompt


def test_the_prompt_skips_plumbing_inputs():
    """RAG context and the raw request are not art direction; including them
    would spend the prompt cap on text the image is not of."""
    from src.orchestrator import worker_handler

    class Agent:
        goal_template = "direction"

    prompt = worker_handler._image_prompt(Agent(), {
        "rag_context": "SHOULD NOT APPEAR",
        "request": {"topic": "ALSO NOT"},
        "editor.output": {"post": "the copy"},
    })
    assert "SHOULD NOT APPEAR" not in prompt
    assert "ALSO NOT" not in prompt
    assert "the copy" in prompt


# ── the branch is actually reached ──────────────────────────────────────────

def test_an_image_member_never_reaches_the_agent_runtime(monkeypatch):
    """The branch existing is not the same as the branch firing.

    Deleting the modality check passed every other test here: the builders,
    the response handling and the step function are all still correct in
    isolation, and an image member would simply be sent to the agent runtime
    as a Converse turn -- which Canvas does not implement. So this drives the
    real pipeline and asserts the image path ran and invoke_agent did not.
    """
    from src.orchestrator import models as m
    from src.orchestrator import worker_handler

    team_doc = {
        "team": {"name": "t", "version": "v1", "owner": "o"},
        "workflow": [{"step": "illustrator", "inputs": ["request"]}],
        "schemas": {},
    }
    globals_ = m.TeamGlobals(north_star="", default_channel="", hard_constraints=[],
                             features={}, rag={"mode": "none"}, artifact_store={},
                             revision={})
    cfg = m.TeamConfig(
        team=team_doc["team"],
        globals=globals_,
        agents=[m.AgentConfig(
            id="illustrator", name="Visual Designer",
            bedrock=m.BedrockRef(agentId="", aliasId="", modality="image",
                                 model_id="amazon.nova-canvas-v1:0"),
            goal_template="art direction", schema_ref="")],
        workflow=team_doc["workflow"],
        schemas={},
    )

    monkeypatch.setattr(worker_handler, "load_team_config", lambda *a, **kw: (cfg, team_doc))

    class FakeDao:
        def put_run_meta(self, *a, **kw): pass
        def put_step(self, *a, **kw): pass
        def put_tasks(self, *a, **kw): pass
    monkeypatch.setattr(worker_handler.DbDao, "from_team_config", staticmethod(lambda *_: FakeDao()))
    monkeypatch.setattr(worker_handler, "save_artifact", lambda *a, **kw: "s3://b/a.json")
    monkeypatch.setattr(worker_handler, "save_bytes", lambda *a, **kw: "s3://b/runs/r/illustrator.png")

    called = {}

    def _fake_generate(prompt, **kw):
        called["image"] = prompt
        return {"bytes": PNG, "model_id": "amazon.nova-canvas-v1:0",
                "prompt": prompt, "width": 8, "height": 8}

    monkeypatch.setattr(worker_handler.bedrock_image, "generate", _fake_generate)

    def _boom(*a, **kw):
        raise AssertionError(
            "an image member was sent to the agent runtime; Canvas does not "
            "implement Converse, so this fails at the first call"
        )
    monkeypatch.setattr(worker_handler, "invoke_agent", _boom)
    monkeypatch.setattr(worker_handler, "invoke_agent_with_metrics", _boom)

    result = worker_handler.run_team_pipeline("t", "v1", {"topic": "x"}, run_id="run-1")

    assert "image" in called, "the image path never ran"
    assert result["steps"]["illustrator"]["image_uri"].endswith(".png")


def test_the_shipped_config_parses_with_its_modality(monkeypatch):
    """The wire from team.json to BedrockRef, through the real loader.

    Every other test here builds a BedrockRef directly, so all of them pass
    with the loader silently dropping `modality` -- the shipped illustrator
    would then arrive as modality="text", be sent to the agent runtime as a
    Converse turn, and fail on a model that does not implement it. This reads
    the file the deploy actually ships and parses it the way the worker does.
    """
    import json as _json
    from pathlib import Path

    from src.orchestrator import config_loader

    repo = Path(__file__).resolve().parents[1]
    doc = _json.loads(
        (repo / "config" / "examples" / "teams" / "tarun_visibility_team"
         / "v1" / "team.json").read_text()
    )
    monkeypatch.setattr(config_loader, "_s3_get_json", lambda *a, **kw: doc)
    monkeypatch.setenv("CONFIG_BUCKET", "b")

    cfg, _ = config_loader.load_team_config("tarun_visibility_team", "v1")
    by_modality = {}
    for agent in cfg.agents:
        by_modality.setdefault(agent.bedrock.modality, []).append(agent.id)

    assert "image" in by_modality, (
        f"no member parsed as an image agent; modalities seen: {sorted(by_modality)}"
    )
    assert len(by_modality["image"]) == 1
    assert by_modality["image"][0].endswith("illustrator")
    # And the rest are still text, or the writers would take the image path.
    assert len(by_modality.get("text", [])) == 4


# ── the image has to be viewable ────────────────────────────────────────────

def test_a_presigned_url_is_offered_beside_the_durable_uri():
    """A browser cannot open s3://, so without this the run produces an image
    nobody can look at. The s3:// URI stays because the signed one expires."""
    from src.orchestrator import storage

    class FakeS3:
        def generate_presigned_url(self, op, Params, ExpiresIn):
            assert op == "get_object"
            assert Params == {"Bucket": "b", "Key": "runs/r/s.png"}
            return "https://b.s3.amazonaws.com/runs/r/s.png?X-Amz-Signature=x"

    storage.s3 = FakeS3()
    assert storage.presign("s3://b/runs/r/s.png").startswith("https://")


def test_a_failed_signature_does_not_fail_the_step(monkeypatch):
    """The artifact is stored and the step has already succeeded. Failing the
    run because a convenience link could not be signed would trade the
    deliverable for the preview of it."""
    from src.orchestrator import storage

    class Boom:
        def generate_presigned_url(self, *a, **kw):
            raise RuntimeError("no credentials")

    storage.s3 = Boom()
    assert storage.presign("s3://b/k.png") == ""


def test_a_non_s3_uri_signs_to_nothing():
    from src.orchestrator import storage
    assert storage.presign("https://example.com/x.png") == ""
    assert storage.presign("") == ""


# ── a failed image must not destroy a finished post ─────────────────────────

def test_a_refused_model_degrades_instead_of_failing_the_run(monkeypatch):
    """What the first real run did: Bedrock refused the image model and the
    whole pipeline ended FAILED, throwing away an approved post because a
    picture of it could not be made. The illustration adorns the deliverable;
    it is not the deliverable."""
    from src.orchestrator import worker_handler

    def _refuse(prompt, **kw):
        raise RuntimeError(
            "An error occurred (ResourceNotFoundException): Access denied. "
            "This Model is marked by provider as Legacy"
        )

    monkeypatch.setattr(worker_handler.bedrock_image, "generate", _refuse)

    class Bedrock:
        model_id = "amazon.nova-canvas-v1:0"

    class Agent:
        goal_template = "art direction"
        bedrock = Bedrock()

    out = worker_handler._run_image_step(Agent(), "s", "r", {"editor.output": {"post": "x"}})
    assert out["error"].startswith("RuntimeError:")
    assert "Legacy" in out["error"]
    # Nothing claims an image exists -- the opposite of an empty success.
    assert out["image_uri"] == ""
    assert out["image_url"] == ""
    # And it names the model that was refused, so the fix is obvious.
    assert out["model_id"] == "amazon.nova-canvas-v1:0"


# ── which model actually gets called ────────────────────────────────────────

def test_the_members_declared_model_wins(monkeypatch):
    """Editing team.json has to change the call.

    generate() read only IMAGE_MODEL_ID, which nothing set, so a member's
    declared model_id was ignored and every request went to the built-in
    default -- changing the config changed nothing and said nothing. That is
    the same trap the text agents had with AGENT_MODEL_ID.
    """
    monkeypatch.setenv("IMAGE_MODEL_ID", "amazon.titan-image-generator-v2:0")
    capture = {}
    bedrock_image.generate("p", declared_model_id="amazon.nova-canvas-v2:0",
                           client=FakeBedrock(capture=capture))
    assert capture["modelId"] == "amazon.nova-canvas-v2:0"


def test_the_stack_default_applies_when_a_member_declares_none(monkeypatch):
    monkeypatch.setenv("IMAGE_MODEL_ID", "amazon.titan-image-generator-v2:0")
    capture = {}
    bedrock_image.generate("p", client=FakeBedrock(capture=capture))
    assert capture["modelId"] == "amazon.titan-image-generator-v2:0"


def test_the_worker_passes_the_declared_model_through(monkeypatch):
    """The wire, not the resolver: _run_image_step must hand it over."""
    from src.orchestrator import worker_handler

    seen = {}

    def _capture(prompt, **kw):
        seen.update(kw)
        return {"bytes": PNG, "model_id": kw.get("declared_model_id", ""),
                "prompt": prompt, "width": 8, "height": 8}

    monkeypatch.setattr(worker_handler.bedrock_image, "generate", _capture)
    monkeypatch.setattr(worker_handler, "save_bytes", lambda *a, **kw: "s3://b/k.png")
    monkeypatch.setattr(worker_handler, "presign", lambda *a, **kw: "")

    class Bedrock:
        model_id = "amazon.nova-canvas-v2:0"

    class Agent:
        goal_template = "d"
        bedrock = Bedrock()

    worker_handler._run_image_step(Agent(), "s", "r", {})
    assert seen.get("declared_model_id") == "amazon.nova-canvas-v2:0", (
        "the member's declared model never reached bedrock_image, so editing "
        "team.json changes nothing"
    )


def test_a_v2_canvas_id_is_a_family_the_builder_knows():
    """The v2 ids a person would reach for must not be refused as unknown."""
    for model in ("amazon.nova-canvas-v2:0", "amazon.titan-image-generator-v2:0"):
        body = bedrock_image.build_body("p", model=model, width=8, height=8)
        assert body["taskType"] == "TEXT_IMAGE"


# ── which service makes the image ───────────────────────────────────────────

def _agent(provider, model="m"):
    class Bedrock:
        image_provider = provider
        model_id = model

    class Agent:
        goal_template = "art direction"
        bedrock = Bedrock()

    return Agent()


def _stub_providers(monkeypatch, worker_handler):
    calls = []

    def bedrock_gen(prompt, **kw):
        calls.append(("bedrock", kw.get("declared_model_id")))
        return {"bytes": PNG, "model_id": "b", "prompt": prompt,
                "width": 8, "height": 8, "content_type": "image/png"}

    def gemini_gen(prompt, **kw):
        calls.append(("gemini", kw.get("declared_model_id")))
        return {"bytes": PNG, "model_id": "g", "prompt": prompt,
                "content_type": "image/png"}

    monkeypatch.setattr(worker_handler.bedrock_image, "generate", bedrock_gen)
    monkeypatch.setattr(worker_handler.gemini_image, "generate", gemini_gen)
    monkeypatch.setattr(worker_handler, "save_bytes", lambda *a, **kw: "s3://b/k.png")
    monkeypatch.setattr(worker_handler, "presign", lambda *a, **kw: "")
    return calls


def test_a_gemini_member_never_reaches_bedrock(monkeypatch):
    """Bedrock refused the illustration twice over model entitlement. Routing
    to Gemini is the point of the provider seam; sending it to Bedrock anyway
    would reproduce the failure the switch exists to avoid."""
    from src.orchestrator import worker_handler

    calls = _stub_providers(monkeypatch, worker_handler)
    out = worker_handler._run_image_step(
        _agent("gemini", "gemini-3.1-flash-lite-image"), "s", "r", {})
    assert calls == [("gemini", "gemini-3.1-flash-lite-image")]
    assert out["provider"] == "gemini"


def test_bedrock_remains_the_default(monkeypatch):
    """A member that declares no provider must not change behaviour."""
    from src.orchestrator import worker_handler

    calls = _stub_providers(monkeypatch, worker_handler)

    class Bedrock:
        model_id = "amazon.nova-canvas-v1:0"

    class Agent:
        goal_template = "d"
        bedrock = Bedrock()

    worker_handler._run_image_step(Agent(), "s", "r", {})
    assert calls == [("bedrock", "amazon.nova-canvas-v1:0")]


def test_an_unknown_provider_degrades_and_names_itself(monkeypatch):
    """A typo must not silently fall through to Bedrock with a Gemini model."""
    from src.orchestrator import worker_handler

    _stub_providers(monkeypatch, worker_handler)
    out = worker_handler._run_image_step(_agent("gemni", "gemini-x"), "s", "r", {})
    assert "unknown image_provider" in out["error"]
    assert out["image_uri"] == ""


def test_the_stored_extension_matches_the_returned_mime(monkeypatch):
    """Gemini may answer JPEG; writing it to a .png key serves a file whose
    extension lies about its contents."""
    from src.orchestrator import worker_handler

    _stub_providers(monkeypatch, worker_handler)
    monkeypatch.setattr(worker_handler.gemini_image, "generate",
                        lambda prompt, **kw: {"bytes": PNG, "model_id": "g",
                                              "prompt": prompt, "content_type": "image/jpeg"})
    saved = {}
    monkeypatch.setattr(worker_handler, "save_bytes",
                        lambda run, step, data, *, extension, content_type: saved.update(
                            extension=extension, content_type=content_type) or "s3://b/k.jpg")

    out = worker_handler._run_image_step(_agent("gemini"), "s", "r", {})
    assert saved == {"extension": "jpg", "content_type": "image/jpeg"}
    assert out["content_type"] == "image/jpeg"


def test_a_gemini_failure_degrades_like_a_bedrock_one(monkeypatch):
    """The post is still the deliverable whichever service refused."""
    from src.orchestrator import worker_handler

    _stub_providers(monkeypatch, worker_handler)

    def refuse(prompt, **kw):
        raise RuntimeError("Gemini HTTP 429: quota exceeded")

    monkeypatch.setattr(worker_handler.gemini_image, "generate", refuse)
    out = worker_handler._run_image_step(_agent("gemini"), "s", "r", {})
    assert "quota exceeded" in out["error"]
    assert out["provider"] == "gemini"
    assert out["image_uri"] == ""


def test_the_shipped_config_parses_with_its_provider(monkeypatch):
    """The wire from team.json to BedrockRef, through the real loader.

    Every other provider test builds a BedrockRef directly, so all of them
    pass with the loader silently dropping `image_provider` -- the shipped
    illustrator would then arrive as "bedrock", be sent to Bedrock with a
    Gemini model id, and fail exactly the way the switch to Gemini exists to
    avoid. This reads the file the deploy ships and parses it as the worker
    does.
    """
    import json as _json
    from pathlib import Path

    from src.orchestrator import config_loader

    repo = Path(__file__).resolve().parents[1]
    doc = _json.loads(
        (repo / "config" / "examples" / "teams" / "tarun_visibility_team"
         / "v1" / "team.json").read_text()
    )
    monkeypatch.setattr(config_loader, "_s3_get_json", lambda *a, **kw: doc)
    monkeypatch.setenv("CONFIG_BUCKET", "b")

    cfg, _ = config_loader.load_team_config("tarun_visibility_team", "v1")
    image_agents = [a for a in cfg.agents if a.bedrock.modality == "image"]
    assert len(image_agents) == 1
    illustrator = image_agents[0]

    assert illustrator.bedrock.image_provider == "gemini", (
        f"parsed provider was {illustrator.bedrock.image_provider!r}; a Gemini "
        "model would be sent to Bedrock"
    )
    # And the pairing survives the parse, not just the file.
    assert illustrator.bedrock.model_id.startswith("gemini-")


def test_a_text_member_never_parses_as_an_image_provider(monkeypatch):
    """The default must stay bedrock for members that say nothing."""
    from src.orchestrator import models as m

    assert m.BedrockRef(agentId="", aliasId="").image_provider == "bedrock"
