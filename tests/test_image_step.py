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

    class Agent:
        goal_template = "art direction"

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
