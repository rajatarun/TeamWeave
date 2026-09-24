"""Run artifacts are filed under a summary of the prompt, not the raw run id.

A bucket of uuids cannot be scanned. The folder is a short extractive summary
plus a suffix of the run id, so two runs about the same thing do not collide
and a person can still tell them apart.
"""
from __future__ import annotations

import os

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from src.orchestrator.storage import request_text, run_folder, save_artifact  # noqa: E402


def test_the_folder_is_the_summary_not_the_run_id():
    run_id = "11111111-2222-4333-8444-555555555555"
    folder = run_folder(run_id, "Plan a weekend in Lisbon with the kids")
    assert folder.startswith("plan-weekend-lisbon")
    assert folder != run_id
    assert run_id not in folder
    assert folder.endswith("11111111")


def test_the_same_prompt_with_another_run_does_not_collide():
    summary = "Explore whether to switch the image provider"
    one = run_folder("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", summary)
    two = run_folder("ffffffff-1111-4222-8333-444444444444", summary)
    assert one != two
    assert one.startswith("explore-whether-switch")
    assert two.startswith("explore-whether-switch")


def test_unsafe_characters_never_reach_the_key():
    folder = run_folder("run/../id", "hello ../world\nplan!")
    assert "/" not in folder and ".." not in folder
    assert folder.startswith("hello-world-plan")


def test_an_empty_prompt_still_gets_a_unique_folder():
    folder = run_folder("abcd1234-ffff-4000-8000-000000000000", "")
    assert folder == "run-abcd1234"
    assert folder != "abcd1234-ffff-4000-8000-000000000000"


def test_carried_fields_are_not_the_name():
    text = request_text({
        "dump": "fix the deploy",
        "previous_output": "a very long previous answer that must not become the folder name",
        "edit_instruction": "make it shorter",
    })
    assert text == "fix the deploy"
    assert "previous" not in run_folder("abc12345", text)


def test_save_artifact_uses_the_summary_folder(monkeypatch):
    saved = {}

    class FakeS3:
        def put_object(self, **kwargs):
            saved.update(kwargs)

    monkeypatch.setattr("src.orchestrator.storage.s3", FakeS3())
    monkeypatch.setenv("ARTIFACT_BUCKET", "artifacts")
    uri = save_artifact(
        "99999999-aaaa-4bbb-8ccc-dddddddddddd",
        "DO_plan",
        {"ok": True},
        summary="Plan the Lisbon weekend",
    )
    assert saved["Key"].startswith("runs/plan-lisbon-weekend-99999999/")
    assert saved["Key"].endswith("/DO_plan.json")
    assert "99999999-aaaa" not in saved["Key"]
    assert saved["Metadata"]["run-id"].startswith("99999999")
    assert uri == f"s3://artifacts/{saved['Key']}"
