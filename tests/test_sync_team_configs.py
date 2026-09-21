"""The sync that was destroying the platform's own provisioning state.

`aws s3 sync config/examples/teams/` overwrote the same S3 key that
provisioning writes agentId/aliasId back into, and a fresh CI checkout always
has the newer mtime, so the sync always won. Every deploy handed
`needs_provisioning` twelve agents with empty ids and forced a full rebuild --
which grew past the CLI timeout in September, and orphaned the previous
deploy's agents each time.

These pin the invariant that fixes it: the repository owns definitions, S3
owns the runtime identifiers, and a merge never loses the latter.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sync_team_configs", REPO / "scripts" / "sync_team_configs.py")
sync_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync_mod)


def local_team():
    return {
        "team": {"name": "t", "version": "v1", "owner": ""},
        "agents": [
            {
                "id": "A1",
                "name": "Analyzer",
                "goal_template": "NEW PROMPT",
                "bedrock": {
                    "agentId": "",
                    "aliasId": "",
                    "model_id": "us.amazon.nova-micro-v1:0",
                    "model_aliases": {"us.amazon.nova-micro-v1:0": ""},
                },
            }
        ],
    }


def remote_team():
    return {
        "team": {"name": "t", "version": "v1", "team_id": "TID-1"},
        "agents": [
            {
                "id": "A1",
                "name": "Analyzer",
                "goal_template": "OLD PROMPT",
                "bedrock": {
                    "agentId": "AGENT123",
                    "aliasId": "ALIAS123",
                    "model_id": "us.amazon.nova-micro-v1:0",
                    "model_aliases": {"us.amazon.nova-micro-v1:0": "MALIAS1"},
                },
            }
        ],
    }


def test_provisioned_ids_survive_the_sync():
    # The whole bug in one assertion.
    merged = sync_mod.merge_team(local_team(), remote_team())
    bedrock = merged["agents"][0]["bedrock"]
    assert bedrock["agentId"] == "AGENT123"
    assert bedrock["aliasId"] == "ALIAS123"


def test_the_repository_still_wins_for_definitions():
    # The point of syncing at all: a changed prompt must reach S3.
    merged = sync_mod.merge_team(local_team(), remote_team())
    assert merged["agents"][0]["goal_template"] == "NEW PROMPT"


def test_model_alias_ids_survive_for_models_still_declared():
    merged = sync_mod.merge_team(local_team(), remote_team())
    assert merged["agents"][0]["bedrock"]["model_aliases"]["us.amazon.nova-micro-v1:0"] == "MALIAS1"


def test_an_alias_for_a_removed_model_is_not_carried_over():
    # A model dropped from the definition should not keep a stale alias alive.
    remote = remote_team()
    remote["agents"][0]["bedrock"]["model_aliases"]["gone.model-v1"] = "STALE"
    merged = sync_mod.merge_team(local_team(), remote)
    assert "gone.model-v1" not in merged["agents"][0]["bedrock"]["model_aliases"]


def test_team_id_is_preserved_rather_than_regenerated():
    # It is referenced elsewhere; regenerating each deploy repoints those.
    merged = sync_mod.merge_team(local_team(), remote_team())
    assert merged["team"]["team_id"] == "TID-1"


def test_a_team_id_set_in_the_repository_is_respected():
    local = local_team()
    local["team"]["team_id"] = "EXPLICIT"
    assert sync_mod.merge_team(local, remote_team())["team"]["team_id"] == "EXPLICIT"


def test_a_brand_new_team_merges_cleanly_against_nothing():
    merged = sync_mod.merge_team(local_team(), {})
    assert merged["agents"][0]["bedrock"]["agentId"] == ""
    assert merged["agents"][0]["goal_template"] == "NEW PROMPT"


def test_a_new_agent_added_to_an_existing_team_gets_no_stale_ids():
    local = local_team()
    local["agents"].append({"id": "A2", "name": "Second", "bedrock": {"agentId": "", "aliasId": ""}})
    merged = sync_mod.merge_team(local, remote_team())
    assert merged["agents"][1]["bedrock"]["agentId"] == ""
    assert merged["agents"][0]["bedrock"]["agentId"] == "AGENT123"


def test_an_agent_removed_from_the_definition_is_dropped():
    remote = remote_team()
    remote["agents"].append({"id": "GONE", "bedrock": {"agentId": "X", "aliasId": "Y"}})
    merged = sync_mod.merge_team(local_team(), remote)
    assert [a["id"] for a in merged["agents"]] == ["A1"]


def test_agents_are_matched_by_id_not_position():
    # Reordering the definition must not hand one agent another's identity.
    local = local_team()
    local["agents"].append({"id": "A2", "name": "Second", "bedrock": {"agentId": "", "aliasId": ""}})
    local["agents"].reverse()
    merged = sync_mod.merge_team(local, remote_team())
    by_id = {a["id"]: a for a in merged["agents"]}
    assert by_id["A1"]["bedrock"]["agentId"] == "AGENT123"
    assert by_id["A2"]["bedrock"]["agentId"] == ""


def test_an_agent_with_only_a_name_still_matches():
    local, remote = local_team(), remote_team()
    for d in (local, remote):
        d["agents"][0].pop("id")
    assert sync_mod.merge_team(local, remote)["agents"][0]["bedrock"]["agentId"] == "AGENT123"


def test_blank_remote_ids_do_not_overwrite_anything():
    remote = remote_team()
    remote["agents"][0]["bedrock"]["agentId"] = "   "
    merged = sync_mod.merge_team(local_team(), remote)
    assert merged["agents"][0]["bedrock"]["agentId"] == ""


def test_the_real_repo_configs_merge_without_losing_ids(tmp_path):
    # Guards the actual shipped team.json files, not just fixtures.
    teams = sorted((REPO / "config" / "examples" / "teams").rglob("team.json"))
    assert teams, "no team configs found"
    for path in teams:
        local = json.loads(path.read_text())
        remote = json.loads(path.read_text())
        for i, agent in enumerate(remote.get("agents", [])):
            agent.setdefault("bedrock", {})["agentId"] = f"AID{i}"
            agent["bedrock"]["aliasId"] = f"ALIAS{i}"
        merged = sync_mod.merge_team(local, remote)
        ids = [a["bedrock"]["agentId"] for a in merged["agents"]]
        assert ids == [f"AID{i}" for i in range(len(ids))], path


def test_an_id_committed_in_the_repository_is_not_wiped_by_an_empty_remote():
    # The guard that matters: if someone commits a provisioned config and S3
    # has no id for that agent, the merge must not blank it out.
    local = local_team()
    local["agents"][0]["bedrock"]["agentId"] = "FROM-REPO"
    local["agents"][0]["bedrock"]["aliasId"] = "ALIAS-REPO"
    remote = remote_team()
    remote["agents"][0]["bedrock"]["agentId"] = ""
    remote["agents"][0]["bedrock"]["aliasId"] = "   "

    merged = sync_mod.merge_team(local, remote)
    assert merged["agents"][0]["bedrock"]["agentId"] == "FROM-REPO"
    assert merged["agents"][0]["bedrock"]["aliasId"] == "ALIAS-REPO"


def test_a_committed_alias_is_not_wiped_by_an_empty_remote_alias():
    local = local_team()
    local["agents"][0]["bedrock"]["model_aliases"]["us.amazon.nova-micro-v1:0"] = "REPO-ALIAS"
    remote = remote_team()
    remote["agents"][0]["bedrock"]["model_aliases"]["us.amazon.nova-micro-v1:0"] = ""
    merged = sync_mod.merge_team(local, remote)
    assert merged["agents"][0]["bedrock"]["model_aliases"]["us.amazon.nova-micro-v1:0"] == "REPO-ALIAS"


def test_an_agentcore_runtime_arn_also_survives_the_sync():
    # The same clobber, one substrate later. agentId/aliasId are Classic;
    # runtimeArn/qualifier are AgentCore. Both are runtime identity, and
    # covering only the first would have reproduced this bug silently on the
    # newer path the first time an agent was moved.
    local, remote = local_team(), remote_team()
    remote["agents"][0]["bedrock"]["runtimeArn"] = "arn:aws:bedrock-agentcore:us-east-1:1:runtime/r"
    remote["agents"][0]["bedrock"]["qualifier"] = "PROD"

    merged = sync_mod.merge_team(local, remote)
    bedrock = merged["agents"][0]["bedrock"]
    assert bedrock["runtimeArn"].endswith("runtime/r")
    assert bedrock["qualifier"] == "PROD"
    # And the Classic pair is still preserved alongside it.
    assert bedrock["agentId"] == "AGENT123"


# ── Pruning teams the repository no longer defines ───────────────────────────
# Deleting a team from the repository removes it from nobody's view unless S3
# is told: GET /teams still lists it, the UI still offers it, and running it
# starts a pipeline whose definition no longer exists here.
#
# This is deliberately narrower than `aws s3 sync --delete`, which is banned
# because it overwrites keys *within* a team and erased the provisioned ids
# that provisioning writes back.

class _FakeS3:
    def __init__(self, keys):
        self._keys = list(keys)
        self.deleted = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        keys = self._keys

        class _P:
            def paginate(self, Bucket, Prefix):  # noqa: N803 - boto3 casing
                yield {"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}

        return _P()

    def delete_object(self, Bucket, Key):  # noqa: N803 - boto3 casing
        self.deleted.append(Key)


def _sync_module():
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "sync_team_configs", _Path(__file__).resolve().parents[1] / "scripts" / "sync_team_configs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KEYS = [
    "teams/tarun_visibility_team/v1/team.json",
    "teams/tarun_improvement_team/v1/team.json",
    "teams/doc_rewrite_team/v1/team.json",
    "teams/doc_rewrite_team/v1/notes.json",
]


def test_it_deletes_a_team_the_repository_no_longer_has():
    mod = _sync_module()
    s3 = _FakeS3(KEYS)
    removed = mod.prune(s3, "b", "teams", {"tarun_visibility_team", "tarun_improvement_team"})
    assert removed == 1
    assert sorted(s3.deleted) == [
        "teams/doc_rewrite_team/v1/notes.json",
        "teams/doc_rewrite_team/v1/team.json",
    ]


def test_it_leaves_every_team_the_repository_still_defines():
    mod = _sync_module()
    s3 = _FakeS3(KEYS)
    mod.prune(s3, "b", "teams", {"tarun_visibility_team", "tarun_improvement_team", "doc_rewrite_team"})
    assert s3.deleted == []


def test_a_dry_run_deletes_nothing():
    mod = _sync_module()
    s3 = _FakeS3(KEYS)
    removed = mod.prune(s3, "b", "teams", {"tarun_visibility_team"}, dry_run=True)
    assert removed == 2          # reported
    assert s3.deleted == []      # not performed


def test_it_only_looks_inside_the_team_prefix():
    """Keys outside the prefix are not this function's business.

    roles.json and departments.json live at the bucket root; a prune that
    treated them as teams would delete the platform's own configuration.
    """
    mod = _sync_module()
    s3 = _FakeS3(KEYS + ["roles.json", "departments.json", "agentcore/app.zip"])
    mod.prune(s3, "b", "teams", {"tarun_visibility_team", "tarun_improvement_team"})
    assert all(k.startswith("teams/doc_rewrite_team/") for k in s3.deleted), s3.deleted


def test_a_loose_file_under_the_prefix_is_not_a_team():
    """A team is a directory. `teams/notes.txt` is not one.

    Splitting the key and taking the first segment makes any loose file look
    like a team whose name the repository does not have — so it would be
    deleted on the next deploy.
    """
    mod = _sync_module()
    s3 = _FakeS3(KEYS + ["teams/README.md", "teams/index.json"])
    mod.prune(s3, "b", "teams", {"tarun_visibility_team", "tarun_improvement_team"})
    assert "teams/README.md" not in s3.deleted
    assert "teams/index.json" not in s3.deleted


def test_the_sync_refuses_to_prune_against_no_local_teams(tmp_path, capsys):
    """An unreadable root must not be read as "delete everything"."""
    mod = _sync_module()
    calls = []
    mod.prune = lambda *a, **k: calls.append(a)  # type: ignore[assignment]
    empty = tmp_path / "teams"
    empty.mkdir()
    mod.sync(empty, "b", "teams", "us-east-1", dry_run=True, do_prune=True)
    assert not calls, "pruned with nothing to compare against"
    assert "refusing to prune" in capsys.readouterr().err
