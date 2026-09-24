"""Object events must not each call StartIngestionJob.

A bulk upload used to invoke the sync function once per object. Bedrock
answered ThrottlingException, the invocation raised, and the objects were
not indexed. The queue gathers the events. One invocation starts one job,
and a job already running puts the messages back so they are tried again
after it finishes.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from botocore.exceptions import ClientError

from src.orchestrator import health_kb, portfolio_kb

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()


class CfnLoader(yaml.SafeLoader):
    """CloudFormation short forms, keeping the argument."""


def _keep(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {"__fn__": suffix, "__arg__": value}


CfnLoader.add_multi_constructor("!", _keep)


class Context:
    def get_remaining_time_in_millis(self):
        return 60_000


def _throttle(message="Rate limit exceeded"):
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": message}},
        "StartIngestionJob",
    )


def _records(*ids):
    return {
        "Records": [
            {
                "messageId": message_id,
                "eventSource": "aws:sqs",
                "body": '{"detail":{"object":{"key":"statement-%s.pdf"}}}' % message_id,
            }
            for message_id in ids
        ]
    }


class Listing:
    def __init__(self, statuses=(), start=None):
        self.statuses = list(statuses)
        self.starts = []
        self.start = start

    def list_ingestion_jobs(self, **_kwargs):
        status = self.statuses.pop(0) if self.statuses else None
        summaries = [{"status": status}] if status else []
        return {"ingestionJobSummaries": summaries}

    def start_ingestion_job(self, **kwargs):
        self.starts.append(kwargs)
        if self.start is not None:
            outcome = self.start
            if isinstance(outcome, list):
                outcome = outcome.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
        return {"ingestionJob": {"ingestionJobId": "job"}}


def test_a_throttled_start_retries_and_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(health_kb, "_backoff_sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(health_kb.random, "uniform", lambda _a, _b: 0.0)
    client = Listing(start=[_throttle(), None])
    result = health_kb.start_sync(client=client, kb_id="KB", source_id="DS")
    assert result == {"kbSync": "started"}
    assert len(client.starts) == 2
    assert sleeps == [health_kb.SYNC_BASE_DELAY_SECONDS]


def test_throttling_that_does_not_clear_returns_the_batch(monkeypatch):
    monkeypatch.setattr(health_kb, "_backoff_sleep", lambda _seconds: None)
    monkeypatch.setattr(health_kb.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KB")
    monkeypatch.setenv("HEALTH_DATA_SOURCE_ID", "DS")
    client = Listing(start=[_throttle()] * health_kb.SYNC_MAX_ATTEMPTS)
    monkeypatch.setattr(health_kb, "_client", lambda *_a, **_k: client)
    result = health_kb.handle_sync_event(_records("a", "b"), Context())
    assert result == {
        "batchItemFailures": [
            {"itemIdentifier": "a"},
            {"itemIdentifier": "b"},
        ]
    }
    assert len(client.starts) == health_kb.SYNC_MAX_ATTEMPTS


def test_a_running_job_returns_every_message_to_the_queue(monkeypatch):
    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    monkeypatch.setenv("HEALTH_DATA_SOURCE_ID", "DSHEALTH")
    client = Listing(statuses=["IN_PROGRESS"])
    monkeypatch.setattr(health_kb, "_client", lambda *_a, **_k: client)
    result = health_kb.handle_sync_event(_records("one", "two", "three"), Context())
    assert result == {
        "batchItemFailures": [
            {"itemIdentifier": "one"},
            {"itemIdentifier": "two"},
            {"itemIdentifier": "three"},
        ]
    }
    assert client.starts == []


def test_conflict_defers_without_a_second_start(monkeypatch):
    monkeypatch.setattr(health_kb, "_backoff_sleep", lambda _seconds: None)
    client = Listing(start=ClientError(
        {"Error": {"Code": "ConflictException", "Message": "already running"}},
        "StartIngestionJob",
    ))
    result = health_kb.start_sync(client=client, kb_id="KB", source_id="DS")
    assert result == {"kbSync": "deferred"}
    assert len(client.starts) == 1


def test_a_batch_of_uploads_starts_one_job(monkeypatch):
    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    monkeypatch.setenv("HEALTH_DATA_SOURCE_ID", "DSHEALTH")
    logged = []
    monkeypatch.setattr(health_kb.log, "info", lambda *a, **k: logged.append((a, k)))
    monkeypatch.setattr(health_kb.log, "warning", lambda *a, **k: logged.append((a, k)))
    client = Listing()
    monkeypatch.setattr(health_kb, "_client", lambda *_a, **_k: client)
    result = health_kb.sync_handler(_records("m1", "m2", "m3", "m4", "m5"), Context())
    assert result == {"batchItemFailures": []}
    assert len(client.starts) == 1
    assert client.starts[0]["knowledgeBaseId"] == "KBHEALTH"
    assert client.starts[0]["dataSourceId"] == "DSHEALTH"
    assert "statement-" not in str(logged)


def test_a_direct_defer_raises_so_the_event_is_not_acked(monkeypatch):
    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    monkeypatch.setenv("HEALTH_DATA_SOURCE_ID", "DSHEALTH")
    client = Listing(statuses=["STARTING"])
    monkeypatch.setattr(health_kb, "_client", lambda *_a, **_k: client)
    with pytest.raises(health_kb.SyncDeferred):
        health_kb.sync_handler(
            {"detail": {"object": {"key": "labs.pdf"}}}, Context())
    assert client.starts == []


def test_portfolio_batch_uses_the_portfolio_base_once(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_KNOWLEDGE_BASE_ID", "KBPORT")
    monkeypatch.setenv("PORTFOLIO_DATA_SOURCE_ID", "DSPORT")
    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    client = Listing()
    monkeypatch.setattr(health_kb, "_client", lambda *_a, **_k: client)
    result = portfolio_kb.sync_handler(_records("p1", "p2"), Context())
    assert result == {"batchItemFailures": []}
    assert len(client.starts) == 1
    assert client.starts[0]["knowledgeBaseId"] == "KBPORT"
    assert client.starts[0]["dataSourceId"] == "DSPORT"


def test_an_in_progress_portfolio_batch_is_returned(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_KNOWLEDGE_BASE_ID", "KBPORT")
    monkeypatch.setenv("PORTFOLIO_DATA_SOURCE_ID", "DSPORT")
    client = Listing(statuses=["IN_PROGRESS"])
    monkeypatch.setattr(health_kb, "_client", lambda *_a, **_k: client)
    result = portfolio_kb.sync_handler(_records("p1", "p2"), Context())
    assert result == {
        "batchItemFailures": [
            {"itemIdentifier": "p1"},
            {"itemIdentifier": "p2"},
        ]
    }
    assert client.starts == []


def test_the_sync_client_retries_and_retrieve_does_not(monkeypatch):
    captured = {}

    class _Shape:
        members = {"managedSearchConfiguration": object()}

    class _Model:
        def shape_for(self, _name):
            return _Shape()

    class _Meta:
        service_model = _Model()

    class _Fake:
        def __init__(self):
            self.meta = _Meta()

    def fake(service, config=None, **_kwargs):
        captured[service] = config
        return _Fake()

    monkeypatch.setattr(health_kb.boto3, "client", fake)
    health_kb._client("bedrock-agent", sync=True)
    health_kb._client("bedrock-agent-runtime")
    sync = captured["bedrock-agent"].retries
    retrieve = captured["bedrock-agent-runtime"].retries
    assert sync["mode"] == "adaptive"
    assert sync["max_attempts"] >= 5
    assert retrieve["max_attempts"] == 0
    assert retrieve.get("mode", "legacy") != "adaptive"


def test_both_sync_paths_coalesce_on_a_queue():
    doc = yaml.load(TEMPLATE, Loader=CfnLoader)
    resources = doc["Resources"]
    # The buckets and the bases stay the resources they already are. A
    # queue is added beside them; renaming either bucket would replace it.
    assert resources["PortfolioBucket"]["Type"] == "AWS::S3::Bucket"
    assert "QueueName" not in resources["PortfolioBucket"].get("Properties", {})
    assert resources["PortfolioKnowledgeBase"]["Type"] == "AWS::CloudFormation::CustomResource"
    assert resources["HealthKnowledgeBase"]["Type"] == "AWS::CloudFormation::CustomResource"

    for prefix, condition in (
        ("Health", "HealthKnowledgeBaseEnabled"),
        ("Portfolio", None),
    ):
        queue = resources[f"{prefix}KbSyncQueue"]
        if condition:
            assert queue["Condition"] == condition
        assert "QueueName" not in queue.get("Properties", {})
        timeout = resources[f"{prefix}KbSyncFunction"]["Properties"]["Timeout"]
        # Six times the function timeout. A reserved-concurrency throttle
        # holds the batch until this expires, and a shorter window can make
        # the message visible while Lambda is still retrying the invocation.
        assert queue["Properties"]["VisibilityTimeout"] >= 6 * timeout
        assert queue["Properties"]["VisibilityTimeout"] == 720
        dlq_name = f"{prefix}KbSyncDlq"
        dlq = resources[dlq_name]
        if condition:
            assert dlq["Condition"] == condition
        assert "QueueName" not in dlq.get("Properties", {})
        assert dlq["Properties"]["MessageRetentionPeriod"] == 1209600
        assert dlq["Properties"]["SqsManagedSseEnabled"] is True
        redrive = queue["Properties"]["RedrivePolicy"]
        assert redrive["maxReceiveCount"] == 1000
        assert redrive["deadLetterTargetArn"] == {
            "__fn__": "GetAtt",
            "__arg__": f"{dlq_name}.Arn",
        }
        function = resources[f"{prefix}KbSyncFunction"]["Properties"]
        assert "FunctionName" not in function
        assert function["ReservedConcurrentExecutions"] == 1
        mapping = resources[f"{prefix}KbSyncMapping"]["Properties"]
        assert mapping["BatchSize"] == 100
        assert mapping["MaximumBatchingWindowInSeconds"] == 60
        assert mapping["FunctionResponseTypes"] == ["ReportBatchItemFailures"]
        # MaximumConcurrency's minimum is 2. Two pollers would both start
        # a job. ReservedConcurrentExecutions is the serialization.
        assert "ScalingConfig" not in mapping
        target = resources[f"{prefix}KbSyncRule"]["Properties"]["Targets"][0]["Arn"]
        assert target == {"__fn__": "GetAtt", "__arg__": f"{prefix}KbSyncQueue.Arn"}
        role = str(resources[f"{prefix}KbSyncRole"])
        assert "bedrock:ListIngestionJobs" in role
        assert "bedrock:StartIngestionJob" in role
        assert "sqs:ReceiveMessage" in role
        assert "sqs:DeleteMessage" in role
        assert "s3:GetObject" not in role
