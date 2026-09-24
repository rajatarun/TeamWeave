"""The financial team returns insights grounded in uploads and in the live web.

Modeled on health_insights_v1. A list of questions is not the deliverable, and
a guaranteed return is not an insight. The portfolio bucket is this stack's;
the web citations come from Gemini Google Search grounding.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema.exceptions import ValidationError

from src.orchestrator.models import AgentConfig, BedrockRef, TeamConfig, TeamGlobals
from src.orchestrator.prompt_builder import build_prompt
from src.orchestrator.schema_validate import (
    FINANCIAL_INSIGHTS,
    settle_health_insights,
    validate_output,
)
from src.orchestrator.tool_registry import execute_tool

REPO = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (REPO / "config/examples/schemas/financial_insights_v1.json").read_text())
RECORD = json.loads(
    (REPO / "config/examples/schemas/financial_record_v1.json").read_text())
TEAM = json.loads(
    (REPO / "config/examples/teams/financial_advisors/v1/team.json").read_text())
TEMPLATE = (REPO / "infra" / "template.yaml").read_text()
WORKFLOW = (REPO / ".github" / "workflows" / "deploy.yml").read_text()

VALID = {
    "summary": "The March statement lists 40 percent of the account in one fund, and the fund's latest quote is below that statement's price.",
    "safety_note": "This is not personalized financial advice. Do your own research. This is not a guaranteed return.",
    "insights": [{
        "title": "One fund is 40 percent of the March statement",
        "finding": "The statement lists the fund at 40 percent of the account, and the cited page quotes it below the statement price.",
        "evidence": [
            {
                "source": "s3://stack-portfolios/march.csv",
                "date": "2026-03-31",
                "value": "40 percent of the account",
                "kind": "portfolio",
            },
            {
                "source": "https://example.com/quote",
                "date": "2026-09-24",
                "value": "quoted below the statement price",
                "kind": "web",
            },
        ],
        "significance": "high",
        "confidence": "medium",
    }],
    "suggestions": [{
        "action": "Compare that fund's weight with the allocation in the previous statement.",
        "rationale": "A single statement does not show whether 40 percent is new.",
        "linked_insight": "One fund is 40 percent of the March statement",
        "priority": "high",
        "timeframe": "this week",
    }],
    "data_gaps": ["No cost-basis excerpt was returned."],
    "follow_ups": [],
}


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


def _team():
    globals_ = TEAM["globals"]
    return TeamConfig(
        team=TEAM["team"],
        globals=TeamGlobals(
            north_star=globals_["north_star"],
            default_channel=globals_["default_channel"],
            hard_constraints=globals_["hard_constraints"],
            features=globals_["features"],
            rag=globals_["rag"],
            artifact_store=globals_["artifact_store"],
            revision=globals_["revision"],
        ),
        agents=[], workflow=[], schemas={},
    )


def test_the_schema_requires_insights_suggestions_and_a_short_safety_note():
    assert SCHEMA["title"] == FINANCIAL_INSIGHTS
    for field in ("summary", "safety_note", "insights", "suggestions", "data_gaps"):
        assert field in SCHEMA["required"], field
    assert "follow_ups" not in SCHEMA["required"]
    assert SCHEMA["properties"]["insights"]["minItems"] == 1
    assert SCHEMA["properties"]["suggestions"]["minItems"] == 1
    assert SCHEMA["properties"]["follow_ups"]["maxItems"] == 2
    assert SCHEMA["properties"]["safety_note"]["maxLength"] == 240
    assert "questions_to_ask" not in SCHEMA["properties"]
    evidence = SCHEMA["properties"]["insights"]["items"]["properties"]["evidence"]
    assert evidence["minItems"] == 1
    item = evidence["items"]
    assert set(item["required"]) == {"source", "date", "value", "kind"}
    assert item["properties"]["kind"]["enum"] == ["portfolio", "web"]
    validate_output(VALID, SCHEMA)


def test_a_question_list_fails_validation():
    asked = json.loads(json.dumps(VALID))
    asked["summary"] = "What should I do with this account?"
    asked["insights"][0]["title"] = "Is this too concentrated?"
    asked["insights"][0]["finding"] = "Should this weight be lower?"
    asked["suggestions"][0]["action"] = "What would a better allocation be?"
    asked["suggestions"][0]["rationale"] = "Is 40 percent a problem?"
    with pytest.raises(ValidationError, match="mostly questions"):
        validate_output(asked, SCHEMA)


def test_a_question_in_data_gaps_fails_validation():
    asked = json.loads(json.dumps(VALID))
    asked["data_gaps"] = ["What was the cost basis?"]
    with pytest.raises(ValidationError, match="mostly questions"):
        validate_output(asked, SCHEMA)


def test_a_guaranteed_return_fails_and_the_disclaimer_does_not():
    claimed = json.loads(json.dumps(VALID))
    claimed["insights"][0]["finding"] = "This rebalance has a guaranteed return of 8 percent."
    with pytest.raises(ValidationError, match="guaranteed return"):
        validate_output(claimed, SCHEMA)
    # VALID already says "not a guaranteed return" and must still validate.
    validate_output(VALID, SCHEMA)
    denied = json.loads(json.dumps(VALID))
    denied["suggestions"][0]["rationale"] = "There is no guaranteed profit in a single statement."
    validate_output(denied, SCHEMA)


def test_a_question_shaped_answer_is_repaired_once():
    repaired = json.loads(json.dumps(VALID))
    calls = []

    def transform(payload, schema, **_kwargs):
        calls.append(payload)
        return repaired

    out, accepted = settle_health_insights(
        {"questions_to_ask": ["Should I sell?"]}, SCHEMA, transform)
    assert accepted is True
    assert out == repaired
    assert len(calls) == 1


def test_a_repair_that_still_promises_a_return_is_not_accepted():
    claimed = json.loads(json.dumps(VALID))
    claimed["summary"] = "A guaranteed profit follows from this allocation."

    def transform(payload, schema, **_kwargs):
        return claimed

    _out, accepted = settle_health_insights(
        {"questions_to_ask": ["Should I sell?"]}, SCHEMA, transform)
    assert accepted is False


def test_prompts_include_portfolio_and_web_grounding():
    team = _team()
    for agent_doc in TEAM["agents"]:
        agent = AgentConfig(
            id=agent_doc["id"],
            name=agent_doc["name"],
            bedrock=BedrockRef(agentId="", aliasId=""),
            goal_template=agent_doc["goal_template"],
            schema_ref=agent_doc["schema_ref"],
        )
        prompt = build_prompt(
            team, agent, {"request": {"focus": "the largest holding"}}, {}, "", "", "")
        for phrase in (
            "knowledge base", "sourceKey", "url", "retrieved_on",
            "found false", "error", "DO THE WORK", "not the answer",
        ):
            assert phrase in prompt, phrase
        lowered = agent_doc["goal_template"].lower()
        assert "questions is not" in lowered
        assert "do not echo" in lowered
        for facet in ("holdings", "allocation", "cost basis", "performance", "risk"):
            assert facet in lowered
    constraints = " ".join(TEAM["globals"]["hard_constraints"]).lower()
    assert "not personalized financial advice" in constraints
    assert "do your own research" in constraints
    assert "guaranteed return" in constraints
    assert "list of questions is not the answer" in constraints
    assert "plan" in constraints and "explore" in constraints


def test_both_steps_retrieve_the_portfolio_and_the_web_before_writing():
    assert {a["schema_ref"] for a in TEAM["agents"]} == {
        "financial_record_v1", "financial_insights_v1",
    }
    assert RECORD["title"] == "financial_record_v1"
    for step in TEAM["workflow"]:
        names = [tool["name"] for tool in step["pre_tools"]]
        assert names == ["query_portfolio", "web_search"]
        for tool in step["pre_tools"]:
            assert tool["args"]["source_key"] == "request.focus"
            assert tool["args"]["facets"] is True


def test_portfolio_facets_cover_holdings_allocation_cost_performance_and_risk(monkeypatch):
    from src.orchestrator import portfolio_kb
    from src.orchestrator.tools import weave_tools

    monkeypatch.setenv("PORTFOLIO_KNOWLEDGE_BASE_ID", "KBPORT")
    seen = []

    def retrieve(question, *, top_k=6, client=None):
        seen.append((question, top_k))
        if question.startswith("Holdings"):
            return {
                "found": True,
                "source": "bedrock-knowledge-base",
                "excerpts": [{
                    "sourceKey": "s3://stack-portfolios/march.csv",
                    "content": "FUND 40 percent on 2026-03-31",
                }],
            }
        return {"found": False, "excerpts": [], "source": "bedrock-knowledge-base"}

    monkeypatch.setattr(portfolio_kb, "retrieve", retrieve)
    result = weave_tools.query_portfolio("largest holding", top_k=4, facets=True)
    assert [q["facet"] for q in result["queries"]] == [
        "holdings", "allocation", "cost_basis", "performance", "risk"]
    assert len(seen) == 5
    assert all(top_k == 4 for _q, top_k in seen)
    assert result["found"] is True
    assert result["excerpts"] == [{
        "sourceKey": "s3://stack-portfolios/march.csv",
        "content": "FUND 40 percent on 2026-03-31",
    }]
    assert "used_for" in result


def test_an_unconfigured_portfolio_base_is_one_error_not_five_searches(monkeypatch):
    from src.orchestrator import portfolio_kb
    from src.orchestrator.tools import weave_tools

    monkeypatch.delenv("PORTFOLIO_KNOWLEDGE_BASE_ID", raising=False)
    monkeypatch.setattr(
        portfolio_kb, "retrieve",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrieved with no base")),
    )
    result = weave_tools.query_portfolio("largest holding", facets=True)
    assert "error" in result
    assert "queries" not in result
    assert "found" not in result


def test_every_portfolio_facet_failing_is_not_an_empty_portfolio(monkeypatch):
    from src.orchestrator import portfolio_kb
    from src.orchestrator.tools import weave_tools

    monkeypatch.setenv("PORTFOLIO_KNOWLEDGE_BASE_ID", "KBPORT")
    monkeypatch.setattr(
        portfolio_kb, "retrieve",
        lambda *a, **k: {"error": "the portfolio knowledge base could not be reached"},
    )
    result = weave_tools.query_portfolio("largest holding", facets=True)
    assert "error" in result
    assert "found" not in result
    assert len(result["searched"]) == 5


def test_empty_portfolio_facets_say_what_was_searched(monkeypatch):
    from src.orchestrator import portfolio_kb
    from src.orchestrator.tools import weave_tools

    monkeypatch.setenv("PORTFOLIO_KNOWLEDGE_BASE_ID", "KBPORT")
    monkeypatch.setattr(
        portfolio_kb, "retrieve",
        lambda *a, **k: {"found": False, "excerpts": [], "source": "bedrock-knowledge-base"},
    )
    result = weave_tools.query_portfolio("largest holding", facets=True)
    assert result["found"] is False
    assert "error" not in result
    assert any("Cost basis" in q for q in result["searched"])


def test_web_search_cites_the_url_and_the_retrieval_date(monkeypatch):
    from src.orchestrator import web_search
    from src.orchestrator.tools import weave_tools

    monkeypatch.setattr(web_search, "api_key", lambda: "test-key")
    captured = {}

    def post(url, body, key):
        captured["url"] = url
        captured["body"] = body
        captured["key"] = key
        return {
            "candidates": [{
                "groundingMetadata": {
                    "groundingChunks": [{
                        "web": {"uri": "https://example.com/rates", "title": "Rates"},
                    }],
                    "groundingSupports": [{
                        "segment": {"text": "The 10-year yield is 4.1 percent."},
                        "groundingChunkIndices": [0],
                    }],
                },
            }],
        }

    monkeypatch.setattr(web_search, "_post", post)
    result = weave_tools.web_search("treasury yields", facets=True)
    assert result["found"] is True
    assert result["retrieved_on"]
    assert result["results"][0]["url"] == "https://example.com/rates"
    assert result["results"][0]["retrieved_on"] == result["retrieved_on"]
    assert "4.1" in result["results"][0]["snippet"]
    assert captured["body"]["tools"] == [{"google_search": {}}]
    assert captured["key"] == "test-key"
    assert "test-key" not in json.dumps({k: v for k, v in result.items() if k != "used_for"})
    assert [q["facet"] for q in result["queries"]] == ["request", "markets", "news", "rates"]


def test_web_search_without_a_url_is_not_a_hit(monkeypatch):
    from src.orchestrator import web_search

    monkeypatch.setattr(web_search, "api_key", lambda: "test-key")
    monkeypatch.setattr(web_search, "_post", lambda *a, **k: {"candidates": [{}]})
    result = web_search.search("treasury yields", facets=False, retrieved_on="2026-09-24")
    assert result["found"] is False
    assert result["results"] == []
    assert result["retrieved_on"] == "2026-09-24"
    assert "error" not in result


def test_a_missing_web_search_secret_is_an_error_and_does_not_call_out(monkeypatch):
    from src.orchestrator import web_search

    monkeypatch.delenv("WEB_SEARCH_SECRET_ARN", raising=False)
    monkeypatch.delenv("GEMINI_SECRET_ARN", raising=False)
    monkeypatch.setattr(
        web_search, "_post",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("called without a key")),
    )
    result = web_search.search("treasury yields", facets=True)
    assert "GEMINI_SECRET_ARN" in result["error"]
    assert "GeminiSecretArn" in result["error"]
    assert "queries" not in result


def test_an_unknown_web_search_provider_is_an_error(monkeypatch):
    from src.orchestrator import web_search

    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "bing")
    result = web_search.search("treasury yields", facets=True)
    assert "bing" in result["error"]
    assert "queries" not in result


def test_only_the_financial_team_may_read_the_portfolio_or_search(monkeypatch):
    from src.orchestrator import tool_registry

    monkeypatch.setitem(
        tool_registry.TOOL_REGISTRY, "query_portfolio",
        lambda **_k: {"found": False})
    monkeypatch.setitem(
        tool_registry.TOOL_REGISTRY, "web_search",
        lambda **_k: {"found": False})
    for name in ("query_portfolio", "web_search"):
        with pytest.raises(PermissionError) as raised:
            execute_tool(name, {"question": "x"}, team="health_prep", caller_token="tok")
        assert "financial_advisors" in str(raised.value)
        assert execute_tool(
            name, {"question": "x", "caller_token": "tok"},
            team="financial_advisors", caller_token="tok",
        ) == {"found": False}


def test_the_portfolio_tool_does_not_receive_the_caller_token(monkeypatch):
    from src.orchestrator import tool_registry

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return {"found": False}

    monkeypatch.setitem(tool_registry.TOOL_REGISTRY, "query_portfolio", fake)
    execute_tool(
        "query_portfolio",
        {"question": "x", "caller_token": "should-not-arrive"},
        team="financial_advisors",
        caller_token="should-not-arrive",
    )
    assert "caller_token" not in seen
    assert seen["question"] == "x"


def test_portfolio_sync_uses_the_portfolio_ids(monkeypatch):
    from src.orchestrator import health_kb, portfolio_kb

    seen = {}

    def start_sync(*, client=None, kb_id=None, source_id=None):
        seen.update(kb_id=kb_id, source_id=source_id)
        return {"kbSync": "started"}

    monkeypatch.setattr(health_kb, "start_sync", start_sync)
    monkeypatch.setenv("PORTFOLIO_KNOWLEDGE_BASE_ID", "KBPORT")
    monkeypatch.setenv("PORTFOLIO_DATA_SOURCE_ID", "DSPORT")
    monkeypatch.setenv("HEALTH_KNOWLEDGE_BASE_ID", "KBHEALTH")
    monkeypatch.setenv("HEALTH_DATA_SOURCE_ID", "DSHEALTH")

    class Context:
        def get_remaining_time_in_millis(self):
            return 60_000

    result = portfolio_kb.sync_handler(
        {"detail": {"object": {"key": "statement.pdf"}}}, Context())
    assert result == {"kbSync": "started"}
    assert seen == {"kb_id": "KBPORT", "source_id": "DSPORT"}


def test_the_data_source_name_defaults_to_health_and_the_portfolio_sets_its_own():
    from src.orchestrator.health_kb_provision import _source_name, data_source_fields

    assert _source_name({}) == "health-s3"
    assert _source_name({"DataSourceName": "portfolio-s3"}) == "portfolio-s3"
    fields = data_source_fields(
        "stack-portfolios", "239571291755",
        name="portfolio-s3", description="Statements uploaded here.",
    )
    assert fields["name"] == "portfolio-s3"
    assert "Statements" in fields["description"]
    health = data_source_fields("health-docs", "239571291755")
    assert health["name"] == "health-s3"


def test_the_template_provisions_the_portfolio_bucket_and_the_managed_base():
    from src.orchestrator.health_kb_provision import EMBEDDING_DIMENSIONS, knowledge_base_configuration

    doc = yaml.load(TEMPLATE, Loader=CfnLoader)
    resources = doc["Resources"]
    bucket = resources["PortfolioBucket"]
    assert "Condition" not in bucket
    props = bucket["Properties"]
    name = str(props["BucketName"])
    assert "AWS::StackName" in name and "portfolios" in name
    assert props["VersioningConfiguration"]["Status"] == "Enabled"
    block = props["PublicAccessBlockConfiguration"]
    assert block["BlockPublicAcls"] is True
    assert block["BlockPublicPolicy"] is True
    assert block["IgnorePublicAcls"] is True
    assert block["RestrictPublicBuckets"] is True
    algorithm = props["BucketEncryption"]["ServerSideEncryptionConfiguration"][0][
        "ServerSideEncryptionByDefault"]["SSEAlgorithm"]
    assert algorithm == "AES256"
    assert props["NotificationConfiguration"]["EventBridgeConfiguration"]["EventBridgeEnabled"] is True

    media = resources["PortfolioMultimodalBucket"]["Properties"]
    assert "portfolio-mm" in str(media["BucketName"])
    assert "AWS::StackName" in str(media["BucketName"])
    assert media["BucketEncryption"]["ServerSideEncryptionConfiguration"][0][
        "ServerSideEncryptionByDefault"]["SSEAlgorithm"] == "AES256"

    kb = resources["PortfolioKnowledgeBase"]
    assert "Condition" not in kb
    assert kb["Type"] == "AWS::CloudFormation::CustomResource"
    kb_props = kb["Properties"]
    assert "amazon.nova-2-multimodal-embeddings-v1:0" in str(kb_props["EmbeddingModelArn"])
    assert kb_props["DataSourceName"] == "portfolio-s3"
    assert kb_props["MultimodalBucket"] == {"__fn__": "Ref", "__arg__": "PortfolioMultimodalBucket"}
    assert kb_props["DocsBucketName"] == {"__fn__": "Ref", "__arg__": "PortfolioBucket"}
    assert "PortfolioMultimodalBucket" not in str(kb_props["DocsBucketName"])

    body = knowledge_base_configuration(
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-multimodal-embeddings-v1:0",
        "stack-portfolio-mm",
    )
    managed = body["managedKnowledgeBaseConfiguration"]
    assert managed["embeddingModelType"] == "CUSTOM"
    bedrock = managed["embeddingModelConfiguration"]["bedrockEmbeddingModelConfiguration"]
    assert bedrock["dimensions"] == 1024
    assert EMBEDDING_DIMENSIONS == 1024
    assert bedrock["embeddingDataType"] == "FLOAT32"
    assert managed["supplementalDataStorageConfiguration"]["storageLocations"][0]["s3Location"]["uri"] == (
        "s3://stack-portfolio-mm/"
    )
    assert "modelConfiguration" not in managed

    provision = resources["PortfolioKbProvisionFunction"]
    assert provision["Properties"]["Handler"] == "src/orchestrator/health_kb_provision.handler"
    assert provision["Properties"]["Timeout"] == 900
    assert "FunctionName" not in provision["Properties"]
    assert "VpcConfig" not in provision["Properties"]
    assert provision["Properties"]["LoggingConfig"]["LogGroup"] == {
        "__fn__": "Ref", "__arg__": "PortfolioKbProvisionLogGroup",
    }
    assert "logs:CreateLogGroup" not in str(resources["PortfolioKbProvisionRole"])

    sync = resources["PortfolioKbSyncFunction"]["Properties"]
    assert sync["Handler"] == "src/orchestrator/portfolio_kb.sync_handler"
    assert "FunctionName" not in sync
    variables = sync["Environment"]["Variables"]
    assert "PortfolioKnowledgeBase" in str(variables["PORTFOLIO_KNOWLEDGE_BASE_ID"])
    assert "DataSourceId" in str(variables["PORTFOLIO_DATA_SOURCE_ID"])
    sync_role = str(resources["PortfolioKbSyncRole"])
    assert "bedrock:StartIngestionJob" in sync_role
    assert "s3:GetObject" not in sync_role

    worker = resources["WorkerFunction"]["Properties"]["Environment"]["Variables"]
    assert "PortfolioKnowledgeBase" in str(worker["PORTFOLIO_KNOWLEDGE_BASE_ID"])
    assert worker["WEB_SEARCH_PROVIDER"] == "gemini"
    assert "GeminiSecretArn" in str(worker["GEMINI_SECRET_ARN"])
    role = str(resources["WorkerRole"])
    assert "PortfolioKnowledgeBase.KnowledgeBaseArn" in role
    assert "bedrock:Retrieve" in role

    kb_role = resources["PortfolioKnowledgeBaseRole"]
    statements = kb_role["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    wrote_statements = False
    for statement in statements:
        actions = statement["Action"]
        actions = actions if isinstance(actions, list) else [actions]
        resource = str(statement["Resource"])
        if "s3:PutObject" in actions:
            assert "PortfolioMultimodalBucket" in resource
            assert "PortfolioBucket" not in resource.replace("PortfolioMultimodalBucket", "")
        if "PortfolioBucket" in resource and "s3:PutObject" in actions:
            wrote_statements = True
    assert not wrote_statements
    assert "amazon.nova-2-multimodal-embeddings-v1:0" in str(kb_role)

    pattern = resources["PortfolioKbSyncRule"]["Properties"]["EventPattern"]
    assert pattern["source"] == ["aws.s3"]
    assert "Object Created" in pattern["detail-type"]
    assert "PortfolioBucket" in str(pattern["detail"])

    assert doc["Outputs"]["PortfolioBucketName"]["Value"]["__fn__"] == "Ref"
    assert "PortfolioKnowledgeBase" in str(doc["Outputs"]["PortfolioKnowledgeBaseId"]["Value"])
    assert "DataSourceId" in str(doc["Outputs"]["PortfolioDataSourceId"]["Value"])
    assert "PortfolioKnowledgeBaseId" in WORKFLOW
    assert "PortfolioDataSourceId" in WORKFLOW
    assert "start-ingestion-job" in WORKFLOW

    runtime = resources["AgentCoreRuntimeFinancialAdvisors"]["Properties"]
    assert runtime["AgentRuntimeName"] == "teamweave_financial_advisors"
    assert runtime["EnvironmentVariables"]["AGENT_TEAM"] == "financial_advisors"
