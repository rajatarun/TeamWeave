"""When a pipeline step may use a weave sibling's tool -- and when it may not.

Each rule names one MCP tool on one sibling, what calling it *does*, and the
condition under which a step should reach for it. The `effect` column is the
one with teeth.

**Read, propose, commit.** Two siblings deliberately split their write path in
half: DataDictionary has `propose_data_element` -> `commit_data_element`, and
ToolWeave has `pre_tool`/`post_tool` -> `commit_api_call`, each returning a
token the second call requires. That design exists so a machine cannot mutate
a catalogue or fire a real REST call on its own reading of a prompt. A pipeline
step running unattended is exactly the caller it is protecting against, so
`effect: "commit"` tools are **refused at execution**, by name, rather than
left to a convention nobody enforces. The refusal is a hard failure and not a
degrade: silently skipping a commit would let a run report success for work it
never did.

`propose` is allowed, and is the useful half: a run can draft a catalogue entry
or plan an API call and hand a person the token, which is the two-phase design
working rather than being worked around.

This table is also what a team config is checked against. A step may only
declare a tool that has a rule, so an agent cannot be handed a capability
nobody wrote down a reason for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

READ, PROPOSE, COMMIT = "read", "propose", "commit"
MCP, HTTP = "mcp", "http"

# Siblings reached over their own HTTP API rather than over MCP, and the
# variable that seeds the address. `mcp_client.SIBLING_ENV` is the MCP half;
# a rule must name a sibling in whichever map matches its transport, or it
# points at a service nothing can supply an address for.
HTTP_SIBLING_ENV = {"contextweave": "CONTEXTWEAVE_URL"}


@dataclass(frozen=True)
class ToolRule:
    tool: str          # the name a team.json step declares
    sibling: str
    mcp_tool: str      # the operation as that sibling names it
    effect: str        # read | propose | commit
    use_when: str
    never_when: str
    transport: str = MCP
    # Empty means any team may declare it. A non-empty tuple is enforced in
    # `execute_tool` against the team the *worker* passes down, not against
    # anything in the config -- team configs live in S3 and are edited without
    # a deploy, so a restriction expressed only there is not a restriction.
    only_teams: tuple = ()
    # The tool is called as the person who started the run, with the bearer
    # token they presented. There is no service credential to fall back to:
    # see `query_health_record` below for why that is the whole design.
    needs_caller_token: bool = False


RULES: Dict[str, ToolRule] = {rule.tool: rule for rule in [
    # ── DataDictionary: the platform's term and schema catalogue ────────────
    ToolRule(
        tool="lookup_data_element",
        sibling="datadictionary", mcp_tool="get_data_element", effect=READ,
        use_when="a step names a specific data element and needs its agreed "
                 "definition rather than the agent's guess at one",
        never_when="the element name is itself what the step is trying to "
                   "decide -- search first, then look up what you found",
    ),
    ToolRule(
        tool="search_data_elements",
        sibling="datadictionary", mcp_tool="search_data_elements", effect=READ,
        use_when="a step has prose and needs to know whether the platform "
                 "already has a term for it, before inventing a second one",
        never_when="the exact element name is already known -- that is a lookup",
    ),
    ToolRule(
        tool="data_elements_for_context",
        sibling="datadictionary", mcp_tool="get_elements_by_context", effect=READ,
        use_when="a step works on a whole domain and needs its vocabulary at "
                 "once, rather than one term at a time",
        never_when="a single term would answer it -- this returns a list and "
                   "costs prompt budget every later step pays for",
    ),
    ToolRule(
        tool="propose_data_element",
        sibling="datadictionary", mcp_tool="propose_data_element", effect=PROPOSE,
        use_when="search found nothing and the run should hand a person a "
                 "drafted entry plus its commit token",
        never_when="search was not run first -- proposing a duplicate is worse "
                   "than proposing nothing",
    ),
    ToolRule(
        tool="commit_data_element",
        sibling="datadictionary", mcp_tool="commit_data_element", effect=COMMIT,
        use_when="never from a pipeline step; a person holds the token",
        never_when="always -- refused at execution",
    ),

    # ── CipherWeave: how a given piece of data should be protected ──────────
    ToolRule(
        tool="encryption_strategy",
        sibling="cipherweave", mcp_tool="get_encryption_strategy", effect=READ,
        use_when="a step is deciding how to store or move something and the "
                 "answer must be the platform's standing policy, not a model's "
                 "recollection of one",
        never_when="the step produces no design or storage decision -- a "
                   "security answer nobody acts on is noise in the prompt",
    ),

    # ── ScreenWeave: what a page actually is, rather than what it claims ────
    ToolRule(
        tool="crawl_site",
        sibling="screenweave", mcp_tool="crawl_url", effect=READ,
        use_when="a step reasons about a live page and needs its real content, "
                 "not the model's memory of that domain",
        never_when="the URL is not public, or the step only needs a page it was "
                   "already given the text of",
    ),
    ToolRule(
        tool="site_metrics",
        sibling="screenweave", mcp_tool="get_metrics", effect=READ,
        use_when="a finding has to be quantified -- load, size, counts -- and a "
                 "crawl for the same session already exists",
        never_when="no crawl has run: this reads a session, it does not make one",
    ),

    # ── ToolWeave: natural language to a safe REST call ─────────────────────
    ToolRule(
        tool="plan_api_call",
        sibling="toolweave", mcp_tool="pre_tool", effect=PROPOSE,
        use_when="a step must turn a request in prose into a concrete, "
                 "reviewable API call against a catalogued spec",
        never_when="the endpoint and payload are already known -- planning them "
                   "again invites a second, different answer",
    ),
    ToolRule(
        tool="commit_api_call",
        sibling="toolweave", mcp_tool="commit_api_call", effect=COMMIT,
        use_when="never from a pipeline step; a person holds the token",
        never_when="always -- refused at execution",
    ),

    # ── ContextWeave: the person's own health record ────────────────────────
    #
    # The only tool here that reads something about a *person* rather than
    # about the platform, and the two extra fields are why it can exist at all.
    #
    # ContextWeave keeps medical records in their own database, behind their
    # own role, behind the only authorizer on that API -- precisely so that a
    # content team asking about "work under pressure" cannot retrieve a chunk
    # of a discharge summary. Wiring a tool to it puts that back at risk in two
    # ways, and each has its own barrier:
    #
    #   `needs_caller_token`  There is no service credential. The call is made
    #       with the bearer token the person presented when they started the
    #       run, so the identity ContextWeave authorises is the identity whose
    #       record it is. A token in the worker's environment would mean *any*
    #       run could read the record, which is ambient authority over exactly
    #       the data that should have none.
    #
    #   `only_teams`  A team config is JSON in S3, edited with no deploy and no
    #       review. Adding this tool to `linkedin_quick_post` there would
    #       otherwise be enough to put a medical record into a draft post. The
    #       allowlist is checked in code against the team the worker is
    #       running, so both barriers have to be removed, and only one of them
    #       is reachable without a commit.
    ToolRule(
        tool="query_health_record",
        sibling="contextweave", mcp_tool="POST /health/query", effect=READ,
        transport=HTTP,
        only_teams=("health_prep",),
        needs_caller_token=True,
        use_when="a health step needs what the person's own uploaded records "
                 "actually say -- a previous result, a date, a prescribed "
                 "change -- rather than the model's guess at their history",
        never_when="any team that is not preparing this person for care; and "
                   "never to answer a clinical question, which the record "
                   "cannot settle and this platform must not attempt",
    ),
]}


def rule_for(tool: str) -> ToolRule:
    if tool not in RULES:
        raise KeyError(
            f"'{tool}' has no registry rule. Every sibling tool a step may use "
            f"is declared in tool_rules.RULES with the condition it is used "
            f"under. Known: {sorted(RULES)}"
        )
    return RULES[tool]


def is_refused(tool: str) -> bool:
    """True for a tool a pipeline step must never call on its own."""
    return tool in RULES and RULES[tool].effect == COMMIT


def tools_by_effect(effect: str) -> List[str]:
    return sorted(name for name, rule in RULES.items() if rule.effect == effect)


def refusal_message(tool: str) -> str:
    rule = RULES[tool]
    return (
        f"'{tool}' maps to {rule.sibling}.{rule.mcp_tool}, which commits. "
        f"{rule.sibling} splits its write path into propose then commit so that "
        f"an unattended caller cannot mutate state on its own reading of a "
        f"prompt -- a pipeline step is that caller. Use "
        f"{', '.join(tools_by_effect(PROPOSE)) or 'the propose half'} and hand "
        f"the token to a person."
    )


def requires_caller_token(tool: str) -> bool:
    """True for a tool that must be called as the person, or not at all."""
    return tool in RULES and RULES[tool].needs_caller_token


def team_allowed(tool: str, team: str) -> bool:
    rule = RULES.get(tool)
    if rule is None or not rule.only_teams:
        return True
    return team in rule.only_teams


def team_refusal_message(tool: str, team: str) -> str:
    rule = RULES[tool]
    return (
        f"'{tool}' reads {rule.sibling}.{rule.mcp_tool} and is restricted to "
        f"{', '.join(rule.only_teams)}; this run is team '{team or '<unset>'}'. "
        f"The restriction is in code rather than in team.json because team "
        f"configs are edited in S3 without a deploy."
    )
