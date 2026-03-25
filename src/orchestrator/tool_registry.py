"""
tool_registry.py
~~~~~~~~~~~~~~~~
Central registry that maps tool names to Python callables and provides
helpers for executing pre/post tools declared in workflow step configs.

Team config usage
-----------------
Workflow steps can declare tools that run before (pre_tools) or after
(post_tools) the Bedrock agent invocation:

  workflow:
    - step: analyzer
      pre_tools:
        - name: parse_document
          args:
            source_key: request.document_text   # dotted path into step_inputs
    - step: formatter
      post_tools:
        - name: reconstruct_document
          args:
            source_key: formatter               # entire prior-step output dict

Pre-tool results are injected into step_inputs under:
  step_inputs["tool_results"][<tool_name>]

Post-tool results are merged (dict.update) into out_json so the agent
output is enriched before artifact persistence.
"""

from typing import Any, Callable, Dict, List

from .logger import get_logger
from .tools.content_tools import (
    analyse_draft_quality,
    compute_optimal_post_time,
    extract_topic_keywords,
    format_approval_decision,
    format_distribution_checklist,
    measure_post_quality,
)
from .tools.document_tools import parse_document, reconstruct_document

log = get_logger("tool_registry")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    # Document rewrite tools
    "parse_document": parse_document,
    "reconstruct_document": reconstruct_document,
    # Visibility team content tools
    "extract_topic_keywords": extract_topic_keywords,
    "analyse_draft_quality": analyse_draft_quality,
    "compute_optimal_post_time": compute_optimal_post_time,
    "measure_post_quality": measure_post_quality,
    "format_distribution_checklist": format_distribution_checklist,
    "format_approval_decision": format_approval_decision,
}


def register_tool(name: str, fn: Callable[..., Any]) -> None:
    """Register a new tool at runtime (useful for tests and extensions)."""
    TOOL_REGISTRY[name] = fn


# ---------------------------------------------------------------------------
# Arg resolution
# ---------------------------------------------------------------------------


def _resolve_source_key(source_key: str, step_inputs: Dict[str, Any]) -> Any:
    """
    Resolve a dotted path against step_inputs, traversing arbitrarily deep.

    Examples:
      "request.document_text"                           → step_inputs["request"]["document_text"]
      "formatter"                                        → step_inputs["formatter"]
      "TVT_DEPT-003_PBM-006_writer_linkedin.drafts"    → step_inputs[agent_id]["drafts"]
      "request.topic"                                    → step_inputs["request"]["topic"]

    Note: step IDs may contain hyphens but never dots, so splitting on "."
    correctly separates step IDs from field names at any depth.
    """
    parts = source_key.split(".")
    val: Any = step_inputs
    for part in parts:
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return None
    return val


def _build_tool_args(tool_cfg: Dict[str, Any], step_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the kwargs dict to pass to a tool function.

    source_key (dotted path):
        Resolved against step_inputs; the resulting value is passed under
        a parameter name derived from the *leaf* segment of the dotted path
        (e.g. ``"request.topic"`` → ``topic=<value>``).

    output_key (optional override):
        Overrides the auto-derived parameter name.  Useful when the step ID
        (last path segment) is too verbose to be a clean function parameter
        (e.g. ``source_key: "TVT_DEPT-001_PBM-001A_director_approve"`` with
        ``output_key: "approval"`` → ``approval=<dict>``).

    All other keys in ``args`` are forwarded unchanged.
    """
    raw_args: Dict[str, Any] = dict(tool_cfg.get("args") or {})
    source_key = raw_args.pop("source_key", None)
    output_key = raw_args.pop("output_key", None)

    resolved: Dict[str, Any] = {}
    if source_key:
        value = _resolve_source_key(source_key, step_inputs)
        # output_key takes precedence; fall back to last segment of dotted path
        param_name = output_key or source_key.split(".")[-1]
        resolved[param_name] = value

    resolved.update(raw_args)
    return resolved


# ---------------------------------------------------------------------------
# Public execution helpers
# ---------------------------------------------------------------------------


def execute_tool(name: str, args: Dict[str, Any]) -> Any:
    """
    Look up *name* in the registry and call it with *args* as kwargs.

    Raises KeyError if the tool is not registered.
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' is not registered. Available: {list(TOOL_REGISTRY)}")
    fn = TOOL_REGISTRY[name]
    log.info("tool_execute name=%s args_keys=%s", name, list(args))
    return fn(**args)


def execute_pre_tools(step_def: Dict[str, Any], step_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run every tool listed under ``step_def["pre_tools"]``.

    Results are injected into ``step_inputs["tool_results"][<tool_name>]``
    so the Bedrock agent can see them in ``STEP_INPUTS_JSON``.

    Returns the (mutated) step_inputs dict.
    """
    pre_tools: List[Dict[str, Any]] = step_def.get("pre_tools") or []
    if not pre_tools:
        return step_inputs

    tool_results: Dict[str, Any] = step_inputs.setdefault("tool_results", {})

    for tool_cfg in pre_tools:
        name = tool_cfg.get("name", "")
        try:
            args = _build_tool_args(tool_cfg, step_inputs)
            result = execute_tool(name, args)
            tool_results[name] = result
            log.info("pre_tool_succeeded name=%s", name)
        except Exception:
            log.exception("pre_tool_failed name=%s — skipping", name)

    return step_inputs


def execute_post_tools(
    step_def: Dict[str, Any],
    out_json: Dict[str, Any],
    step_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run every tool listed under ``step_def["post_tools"]``.

    Each tool receives the agent's output (``out_json``) merged with any
    ``args`` from the config.  The tool's return dict is merged back into
    ``out_json`` so downstream steps see the enriched output.

    Returns the (mutated) out_json dict.
    """
    post_tools: List[Dict[str, Any]] = step_def.get("post_tools") or []
    if not post_tools:
        return out_json

    # Make the current step output available to arg resolution
    step_id = step_def.get("step", "")
    if step_id:
        step_inputs = {**step_inputs, step_id: out_json}

    for tool_cfg in post_tools:
        name = tool_cfg.get("name", "")
        try:
            args = _build_tool_args(tool_cfg, step_inputs)
            # For reconstruct_document the key arg is "sections"
            # If caller used source_key pointing to a step output dict,
            # auto-extract the "sections" list from it.
            for param, val in list(args.items()):
                if isinstance(val, dict) and "sections" in val and param != "sections":
                    args["sections"] = val["sections"]
                    del args[param]
            result = execute_tool(name, args)
            if isinstance(result, dict):
                out_json.update(result)
            log.info("post_tool_succeeded name=%s", name)
        except Exception:
            log.exception("post_tool_failed name=%s — skipping", name)

    return out_json
