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
from .tools.document_tools import parse_document, reconstruct_document

log = get_logger("tool_registry")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "parse_document": parse_document,
    "reconstruct_document": reconstruct_document,
}


def register_tool(name: str, fn: Callable[..., Any]) -> None:
    """Register a new tool at runtime (useful for tests and extensions)."""
    TOOL_REGISTRY[name] = fn


# ---------------------------------------------------------------------------
# Arg resolution
# ---------------------------------------------------------------------------


def _resolve_source_key(source_key: str, step_inputs: Dict[str, Any]) -> Any:
    """
    Resolve a dotted path like ``"request.document_text"`` against step_inputs.

    Examples:
      "request.document_text" → step_inputs["request"]["document_text"]
      "formatter"             → step_inputs["formatter"]
    """
    parts = source_key.split(".", 1)
    root = step_inputs.get(parts[0])
    if len(parts) == 1:
        return root
    if isinstance(root, dict):
        return root.get(parts[1])
    return None


def _build_tool_args(tool_cfg: Dict[str, Any], step_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the kwargs dict to pass to a tool function.

    If the tool config has a ``source_key``, the referenced value is
    resolved and passed as the first positional-equivalent kwarg named
    after the key's leaf segment.  All other keys in ``args`` are passed
    through unchanged.
    """
    raw_args: Dict[str, Any] = dict(tool_cfg.get("args") or {})
    source_key = raw_args.pop("source_key", None)

    resolved: Dict[str, Any] = {}
    if source_key:
        value = _resolve_source_key(source_key, step_inputs)
        # Derive a param name from the leaf of the dotted path
        param_name = source_key.split(".")[-1]
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
