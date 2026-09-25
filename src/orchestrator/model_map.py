"""The one map from a use-case category to the model that runs it.

Team configs name a category. ``resolve_model`` is what an invocation calls
when it needs an id. A model id written on an agent is an override: it is
logged, and the category's fallbacks still follow it.

Prices live on the model records. A category's cost tier is its primary's
tier, so changing a price is one edit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .logger import get_logger

log = get_logger("model_map")

_MAP_PATH = Path(__file__).resolve().parents[2] / "config" / "model_map.yaml"

_PROVIDERS = frozenset({"bedrock", "gemini"})
_COST_TIERS = frozenset({"low", "standard", "high", "infra"})
_LATENCY_TIERS = frozenset({"fastest", "fast", "medium", "slow"})
_MODEL_FIELDS = (
    "provider",
    "cost_per_1m_input",
    "cost_per_1m_output",
    "cost_tier",
    "latency_tier",
    "price_source",
    "price_checked",
)

_UNAVAILABLE_CODES = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceQuotaExceededException",
    "ModelNotReadyException",
    "ResourceNotFoundException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "ModelErrorException",
})

_cache: Optional[Dict[str, Any]] = None


class ModelUnavailable(RuntimeError):
    """The named model cannot take this call. The next fallback should."""


@dataclass
class ModelSpec:
    model_id: str
    provider: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    cost_tier: str
    latency_tier: str
    price_source: str
    price_checked: str
    cost_per_image: float = 0.0


@dataclass
class ResolvedModel:
    category: str
    requested_category: str
    used_default: bool
    override: bool
    primary: ModelSpec
    fallbacks: List[ModelSpec] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.0
    rationale: str = ""

    @property
    def model_id(self) -> str:
        return self.primary.model_id

    @property
    def provider(self) -> str:
        return self.primary.provider

    @property
    def chain(self) -> List[ModelSpec]:
        return [self.primary, *self.fallbacks]

    @property
    def cost_tier(self) -> str:
        return self.primary.cost_tier

    @property
    def cost_per_1m_input(self) -> float:
        return self.primary.cost_per_1m_input

    @property
    def cost_per_1m_output(self) -> float:
        return self.primary.cost_per_1m_output


def map_path() -> Path:
    override = (os.environ.get("MODEL_MAP_PATH") or "").strip()
    return Path(override) if override else _MAP_PATH


def load_model_map(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate the map. The parsed document is cached per path."""
    global _cache
    target = path or map_path()
    if path is None and _cache is not None:
        return _cache
    raw = yaml.safe_load(target.read_text())
    errors = validate_model_map(raw)
    if errors:
        raise ValueError("model map is invalid: " + "; ".join(errors))
    if path is None:
        _cache = raw
    return raw


def reset_cache() -> None:
    global _cache
    _cache = None


def validate_model_map(doc: Any) -> List[str]:
    """Return human-readable problems. An empty list means the map can load."""
    errors: List[str] = []
    if not isinstance(doc, dict):
        return ["the map is not a mapping"]
    models = doc.get("models")
    categories = doc.get("categories")
    if not isinstance(models, dict) or not models:
        errors.append("models must be a non-empty mapping")
        models = {}
    if not isinstance(categories, dict) or not categories:
        errors.append("categories must be a non-empty mapping")
        categories = {}
    if "default" not in categories:
        errors.append("categories.default is required")
    for model_id, spec in models.items():
        if not isinstance(spec, dict):
            errors.append(f"{model_id} is not a mapping")
            continue
        for key in _MODEL_FIELDS:
            if key not in spec:
                errors.append(f"{model_id} is missing {key}")
        provider = spec.get("provider")
        if provider not in _PROVIDERS:
            errors.append(f"{model_id} provider {provider!r} is not bedrock or gemini")
        if spec.get("cost_tier") not in _COST_TIERS:
            errors.append(f"{model_id} cost_tier {spec.get('cost_tier')!r} is unknown")
        if spec.get("latency_tier") not in _LATENCY_TIERS:
            errors.append(f"{model_id} latency_tier {spec.get('latency_tier')!r} is unknown")
        for price in ("cost_per_1m_input", "cost_per_1m_output"):
            try:
                if float(spec.get(price)) < 0:
                    errors.append(f"{model_id} {price} is negative")
            except (TypeError, ValueError):
                errors.append(f"{model_id} {price} is not a number")
    for name, cat in categories.items():
        if not isinstance(cat, dict):
            errors.append(f"category {name} is not a mapping")
            continue
        primary = cat.get("primary")
        if primary not in models:
            errors.append(f"category {name} primary {primary!r} is not in models")
        fallbacks = cat.get("fallbacks", [])
        if not isinstance(fallbacks, list):
            errors.append(f"category {name} fallbacks is not a list")
            fallbacks = []
        for item in fallbacks:
            if item not in models:
                errors.append(f"category {name} fallback {item!r} is not in models")
        if not str(cat.get("rationale") or "").strip():
            errors.append(f"category {name} has no rationale")
        tier = cat.get("cost_tier")
        if tier not in _COST_TIERS:
            errors.append(f"category {name} cost_tier {tier!r} is unknown")
        elif primary in models and isinstance(models.get(primary), dict):
            if models[primary].get("cost_tier") != tier:
                errors.append(
                    f"category {name} cost_tier {tier!r} does not match "
                    f"its primary's {models[primary].get('cost_tier')!r}"
                )
        try:
            int(cat.get("max_tokens"))
        except (TypeError, ValueError):
            errors.append(f"category {name} max_tokens is not an integer")
        try:
            float(cat.get("temperature"))
        except (TypeError, ValueError):
            errors.append(f"category {name} temperature is not a number")
    return errors


def model_ids() -> set:
    return set(load_model_map().get("models") or {})


def categories() -> List[str]:
    return sorted((load_model_map().get("categories") or {}).keys())


def _spec(model_id: str) -> ModelSpec:
    raw = load_model_map()["models"][model_id]
    return ModelSpec(
        model_id=model_id,
        provider=str(raw["provider"]),
        cost_per_1m_input=float(raw["cost_per_1m_input"]),
        cost_per_1m_output=float(raw["cost_per_1m_output"]),
        cost_tier=str(raw["cost_tier"]),
        latency_tier=str(raw["latency_tier"]),
        price_source=str(raw["price_source"]),
        price_checked=str(raw["price_checked"]),
        cost_per_image=float(raw.get("cost_per_image") or 0),
    )


def _override_spec(model_id: str) -> ModelSpec:
    known = (load_model_map().get("models") or {})
    if model_id in known:
        return _spec(model_id)
    provider = "gemini" if model_id.startswith("gemini-") else "bedrock"
    log.warning(
        "model_override_not_in_map",
        extra={"model_id": model_id, "provider": provider},
    )
    return ModelSpec(
        model_id=model_id,
        provider=provider,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        cost_tier="infra",
        latency_tier="medium",
        price_source="",
        price_checked="",
    )


def resolve_model(category: str, override: str = "") -> ResolvedModel:
    """The model chain for this category.

    An unknown category resolves to ``default`` and logs a warning. A
    non-empty ``override`` becomes the primary and is logged; the category's
    own fallbacks still follow it, so a throttle can leave the override.
    """
    doc = load_model_map()
    cats = doc["categories"]
    requested = (category or "").strip() or "default"
    used_default = False
    if requested not in cats:
        log.warning(
            "unknown_model_category",
            extra={"category": requested, "using": "default"},
        )
        requested_resolved = "default"
        used_default = True
    else:
        requested_resolved = requested
    cat = cats[requested_resolved]
    fallback_ids = [str(item) for item in cat.get("fallbacks") or []]
    override_id = (override or "").strip()
    if override_id:
        log.warning(
            "model_id_override",
            extra={"category": requested_resolved, "model_id": override_id},
        )
        primary = _override_spec(override_id)
        fallbacks = [_spec(item) for item in fallback_ids if item != override_id]
    else:
        primary = _spec(str(cat["primary"]))
        fallbacks = [_spec(item) for item in fallback_ids]
    return ResolvedModel(
        category=requested_resolved,
        requested_category=requested,
        used_default=used_default,
        override=bool(override_id),
        primary=primary,
        fallbacks=fallbacks,
        max_tokens=int(cat.get("max_tokens") or 0),
        temperature=float(cat.get("temperature") or 0),
        rationale=str(cat.get("rationale") or "").strip(),
    )


def estimate_cost(model_id: str, prompt_tokens: int = 0,
                  completion_tokens: int = 0, *, images: int = 0) -> float:
    """Dollars for this call, from the map's per-million-token prices.

    An image model with ``cost_per_image`` uses that when the call reports
    images and no output tokens. An id that is not in the map estimates 0.
    """
    known = load_model_map().get("models") or {}
    spec = known.get(model_id)
    if not isinstance(spec, dict):
        return 0.0
    prompt = max(int(prompt_tokens or 0), 0)
    completion = max(int(completion_tokens or 0), 0)
    cost = (
        prompt / 1_000_000 * float(spec["cost_per_1m_input"])
        + completion / 1_000_000 * float(spec["cost_per_1m_output"])
    )
    per_image = float(spec.get("cost_per_image") or 0)
    if images and per_image and completion == 0:
        cost += per_image * int(images)
    return round(cost, 8)


def is_model_unavailable(exc: BaseException) -> bool:
    """True when the next model in the chain should be tried.

    A malformed prompt is not this. A throttle, a missing grant, an unknown
    id, and a legacy model are.
    """
    if isinstance(exc, ModelUnavailable):
        return True
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(((response.get("Error") or {}).get("Code")) or "")
    message = str(exc).lower()
    if code in _UNAVAILABLE_CODES:
        return True
    if code == "ValidationException":
        if "malformed" in message:
            return False
        return any(phrase in message for phrase in (
            "model identifier",
            "invalid model",
            "legacy",
            "not supported",
        ))
    if code == "AccessDeniedException":
        return "model" in message
    return any(phrase in message for phrase in (
        "throttl",
        "too many requests",
        "model identifier is invalid",
        "marked by provider as legacy",
        "not authorized to invoke",
        "could not be found",
        "model access",
    ))


def emit_model_event(record: Dict[str, Any]) -> None:
    """Tell the observatory which model ran. A down observatory does not fail the turn."""
    try:
        log.info("model_selection", extra={k: v for k, v in record.items() if v is not None})
    except Exception:  # noqa: BLE001
        return
    try:
        from .mcp_observatory import record_model_selection
        record_model_selection(record)
    except Exception as exc:  # noqa: BLE001
        log.warning("model_selection_emit_failed", extra={"err": str(exc)[:200]})
