"""Strategy registry: name → apply(config, context) → context.

New method = new module file + ``register("name", apply_fn, ...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import MixUpConfig, KNOWN_BOOL_FLAGS

ApplyFn = Callable[["MixUpConfig", Dict[str, Any]], Dict[str, Any]]


@dataclass
class ModuleSpec:
    name: str
    apply: ApplyFn
    stage: str  # reward | advantage | mask | prune | loss | generation | reserved
    priority: str  # P0 | P1 | P2 | reserved
    paper: str = ""
    enabled_attr: str = ""


_REGISTRY: Dict[str, ModuleSpec] = {}


def register(
    name: str,
    *,
    stage: str,
    priority: str,
    paper: str = "",
    enabled_attr: Optional[str] = None,
):
    """Decorator: ``@register("cppo", stage="prune", priority="P0")``."""

    def decorator(apply: ApplyFn) -> ApplyFn:
        attr = enabled_attr or name
        if name in _REGISTRY:
            raise ValueError(f"Module already registered: {name}")
        _REGISTRY[name] = ModuleSpec(
            name=name,
            apply=apply,
            stage=stage,
            priority=priority,
            paper=paper,
            enabled_attr=attr,
        )
        return apply

    return decorator


def get_module(name: str) -> ModuleSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown MixUp module {name!r}. "
            f"Registered: {sorted(_REGISTRY)}. "
            f"Config flags: {list(KNOWN_BOOL_FLAGS)}."
        )
    return _REGISTRY[name]


def list_modules() -> List[ModuleSpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def is_enabled(cfg: MixUpConfig, spec: ModuleSpec) -> bool:
    return bool(getattr(cfg, spec.enabled_attr, False))


# Pipeline order inside compute_loss (after rewards_per_func is available).
PIPELINE_ORDER: Sequence[str] = (
    "step_grpo",          # reserved: may rewrite rewards_per_func
    "project_baseline",   # length shaping on rewards (pre-adv)
    "mo_grpo",            # normalize-then-sum → advantages
    "ngrpo",              # virtual max advantage (if not mo, or post)
    "mapo",               # mixed advantage
    "gfpo",               # top-k mask on advantages
    "project_baseline_reweight",  # difficulty reweight + clamp (internal stage)
    "dagrpo",             # distinctiveness mask
    "cppo",               # prune batch
    "gmpo",               # loss aggregation flag
    "grpo_a",             # generation stub (no-op in loss)
)


def apply_registered(
    cfg: MixUpConfig,
    ctx: Dict[str, Any],
    *,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run enabled modules in pipeline order (or a custom name list)."""
    cfg.validate()
    order = list(names) if names is not None else list(PIPELINE_ORDER)
    for name in order:
        if name == "project_baseline_reweight":
            if not cfg.project_baseline:
                continue
            from . import project_baseline as pb

            ctx = pb.apply_reweight(cfg, ctx)
            continue
        if name not in _REGISTRY:
            continue
        spec = _REGISTRY[name]
        if not is_enabled(cfg, spec):
            continue
        ctx = spec.apply(cfg, ctx)
    return ctx


def enabled_strategies(cfg: MixUpConfig) -> List[str]:
    return cfg.enabled()


# Backward-compatible re-export
__all__ = [
    "MixUpConfig",
    "ModuleSpec",
    "register",
    "get_module",
    "list_modules",
    "apply_registered",
    "enabled_strategies",
    "PIPELINE_ORDER",
]
