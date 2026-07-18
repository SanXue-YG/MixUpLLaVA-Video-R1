"""GRPO-A / G²RPO-A: Guided GRPO with adaptive guidance (P2 stub).

Paper: ``doc/GRPO-A-Guided-Adaptive-Guidance.pdf``
Needs ground-truth reasoning prefixes — uncommon for NextQA video.
This module exposes config + a helper to build guided prompts; generation
hook is left for Phase 4/6 when GT CoT is available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    from .config import MixUpConfig


def plan_guided_indices(num_generations: int, alpha: float) -> List[int]:
    """Indices that should receive guidance prefix (first ⌊αG⌋)."""
    n = max(0, min(num_generations, int(num_generations * alpha)))
    return list(range(n))


def apply_guidance_prefix(prompt: str, guidance: str, ell: int) -> str:
    g = guidance[:ell] if ell > 0 else ""
    if not g:
        return prompt
    return prompt + g


@register(
    "grpo_a",
    stage="generation",
    priority="P2",
    paper="doc/GRPO-A-Guided-Adaptive-Guidance.pdf",
    enabled_attr="grpo_a",
)
def apply(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    ctx["grpo_a_stub"] = True
    ctx["grpo_a_guided_indices"] = plan_guided_indices(
        int(ctx.get("num_generations", cfg.num_generations)),
        cfg.grpo_a_alpha,
    )
    ctx["grpo_a_ell"] = cfg.grpo_a_ell
    # No change to advantages/loss until generation is wired with GT prefixes.
    return ctx
