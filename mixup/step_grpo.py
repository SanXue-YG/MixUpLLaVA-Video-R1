"""StepGRPO (R1-VL): step-wise process rewards — reserved stub.

Paper: ``doc/R1-VL-StepGRPO.pdf``
Requires step segmentation + soft matching to reference key steps.
Phase 3: interface + documentation only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    from .config import MixUpConfig


def split_steps(text: str) -> List[str]:
    """Heuristic step split (newline / 'Step N:'). Not production-grade."""
    import re

    parts = re.split(r"(?:\\n\\n+|\\n(?=Step\\s*\\d)|(?=Step\\s*\\d))", text)
    return [p.strip() for p in parts if p and p.strip()]


def step_rar_score(
    steps: Sequence[str],
    key_steps: Sequence[str],
    final_correct: bool,
    alpha: float = 0.5,
) -> float:
    """Minimal StepRAR-style score when key_steps are provided."""
    if not key_steps:
        return 1.0 if final_correct else 0.0
    matched = 0
    lower_steps = [s.lower() for s in steps]
    for k in key_steps:
        kl = k.lower()
        if any(kl in s or s in kl for s in lower_steps):
            matched += 1
    frac = matched / max(len(key_steps), 1)
    if final_correct:
        return 1.0 + alpha * frac
    return alpha * frac


@register(
    "step_grpo",
    stage="reserved",
    priority="reserved",
    paper="doc/R1-VL-StepGRPO.pdf",
    enabled_attr="step_grpo",
)
def apply(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    ctx["step_grpo_stub"] = True
    ctx["step_grpo_alpha"] = cfg.step_grpo_alpha
    # Without key_steps annotations, leave rewards unchanged.
    return ctx
