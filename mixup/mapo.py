"""MAPO: Mixed Advantage Policy Optimization (P2).

Paper: ``doc/MAPO-Mixed-Advantage-Policy-Optimization.pdf``
Mix z-score and percent-deviation advantages by group success rate p.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    import torch
    from .config import MixUpConfig


def mixed_advantages(
    rewards: "torch.Tensor",
    num_generations: int,
    success: "torch.Tensor",
    eps: float = 1e-4,
) -> "torch.Tensor":
    import torch

    G = num_generations
    r = rewards.view(-1, G)
    s = success.view(-1, G).float()
    p = s.mean(dim=1)  # [B]
    lam = 1.0 - 4.0 * p * (1.0 - p)  # [B]
    mu = r.mean(dim=1, keepdim=True)
    sd = r.std(dim=1, keepdim=True, unbiased=False)
    A_std = (r - mu) / (sd + eps)
    A_apd = (r - mu) / (mu.abs() + eps)
    lam_b = lam.view(-1, 1)
    adv = (1.0 - lam_b) * A_std + lam_b * A_apd
    return adv.view(-1)


@register(
    "mapo",
    stage="advantage",
    priority="P2",
    paper="doc/MAPO-Mixed-Advantage-Policy-Optimization.pdf",
    enabled_attr="mapo",
)
def apply(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    # skip if GFPO/MO already set and user wants exclusivity — MAPO replaces base A
    rpf = ctx.get("rewards_per_func")
    if rpf is not None and rpf.ndim == 2 and rpf.size(1) >= 1:
        success = (rpf[:, 0] > 0)
    else:
        success = ctx["rewards"] > 0
    ctx["advantages"] = mixed_advantages(
        ctx["rewards"], int(ctx["num_generations"]), success
    )
    ctx["mapo_applied"] = True
    return ctx
