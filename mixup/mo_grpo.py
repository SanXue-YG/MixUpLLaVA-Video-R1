"""MO-GRPO: Multi-Objective normalize-then-sum advantages.

Paper: ``doc/MO-GRPO-Multi-Objective-GRPO.pdf``
Avoids high-variance reward components dominating after a single sum-then-normalize.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    import torch
    from .config import MixUpConfig


def normalize_then_sum(
    rewards_per_func: "torch.Tensor",
    num_generations: int,
    eps: float = 1e-4,
) -> "torch.Tensor":
    """``rewards_per_func``: [N, K] → advantages [N]."""
    import torch

    G = num_generations
    N, K = rewards_per_func.shape
    assert N % G == 0
    adv = torch.zeros(N, device=rewards_per_func.device, dtype=rewards_per_func.dtype)
    for k in range(K):
        r = rewards_per_func[:, k]
        rg = r.view(-1, G)
        mu = rg.mean(dim=1).repeat_interleave(G, dim=0)
        sd = rg.std(dim=1, unbiased=False).repeat_interleave(G, dim=0)
        adv = adv + (r - mu) / (sd + eps)
    return adv


@register(
    "mo_grpo",
    stage="advantage",
    priority="P1",
    paper="doc/MO-GRPO-Multi-Objective-GRPO.pdf",
    enabled_attr="mo_grpo",
)
def apply(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    rpf = ctx.get("rewards_per_func")
    if rpf is None:
        raise ValueError("mo_grpo requires ctx['rewards_per_func'] of shape [N, K]")
    # Optionally append length bonus as extra objective if present
    import torch

    extras = []
    if ctx.get("length_bonus") is not None:
        extras.append(ctx["length_bonus"].unsqueeze(1))
    if extras:
        rpf = torch.cat([rpf] + extras, dim=1)
    ctx["advantages"] = normalize_then_sum(rpf, int(ctx["num_generations"]))
    ctx["mo_grpo_applied"] = True
    return ctx
