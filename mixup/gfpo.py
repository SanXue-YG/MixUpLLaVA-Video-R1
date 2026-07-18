"""GFPO: Group Filtered Policy Optimization (Sample More to Think Less).

Paper: ``doc/GFPO-Sample-More-Think-Less.pdf``
Core: keep Top-k by metric; only Top-k enter mean/std (or zero-out others' A).
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    import torch
    from .config import MixUpConfig


def metric_scores(
    rewards: "torch.Tensor",
    completion_mask: "torch.Tensor",
    metric: str,
) -> "torch.Tensor":
    import torch

    lengths = completion_mask.sum(dim=1).float().clamp_min(1.0)
    if metric == "shortest":
        return -lengths
    if metric == "token_efficiency":
        return rewards / lengths
    raise ValueError(f"Unknown GFPO metric: {metric}")


def apply_topk_mask(
    rewards: "torch.Tensor",
    completion_mask: "torch.Tensor",
    num_generations: int,
    top_k: int,
    metric: str = "shortest",
) -> "torch.Tensor":
    """Return advantages with non-topk zeroed; mean/std over top-k only."""
    import torch

    G = num_generations
    n = rewards.numel()
    assert n % G == 0
    B = n // G
    k = max(1, min(top_k, G))
    scores = metric_scores(rewards, completion_mask, metric).view(B, G)
    r = rewards.view(B, G)
    adv = torch.zeros_like(r)
    for b in range(B):
        _, idx = torch.topk(scores[b], k, largest=True)
        sel = r[b, idx]
        mu = sel.mean()
        sd = sel.std(unbiased=False) if k > 1 else torch.tensor(0.0, device=r.device)
        a = (r[b] - mu) / (sd + 1e-4)
        mask = torch.zeros(G, dtype=torch.bool, device=r.device)
        mask[idx] = True
        a = torch.where(mask, a, torch.zeros_like(a))
        adv[b] = a
    return adv.view(-1)


@register(
    "gfpo",
    stage="advantage",
    priority="P1",
    paper="doc/GFPO-Sample-More-Think-Less.pdf",
    enabled_attr="gfpo",
)
def apply(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Replace / overlay advantages with GFPO Top-k masked advantages."""
    advantages = apply_topk_mask(
        ctx["rewards"],
        ctx["completion_mask"],
        int(ctx["num_generations"]),
        cfg.gfpo_top_k,
        cfg.gfpo_metric,
    )
    ctx["advantages"] = advantages
    ctx["gfpo_applied"] = True
    return ctx
