"""GMPO: Geometric-Mean Policy Optimization (P2).

Paper: ``doc/GMPO-Geometric-Mean-Policy-Optimization.pdf``
Sequence-level geometric mean of clipped token objectives.
Note: with TinyLLaVA's on-policy ρ≈1 simplification, gains are limited until
real π_old log-probs are available.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    import torch
    from .config import MixUpConfig


def geometric_mean_loss(
    per_token_logps: "torch.Tensor",
    old_per_token_logps: Optional["torch.Tensor"],
    advantages: "torch.Tensor",
    completion_mask: "torch.Tensor",
    beta_kl: "torch.Tensor",
    per_token_kl: "torch.Tensor",
    eps: float = 0.2,
) -> "torch.Tensor":
    """Return scalar loss (−geo-mean objective + KL)."""
    import torch

    if old_per_token_logps is None:
        # on-policy fallback: ρ=1 → token term = A (broadcast)
        ratio = torch.ones_like(per_token_logps)
    else:
        ratio = torch.exp(per_token_logps - old_per_token_logps)

    A = advantages.unsqueeze(1)
    unclipped = ratio * A
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * A
    token_obj = torch.minimum(unclipped, clipped) if (A >= 0).all() else torch.where(
        A >= 0, torch.minimum(unclipped, clipped), torch.maximum(unclipped, clipped)
    )
    # log-space geometric mean over completion tokens
    abs_obj = token_obj.abs().clamp_min(1e-8)
    log_term = torch.log(abs_obj)
    denom = completion_mask.sum(dim=1).clamp_min(1.0)
    geo = torch.exp((log_term * completion_mask).sum(dim=1) / denom)
    seq_obj = geo * torch.sign(advantages + 1e-12)
    kl_term = ((per_token_kl * completion_mask).sum(dim=1) / denom)
    return -(seq_obj - beta_kl * kl_term).mean()


@register(
    "gmpo",
    stage="loss",
    priority="P2",
    paper="doc/GMPO-Geometric-Mean-Policy-Optimization.pdf",
    enabled_attr="gmpo",
)
def apply(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    ctx["use_gmpo_loss"] = True
    ctx["gmpo_eps"] = cfg.gmpo_eps
    return ctx
