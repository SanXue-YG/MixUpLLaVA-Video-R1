"""NGRPO: Negative-enhanced Group Relative Policy Optimization.

Paper: ``doc/NGRPO-Negative-Enhanced-GRPO.pdf``
Core: virtual r_max in group stats so all-wrong groups get negative advantages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    import torch
    from .config import MixUpConfig


def advantages_with_virtual_max(
    rewards: "torch.Tensor",
    num_generations: int,
    r_max: float = 2.0,
    eps: float = 1e-4,
) -> "torch.Tensor":
    """Group advantage using R' = R ∪ {r_max} for mean/std."""
    import torch

    G = num_generations
    r = rewards.view(-1, G)
    B = r.size(0)
    r_max_t = torch.full((B, 1), r_max, device=rewards.device, dtype=rewards.dtype)
    r_aug = torch.cat([r, r_max_t], dim=1)  # [B, G+1]
    mu = r_aug.mean(dim=1, keepdim=True)
    sd = r_aug.std(dim=1, keepdim=True, unbiased=False)
    adv = (r - mu) / (sd + eps)
    return adv.view(-1)


def asymmetric_clip_objective(
    ratio: "torch.Tensor",
    advantages: "torch.Tensor",
    eps_pos: float = 0.24,
    eps_neg: float = 0.16,
) -> "torch.Tensor":
    """Asymmetric PPO-style clip on ρ·A (needs real importance ratio)."""
    import torch

    A = advantages
    pos = A >= 0
    # unclipped
    obj = ratio * A
    # clipped bounds
    clipped_pos = torch.minimum(obj, (1.0 + eps_pos) * A)
    clipped_neg = torch.maximum(obj, (1.0 - eps_neg) * A)
    return torch.where(pos, clipped_pos, clipped_neg)


@register(
    "ngrpo",
    stage="advantage",
    priority="P1",
    paper="doc/NGRPO-Negative-Enhanced-GRPO.pdf",
    enabled_attr="ngrpo",
)
def apply(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    # If MO-GRPO already set advantages, only fix all-wrong groups via virtual-max on rewards
    if ctx.get("mo_grpo_applied") and "advantages" in ctx:
        import torch

        G = int(ctx["num_generations"])
        rewards = ctx["rewards"]
        r = rewards.view(-1, G)
        # all non-positive ≈ all wrong for coupled reward
        all_wrong = (r <= 0).all(dim=1)
        if all_wrong.any():
            adv_n = advantages_with_virtual_max(rewards, G, cfg.ngrpo_r_max)
            adv = ctx["advantages"].view(-1, G)
            aw = all_wrong.unsqueeze(1).expand_as(adv)
            ctx["advantages"] = torch.where(aw, adv_n.view(-1, G), adv).view(-1)
        ctx["ngrpo_applied"] = True
        return ctx

    ctx["advantages"] = advantages_with_virtual_max(
        ctx["rewards"],
        int(ctx["num_generations"]),
        cfg.ngrpo_r_max,
    )
    ctx["ngrpo_applied"] = True
    if cfg.ngrpo_apply_asym_clip:
        ctx["use_ngrpo_asym_clip"] = True
        ctx["ngrpo_eps_pos"] = cfg.ngrpo_eps_pos
        ctx["ngrpo_eps_neg"] = cfg.ngrpo_eps_neg
    return ctx
