"""Project baseline enhancement B — difficulty-aware + length shaping.

Survey §2.3 / strategy B. Paper folder: project design (not a public GRPO paper).
Full trainer reference: ``mixup/patches/tinyllava_trainer_reason_strategy_b.py``.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    import torch
    from .config import MixUpConfig


def describe() -> str:
    return (
        "project_baseline: difficulty-aware advantage reweight + "
        "adaptive length reward shaping + optional advantage clamp / beta↑"
    )


def apply_length_shaping(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Add length reward term into ``ctx['rewards']`` using completion lengths."""
    import torch

    rewards = ctx["rewards"]
    completion_mask = ctx["completion_mask"]
    G = int(ctx["num_generations"])
    device = rewards.device

    lengths = completion_mask.sum(dim=1).float()
    # difficulty proxy from accuracy column if present
    rpf = ctx.get("rewards_per_func")
    if rpf is not None and rpf.ndim == 2 and rpf.size(1) >= 1:
        acc = (rpf[:, 0] > 0).float()
        p = acc.view(-1, G).mean(dim=1)
    else:
        # fallback: treat positive reward as success
        succ = (rewards > 0).float()
        p = succ.view(-1, G).mean(dim=1)
    diff = (1.0 - p).clamp(0.0, 1.0)  # [B]
    L_tgt = cfg.baseline_L_easy * (1.0 - diff) + cfg.baseline_L_hard * diff
    L_tgt = L_tgt.repeat_interleave(G, dim=0)
    r_len = cfg.baseline_length_coef * torch.tanh((L_tgt - lengths) / cfg.baseline_length_tau)
    ctx["rewards"] = rewards + r_len
    ctx["length_bonus"] = r_len
    ctx["difficulty"] = diff.repeat_interleave(G, dim=0)
    return ctx


def apply_reweight(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Difficulty reweight + clamp after advantages are computed."""
    import torch

    advantages = ctx["advantages"]
    G = int(ctx["num_generations"])
    if "difficulty" in ctx:
        diff = ctx["difficulty"]
    else:
        rpf = ctx.get("rewards_per_func")
        rewards = ctx["rewards"]
        if rpf is not None and rpf.ndim == 2 and rpf.size(1) >= 1:
            acc = (rpf[:, 0] > 0).float()
            p = acc.view(-1, G).mean(dim=1)
        else:
            p = (rewards > 0).float().view(-1, G).mean(dim=1)
        diff = (1.0 - p).clamp(0.0, 1.0).repeat_interleave(G, dim=0)

    w = 1.0 + cfg.baseline_lambda_diff * diff
    advantages = advantages * w
    if cfg.baseline_advantage_clamp and cfg.baseline_advantage_clamp > 0:
        advantages = advantages.clamp(-cfg.baseline_advantage_clamp, cfg.baseline_advantage_clamp)
    ctx["advantages"] = advantages
    ctx["advantage_weights"] = w
    if cfg.baseline_disable_noise:
        ctx["skip_advantage_noise"] = True
    ctx["preferred_beta"] = cfg.baseline_beta
    return ctx


@register(
    "project_baseline",
    stage="reward",
    priority="P0",
    paper="survey §2.3 / strategy B",
    enabled_attr="project_baseline",
)
def apply(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Registry entry: length shaping only (reweight runs later in pipeline)."""
    return apply_length_shaping(cfg, ctx)
