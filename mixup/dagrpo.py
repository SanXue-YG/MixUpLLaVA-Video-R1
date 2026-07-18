"""DaGRPO: Distinctiveness-Aware GRPO (P2).

Paper: ``doc/DaGRPO-Distinctiveness-Aware-GRPO.pdf``
Masks low-distinctiveness samples that cause pos/neg gradient conflict.
Off-policy anchors: stub (optional later).
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    from .config import MixUpConfig


def distinctiveness_mask(
    advantages: "torch.Tensor",
    quality: "torch.Tensor",
    num_generations: int,
    delta: float = 0.1,
) -> "torch.Tensor":
    import torch

    G = num_generations
    A = advantages.view(-1, G)
    S = quality.view(-1, G)
    B = A.size(0)
    lam = torch.zeros_like(A)
    for b in range(B):
        pos = A[b] > 0
        neg = A[b] < 0
        s_max_neg = S[b, neg].max() if neg.any() else torch.tensor(float("-inf"), device=A.device)
        s_min_pos = S[b, pos].min() if pos.any() else torch.tensor(float("inf"), device=A.device)
        keep_pos = pos & ((S[b] - s_max_neg) >= delta)
        keep_neg = neg & ((s_min_pos - S[b]) >= delta)
        lam[b] = (keep_pos | keep_neg).float()
        # if everything masked, keep original signs (avoid empty group)
        if lam[b].sum() == 0:
            lam[b] = (A[b] != 0).float()
    return lam.view(-1)


@register(
    "dagrpo",
    stage="mask",
    priority="P2",
    paper="doc/DaGRPO-Distinctiveness-Aware-GRPO.pdf",
    enabled_attr="dagrpo",
)
def apply(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    advantages = ctx["advantages"]
    # quality score S: prefer accuracy reward, else rewards
    rpf = ctx.get("rewards_per_func")
    if rpf is not None and rpf.ndim == 2 and rpf.size(1) >= 1:
        quality = rpf[:, 0]
    else:
        quality = ctx["rewards"]
    lam = distinctiveness_mask(
        advantages, quality, int(ctx["num_generations"]), cfg.dagrpo_delta
    )
    ctx["advantages"] = advantages * lam
    ctx["dagrpo_lambda"] = lam
    return ctx
