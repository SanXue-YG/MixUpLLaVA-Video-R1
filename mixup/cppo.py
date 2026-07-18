"""CPPO: Completion Pruning Policy Optimization helpers.

Paper: ``doc/CPPO-Accelerating-GRPO-Reasoning-Models.pdf``
Core idea: after computing group advantages, drop completions with the
smallest |advantage|, then run policy/ref forward only on kept rows.

Integration (Colab / Drive REPO):
  1. Use MixUp trainer patch via ``mixup.trainer_mixup.apply_mixup_to_repo``, OR
  2. Copy ``mixup/patches/tinyllava_trainer_reason_cppo.py`` / ``*_mixup.py``.

``pruning_rate=0`` → GRPO baseline (no prune).
``pruning_rate=0.5`` → drop ~50% lowest-|advantage| completions (keep ≥1).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from .registry import register

if TYPE_CHECKING:
    import torch
    from .config import MixUpConfig


def build_keep_mask(
    advantages: "torch.Tensor",
    pruning_rate: float,
) -> "torch.Tensor":
    """Return boolean mask over completions (True = keep for gradient)."""
    import torch

    n = advantages.numel()
    if pruning_rate <= 0 or n <= 1:
        return torch.ones(n, dtype=torch.bool, device=advantages.device)

    k_prune = max(0, int(n * pruning_rate))
    k_prune = min(k_prune, n - 1)  # keep at least 1
    if k_prune <= 0:
        return torch.ones(n, dtype=torch.bool, device=advantages.device)

    abs_adv = torch.abs(advantages)
    _, idx_smallest = torch.topk(abs_adv, k_prune, largest=False)
    keep_mask = torch.ones(n, dtype=torch.bool, device=advantages.device)
    keep_mask[idx_smallest] = False
    return keep_mask


def apply_completion_pruning(
    advantages: "torch.Tensor",
    prompt_inputs: Dict[str, Any],
    combined_ids: "torch.Tensor",
    completion_mask: "torch.Tensor",
    pruning_rate: float,
) -> Tuple["torch.Tensor", Dict[str, Any], "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
    """Prune batch tensors by smallest |advantage|."""
    keep_mask = build_keep_mask(advantages, pruning_rate)
    if keep_mask.all():
        return advantages, prompt_inputs, combined_ids, completion_mask, None

    advantages = advantages[keep_mask]
    if "video" in prompt_inputs and prompt_inputs["video"] is not None:
        prompt_inputs = dict(prompt_inputs)
        prompt_inputs["video"] = prompt_inputs["video"][keep_mask]
    combined_ids = combined_ids[keep_mask]
    completion_mask = completion_mask[keep_mask]
    return advantages, prompt_inputs, combined_ids, completion_mask, keep_mask


def summarize_pruning(keep_mask: Optional["torch.Tensor"], pruning_rate: float) -> str:
    if keep_mask is None:
        return f"cppo off or no-op (rate={pruning_rate})"
    kept = int(keep_mask.sum().item())
    total = int(keep_mask.numel())
    return f"cppo rate={pruning_rate}: keep {kept}/{total}"


@register(
    "cppo",
    stage="prune",
    priority="P0",
    paper="doc/CPPO-Accelerating-GRPO-Reasoning-Models.pdf",
    enabled_attr="cppo",
)
def apply(cfg: "MixUpConfig", ctx: Dict[str, Any]) -> Dict[str, Any]:
    rate = cfg.cppo_pruning_rate if cfg.cppo else 0.0
    advantages, prompt_inputs, combined_ids, completion_mask, keep_mask = apply_completion_pruning(
        ctx["advantages"],
        ctx.get("prompt_inputs") or {},
        ctx["combined_ids"],
        ctx["completion_mask"],
        rate,
    )
    ctx["advantages"] = advantages
    ctx["prompt_inputs"] = prompt_inputs
    ctx["combined_ids"] = combined_ids
    ctx["completion_mask"] = completion_mask
    ctx["cppo_keep_mask"] = keep_mask
    ctx["cppo_summary"] = summarize_pruning(keep_mask, rate)
    if keep_mask is not None:
        if "rewards" in ctx:
            ctx["rewards"] = ctx["rewards"][keep_mask]
        if "rewards_per_func" in ctx and ctx["rewards_per_func"] is not None:
            ctx["rewards_per_func"] = ctx["rewards_per_func"][keep_mask]
    return ctx
