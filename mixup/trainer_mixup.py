"""Compose MixUp modules into TinyLLaVA GRPO ``compute_loss`` hooks.

Typical use inside a trainer patch::

    from mixup.trainer_mixup import ensure_modules_loaded, run_advantage_pipeline
    ensure_modules_loaded()
    ctx = run_advantage_pipeline(cfg, rewards_per_func=..., completion_mask=..., ...)

Also provides ``apply_mixup_to_repo`` to copy the MixUp trainer onto Drive REPO.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

from .config import MixUpConfig

_MODULES_LOADED = False


def ensure_modules_loaded() -> None:
    """Import all strategy modules so ``@register`` hooks run."""
    global _MODULES_LOADED
    if _MODULES_LOADED:
        return
    from . import (  # noqa: F401
        cppo,
        dagrpo,
        gfpo,
        gmpo,
        grpo_a,
        mapo,
        mo_grpo,
        ngrpo,
        project_baseline,
        step_grpo,
    )
    _MODULES_LOADED = True


def standard_grpo_advantages(
    rewards: "Any",
    num_generations: int,
    eps: float = 1e-4,
    add_noise: bool = True,
    noise_std: float = 0.02,
) -> "Any":
    import torch

    mean = rewards.view(-1, num_generations).mean(dim=1)
    std = rewards.view(-1, num_generations).std(dim=1, unbiased=False)
    mean = mean.repeat_interleave(num_generations, dim=0)
    std = std.repeat_interleave(num_generations, dim=0)
    advantages = (rewards - mean) / (std + eps)
    if add_noise:
        advantages = advantages + torch.randn_like(advantages) * noise_std
    return advantages


def compose_scalar_rewards(
    rewards_per_func: "Any",
    *,
    style: str = "coupled",
) -> "Any":
    """Match TinyLLaVA Phase1 reward coupling unless ``style='sum'``."""
    import torch

    if style == "sum":
        return rewards_per_func.sum(dim=1)
    acc = rewards_per_func[:, 0]
    fmt = rewards_per_func[:, 1] if rewards_per_func.size(1) > 1 else torch.zeros_like(acc)
    rewards = acc + (2 * acc - 1) * fmt
    rewards = torch.where(
        rewards == 0,
        torch.tensor(-2.0, device=rewards.device, dtype=rewards.dtype),
        rewards,
    )
    return rewards


def run_advantage_pipeline(
    cfg: MixUpConfig,
    *,
    rewards_per_func: "Any",
    completion_mask: "Any",
    num_generations: int,
    combined_ids: Optional["Any"] = None,
    prompt_inputs: Optional[Dict[str, Any]] = None,
    reward_style: str = "coupled",
) -> Dict[str, Any]:
    """Build rewards + advantages (+ optional CPPO prune) via registered modules."""
    ensure_modules_loaded()
    from .registry import apply_registered

    cfg.validate()
    rewards = compose_scalar_rewards(rewards_per_func, style=reward_style)
    ctx: Dict[str, Any] = {
        "rewards_per_func": rewards_per_func,
        "rewards": rewards,
        "completion_mask": completion_mask,
        "num_generations": num_generations,
        "combined_ids": combined_ids,
        "prompt_inputs": prompt_inputs or {},
    }

    # Reward-stage modules (length shaping, step stub, …)
    ctx = apply_registered(
        cfg,
        ctx,
        names=("step_grpo", "project_baseline"),
    )

    # Advantage producers — only run ones that are enabled; if none, standard GRPO
    adv_names = []
    if cfg.mo_grpo:
        adv_names.append("mo_grpo")
    if cfg.ngrpo:
        adv_names.append("ngrpo")
    if cfg.mapo and not cfg.mo_grpo and not cfg.ngrpo:
        adv_names.append("mapo")
    if cfg.gfpo:
        adv_names.append("gfpo")
    if cfg.grpo_a:
        adv_names.append("grpo_a")

    if adv_names:
        ctx = apply_registered(cfg, ctx, names=tuple(adv_names))
    if "advantages" not in ctx:
        add_noise = not (cfg.project_baseline and cfg.baseline_disable_noise)
        ctx["advantages"] = standard_grpo_advantages(
            ctx["rewards"], num_generations, add_noise=add_noise
        )

    # Post-advantage: baseline reweight, dagrpo, cppo, gmpo flags
    post = ["project_baseline_reweight", "dagrpo", "gmpo"]
    if combined_ids is not None and cfg.cppo:
        post.append("cppo")
    ctx = apply_registered(cfg, ctx, names=tuple(post))
    return ctx


def snapshot_config(cfg: MixUpConfig, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def apply_mixup_to_repo(
    mixup_repo: Union[str, Path],
    tinyllava_repo: Union[str, Path],
    cfg: Optional[MixUpConfig] = None,
    *,
    patch_name: str = "tinyllava_trainer_reason_mixup.py",
) -> str:
    """Copy MixUp trainer patch onto upstream ``tinyllava_trainer_reason.py``.

    Returns destination path. Optionally writes ``mixup_config.json`` next to train/.
    """
    mixup_repo = Path(mixup_repo)
    tinyllava_repo = Path(tinyllava_repo)
    src = mixup_repo / "mixup" / "patches" / patch_name
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = tinyllava_repo / "tinyllava" / "train" / "tinyllava_trainer_reason.py"
    shutil.copy2(src, dst)

    # Ensure mixup package is importable from REPO (symlink or path file)
    site = tinyllava_repo / "tinyllava" / "train" / "_mixup_path.pth"
    site.write_text(str(mixup_repo.resolve()) + "\n", encoding="utf-8")

    if cfg is not None:
        snap = tinyllava_repo / "tinyllava" / "train" / "mixup_config.json"
        snapshot_config(cfg, snap)
        # Patch trainer defaults via a small sidecar read by mixup trainer
    return str(dst)


def load_mixup_config_sidecar(train_dir: Union[str, Path]) -> Optional[MixUpConfig]:
    path = Path(train_dir) / "mixup_config.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return MixUpConfig.from_dict(data, strict=False)


def cfg_from_mapping(data: Mapping[str, Any]) -> MixUpConfig:
    return MixUpConfig.from_dict(dict(data), strict=True)
