#!/usr/bin/env python3
"""Smoke tests for MixUp modules (CPU). Run: python -m mixup.tests_smoke"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import torch
    from mixup.config import MixUpConfig
    from mixup.trainer_mixup import ensure_modules_loaded, run_advantage_pipeline
    from mixup.registry import list_modules
    from mixup.presets import load_preset, list_presets

    ensure_modules_loaded()
    mods = {m.name: m for m in list_modules()}
    for need in ("project_baseline", "cppo", "gfpo", "ngrpo", "mo_grpo"):
        assert need in mods, f"missing module {need}"
    print("registered:", sorted(mods))

    G = 8
    N = G
    acc = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    fmt = torch.tensor([0.8, 0.7, 0.6, 0.5, 0.9, 0.4, 0.3, 0.2])
    rpf = torch.stack([acc, fmt], dim=1)
    mask = torch.ones(N, 16)
    ids = torch.randint(0, 100, (N, 20))

    # M1 default = B + CPPO
    cfg = MixUpConfig()
    assert cfg.project_baseline and cfg.cppo
    ctx = run_advantage_pipeline(
        cfg,
        rewards_per_func=rpf,
        completion_mask=mask,
        num_generations=G,
        combined_ids=ids,
        prompt_inputs={"video": torch.zeros(N, 1)},
    )
    assert "length_bonus" in ctx
    assert ctx["advantages"].numel() <= N
    print("M1 default B+CPPO OK", cfg.enabled())

    # 加开 NGRPO 时 B/CPPO 仍保留
    cfg = MixUpConfig.from_dict({"ngrpo": True})
    assert cfg.project_baseline and cfg.cppo and cfg.ngrpo
    print("stack ngrpo keeps defaults OK", cfg.enabled())

    # 显式关闭 B，仅 CPPO
    cfg = MixUpConfig(project_baseline=False, cppo=True, cppo_pruning_rate=0.5)
    assert cfg.dropped_project_defaults() == ["project_baseline"]
    ctx = run_advantage_pipeline(
        cfg,
        rewards_per_func=rpf,
        completion_mask=mask,
        num_generations=G,
        combined_ids=ids,
        prompt_inputs={"video": torch.zeros(N, 1)},
    )
    assert ctx["advantages"].numel() < N
    print("CPPO-only ablation OK", ctx.get("cppo_summary"))

    # GFPO（仍带默认 B+CPPO；测 Top-k 用关闭 CPPO 以免剪枝干扰计数）
    cfg = MixUpConfig(project_baseline=False, cppo=False, gfpo=True, gfpo_top_k=4)
    ctx = run_advantage_pipeline(cfg, rewards_per_func=rpf, completion_mask=mask, num_generations=G)
    assert (ctx["advantages"] == 0).sum() >= 4
    print("GFPO OK nonzeros", int((ctx["advantages"] != 0).sum()))

    # NGRPO all-wrong（关 B/CPPO 以免耦合 reward 干扰）
    rpf0 = torch.zeros(N, 2)
    cfg = MixUpConfig(project_baseline=False, cppo=False, ngrpo=True, ngrpo_r_max=2.0)
    ctx = run_advantage_pipeline(cfg, rewards_per_func=rpf0, completion_mask=mask, num_generations=G)
    assert (ctx["advantages"] < 0).all()
    print("NGRPO OK all-neg", float(ctx["advantages"].mean()))

    # MO-GRPO
    cfg = MixUpConfig(project_baseline=False, cppo=False, mo_grpo=True)
    ctx = run_advantage_pipeline(cfg, rewards_per_func=rpf, completion_mask=mask, num_generations=G)
    assert ctx.get("mo_grpo_applied")
    print("MO-GRPO OK")

    # presets
    assert "m5q_quality" in list_presets()
    p = load_preset("m1_baseline_b")
    assert p.project_baseline and p.cppo
    p = load_preset("m5q_quality")
    assert p.mo_grpo and p.ngrpo and p.project_baseline and p.cppo
    ctx = run_advantage_pipeline(
        p,
        rewards_per_func=rpf,
        completion_mask=mask,
        num_generations=G,
        combined_ids=ids,
        prompt_inputs={"video": torch.zeros(N, 1)},
    )
    print("preset m1/m5q OK", p.enabled(), "adv", tuple(ctx["advantages"].shape))

    ab = load_preset("ablation_grpo_vanilla")
    assert not ab.project_baseline and not ab.cppo

    try:
        MixUpConfig.from_dict({"not_a_module": True}, strict=True)
        raise AssertionError("expected KeyError")
    except KeyError as e:
        print("unknown key OK", str(e)[:60])

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
