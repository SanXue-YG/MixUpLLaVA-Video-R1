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


def _test_selector() -> None:
    import tempfile
    from pathlib import Path
    from mixup.selector import prepare_run, build_mixup_config
    from mixup.training_entry import prepare_training

    cfg = build_mixup_config(mixup={"ngrpo": True})
    assert cfg.project_baseline and cfg.cppo and cfg.ngrpo

    cfg = build_mixup_config(preset="speed")
    assert (not cfg.project_baseline) and cfg.cppo

    with tempfile.TemporaryDirectory() as td:
        plan = prepare_run(
            preset="m1",
            mixup={"mo_grpo": True},
            out_base=td,
            dry_run=False,
            tag="test",
        )
        assert plan.mixup.mo_grpo and plan.mixup.cppo
        assert Path(plan.snapshot_path).is_file()
        assert Path(plan.report_stub_path).is_file()
        assert "MO" in plan.run_id or "mo" in plan.run_id.lower() or "MO" in "".join(plan.enabled)
        print("selector OK", plan.run_id)

    plan2 = prepare_training(preset="m5q", dry_run=True, tag="dry")
    assert plan2.mixup.ngrpo and plan2.mixup.mo_grpo
    print("prepare_training dry_run OK", plan2.enabled)


def _test_eval_training() -> None:
    import json
    import tempfile
    from pathlib import Path
    from mixup.eval_training import collect_training_metrics, evaluate_run, compare_training_metrics
    from mixup.eval_benchmarks import build_benchmark_plan, run_benchmark, SPACE_WARNING

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run_a = td / "run_a"
        run_b = td / "run_b"
        run_a.mkdir()
        run_b.mkdir()
        hist = [
            {"loss": 0.1, "reward": 0.2, "rewards/accuracy_reward": 0.3, "kl": 0.01},
            {"loss": 0.05, "reward": 0.4, "rewards/accuracy_reward": 0.5, "kl": 0.02},
            {"train_runtime": 100.0, "train_steps_per_second": 0.01, "train_loss": 0.05},
        ]
        (run_a / "trainer_state.json").write_text(
            json.dumps({"log_history": hist}), encoding="utf-8"
        )
        hist_b = [
            {"loss": 0.08, "reward": 0.5, "rewards/accuracy_reward": 0.6, "kl": 0.01},
            {"loss": 0.04, "reward": 0.55, "rewards/accuracy_reward": 0.65, "kl": 0.015},
            {"train_runtime": 80.0, "train_steps_per_second": 0.012, "train_loss": 0.04},
        ]
        (run_b / "trainer_state.json").write_text(
            json.dumps({"log_history": hist_b}), encoding="utf-8"
        )
        m = evaluate_run(run_b, baseline_dir=run_a)
        assert m.summary.get("train_steps_per_second") == 0.012
        assert (run_b / "training_eval_report.md").is_file()
        print("eval_training OK", m.means.get("reward"))

    plan = build_benchmark_plan(
        eval_root=td if "td" in dir() else tempfile.mkdtemp(),
        model_path="/tmp/model",
        tinyllava_repo="/tmp/repo",
    )
    # use fresh temp for plan
    with tempfile.TemporaryDirectory() as ed:
        plan = build_benchmark_plan(
            eval_root=ed, model_path="/tmp/model", tinyllava_repo="/tmp/repo"
        )
        assert not plan.any_ready()
        r = run_benchmark(
            "videomme",
            mixup_repo=Path(__file__).resolve().parents[1],
            tinyllava_repo="/tmp/repo",
            model_path="/tmp/model",
            eval_root=ed,
            dry_run=True,
        )
        assert r["status"] == "dry_run"
        assert "600" in SPACE_WARNING or "GB" in SPACE_WARNING
        print("eval_benchmarks dry_run OK")


if __name__ == "__main__":
    main()
    from pathlib import Path
    _test_selector()
    print("SELECTOR TESTS PASSED")
    _test_eval_training()
    print("EVAL TESTS PASSED")
