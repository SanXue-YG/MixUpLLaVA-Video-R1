"""Phase 4 optimization-strategy selector.

Build a runnable experiment plan from:
  - preset name (m1 / speed / m5 / m5q / …), and/or
  - MixUp toggle dict (B+CPPO on by default), and/or
  - tier yaml overrides (C1 …)

Does **not** require launching DeepSpeed; ``prepare_run`` writes snapshots and
optionally patches the upstream trainer for when training is needed later.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Union

from .config import MixUpConfig, PROJECT_DEFAULT_FLAGS
from .config_loader import default_c1_path, load_tier_config, notebook_constants
from .presets import PRESET_ALIASES, list_presets, load_preset, resolve_preset_name
from .trainer_mixup import apply_mixup_to_repo, snapshot_config

PathLike = Union[str, Path]


def _slug_enabled(cfg: MixUpConfig) -> str:
    parts: List[str] = []
    if cfg.project_baseline:
        parts.append("B")
    if cfg.cppo:
        parts.append(f"CPPO{int(cfg.cppo_pruning_rate * 100):02d}")
    if cfg.gfpo:
        parts.append(f"GFPO{cfg.gfpo_top_k}")
    if cfg.ngrpo:
        parts.append("NGRPO")
    if cfg.mo_grpo:
        parts.append("MO")
    if cfg.dagrpo:
        parts.append("Da")
    if cfg.gmpo:
        parts.append("GMPO")
    if cfg.mapo:
        parts.append("MAPO")
    if cfg.grpo_a:
        parts.append("GRPOA")
    if cfg.step_grpo:
        parts.append("STEP")
    if not parts:
        parts.append("vanilla")
    return "_".join(parts)


def make_run_id(
    cfg: MixUpConfig,
    *,
    tier: str = "C1",
    tag: str = "",
    timestamp: Optional[str] = None,
) -> str:
    """e.g. ``20260718_143022_C1_B_CPPO50_NGRPO`` or with custom tag."""
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _slug_enabled(cfg)
    bits = [ts, tier, slug]
    if tag:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")
        if safe:
            bits.append(safe)
    return "_".join(bits)


@dataclass
class RunPlan:
    """Materialized selector output — safe to serialize and hand to Phase 6."""

    run_id: str
    mixup: MixUpConfig
    tier_name: str = "C1"
    tier_config: Dict[str, Any] = field(default_factory=dict)
    preset: Optional[str] = None
    output_dir: str = ""
    snapshot_path: str = ""
    report_stub_path: str = ""
    dropped_project_defaults: List[str] = field(default_factory=list)
    enabled: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mixup": self.mixup.to_dict(),
            "tier_name": self.tier_name,
            "tier_config": self.tier_config,
            "preset": self.preset,
            "output_dir": self.output_dir,
            "snapshot_path": self.snapshot_path,
            "report_stub_path": self.report_stub_path,
            "dropped_project_defaults": list(self.dropped_project_defaults),
            "enabled": list(self.enabled),
            "notes": self.notes,
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        drop = (
            f" (dropped defaults: {self.dropped_project_defaults})"
            if self.dropped_project_defaults
            else ""
        )
        return (
            f"run_id={self.run_id}\n"
            f"enabled={self.enabled}{drop}\n"
            f"output_dir={self.output_dir}\n"
            f"snapshot={self.snapshot_path}"
        )


def build_mixup_config(
    *,
    preset: Optional[str] = None,
    mixup: Optional[Mapping[str, Any]] = None,
    base: Optional[MixUpConfig] = None,
) -> MixUpConfig:
    """Resolve MixUpConfig: preset → base → overlay ``mixup`` dict.

    Unspecified flags keep defaults (B+CPPO on). Explicit ``false`` turns off.
    """
    if base is not None:
        cfg = MixUpConfig.from_dict(base.to_dict(), strict=False)
        resolved_preset = None
    elif preset:
        resolved_preset = resolve_preset_name(preset)
        cfg = load_preset(resolved_preset)
    else:
        cfg = MixUpConfig()  # M1 defaults
        resolved_preset = None

    if mixup:
        overlay = MixUpConfig.from_dict(dict(mixup), strict=True)
        # Merge: only keys present in ``mixup`` override
        merged = cfg.to_dict()
        for k in mixup.keys():
            key = "gfpo_top_k" if k == "gfpo_topk" else k
            if key in overlay.to_dict():
                merged[key] = getattr(overlay, key)
        cfg = MixUpConfig.from_dict(merged, strict=False)

    cfg.validate()
    return cfg


def select_strategy(
    *,
    preset: Optional[str] = None,
    mixup: Optional[Mapping[str, Any]] = None,
    tier_yaml: Optional[PathLike] = None,
    tier_overrides: Optional[Mapping[str, Any]] = None,
    project_dir: Optional[PathLike] = None,
    out_base: Optional[PathLike] = None,
    tag: str = "",
    run_id: Optional[str] = None,
    notes: str = "",
) -> RunPlan:
    """Create a ``RunPlan`` (paths + config) without starting training."""
    cfg = build_mixup_config(preset=preset, mixup=mixup)
    resolved = resolve_preset_name(preset) if preset else None

    if tier_yaml is None:
        tier_yaml = default_c1_path()
    tier = load_tier_config(tier_yaml, overrides=tier_overrides)
    tier_name = str(tier.get("tier") or "C1")

    paths = tier.get("paths") or {}
    if project_dir is not None:
        paths = dict(paths)
        paths["project_dir"] = str(project_dir)
    if out_base is not None:
        base_out = Path(out_base)
    else:
        base_out = Path(paths.get("out_base") or "outputs")

    rid = run_id or make_run_id(cfg, tier=tier_name, tag=tag)
    output_dir = base_out / rid
    snapshot_path = output_dir / "mixup_config.json"
    report_stub = output_dir / "PHASE4_REPORT_STUB.md"

    plan = RunPlan(
        run_id=rid,
        mixup=cfg,
        tier_name=tier_name,
        tier_config=tier,
        preset=resolved,
        output_dir=str(output_dir),
        snapshot_path=str(snapshot_path),
        report_stub_path=str(report_stub),
        dropped_project_defaults=cfg.dropped_project_defaults(),
        enabled=cfg.enabled(),
        notes=notes,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return plan


def write_run_artifacts(plan: RunPlan, *, dry_run: bool = False) -> RunPlan:
    """Write output dir, mixup snapshot, run_plan.json, and report stub."""
    if dry_run:
        return plan
    out = Path(plan.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_config(plan.mixup, plan.snapshot_path)
    (out / "run_plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    _write_report_stub(plan)
    return plan


def _write_report_stub(plan: RunPlan) -> None:
    drop = plan.dropped_project_defaults
    drop_line = (
        f"- **已显式关闭的项目默认**：`{', '.join(drop)}`"
        if drop
        else "- **项目默认（B / CPPO）**：均保持开启"
    )
    body = f"""# Phase 4 运行记录（草稿）— `{plan.run_id}`

- **创建时间（UTC）**：{plan.created_at}
- **档位**：{plan.tier_name}
- **预设**：{plan.preset or "（自定义）"}
- **启用模块**：{', '.join(plan.enabled)}
{drop_line}
- **输出目录**：`{plan.output_dir}`
- **配置快照**：`{plan.snapshot_path}`
- **备注**：{plan.notes or "—"}

## 训练期指标（开训后填写）

| 指标 | 数值 |
|------|------|
| train_runtime | |
| train_steps_per_second | |
| reward 均值 | |
| accuracy_reward 均值 | |
| format_reward 均值 | |

## 与 M1 / Phase1 对照

| 对照 | 说明 |
|------|------|
| M1（B+CPPO） | |
| Phase1 A2 / C2 | 可选附录 |

完整模板见仓库 `docs/PHASE4_REPORT.md`。
"""
    Path(plan.report_stub_path).write_text(body, encoding="utf-8")


def prepare_run(
    *,
    preset: Optional[str] = None,
    mixup: Optional[Mapping[str, Any]] = None,
    tier_yaml: Optional[PathLike] = None,
    tier_overrides: Optional[Mapping[str, Any]] = None,
    project_dir: Optional[PathLike] = None,
    out_base: Optional[PathLike] = None,
    mixup_repo: Optional[PathLike] = None,
    tinyllava_repo: Optional[PathLike] = None,
    apply_patch: bool = False,
    tag: str = "",
    run_id: Optional[str] = None,
    notes: str = "",
    dry_run: bool = False,
) -> RunPlan:
    """Select strategy → write artifacts → optionally patch upstream trainer.

    ``apply_patch=True`` requires ``mixup_repo`` and ``tinyllava_repo`` (Colab Drive).
    Training itself is **not** started here (Phase 4 能力优先 / 开训可延后).
    """
    plan = select_strategy(
        preset=preset,
        mixup=mixup,
        tier_yaml=tier_yaml,
        tier_overrides=tier_overrides,
        project_dir=project_dir,
        out_base=out_base,
        tag=tag,
        run_id=run_id,
        notes=notes,
    )
    write_run_artifacts(plan, dry_run=dry_run)

    if apply_patch:
        if not mixup_repo or not tinyllava_repo:
            raise ValueError("apply_patch=True requires mixup_repo and tinyllava_repo")
        if not dry_run:
            apply_mixup_to_repo(mixup_repo, tinyllava_repo, plan.mixup)
            # also mirror snapshot under output
            Path(plan.output_dir).mkdir(parents=True, exist_ok=True)

    return plan


def apply_plan_to_notebook(plan: RunPlan, namespace: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Inject tier constants + MIXUP / RUN_ID into Notebook ``globals()``."""
    consts = notebook_constants(plan.tier_config)
    namespace.update(consts)
    namespace["MIXUP_CFG"] = plan.mixup
    namespace["MIXUP"] = plan.mixup.to_dict()
    namespace["RUN_ID"] = plan.run_id
    namespace["RUN_OUTPUT_DIR"] = plan.output_dir
    namespace["RUN_PLAN"] = plan
    return consts


def list_selector_presets() -> Dict[str, str]:
    """alias → yaml stem (for UI / docs)."""
    return dict(PRESET_ALIASES)
