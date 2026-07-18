"""Phase 6 console report helpers — baseline & compare markdown.

Used by ``notebooks/mixup_console.ipynb``. Training itself stays optional;
these helpers only need run dirs / RunPlan snapshots when available.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from .eval_training import (
    TrainingMetrics,
    collect_training_metrics,
    compare_training_metrics,
    evaluate_run,
)
from .selector import RunPlan

PathLike = Union[str, Path]

PHASE1_REFERENCE = {
    "A2": {
        "steps_per_second": 0.008,
        "train_runtime": 6357.3,
        "reward_mean": 0.190,
        "note": "GRPO g=8, Phase1",
    },
    "C2": {
        "steps_per_second": 0.011,
        "train_runtime": 4616.4,
        "reward_mean": 0.321,
        "note": "CPPO p=0.5 g=8, Phase1",
    },
}


def _read_mixup_snapshot(run_dir: Path) -> Dict[str, Any]:
    for name in ("mixup_config.json", "run_plan.json"):
        p = run_dir / name
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if name == "run_plan.json" and "mixup" in data:
                return data
            return {"mixup": data} if "project_baseline" in data or "cppo" in data else data
    return {}


def write_baseline_report(
    plan: RunPlan,
    *,
    metrics: Optional[TrainingMetrics] = None,
    out_path: Optional[PathLike] = None,
    extra_notes: str = "",
) -> Path:
    """Write ``baseline_report.md`` under the baseline run output dir."""
    out_dir = Path(plan.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = Path(out_path) if out_path else out_dir / "baseline_report.md"

    if metrics is None:
        # Prefer live metrics if trainer_state exists; else stub table
        state = out_dir / "trainer_state.json"
        if state.is_file() or any(out_dir.glob("checkpoint-*/trainer_state.json")):
            metrics = collect_training_metrics(out_dir, run_id=plan.run_id)

    drop = plan.dropped_project_defaults
    drop_line = (
        f"- **已关闭的项目默认**：`{', '.join(drop)}`"
        if drop
        else "- **项目默认（B / CPPO）**：均保持开启"
    )

    lines = [
        f"# 基线报告 — `{plan.run_id}`",
        "",
        f"- **UTC**：{datetime.now(timezone.utc).isoformat()}",
        f"- **预设**：{plan.preset or '自定义'}",
        f"- **档位**：{plan.tier_name}",
        f"- **启用**：{', '.join(plan.enabled)}",
        drop_line,
        f"- **输出**：`{plan.output_dir}`",
        f"- **快照**：`{plan.snapshot_path}`",
        f"- **备注**：{extra_notes or plan.notes or '—'}",
        "",
        "## MixUp 开关",
        "",
        "```json",
        json.dumps(plan.mixup.to_dict(), indent=2, ensure_ascii=False),
        "```",
        "",
        "## 训练期指标",
        "",
    ]
    if metrics is not None:
        lines += [
            "| 指标 | 值 |",
            "|------|----|",
        ]
        for k, v in sorted(metrics.summary.items()):
            lines.append(f"| `{k}` | {v} |")
        lines += ["", "### 过程均值", "", "| 指标 | 均值 |", "|------|------|"]
        for k, v in sorted(metrics.means.items()):
            lines.append(f"| `{k}` | {v} |")
    else:
        lines += [
            "> 尚未检测到 `trainer_state.json`。开训并落盘后重跑本单元格，或调用 "
            "`evaluate_run(plan.output_dir)`。",
            "",
            "| 指标 | 值 |",
            "|------|----|",
            "| train_steps_per_second | （待填） |",
            "| train_runtime | （待填） |",
            "| reward 均值 | （待填） |",
        ]

    lines += [
        "",
        "## 可选：四基准总评估",
        "",
        "默认跳过（数据约 ~600GB）。需要时见 `docs/PHASE5_BENCHMARKS.md`。",
        "",
        "## Phase1 参考（同档 C1 历史）",
        "",
        "| | A2 | C2 |",
        "|--|----|----|",
        f"| steps/s | {PHASE1_REFERENCE['A2']['steps_per_second']} | {PHASE1_REFERENCE['C2']['steps_per_second']} |",
        f"| runtime | {PHASE1_REFERENCE['A2']['train_runtime']} | {PHASE1_REFERENCE['C2']['train_runtime']} |",
        f"| reward | {PHASE1_REFERENCE['A2']['reward_mean']} | {PHASE1_REFERENCE['C2']['reward_mean']} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_compare_report(
    baseline_plan: RunPlan,
    candidate_plan: RunPlan,
    *,
    baseline_metrics: Optional[TrainingMetrics] = None,
    candidate_metrics: Optional[TrainingMetrics] = None,
    out_path: Optional[PathLike] = None,
) -> Path:
    """Write ``compare_vs_baseline.md`` under the candidate run dir."""
    cand_dir = Path(candidate_plan.output_dir)
    cand_dir.mkdir(parents=True, exist_ok=True)
    path = Path(out_path) if out_path else cand_dir / "compare_vs_baseline.md"

    base_dir = Path(baseline_plan.output_dir)
    if baseline_metrics is None and base_dir.is_dir():
        try:
            baseline_metrics = collect_training_metrics(base_dir, run_id=baseline_plan.run_id)
        except Exception:
            baseline_metrics = None
    if candidate_metrics is None and cand_dir.is_dir():
        try:
            candidate_metrics = collect_training_metrics(cand_dir, run_id=candidate_plan.run_id)
        except Exception:
            candidate_metrics = None

    compare = None
    if baseline_metrics is not None and candidate_metrics is not None:
        if baseline_metrics.summary or baseline_metrics.means:
            if candidate_metrics.summary or candidate_metrics.means:
                compare = compare_training_metrics(baseline_metrics, candidate_metrics)

    lines = [
        f"# 对比报告 — `{candidate_plan.run_id}` vs `{baseline_plan.run_id}`",
        "",
        f"- **UTC**：{datetime.now(timezone.utc).isoformat()}",
        f"- **基线启用**：{', '.join(baseline_plan.enabled)}",
        f"- **方案启用**：{', '.join(candidate_plan.enabled)}",
        f"- **基线 dropped defaults**：{baseline_plan.dropped_project_defaults or '无'}",
        f"- **方案 dropped defaults**：{candidate_plan.dropped_project_defaults or '无'}",
        "",
        "## 训练期对比",
        "",
    ]
    if compare:
        lines += [
            "| 指标 | 基线 | 方案 | Δ | 相对 |",
            "|------|------|------|---|------|",
        ]
        for row in compare["rows"]:
            rel = row.get("rel")
            rel_s = f"{rel * 100:.1f}%" if isinstance(rel, float) else "—"
            lines.append(
                f"| `{row['metric']}` | {row.get('baseline')} | {row.get('candidate')} | "
                f"{row.get('delta')} | {rel_s} |"
            )
    else:
        lines += [
            "> 缺少一侧训练指标时无法自动算 Δ。请先对两边 `evaluate_run`，或完成开训。",
            "",
            "| 指标 | 基线 | 方案 | Δ |",
            "|------|------|------|---|",
            "| train_steps_per_second | | | |",
            "| reward | | | |",
            "| accuracy_reward | | | |",
        ]

    lines += [
        "",
        "## 配置差异（方案）",
        "",
        "```json",
        json.dumps(candidate_plan.mixup.to_dict(), indent=2, ensure_ascii=False),
        "```",
        "",
        "## 可选四基准 Δ",
        "",
        "未跑总评估则留空。见 `docs/PHASE5_BENCHMARKS.md`。",
        "",
        "| 基准 | 基线 | 方案 | Δ |",
        "|------|------|------|---|",
        "| Video-MME | | | |",
        "| MVBench | | | |",
        "| MLVU | | | |",
        "| MMVU | | | |",
        "",
        "## 结论（填写）",
        "",
        "- 相对基线：效率 …；reward/acc …",
        "- 是否保留该组合 / 是否值得下四基准：",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    # also dump compare json if available
    if compare:
        (cand_dir / "compare_vs_baseline.json").write_text(
            json.dumps(compare, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return path


def refresh_training_eval(run_dir: PathLike, baseline_dir: Optional[PathLike] = None) -> TrainingMetrics:
    """Thin wrapper for console cells."""
    return evaluate_run(run_dir, baseline_dir=baseline_dir)
