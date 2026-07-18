"""Phase 5 — training-period evaluation pipeline (primary).

Collects HF / GRPO training metrics from a run directory and compares runs.
Does **not** require Video-MME / MVBench / MLVU / MMVU (~600GB); those live in
``mixup.eval_benchmarks`` as an optional path.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]

# Keys we care about for MixUp reports
TRAIN_METRIC_KEYS = (
    "loss",
    "reward",
    "rewards/accuracy_reward",
    "rewards/format_reward",
    "reward_std",
    "kl",
    "completion_length",
    "learning_rate",
    "grad_norm",
)

SUMMARY_KEYS = (
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "train_loss",
    "epoch",
)


@dataclass
class TrainingMetrics:
    """Aggregated training-period metrics for one run."""

    run_dir: str
    run_id: str = ""
    n_steps: int = 0
    summary: Dict[str, float] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    last: Dict[str, float] = field(default_factory=dict)
    last_n_means: Dict[str, float] = field(default_factory=dict)
    last_n: int = 10
    history_len: int = 0
    sources: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _find_trainer_state(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "trainer_state.json"
    if direct.is_file():
        return direct
    # checkpoint-* dirs
    cands = sorted(run_dir.glob("checkpoint-*/trainer_state.json"))
    return cands[-1] if cands else None


def _parse_trainer_state(path: Path) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data.get("log_history") or []
    summary: Dict[str, float] = {}
    # last entry often has train_runtime etc.
    for row in reversed(history):
        for k in SUMMARY_KEYS:
            if k in row and k not in summary:
                try:
                    summary[k] = float(row[k])
                except (TypeError, ValueError):
                    pass
        if len(summary) >= 3:
            break
    return summary, list(history)


_DICT_LINE = re.compile(r"\{[^{}]*'(?:loss|reward|train_runtime)[^{}]*\}")


def _parse_log_text(text: str) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Best-effort parse of Python-dict log lines from training stdout."""
    history: List[Dict[str, Any]] = []
    summary: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not ((line.startswith("{") and "loss" in line) or "train_runtime" in line):
            # also allow embedded dicts
            m = _DICT_LINE.search(line)
            if not m:
                continue
            line = m.group(0)
        try:
            # logs use single quotes
            obj = eval(line, {"__builtins__": {}}, {})  # noqa: S307 — controlled training logs
        except Exception:
            try:
                obj = json.loads(line.replace("'", '"'))
            except Exception:
                continue
        if not isinstance(obj, dict):
            continue
        if "train_runtime" in obj:
            for k, v in obj.items():
                if isinstance(v, (int, float)):
                    summary[k] = float(v)
        elif "loss" in obj or "reward" in obj:
            history.append(obj)
    return summary, history


def _aggregate_history(
    history: List[Dict[str, Any]],
    *,
    last_n: int = 10,
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float], int]:
    buckets: Dict[str, List[float]] = {k: [] for k in TRAIN_METRIC_KEYS}
    for row in history:
        for k in TRAIN_METRIC_KEYS:
            if k in row and isinstance(row[k], (int, float)):
                buckets[k].append(float(row[k]))
    means: Dict[str, float] = {}
    last: Dict[str, float] = {}
    last_n_means: Dict[str, float] = {}
    for k, vals in buckets.items():
        m = _mean(vals)
        if m is not None:
            means[k] = m
        if vals:
            last[k] = vals[-1]
            ln = _mean(vals[-last_n:])
            if ln is not None:
                last_n_means[k] = ln
    n_steps = 0
    for row in history:
        if "epoch" in row or "loss" in row:
            n_steps += 1
    return means, last, last_n_means, n_steps


def collect_training_metrics(
    run_dir: PathLike,
    *,
    run_id: str = "",
    last_n: int = 10,
    log_glob: str = "*.log",
) -> TrainingMetrics:
    """Load metrics from ``trainer_state.json`` and/or training log files."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    sources: List[str] = []
    summary: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []

    state = _find_trainer_state(run_dir)
    if state is not None:
        summary, history = _parse_trainer_state(state)
        try:
            sources.append(str(state.relative_to(run_dir)))
        except ValueError:
            sources.append(str(state))

    # Also scan logs if history empty or to enrich summary
    for log_path in sorted(run_dir.glob(log_glob)):
        s2, h2 = _parse_log_text(log_path.read_text(encoding="utf-8", errors="ignore"))
        if s2:
            summary.update(s2)
            sources.append(log_path.name)
        if h2 and not history:
            history = h2
            sources.append(log_path.name)

    # all_results.json / train_results.json
    for name in ("all_results.json", "train_results.json", "trainer_results.json"):
        p = run_dir / name
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        summary[k] = float(v)
                sources.append(name)
            except Exception:
                pass

    means, last, last_n_means, n_steps = _aggregate_history(history, last_n=last_n)
    rid = run_id or run_dir.name
    notes = ""
    if not history and not summary:
        notes = "No trainer_state/log metrics found — run may be incomplete or path wrong."

    return TrainingMetrics(
        run_dir=str(run_dir),
        run_id=rid,
        n_steps=n_steps,
        summary=summary,
        means=means,
        last=last,
        last_n_means=last_n_means,
        last_n=last_n,
        history_len=len(history),
        sources=sources,
        notes=notes,
    )


def compare_training_metrics(
    baseline: TrainingMetrics,
    candidate: TrainingMetrics,
    *,
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute absolute / relative deltas (candidate − baseline)."""
    keys = list(keys) if keys else [
        "train_steps_per_second",
        "train_runtime",
        "reward",
        "rewards/accuracy_reward",
        "rewards/format_reward",
        "kl",
        "loss",
    ]
    rows = []
    for k in keys:
        b = baseline.summary.get(k, baseline.means.get(k))
        c = candidate.summary.get(k, candidate.means.get(k))
        if b is None and c is None:
            continue
        delta = None
        rel = None
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            delta = float(c) - float(b)
            if b != 0:
                rel = delta / float(b)
        rows.append({"metric": k, "baseline": b, "candidate": c, "delta": delta, "rel": rel})
    return {
        "baseline_run": baseline.run_id,
        "candidate_run": candidate.run_id,
        "rows": rows,
    }


def write_training_eval_report(
    metrics: TrainingMetrics,
    out_path: PathLike,
    *,
    compare: Optional[Dict[str, Any]] = None,
    title: str = "Phase 5 训练期评估",
) -> Path:
    """Write markdown report (+ sidecar JSON next to it)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")

    payload = metrics.to_dict()
    if compare:
        payload["compare"] = compare
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# {title} — `{metrics.run_id}`",
        "",
        f"- **run_dir**: `{metrics.run_dir}`",
        f"- **steps (log rows)**: {metrics.n_steps}",
        f"- **sources**: {', '.join(metrics.sources) or '—'}",
        f"- **notes**: {metrics.notes or '—'}",
        "",
        "## HF summary",
        "",
        "| 指标 | 值 |",
        "|------|----|",
    ]
    for k in SUMMARY_KEYS:
        if k in metrics.summary:
            lines.append(f"| `{k}` | {metrics.summary[k]} |")
    if not any(k in metrics.summary for k in SUMMARY_KEYS):
        lines.append("| — | （无） |")

    lines += [
        "",
        "## 训练过程均值 / 末值",
        "",
        f"| 指标 | 全程均值 | 末 {metrics.last_n} step 均值 | 末值 |",
        "|------|----------|------------------------------|------|",
    ]
    keys = sorted(set(metrics.means) | set(metrics.last) | set(metrics.last_n_means))
    for k in keys:
        lines.append(
            f"| `{k}` | {metrics.means.get(k, '—')} | "
            f"{metrics.last_n_means.get(k, '—')} | {metrics.last.get(k, '—')} |"
        )
    if not keys:
        lines.append("| — | — | — | — |")

    if compare:
        lines += [
            "",
            f"## 相对对照（{compare.get('candidate_run')} − {compare.get('baseline_run')}）",
            "",
            "| 指标 | baseline | candidate | Δ | 相对 |",
            "|------|----------|-----------|---|------|",
        ]
        for row in compare.get("rows") or []:
            rel = row.get("rel")
            rel_s = f"{rel*100:.1f}%" if isinstance(rel, float) else "—"
            lines.append(
                f"| `{row['metric']}` | {row.get('baseline')} | {row.get('candidate')} | "
                f"{row.get('delta')} | {rel_s} |"
            )

    lines += [
        "",
        "## 说明",
        "",
        "- 本报告为 **训练期评估**（快筛）。",
        "- **总评估**（Video-MME / MVBench / MLVU / MMVU）可选，数据约数百 GB，见 `docs/PHASE5_BENCHMARKS.md`。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def evaluate_run(
    run_dir: PathLike,
    *,
    baseline_dir: Optional[PathLike] = None,
    out_dir: Optional[PathLike] = None,
    run_id: str = "",
    last_n: int = 10,
) -> TrainingMetrics:
    """Collect metrics, optionally compare to baseline, write report under run or ``out_dir``."""
    metrics = collect_training_metrics(run_dir, run_id=run_id, last_n=last_n)
    compare = None
    if baseline_dir is not None:
        base = collect_training_metrics(baseline_dir)
        compare = compare_training_metrics(base, metrics)

    dest_dir = Path(out_dir) if out_dir else Path(run_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    write_training_eval_report(
        metrics,
        dest_dir / "training_eval_report.md",
        compare=compare,
    )
    return metrics
