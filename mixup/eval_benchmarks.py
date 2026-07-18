"""Optional full-benchmark evaluation (Video-MME / MVBench / MLVU / MMVU).

Aligned with upstream TinyLLaVA-Video-R1:
https://github.com/ZhangXJ199/TinyLLaVA-Video-R1

**Not required for Phase 5 acceptance.** Four benchmarks together are on the
order of **~600GB**; only run after explicitly downloading data.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]

BENCHMARKS = ("videomme", "mvbench", "mlvu", "mmvu")

# Relative layout under EVAL_ROOT (experimenter-configurable)
DEFAULT_LAYOUT = {
    "videomme": "Video-MME",
    "mvbench": "MVBench",
    "mlvu": "MLVU",
    "mmvu": "MMVU",
}

SPACE_WARNING = (
    "Full Video-MME + MVBench + MLVU + MMVU is ~600GB. "
    "Prefer training-period eval (mixup.eval_training) for small-scale work. "
    "See docs/PHASE5_BENCHMARKS.md."
)


@dataclass
class BenchmarkStatus:
    name: str
    data_dir: str
    ready: bool
    detail: str = ""


@dataclass
class BenchmarkPlan:
    eval_root: str
    model_path: str
    model_name: str
    tinyllava_repo: str
    benchmarks: List[str] = field(default_factory=lambda: list(BENCHMARKS))
    statuses: List[BenchmarkStatus] = field(default_factory=list)
    space_warning: str = SPACE_WARNING

    def any_ready(self) -> bool:
        return any(s.ready for s in self.statuses)

    def to_dict(self) -> dict:
        return {
            "eval_root": self.eval_root,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "tinyllava_repo": self.tinyllava_repo,
            "benchmarks": self.benchmarks,
            "statuses": [s.__dict__ for s in self.statuses],
            "space_warning": self.space_warning,
        }


def check_benchmark_data(eval_root: PathLike, name: str) -> BenchmarkStatus:
    """Heuristic readiness check (directory exists and is non-empty)."""
    root = Path(eval_root)
    sub = root / DEFAULT_LAYOUT.get(name, name)
    if not sub.exists():
        return BenchmarkStatus(name, str(sub), False, "missing directory")
    # non-empty
    try:
        next(sub.iterdir())
        return BenchmarkStatus(name, str(sub), True, "present")
    except StopIteration:
        return BenchmarkStatus(name, str(sub), False, "empty directory")


def build_benchmark_plan(
    *,
    eval_root: PathLike,
    model_path: PathLike,
    tinyllava_repo: PathLike,
    model_name: Optional[str] = None,
    benchmarks: Optional[Sequence[str]] = None,
) -> BenchmarkPlan:
    benches = list(benchmarks) if benchmarks else list(BENCHMARKS)
    for b in benches:
        if b not in BENCHMARKS:
            raise KeyError(f"Unknown benchmark {b!r}. Known: {BENCHMARKS}")
    plan = BenchmarkPlan(
        eval_root=str(eval_root),
        model_path=str(model_path),
        model_name=model_name or Path(model_path).name,
        tinyllava_repo=str(tinyllava_repo),
        benchmarks=benches,
    )
    plan.statuses = [check_benchmark_data(eval_root, b) for b in benches]
    return plan


def script_path(mixup_repo: PathLike, name: str) -> Path:
    return Path(mixup_repo) / "scripts" / "eval" / f"run_{name}.sh"


def run_benchmark(
    name: str,
    *,
    mixup_repo: PathLike,
    tinyllava_repo: PathLike,
    model_path: PathLike,
    eval_root: PathLike,
    model_name: Optional[str] = None,
    dry_run: bool = True,
    confirm_large_download: bool = False,
) -> Dict[str, object]:
    """Invoke wrapper shell that calls upstream ``tinyllava.eval.*``.

    Default ``dry_run=True`` — only prints the command. Set dry_run=False and
    ensure data is present before a real run.
    """
    if name not in BENCHMARKS:
        raise KeyError(name)
    if not confirm_large_download and not dry_run:
        raise RuntimeError(
            SPACE_WARNING
            + " Pass confirm_large_download=True after data is on disk."
        )

    plan = build_benchmark_plan(
        eval_root=eval_root,
        model_path=model_path,
        tinyllava_repo=tinyllava_repo,
        model_name=model_name,
        benchmarks=[name],
    )
    st = plan.statuses[0]
    sh = script_path(mixup_repo, name)
    env = os.environ.copy()
    env.update(
        {
            "MODEL_PATH": str(model_path),
            "MODEL_NAME": plan.model_name,
            "EVAL_DIR": st.data_dir,
            "TINYLLAVA_REPO": str(tinyllava_repo),
        }
    )
    cmd = ["bash", str(sh)]
    result: Dict[str, object] = {
        "benchmark": name,
        "ready": st.ready,
        "cmd": cmd,
        "env": {k: env[k] for k in ("MODEL_PATH", "MODEL_NAME", "EVAL_DIR", "TINYLLAVA_REPO")},
        "dry_run": dry_run,
        "warning": SPACE_WARNING,
    }
    if dry_run:
        result["status"] = "dry_run"
        return result
    if not st.ready:
        result["status"] = "skipped_missing_data"
        return result
    if not sh.is_file():
        raise FileNotFoundError(sh)
    proc = subprocess.run(
        cmd,
        cwd=str(tinyllava_repo),
        env=env,
        capture_output=True,
        text=True,
    )
    result["returncode"] = proc.returncode
    result["stdout_tail"] = proc.stdout[-2000:]
    result["stderr_tail"] = proc.stderr[-2000:]
    result["status"] = "ok" if proc.returncode == 0 else "failed"
    return result


def write_benchmark_plan(plan: BenchmarkPlan, out_path: PathLike) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
