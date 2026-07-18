"""Training entry for Phase 4/6 — prepare MixUp run, optionally launch later.

``prepare_training`` wraps the selector + repo patch. Actual DeepSpeed launch
stays in Colab Notebooks (Phase1-style ``run_training``) so Phase 4 can finish
without A100 time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

from .config import MixUpConfig
from .selector import RunPlan, apply_plan_to_notebook, prepare_run

PathLike = Union[str, Path]


def apply_mixup(
    mixup: Optional[Mapping[str, Any]] = None,
    *,
    preset: Optional[str] = None,
    mixup_repo: Optional[PathLike] = None,
    tinyllava_repo: Optional[PathLike] = None,
    apply_patch: bool = True,
    **prepare_kwargs: Any,
) -> RunPlan:
    """Convenience: build plan from toggles/preset and patch upstream trainer.

    Example::

        from mixup.training_entry import apply_mixup
        plan = apply_mixup({"ngrpo": True}, mixup_repo=MIXUP, tinyllava_repo=REPO)
        print(plan.summary())
    """
    return prepare_run(
        preset=preset,
        mixup=mixup,
        mixup_repo=mixup_repo,
        tinyllava_repo=tinyllava_repo,
        apply_patch=apply_patch and mixup_repo is not None and tinyllava_repo is not None,
        **prepare_kwargs,
    )


def prepare_training(
    *,
    preset: str = "m1",
    mixup: Optional[Mapping[str, Any]] = None,
    project_dir: Optional[PathLike] = None,
    mixup_repo: Optional[PathLike] = None,
    tinyllava_repo: Optional[PathLike] = None,
    namespace: Optional[MutableMapping[str, Any]] = None,
    apply_patch: bool = False,
    dry_run: bool = False,
    tag: str = "",
    notes: str = "",
    tier_overrides: Optional[Mapping[str, Any]] = None,
) -> RunPlan:
    """Full Phase-4 entry used by Notebook / future console.

    1. Resolve preset + overlays (B+CPPO default)
    2. Write ``outputs/{run_id}/`` snapshot + report stub
    3. Optionally patch TinyLLaVA trainer
    4. Optionally inject constants into ``globals()``
    """
    if project_dir is not None and mixup_repo is None:
        mixup_repo = Path(project_dir) / "repo" / "MixUpLLaVA-Video-R1"
    if project_dir is not None and tinyllava_repo is None:
        tinyllava_repo = Path(project_dir) / "repo" / "TinyLLaVA-Video-R1"
    if project_dir is not None and "out_base" not in (tier_overrides or {}):
        tier_overrides = dict(tier_overrides or {})
        # out_base override via select_strategy out_base arg
        pass

    out_base = None
    if project_dir is not None:
        out_base = Path(project_dir) / "outputs"

    plan = prepare_run(
        preset=preset,
        mixup=mixup,
        project_dir=project_dir,
        out_base=out_base,
        mixup_repo=mixup_repo,
        tinyllava_repo=tinyllava_repo,
        apply_patch=apply_patch,
        dry_run=dry_run,
        tag=tag,
        notes=notes,
        tier_overrides=tier_overrides,
    )
    if namespace is not None:
        apply_plan_to_notebook(plan, namespace)
    return plan


def describe_launch(plan: RunPlan) -> str:
    """Human-readable next steps for Colab (no subprocess)."""
    return (
        f"# Next (Colab A100, when ready to train)\n"
        f"# 1. Confirm patch applied / mixup_config.json present\n"
        f"# 2. OUT = {plan.output_dir}\n"
        f"# 3. Reuse Phase1 run_training(OUT, run_name=plan.run_id, ...)\n"
        f"# 4. Fill {plan.report_stub_path}\n"
        f"# enabled: {plan.enabled}\n"
    )
