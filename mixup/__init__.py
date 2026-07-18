"""MixUp GRPO strategy modules for TinyLLaVA-Video-R1."""

from .config import MixUpConfig, PROJECT_DEFAULT_FLAGS, merge_mixup_section
from .config_loader import (
    apply_config_to_notebook,
    default_c1_path,
    load_tier_config,
    notebook_constants,
)
from .eval_benchmarks import build_benchmark_plan, run_benchmark
from .eval_training import (
    collect_training_metrics,
    compare_training_metrics,
    evaluate_run,
    write_training_eval_report,
)
from .presets import PRESET_ALIASES, list_presets, load_preset
from .registry import (
    apply_registered,
    enabled_strategies,
    get_module,
    list_modules,
    register,
)
from .selector import (
    RunPlan,
    apply_plan_to_notebook,
    build_mixup_config,
    list_selector_presets,
    prepare_run,
    select_strategy,
)
from .trainer_mixup import (
    apply_mixup_to_repo,
    ensure_modules_loaded,
    run_advantage_pipeline,
    snapshot_config,
)
from .training_entry import apply_mixup, describe_launch, prepare_training

__all__ = [
    "MixUpConfig",
    "PROJECT_DEFAULT_FLAGS",
    "merge_mixup_section",
    "load_tier_config",
    "notebook_constants",
    "apply_config_to_notebook",
    "default_c1_path",
    "register",
    "get_module",
    "list_modules",
    "apply_registered",
    "enabled_strategies",
    "ensure_modules_loaded",
    "run_advantage_pipeline",
    "apply_mixup_to_repo",
    "snapshot_config",
    "RunPlan",
    "select_strategy",
    "prepare_run",
    "build_mixup_config",
    "apply_plan_to_notebook",
    "list_selector_presets",
    "apply_mixup",
    "prepare_training",
    "describe_launch",
    "PRESET_ALIASES",
    "load_preset",
    "list_presets",
    "collect_training_metrics",
    "compare_training_metrics",
    "evaluate_run",
    "write_training_eval_report",
    "build_benchmark_plan",
    "run_benchmark",
]
