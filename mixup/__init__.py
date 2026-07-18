"""MixUp GRPO strategy modules for TinyLLaVA-Video-R1."""

from .config import MixUpConfig, PROJECT_DEFAULT_FLAGS, merge_mixup_section
from .config_loader import (
    apply_config_to_notebook,
    default_c1_path,
    load_tier_config,
    notebook_constants,
)
from .registry import (
    apply_registered,
    enabled_strategies,
    get_module,
    list_modules,
    register,
)
from .trainer_mixup import (
    apply_mixup_to_repo,
    ensure_modules_loaded,
    run_advantage_pipeline,
    snapshot_config,
)

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
]
