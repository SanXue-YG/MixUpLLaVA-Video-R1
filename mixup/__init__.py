"""MixUp GRPO strategy modules for TinyLLaVA-Video-R1.

These modules are designed to be composed via ``registry`` and applied
to the upstream ``tinyllava_trainer_reason.py`` (on Google Drive REPO)
during Colab training.
"""

from .registry import MixUpConfig, enabled_strategies
from .config_loader import (
    apply_config_to_notebook,
    default_c1_path,
    load_tier_config,
    notebook_constants,
)

__all__ = [
    "MixUpConfig",
    "enabled_strategies",
    "load_tier_config",
    "notebook_constants",
    "apply_config_to_notebook",
    "default_c1_path",
]
