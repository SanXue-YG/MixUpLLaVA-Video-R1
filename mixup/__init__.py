"""MixUp GRPO strategy modules for TinyLLaVA-Video-R1.

These modules are designed to be composed via ``registry`` and applied
to the upstream ``tinyllava_trainer_reason.py`` (on Google Drive REPO)
during Colab training.
"""

from .registry import MixUpConfig, enabled_strategies

__all__ = ["MixUpConfig", "enabled_strategies"]
