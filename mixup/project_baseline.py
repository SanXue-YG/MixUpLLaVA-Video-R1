"""Project baseline enhancement (strategy B style) — stub for Phase 3.

Planned pieces (from survey report §2.3 / prior Colab strategy B):
  - milder reward shaping (accuracy + format coupling)
  - advantage clamp instead of additive noise
  - slightly larger KL beta

Full trainer reference: ``mixup/patches/tinyllava_trainer_reason_strategy_b.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProjectBaselineConfig:
    beta: float = 0.02
    advantage_clamp: float = 3.0
    use_mild_reward: bool = True


def describe() -> str:
    return (
        "project_baseline: difficulty/length-aware reward shaping + "
        "advantage clamp + beta↑ (see patches/strategy_b)"
    )
