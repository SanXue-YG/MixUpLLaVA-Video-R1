"""Strategy registry for MixUp combinations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MixUpConfig:
    """Toggle MixUp strategies for a training run.

    Example:
        cfg = MixUpConfig(cppo=True, cppo_pruning_rate=0.5, num_generations=8)
    """

    # A. project baseline enhancement (strategy B style)
    project_baseline: bool = False

    # B. CPPO completion pruning
    cppo: bool = False
    cppo_pruning_rate: float = 0.5

    # C / D (Phase 3+)
    gfpo: bool = False
    gfpo_topk: int = 2
    ngrpo: bool = False

    # trainer memory profile
    num_generations: int = 8
    num_frame: int = 8

    notes: str = ""

    def enabled(self) -> List[str]:
        names = []
        if self.project_baseline:
            names.append("project_baseline")
        if self.cppo:
            names.append(f"cppo(p={self.cppo_pruning_rate})")
        if self.gfpo:
            names.append(f"gfpo(k={self.gfpo_topk})")
        if self.ngrpo:
            names.append("ngrpo")
        return names or ["baseline"]


def enabled_strategies(cfg: MixUpConfig) -> List[str]:
    return cfg.enabled()
