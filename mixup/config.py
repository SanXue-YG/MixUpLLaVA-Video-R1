"""MixUpConfig schema: boolean toggles + hyperparameters for each module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional


KNOWN_BOOL_FLAGS = (
    "project_baseline",
    "cppo",
    "gfpo",
    "ngrpo",
    "mo_grpo",
    "dagrpo",
    "gmpo",
    "mapo",
    "grpo_a",
    "step_grpo",
)

# 项目组默认保留模块：B（自研质量向）+ CPPO（原定效率目标）。
# 后续加开其它策略时二者默认保持开启；仅当配置里显式写 false 时才关闭。
PROJECT_DEFAULT_FLAGS = ("project_baseline", "cppo")


@dataclass
class MixUpConfig:
    """Toggle MixUp strategies for a training run.

    **项目默认**：``project_baseline=True`` 且 ``cppo=True``（除非显式关闭）。
    加开 NGRPO/GFPO/… 时不必重复写 B/CPPO；要做消融再设 ``False``。

    Example::

        cfg = MixUpConfig()  # == M1：B + CPPO
        cfg = MixUpConfig.from_dict({"ngrpo": True})  # B+CPPO+NGRPO
        cfg = MixUpConfig(project_baseline=False, cppo=True)  # 显式去掉 B（如纯测 CPPO）
    """

    # --- P0（项目默认开启）---
    project_baseline: bool = True
    baseline_lambda_diff: float = 1.0
    baseline_length_coef: float = 0.1
    baseline_length_tau: float = 32.0
    baseline_L_easy: float = 64.0
    baseline_L_hard: float = 256.0
    baseline_advantage_clamp: float = 3.0
    baseline_beta: float = 0.02
    baseline_disable_noise: bool = True

    cppo: bool = True
    cppo_pruning_rate: float = 0.5

    # --- P1 ---
    gfpo: bool = False
    gfpo_top_k: int = 4
    gfpo_metric: str = "shortest"  # shortest | token_efficiency

    ngrpo: bool = False
    ngrpo_r_max: float = 2.0
    ngrpo_eps_pos: float = 0.24
    ngrpo_eps_neg: float = 0.16
    ngrpo_apply_asym_clip: bool = False  # needs real importance ratio ρ

    mo_grpo: bool = False

    # --- P2 (implemented or clear stub) ---
    dagrpo: bool = False
    dagrpo_delta: float = 0.1

    gmpo: bool = False
    gmpo_eps: float = 0.2

    mapo: bool = False

    grpo_a: bool = False
    grpo_a_alpha: float = 0.5
    grpo_a_ell: int = 100

    # --- reserved ---
    step_grpo: bool = False
    step_grpo_alpha: float = 0.5

    # trainer memory profile (also in tier yaml)
    num_generations: int = 8
    num_frame: int = 8

    notes: str = ""
    # unknown keys collected from from_dict for clearer errors
    _unknown: List[str] = field(default_factory=list, repr=False)

    def enabled(self) -> List[str]:
        names: List[str] = []
        if self.project_baseline:
            names.append("project_baseline")
        if self.cppo:
            names.append(f"cppo(p={self.cppo_pruning_rate})")
        if self.gfpo:
            names.append(f"gfpo(k={self.gfpo_top_k},{self.gfpo_metric})")
        if self.ngrpo:
            names.append("ngrpo")
        if self.mo_grpo:
            names.append("mo_grpo")
        if self.dagrpo:
            names.append("dagrpo")
        if self.gmpo:
            names.append("gmpo")
        if self.mapo:
            names.append("mapo")
        if self.grpo_a:
            names.append("grpo_a")
        if self.step_grpo:
            names.append("step_grpo")
        return names or ["vanilla_grpo"]

    def dropped_project_defaults(self) -> List[str]:
        """Return which of B/CPPO were explicitly turned off (for reports)."""
        dropped = []
        if not self.project_baseline:
            dropped.append("project_baseline")
        if not self.cppo:
            dropped.append("cppo")
        return dropped

    def ensure_project_defaults(self) -> "MixUpConfig":
        """Force-enable B+CPPO (e.g. after an ablation preset). Returns self."""
        self.project_baseline = True
        self.cppo = True
        return self

    @classmethod
    def project_baseline_config(cls, **overrides: Any) -> "MixUpConfig":
        """M1 / 选择器默认：B + CPPO，再叠加 ``overrides``（如 ngrpo=True）。"""
        return cls(**overrides)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_unknown", None)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, strict: bool = True) -> "MixUpConfig":
        """Build config from a plain dict (Notebook / YAML mixup section).

        Unspecified flags keep **dataclass defaults**（故 B/CPPO 默认为 True）。
        仅当 dict/YAML 里显式写 ``project_baseline: false`` / ``cppo: false`` 时关闭。

        Aliases: ``gfpo_topk`` → ``gfpo_top_k``.
        Unknown keys raise if ``strict`` (default).
        """
        raw = dict(data)
        if "gfpo_topk" in raw and "gfpo_top_k" not in raw:
            raw["gfpo_top_k"] = raw.pop("gfpo_topk")
        known = {f.name for f in fields(cls) if f.name != "_unknown"}
        unknown = [k for k in raw if k not in known]
        if unknown and strict:
            raise KeyError(
                f"Unknown MixUpConfig key(s): {unknown}. "
                f"Known flags: {list(KNOWN_BOOL_FLAGS)}; "
                f"see mixup.config.MixUpConfig fields. "
                f"Project defaults (on unless false): {list(PROJECT_DEFAULT_FLAGS)}."
            )
        filtered = {k: v for k, v in raw.items() if k in known}
        cfg = cls(**filtered)
        cfg._unknown = unknown
        return cfg

    def validate(self) -> None:
        if self.cppo and not (0.0 <= self.cppo_pruning_rate < 1.0):
            raise ValueError("cppo_pruning_rate must be in [0, 1)")
        if self.gfpo and self.gfpo_top_k < 1:
            raise ValueError("gfpo_top_k must be >= 1")
        if self.gfpo_metric not in ("shortest", "token_efficiency"):
            raise ValueError("gfpo_metric must be 'shortest' or 'token_efficiency'")
        if self.grpo_a and not (0.0 < self.grpo_a_alpha <= 1.0):
            raise ValueError("grpo_a_alpha must be in (0, 1]")


def merge_mixup_section(tier_cfg: Mapping[str, Any]) -> MixUpConfig:
    """Build MixUpConfig from tier yaml ``mixup:`` section (+ optional training gens)."""
    section = dict(tier_cfg.get("mixup") or {})
    training = tier_cfg.get("training") or {}
    if "num_generations" in training and "num_generations" not in section:
        section["num_generations"] = training["num_generations"]
    if "num_frame_trainer" in training and "num_frame" not in section:
        section["num_frame"] = training["num_frame_trainer"]
    return MixUpConfig.from_dict(section, strict=False)
