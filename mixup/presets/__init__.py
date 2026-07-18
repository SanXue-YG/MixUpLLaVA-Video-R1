"""Load MixUp presets from ``mixup/presets/*.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

from ..config import MixUpConfig
from ..config_loader import _load_yaml_file

# Short aliases → preset yaml stem
PRESET_ALIASES: Dict[str, str] = {
    "m1": "m1_baseline_b",
    "default": "m1_baseline_b",
    "baseline": "m1_baseline_b",
    "speed": "speed_cppo",
    "cppo": "speed_cppo",
    "m5": "m5_b_cppo_ngrpo_gfpo",
    "m5q": "m5q_quality",
    "quality": "m5q_quality",
    "b_only": "ablation_b_only",
    "vanilla": "ablation_grpo_vanilla",
    "a2": "ablation_grpo_vanilla",
}


def presets_dir() -> Path:
    return Path(__file__).resolve().parent


def list_presets() -> list[str]:
    return sorted(p.stem for p in presets_dir().glob("*.yaml"))


def resolve_preset_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key in PRESET_ALIASES:
        return PRESET_ALIASES[key]
    available = list_presets()
    if key in available:
        return key
    raise KeyError(
        f"Unknown preset {name!r}. Aliases: {sorted(PRESET_ALIASES)}. "
        f"Files: {available}."
    )


def load_preset(name: str) -> MixUpConfig:
    """Load by stem or alias, e.g. ``m1`` / ``m1_baseline_b`` / ``m5q``."""
    stem = resolve_preset_name(name)
    path = presets_dir() / f"{stem}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Preset not found: {path}. Available: {list_presets()}"
        )
    raw = _load_yaml_file(path)
    return MixUpConfig.from_dict(raw, strict=False)


def load_preset_path(path: Union[str, Path]) -> MixUpConfig:
    raw = _load_yaml_file(Path(path))
    return MixUpConfig.from_dict(raw, strict=False)
