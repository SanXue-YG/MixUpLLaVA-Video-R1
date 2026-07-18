"""Load MixUp presets from ``mixup/presets/*.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..config import MixUpConfig
from ..config_loader import _load_yaml_file


def presets_dir() -> Path:
    return Path(__file__).resolve().parent


def list_presets() -> list[str]:
    return sorted(p.stem for p in presets_dir().glob("*.yaml"))


def load_preset(name: str) -> MixUpConfig:
    """Load by stem, e.g. ``load_preset("m1_baseline_b")``."""
    path = presets_dir() / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Preset not found: {path}. Available: {list_presets()}"
        )
    raw = _load_yaml_file(path)
    return MixUpConfig.from_dict(raw, strict=False)


def load_preset_path(path: Union[str, Path]) -> MixUpConfig:
    raw = _load_yaml_file(Path(path))
    return MixUpConfig.from_dict(raw, strict=False)
