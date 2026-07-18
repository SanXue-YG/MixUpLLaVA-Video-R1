"""Load Phase-2 tier YAML (C1/C0) and apply notebook overrides.

Example::

    from mixup.config_loader import load_tier_config, apply_config_to_notebook

    cfg = load_tier_config("configs/colab_c1.yaml", overrides={"max_steps": 20})
    apply_config_to_notebook(cfg, globals())
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union

PathLike = Union[str, Path]

# Flat keys accepted in ``overrides`` (alias → dotted path under loaded cfg).
_OVERRIDE_ALIASES: dict[str, tuple[str, ...]] = {
    "num_generations": ("training", "num_generations"),
    "num_frames": ("training", "num_frames"),
    "num_queries": ("training", "num_queries"),
    "model_max_length": ("training", "model_max_length"),
    "num_frame_trainer": ("training", "num_frame_trainer"),
    "max_steps": ("training", "max_steps"),
    "learning_rate": ("training", "learning_rate"),
    "save_steps": ("checkpointing", "save_steps"),
    "save_total_limit": ("checkpointing", "save_total_limit"),
    "logging_steps": ("checkpointing", "logging_steps"),
    "enable_resume": ("checkpointing", "enable_resume"),
    "project_dir": ("paths", "project_dir"),
    "data_root": ("paths", "data_root"),
    "video_folder": ("paths", "video_folder"),
    "ckpt": ("paths", "ckpt"),
    "out_base": ("paths", "out_base"),
    "dataset_name": ("paths", "dataset_name"),
    "cppo_pruning_rate": ("mixup", "cppo_pruning_rate"),
    "project_baseline": ("mixup", "project_baseline"),
    "cppo": ("mixup", "cppo"),
    "gfpo": ("mixup", "gfpo"),
    "gfpo_top_k": ("mixup", "gfpo_top_k"),
    "ngrpo": ("mixup", "ngrpo"),
    "mo_grpo": ("mixup", "mo_grpo"),
    "offload_param": ("deepspeed", "offload_param"),
    "offload_optimizer": ("deepspeed", "offload_optimizer"),
    "zero_stage": ("deepspeed", "zero_stage"),
}


def _parse_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s or s == "null" or s == "~":
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        if any(c in s for c in (".", "e", "E")):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal YAML subset loader (mappings + list-of-scalars) when PyYAML is absent."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_list_key: Optional[tuple[dict, str]] = None

    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if content.startswith("#"):
            continue

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ValueError(f"line {lineno}: invalid nesting")

        if content.startswith("- "):
            if pending_list_key is None:
                raise ValueError(f"line {lineno}: list item without key")
            d, key = pending_list_key
            if not isinstance(d.get(key), list):
                d[key] = []
            d[key].append(_parse_scalar(content[2:]))
            continue

        pending_list_key = None
        if ":" not in content:
            raise ValueError(f"line {lineno}: expected key:")
        key, _, rest = content.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest.startswith("#"):
            rest = ""
        elif " #" in rest:
            rest = rest.split(" #", 1)[0].strip()

        if rest == "":
            # nested mapping or list follows
            parent[key] = {}
            stack.append((indent, parent[key]))
            pending_list_key = (parent, key)
        else:
            parent[key] = _parse_scalar(rest)

    return root


def _load_yaml_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ImportError:
        data = _load_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def _deep_merge(base: dict, overlay: Mapping[str, Any]) -> dict:
    out = deepcopy(base)
    for k, v in overlay.items():
        if k == "extends":
            continue
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _set_path(cfg: dict, path: tuple[str, ...], value: Any) -> None:
    cur: dict = cfg
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _resolve_extends(path: Path, raw: dict) -> dict:
    parent_name = raw.get("extends")
    if not parent_name:
        return raw
    parent_path = (path.parent / parent_name).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"extends target not found: {parent_path}")
    parent = load_tier_config(parent_path, overrides=None, _resolving=True)
    merged = _deep_merge(parent, raw)
    merged.pop("extends", None)
    return merged


def load_tier_config(
    path: PathLike,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    _resolving: bool = False,
) -> dict[str, Any]:
    """Load a tier YAML; optionally apply flat or nested ``overrides``.

    ``overrides`` may use short aliases (``max_steps``, ``num_generations``, …)
    or nested dicts matching the YAML structure (``training``, ``mixup``, …).
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    raw = _load_yaml_file(path)

    if not _resolving:
        cfg = _resolve_extends(path, raw)
    else:
        cfg = deepcopy(raw)
        cfg.pop("extends", None)

    if overrides:
        nested: dict[str, Any] = {}
        flat: dict[str, Any] = {}
        for k, v in overrides.items():
            if k in ("training", "paths", "checkpointing", "deepspeed", "mixup", "notes"):
                nested[k] = v
            else:
                flat[k] = v
        if nested:
            cfg = _deep_merge(cfg, nested)
        for k, v in flat.items():
            if k not in _OVERRIDE_ALIASES:
                raise KeyError(
                    f"Unknown override key {k!r}. "
                    f"Known: {sorted(_OVERRIDE_ALIASES)}"
                )
            _set_path(cfg, _OVERRIDE_ALIASES[k], v)

    cfg["_config_path"] = str(path)
    return cfg


def notebook_constants(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Map loaded config → Phase1-style uppercase names for Notebook cells."""
    t = cfg.get("training") or {}
    c = cfg.get("checkpointing") or {}
    p = cfg.get("paths") or {}
    m = cfg.get("mixup") or {}
    return {
        "tier": cfg.get("tier", "C1"),
        "PROJECT_DIR": p.get("project_dir"),
        "DATA_ROOT": p.get("data_root"),
        "VIDEO_FOLDER": p.get("video_folder"),
        "CKPT": p.get("ckpt"),
        "OUT_BASE": p.get("out_base"),
        "NUM_GENERATIONS": t.get("num_generations"),
        "NUM_FRAMES": t.get("num_frames"),
        "NUM_QUERIES": t.get("num_queries"),
        "MODEL_MAX_LENGTH": t.get("model_max_length"),
        "NUM_FRAME_TRAINER": t.get("num_frame_trainer"),
        "MAX_STEPS": t.get("max_steps"),
        "LEARNING_RATE": t.get("learning_rate"),
        "SAVE_STEPS": c.get("save_steps"),
        "SAVE_TOTAL_LIMIT": c.get("save_total_limit"),
        "LOGGING_STEPS": c.get("logging_steps"),
        "ENABLE_RESUME": c.get("enable_resume"),
        "CPPO_PRUNING_RATE": m.get("cppo_pruning_rate"),
        "MIXUP": dict(m),
    }


def apply_config_to_notebook(
    cfg: Mapping[str, Any],
    namespace: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Write notebook constants into ``globals()`` (or any mapping). Returns the dict applied."""
    constants = notebook_constants(cfg)
    namespace.update(constants)
    return constants


def default_c1_path(repo_root: Optional[PathLike] = None) -> Path:
    """Resolve ``configs/colab_c1.yaml`` relative to repo root or this package."""
    if repo_root is not None:
        return Path(repo_root).resolve() / "configs" / "colab_c1.yaml"
    # mixup/ -> repo root
    return Path(__file__).resolve().parent.parent / "configs" / "colab_c1.yaml"
