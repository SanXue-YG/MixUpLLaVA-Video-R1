# Trainer patches (reference copies)

These files are **drop-in / reference** trainers from prior Colab experiments.
On Google Drive, the live training file is:

```text
{PROJECT_DIR}/repo/TinyLLaVA-Video-R1/tinyllava/train/tinyllava_trainer_reason.py
```

| File | Role |
|------|------|
| `tinyllava_trainer_reason_baseline.py` | Strategy A — original GRPO (no CPPO) |
| `tinyllava_trainer_reason_cppo.py` | CPPO-ready trainer (`cppo_pruning_rate`, prune then forward) |
| `tinyllava_trainer_reason_strategy_b.py` | Milder reward + advantage clamp + beta=0.02 |
| **`tinyllava_trainer_reason_mixup.py`** | **Phase 3 MixUp**：读 `mixup_config.json`，调用 `run_advantage_pipeline` |

## How to apply on Colab (Phase 1 CPPO-only)

```python
import shutil, os
REPO = "/content/drive/MyDrive/MixUpLLaVA-video-r1/repo/TinyLLaVA-Video-R1"
MIXUP = "/content/drive/MyDrive/MixUpLLaVA-video-r1/repo/MixUpLLaVA-Video-R1"
dst = os.path.join(REPO, "tinyllava/train/tinyllava_trainer_reason.py")

# Baseline (A2)
shutil.copy(
    os.path.join(MIXUP, "mixup/patches/tinyllava_trainer_reason_baseline.py"),
    dst,
)

# Or CPPO (C2) — then set cppo_pruning_rate=0.5 in __init__ / Notebook patch
shutil.copy(
    os.path.join(MIXUP, "mixup/patches/tinyllava_trainer_reason_cppo.py"),
    dst,
)
```

## How to apply MixUp (Phase 3+)

```python
from mixup import apply_mixup_to_repo
from mixup.presets import load_preset

cfg = load_preset("m1_baseline_b")  # or speed_cppo / m5q_quality / ...
apply_mixup_to_repo(MIXUP, REPO, cfg)
```

Do **not** stack strategy B reward changes on top of CPPO when measuring pure speedup
（测纯 CPPO 加速时用 `speed_cppo` 预设，关闭 `project_baseline`）。
