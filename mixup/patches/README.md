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

## How to apply on Colab (Phase 1)

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

Do **not** stack strategy B reward changes on top of CPPO when measuring pure speedup.
