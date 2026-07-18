# 换环境改参清单 + Phase1 实测吞吐（Phase 2）

- **默认档**：[configs/colab_c1.yaml](../configs/colab_c1.yaml)（= Phase1 已验证）  
- **约定**：与 Phase1 A2/C2 并表 → 保持 C1；其它环境可改参，**单独成组**  
- **加载**：见 [configs/README.md](../configs/README.md) / `mixup.config_loader`

---

## 1. Phase1 实测（Colab A100-80GB · C1 · 50 step · small50）

| 策略 | `train_runtime` | `steps/s` | Cell 墙钟 | reward 均值（报告） |
|------|-----------------|-----------|-----------|-------------------|
| A2 GRPO（p=0） | 6357 s | **0.008** | ≈113 min | 0.190 |
| C2 CPPO（p=0.5） | 4616 s | **0.011**（相对 A2 **+37.5%**） | ≈83 min | 0.321 |

去存盘尖峰校正：时长约 **−32%** / 吞吐约 **+47%**（详见 [`PHASE1_REPORT.md`](./PHASE1_REPORT.md)）。

**显存**：Phase1 未单独记录峰值 GB；A100-80GB + ZeRO-3 **参数 offload**、`g=8`、`frames=2`、`queries=32`、`max_length=256` 可稳定跑完。若日志曾出现接近满显存，优先降 `num_generations`（→ C0）而非先动数据规模。

**存盘成本**：`SAVE_STEPS=10` 时约每 10 step 写 ZeRO-3 checkpoint，单次额外约 **2–2.5 min**（Drive）；`SAVE_TOTAL_LIMIT=1` 并及时清空回收站。

---

## 2. 换部署环境时怎么改（清单）

按顺序排查，改完在报告写清 **环境 + 改动项 + 档位名**：

| 症状 / 目标 | 优先改什么 | 建议值 | 备注 |
|-------------|------------|--------|------|
| OOM | `num_generations` | 8 → **4**（C0） | 最快降显存 |
| 仍 OOM | `num_frames` / `num_queries` / `model_max_length` | 再降或保持 2/32/256 | 升参须单独成组 |
| 仍 OOM | DeepSpeed | 保持 param offload；确认 batch=1 | 勿盲目关 offload |
| 冒烟太慢 | `max_steps` | 10–20 | 不作严肃对比 |
| 显存很富余 | `num_generations` 或 frames/queries | 试 16 或 C2 挑战档 | **单独成表** |
| 多卡 / 更大显存 | `offload_param` | `none` 可能更快 | 单独记 |
| 扩数据 | `video_folder` / dataset | 10% / 全量 | **新实验组** |
| 调 CPPO | `cppo_pruning_rate` | 0.3 / 0.5 / 0.7 | 与档位超参分开记 |

**推荐路径**：新环境先 **C1 冒烟**（可 `max_steps=10`）→ 再按需 C0 / 自定义。

---

## 3. 档位速查

| 档位 | g | frames | queries | max_length | 何时用 |
|------|---|--------|---------|------------|--------|
| **C1** | 8 | 2 | 32 | 256 | 默认；与 Phase1 可比 |
| C0 | 4 | 2 | 32 | 256 | 调试 / OOM |
| C2 挑战 | 8–16 | 2–4 | 32–64 | 512 | 显存有余；单独成表 |
| 自定义 | 自定 | … | … | … | 报告注明 |

---

## 4. Notebook 最小覆盖片段

```python
from pathlib import Path
import sys
MIXUP_REPO = Path("/content/drive/MyDrive/MixUpLLaVA-video-r1/repo/MixUpLLaVA-Video-R1")
sys.path.insert(0, str(MIXUP_REPO))

from mixup.config_loader import load_tier_config, apply_config_to_notebook

cfg = load_tier_config(
    MIXUP_REPO / "configs/colab_c1.yaml",
    overrides={"max_steps": 20},  # 例：仅缩短步数
)
apply_config_to_notebook(cfg, globals())
print(f"tier={tier} g={NUM_GENERATIONS} steps={MAX_STEPS}")
```

OOM 时改为 `configs/colab_c0.yaml`，或 `overrides={"num_generations": 4}`。
