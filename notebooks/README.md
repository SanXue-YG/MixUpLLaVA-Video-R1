# notebooks/ — Colab 入口

| Notebook | 用途 |
|----------|------|
| **`phase1-cppo-g8.ipynb`** | Phase 1：A2 / C2（g=8）效率对比 |
| **`phase4-selector.ipynb`** | **Phase 4 选择器**：预设/开关 → `run_id` + 快照（开训可延后） |
| `mixup-GRPO优化.ipynb` | 与 phase1 相同内容（兼容旧文件名） |

## Phase 4：选择器（推荐）

```python
from mixup.training_entry import prepare_training
plan = prepare_training(
    preset="m1",                 # 或 m5 / m5q / speed
    mixup={"ngrpo": True},       # 叠在默认 B+CPPO 上
    project_dir=PROJECT_DIR,
    apply_patch=False,           # Colab 确认路径后再 True
)
print(plan.summary())
```

报告模板：[`docs/PHASE4_REPORT.md`](../docs/PHASE4_REPORT.md)。

## Phase 2：默认档位

后续实验请以仓库 **`configs/colab_c1.yaml`** 为默认（与 Phase1 常量对齐）。换环境可覆盖，勿与 Phase1 默认档混比。

```python
from pathlib import Path
import sys
MIXUP_REPO = Path("/content/drive/MyDrive/MixUpLLaVA-video-r1/repo/MixUpLLaVA-Video-R1")
sys.path.insert(0, str(MIXUP_REPO))

from mixup.config_loader import load_tier_config, apply_config_to_notebook

cfg = load_tier_config(
    MIXUP_REPO / "configs/colab_c1.yaml",
    overrides={},  # 例: {"max_steps": 20} 或 {"num_generations": 4}
)
apply_config_to_notebook(cfg, globals())
```

- 档位说明：[`configs/README.md`](../configs/README.md)  
- 换环境清单 / Phase1 吞吐：[`docs/MEMORY_BENCHMARK_COLAB.md`](../docs/MEMORY_BENCHMARK_COLAB.md)

## Phase 1 流程摘要

1. 挂载 Drive → 确认 A100  
2. `PROJECT_DIR=/content/drive/MyDrive/MixUpLLaVA-video-r1`  
3. Phase 0 验收 + `git clone/pull` MixUp 仓库  
4. 小数据集 → 装依赖 → 从 `mixup/patches` 覆盖 trainer  
5. 重跑 §6（进度条 + 断点）→ 训练 **A2** → **C2** → 对比（可粘贴 schedule §8）  

## 断点续训 / 进度

- `SAVE_STEPS=10`：约每 10 step 写入 `checkpoint-*`（`SAVE_TOTAL_LIMIT=1` 只留最新）  
- `ENABLE_RESUME=True`：有 checkpoint 时自动续训（如 A2 从 `checkpoint-40` → 50）  
- 日志流式输出 + 简易进度条 / ETA  
- 断连后：按重连顺序跑完 §0–§6，再跑对应训练单元；**清空 Drive 回收站** 避免占配额  

共享包：https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing  

计划详见仓库根目录 `schedule.md`。
