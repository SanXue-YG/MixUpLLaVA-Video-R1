# configs/ — 训练档位默认值（Phase 2）

| 文件 | 档位 | 用途 |
|------|------|------|
| **`colab_c1.yaml`** | **C1（默认）** | Phase1 已验证；推荐起点；与 A2/C2 同档可比 |
| `colab_c0.yaml` | C0 | 调试 / OOM / 小显存（`num_generations=4`） |

C2「挑战档」（g=8–16、frames=2–4、queries=32–64、`max_length=512`）不提供死锁 yaml：在 C1 上用 `overrides` 升参，**单独成组**记录。

## 约定

1. **默认 = C1**；其它部署环境可改参。  
2. 与 Phase1 并表 → 保持 C1 核心超参不变。  
3. 改档 / 换数据规模 → 新 `run_id`，报告单独成组。

## Notebook 覆盖示例

```python
from pathlib import Path
from mixup.config_loader import load_tier_config, apply_config_to_notebook

MIXUP_REPO = Path("/content/drive/MyDrive/MixUpLLaVA-video-r1/repo/MixUpLLaVA-Video-R1")
cfg = load_tier_config(
    MIXUP_REPO / "configs/colab_c1.yaml",
    overrides={
        # 仅改需要的字段；未写的保持 yaml 默认
        "max_steps": 20,
        # "num_generations": 4,   # OOM 时
        # "cppo_pruning_rate": 0.5,
    },
)
# 写入与 Phase1 Notebook 同名的大写常量，便于后续单元复用
g = apply_config_to_notebook(cfg, globals())
print(g["NUM_GENERATIONS"], g["MAX_STEPS"], g["tier"])
```

换档：把路径换成 `configs/colab_c0.yaml`，或继续用 C1 + `overrides`。

更完整的换环境清单与 Phase1 实测吞吐见 [`docs/MEMORY_BENCHMARK_COLAB.md`](../docs/MEMORY_BENCHMARK_COLAB.md)。
