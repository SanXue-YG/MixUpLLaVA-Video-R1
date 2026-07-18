# MixUp 模块库（Phase 3）

可组合 GRPO 改进模块，嵌入 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) 的 `compute_loss` 流水线。  
原论文 PDF 见仓库 [`doc/`](../doc/README.md)。

## 快速使用

```python
from mixup import MixUpConfig, ensure_modules_loaded, run_advantage_pipeline, apply_mixup_to_repo
from mixup.presets import load_preset

ensure_modules_loaded()
cfg = MixUpConfig()                         # M1：默认 B + CPPO
cfg = MixUpConfig.from_dict({"ngrpo": True}) # 叠 NGRPO，B/CPPO 仍在
cfg = load_preset("m1_baseline_b")          # 同上
# 显式去掉其一：MixUpConfig(project_baseline=False) 或 cppo=False
```

## 项目默认：B + CPPO

| 模块 | 角色 | 默认 |
|------|------|------|
| **B** `project_baseline` | 项目组自研质量向（难度 + 长度） | **开** |
| **CPPO** | 项目组原定效率目标（Phase1 已验证） | **开** |

后续勾选 GFPO / NGRPO / MO-GRPO 等时，**二者保持开启**，除非在 dict/YAML/选择器里显式设为 `false`。  
报告可用 `cfg.dropped_project_defaults()` 列出被关掉的默认模块。

消融预设：`ablation_b_only`、`speed_cppo`（仅 CPPO）、`ablation_grpo_vanilla`（纯 GRPO，对齐 Phase1 A2）。

冒烟（需 PyTorch）：

```bash
python -m mixup.tests_smoke
```

## 模块一览

| 开关 | 优先级 | 改动层 | 调研 / 论文 | 状态 |
|------|--------|--------|-------------|------|
| `project_baseline` | P0 | reward + advantage | 调研 §2.3 | ✅ |
| `cppo` | P0 | prune | `doc/CPPO-*.pdf` · §3.1.1 | ✅ Phase1 验证 |
| `gfpo` | P1 | advantage mask | `doc/GFPO-*.pdf` · §3.1.2 | ✅ |
| `ngrpo` | P1 | advantage | `doc/NGRPO-*.pdf` · §3.3.2 | ✅ |
| `mo_grpo` | P1 | advantage | `doc/MO-GRPO-*.pdf` · §3.5 | ✅ |
| `dagrpo` | P2 | mask | `doc/DaGRPO-*.pdf` | ✅ 可用（锚点简化） |
| `gmpo` | P2 | loss | `doc/GMPO-*.pdf` | ✅（需真实 ρ 才充分；当前 on-policy 回退） |
| `mapo` | P2 | advantage | `doc/MAPO-*.pdf` | ✅（与 MO/NGRPO 互斥优先） |
| `grpo_a` | P2 | generation | `doc/GRPO-A-*.pdf` | stub（需 GT 引导前缀） |
| `step_grpo` | 预留 | reward | `doc/R1-VL-StepGRPO.pdf` | stub |

## 推荐叠加顺序

```text
rewards_per_func
  → project_baseline（长度 shaping）
  → mo_grpo | ngrpo | mapo | 标准 GRPO
  → gfpo（Top-k 掩码，可覆盖优势）
  → project_baseline（难度 reweight + clamp）
  → dagrpo
  → cppo（剪枝后再算 logps）
  → gmpo（可选 loss）
```

建议：**不要**同时开 `mo_grpo` 与 `mapo`；`gfpo` 与 `mo_grpo` 同时开时 GFPO 会按 rewards 重算 Top-k 优势。

## 超参（常用）

| 字段 | 默认 | 说明 |
|------|------|------|
| `cppo_pruning_rate` | 0.5 | Phase1 C2 |
| `gfpo_top_k` / `gfpo_metric` | 4 / `shortest` | 或 `token_efficiency` |
| `ngrpo_r_max` | 2.0 | 与 acc+format 耦合上界对齐 |
| `baseline_lambda_diff` | 1.0 | 难度加权强度 |
| `baseline_beta` | 0.02 | 开 B 时 trainer β |

完整字段见 `mixup/config.py` · `MixUpConfig`。

## 预设（`mixup/presets/`）

| 文件 | 含义 |
|------|------|
| `m1_baseline_b.yaml` | **默认基线 M1 = B + CPPO** |
| `speed_cppo.yaml` | 消融：仅 CPPO（显式关 B；对齐 Phase1 C2） |
| `ablation_b_only.yaml` | 消融：仅 B |
| `ablation_grpo_vanilla.yaml` | 消融：纯 GRPO（关 B+CPPO；对齐 Phase1 A2） |
| `m5_b_cppo_ngrpo_gfpo.yaml` | 可选 M5 风格（含默认 B+CPPO） |
| `m5q_quality.yaml` | B+CPPO+NGRPO+MO |

## 打补丁到上游

```python
from mixup import MixUpConfig, apply_mixup_to_repo
from mixup.presets import load_preset

cfg = load_preset("speed_cppo")
apply_mixup_to_repo(
    "/content/drive/MyDrive/MixUpLLaVA-video-r1/repo/MixUpLLaVA-Video-R1",
    "/content/drive/MyDrive/MixUpLLaVA-video-r1/repo/TinyLLaVA-Video-R1",
    cfg,
)
# 写入 tinyllava/train/tinyllava_trainer_reason.py
# 与 mixup_config.json、_mixup_path.pth
```

训练前把 `num_generations` / `num_frame` 仍按 Phase2 `configs/colab_c1.yaml`（或 Notebook）设置。

## 扩展新方法

1. 新建 `mixup/my_method.py`，实现 `apply(cfg, ctx) -> ctx`  
2. `@register("my_method", stage="advantage", priority="P2", paper="doc/...")`  
3. 在 `MixUpConfig` 增加布尔开关与超参  
4. 在 `trainer_mixup.ensure_modules_loaded` 中 `import`  
5. 更新本文档一行  

不必改 Notebook 训练主循环。

## 目录

```text
mixup/
├── config.py / registry.py / trainer_mixup.py / config_loader.py
├── project_baseline.py / cppo.py / gfpo.py / ngrpo.py / mo_grpo.py
├── dagrpo.py / gmpo.py / mapo.py / grpo_a.py / step_grpo.py
├── presets/
├── patches/tinyllava_trainer_reason_mixup.py
└── tests_smoke.py
```
