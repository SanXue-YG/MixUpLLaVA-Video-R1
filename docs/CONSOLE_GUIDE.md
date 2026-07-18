# 总控制台使用指南（Phase 6）

主入口 Notebook：[`notebooks/mixup_console.ipynb`](../notebooks/mixup_console.ipynb)

## 这是什么

**MixUpLLaVA-Video-R1** 的使用者控制台：在同一 Notebook 里完成

1. 环境 / 默认档 C1  
2. 选择优化策略（默认 **B + CPPO**）  
3. 生成基线 run 与优化方案 run  
4. （按需）开训  
5. **训练期**评估与对比报告  
6. （可选）四基准总评估  

开源叙事：**可组合 GRPO 实验台**，而不是单一神秘配方。

## 快速开始

### Colab

1. 挂载 Drive，确认 `PROJECT_DIR=/content/drive/MyDrive/MixUpLLaVA-video-r1`  
2. 打开 `repo/MixUpLLaVA-Video-R1/notebooks/mixup_console.ipynb`  
3. 自上而下运行；`APPLY_PATCH` / 开训单元格按需打开  

### 本地（无 GPU 也可验收流程）

```bash
cd MixUpLLaVA-Video-R1
# 在 Jupyter 中打开 notebooks/mixup_console.ipynb
# 路径会自动回退到仓库根；可生成 plan + 报告骨架
```

## 默认约定

| 项 | 值 |
|----|-----|
| 档位 | C1（`configs/colab_c1.yaml`） |
| 基线策略 | M1 = B + CPPO |
| 加开模块 | 除非显式 `false`，B/CPPO 保持 |
| 评测 | 训练期为主；四基准见 `PHASE5_BENCHMARKS.md`（~600GB） |

## 输出落盘

```text
outputs/{run_id}/
  mixup_config.json
  run_plan.json
  PHASE4_REPORT_STUB.md
  baseline_report.md          # 基线 run
  training_eval_report.md     # 有 trainer_state 时
  compare_vs_baseline.md      # 方案 run
```

## Phase1 效率参考（同档 C1）

| | A2 GRPO | C2 CPPO |
|--|---------|---------|
| steps/s | 0.008 | 0.011（+37.5%） |
| runtime | ~6357 s | ~4616 s |
| reward | 0.190 | 0.321 |

## 相关文档

- 模块：`docs/MIXUP_MODULES.md`  
- 选择器：`notebooks/phase4-selector.ipynb`  
- 训练期评测：`docs/PHASE5_EVAL_REPORT.md`  
- 计划：`schedule.md`
