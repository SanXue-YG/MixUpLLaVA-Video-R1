# Phase 4 报告模板：优化方案选择器运行记录

> 由选择器生成的草稿见各次 `outputs/{run_id}/PHASE4_REPORT_STUB.md`。正式结论可复制本模板填写。

---

## 元信息

| 项 | 内容 |
|----|------|
| run_id | |
| 日期 | |
| 环境 | Colab A100 / 其它 |
| 档位 | C1（默认）/ C0 / 自定义 |
| 预设 | m1 / speed / m5 / m5q / 自定义 |
| 配置快照路径 | `outputs/.../mixup_config.json` |

## 开关快照

| 模块 | 状态 | 超参 |
|------|------|------|
| project_baseline（B） | 开 / **关（显式）** | |
| cppo | 开 / **关（显式）** | pruning_rate= |
| gfpo | | top_k= / metric= |
| ngrpo | | r_max= |
| mo_grpo | | |
| 其它 | | |

**项目默认检查**：B 与 CPPO 是否均为开启？若关闭，写明原因（消融 / Phase1 对照）。

`dropped_project_defaults`: `______________`

## 训练期结果

| 指标 | 本 run | M1（若有） | Phase1 A2 | Phase1 C2 |
|------|--------|------------|-----------|-----------|
| train_steps_per_second | | | 0.008 | 0.011 |
| train_runtime (s) | | | 6357 | 4616 |
| reward 均值 | | | 0.190 | 0.321 |
| accuracy_reward | | | | |
| format_reward | | | | |

## 结论（本 run）

- 相对 M1：效率 / reward / …
- 下一步：是否进入 Phase 5 四基准；或改开关再跑

## 复现

```python
from mixup.training_entry import prepare_training
plan = prepare_training(preset="m1", mixup={"ngrpo": True}, project_dir=PROJECT_DIR)
# 快照与报告草稿已写入 plan.output_dir
```
