# Phase 5 评测报告模板

> **默认路径 = 训练期评估**（快筛，几乎零额外存储）。  
> **总评估**（Video-MME / MVBench / MLVU / MMVU）为可选；四基准合计约 **~600GB**，见 [`PHASE5_BENCHMARKS.md`](./PHASE5_BENCHMARKS.md)。

---

## A. 训练期评估（必填 / 推荐）

| 项 | 内容 |
|----|------|
| 候选 run_id | |
| 基线 run_id（如 M1） | |
| 档位 | C1 / … |
| MixUp 开关 | |
| 是否关掉项目默认 B/CPPO | |

### 效率

| 指标 | 基线 | 候选 | Δ / 相对 |
|------|------|------|----------|
| `train_steps_per_second` | | | |
| `train_runtime` (s) | | | |

### 质量代理

| 指标 | 基线均值 | 候选均值 | 末 10 step | 备注 |
|------|----------|----------|------------|------|
| `reward` | | | | |
| `rewards/accuracy_reward` | | | | |
| `rewards/format_reward` | | | | |
| `kl` / `loss` | | | | |

自动生成：`mixup.eval_training.evaluate_run(run_dir, baseline_dir=...)`  
→ `training_eval_report.md`。

### 结论（训练期）

- 相对基线：效率 …；reward/acc …
- 是否值得进入可选总评估：是 / 否

---

## B. 总评估（可选）

仅在已下载基准、磁盘充足时填写。协议对齐 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)。

| 基准 | 基线 | 候选 | Δ |
|------|------|------|---|
| Video-MME | | | |
| MVBench | | | |
| MLVU | | | |
| MMVU | | | |

数据路径：`EVAL_ROOT=...`；脚本：`scripts/eval/run_*.sh`。

---

## C. Phase1 附录（可选）

| | A2 | C2 |
|--|----|----|
| steps/s | 0.008 | 0.011 |
| runtime (s) | 6357 | 4616 |
| reward 均值 | 0.190 | 0.321 |
