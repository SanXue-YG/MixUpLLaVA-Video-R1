# MixUpLLaVA-Video-R1 项目计划表

> **项目目标**：在 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) 基线上，将调研报告第 3 节「优化方法论」中的多种 GRPO 改进方案进行**模块化实现 + 混合组合 + 全量评估**，形成可复现的开源项目 **MixUpLLaVA-Video-R1**。
>
> **GitHub 仓库**：https://github.com/SanXue-YG/MixUpLLaVA-Video-R1  
> **上游基线**：https://github.com/ZhangXJ199/TinyLLaVA-Video-R1  
> **方法论依据**：`doc/基于Video-R1的视频交通异常行为检测项目中的GRPO优化调研报告.pdf` 第 3 节  
> **前期成果**：`实验/TinyLLaVA-Video-R1 优化测试/`（Colab + 50 条子集 CPPO 验证，加速约 0.7%）

---

## 0. 项目背景与变更说明

| 维度 | 前期（Week 7–8，Colab） | 当前（MixUpLLaVA-Video-R1，本地） |
|------|-------------------------|-----------------------------------|
| 运行环境 | Mac + Colab + Google Drive | Windows 本地，RTX 4080 Laptop **12 GB** |
| 数据规模 | 50 条最小子集 | 全量 **5,496** 条（`nextqa_0-30s.jsonl`） |
| 优化范围 | 仅 CPPO（+ 策略 B 草稿） | **混合优化**：效率 + 稳定性 + 项目内 reward shaping |
| 显存策略 | `num_generations=4, num_frames=2, num_queries=32` | 按 12 GB 重新标定，分档配置 |
| 交付形态 | Drive 笔记 + 本地实验文件夹 | 规范 Git 仓库 + 实验脚本 + 对比报告 |

### 混合优化（MixUp）策略组合（拟采用）

根据调研报告表 1 的互补性，本项目的「MixUp」指以下**可叠加模块**的组合评估，而非简单堆叠所有论文：

| 层级 | 模块 | 来源 | 主要作用 | 实现优先级 |
|------|------|------|----------|------------|
| **A. 项目基线增强** | Difficulty-aware advantage + 自适应长度 reward | 调研报告 §2.3 / 前期策略 B | 提升高难度样本学习、控制推理长度 | P0（已有草稿） |
| **B. 效率** | CPPO（优势剪枝） | 调研报告 §3.1 | 降低 completion 计算量、加速训练 | P0（已复现） |
| **C. 效率/简洁** | GFPO（Top-k 优势掩码） | 调研报告 §3.1 | 抑制冗余推理、配合「sample more, think less」 | P1 |
| **D. 稳定性** | NGRPO（负信号增强） | 调研报告 §3.3 | 缓解全错 group 无梯度问题 | P1 |
| **E. 稳定性** | DaGRPO（梯度冲突掩码） | 调研报告 §3.4 | 减少正负样本梯度抵消 | P2 |
| **F. 数值稳定** | GMPO / MAPO | 调研报告 §3.4 | Token 级极端比率、优势噪声 | P2（可选） |
| **G. 多目标** | MO-GRPO | 调研报告 §3.5 | 格式/正确性等多奖励公平聚合 | P2（可选） |

**最终 MixUp 配方（待消融后确定）**：`基线增强(B) + CPPO + NGRPO` 或 `基线增强(B) + CPPO + GFPO` 等 2–3 个互补模块的组合。

**暂不纳入首期实现**（改动面大）：StepGRPO（需 step-wise 奖励管线）、GRPO-A（需引导样本生成）、GFPO 动态难度采样（二期）。

---

## 1. 里程碑总览

```
Phase 0  环境与仓库        ████░░░░░░  Week 1
Phase 1  基线跑通          ██████░░░░  Week 1–2
Phase 2  显存与配置标定    ██████░░░░  Week 2
Phase 3  单策略复现/实现   ████████░░  Week 2–4
Phase 4  MixUp 混合实验    ████████░░  Week 4–5
Phase 5  全量训练与评估    ██████████  Week 5–7
Phase 6  开源发布与文档    ██████████  Week 7–8
```

---

## 2. 分阶段详细计划

### Phase 0：环境与仓库初始化（第 1 周 · 1–2 天）

**目标**：本地可安装、可导入、数据与 checkpoint 就位，Git 与上游对齐。

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 0.1 | 克隆上游 `TinyLLaVA-Video-R1` 至项目根目录（或 submodule） | `tinyllava/` 源码树 | ⬜ |
| 0.2 | 解压 `Dataset/NextQA.zip`，核对目录结构符合上游 README | `data/dataset/NextQA/`, `nextqa_0-30s.jsonl` | ⬜ |
| 0.3 | 解压 `checkpoints/`，放置 Cold-Start 权重 | `checkpoints/TinyLLaVA-Video-Coldstart/` | ⬜ |
| 0.4 | 创建 conda 环境 `mixup_video_r1`（Python 3.10）并 `pip install -e .` | `environment.yml` 或安装说明 | ⬜ |
| 0.5 | 安装 `flash-attn`（若 Windows 编译失败，记录 fallback：`sdpa`） | 安装日志 | ⬜ |
| 0.6 | 初始化 Git，关联 `SanXue-YG/MixUpLLaVA-Video-R1`，首次 push 骨架 | README、.gitignore、本计划表 | ⬜ |
| 0.7 | 从 `实验/` 迁移 CPPO 补丁与策略 B 草稿至 `mixup/` 模块 | `mixup/patches/` 或 `mixup/strategies/` | ⬜ |

**验收标准**：`python -c "import tinyllava"` 成功；`nvidia-smi` 识别 4080；数据集路径与 jsonl 行数 ≈ 5496。

---

### Phase 1：基线复现（第 1–2 周 · 3–5 天）

**目标**：在本地单卡跑通 **原始 GRPO**（无优化），建立公平对比基准。

| # | 任务 | 说明 | 交付物 |
|---|------|------|--------|
| 1.1 | 编写 `scripts/train/baseline_grpo_local.sh` | 单卡 DeepSpeed ZeRO-3 + CPU offload | 训练脚本 |
| 1.2 | 固定实验约定 | 种子 `42`；日志、checkpoint、配置快照进 `outputs/` | `docs/EXPERIMENT.md` |
| 1.3 | **Smoke test**：50 条子集，≤100 steps | 验证流程不 OOM、能存 checkpoint | `outputs/smoke_baseline_50/` |
| 1.4 | **中规模**：10% 子集（≈550 条），500–1000 steps | 对比 Colab 前期结果量级 | `outputs/baseline_10p_seed42/` |
| 1.5 | Sanity eval | 从子集抽 50–100 条做推理，统计 parse rate / accuracy | `outputs/baseline_eval_sanity.jsonl` |

**对比指标（每个实验必记）**：

- `train_loss`, `reward`, `reward_std`, `kl`
- `train_runtime`, `steps_per_second`
- GPU 峰值显存（`torch.cuda.max_memory_allocated`）

---

### Phase 2：本地显存标定（第 2 周 · 2–3 天）

**目标**：针对 **12 GB 4080** 确定可稳定训练的配置档位，替代 Colab 临时减负参数。

> 上游默认（8×A100-40G）：`num_frames=16`, `num_queries=512`, `num_generations=8`, `model_max_length=3072`  
> Colab 减负：`num_generations=4`, `num_frames=2`, `num_queries=32`, `model_max_length=256`  
> **本地需重新网格搜索**，在「能跑」与「不过分偏离论文设定」之间折中。

| 档位 | num_gen | num_frames | num_queries | max_length | 用途 |
|------|---------|------------|-------------|------------|------|
| L0 极限 | 2 | 2 | 32 | 512 | 调试 / smoke |
| L1 推荐 | 4 | 4 | 64 | 1024 | 日常实验 |
| L2 挑战 | 4 | 8 | 128 | 2048 | 全量训练（若 OOM 回退 L1） |

| # | 任务 | 交付物 |
|---|------|--------|
| 2.1 | 在 `tinyllava_trainer_reason.py` 抽离显存相关参数至配置文件 | `configs/memory_profile_l1.yaml` |
| 2.2 | 实现统一入口 `--memory_profile L0|L1|L2` | 训练脚本参数 |
| 2.3 | 记录各档位最大显存与 steps/s | `docs/MEMORY_BENCHMARK.md` |
| 2.4 | Windows 下 DeepSpeed 兼容性验证（必要时 WSL2） | 环境说明 |

---

### Phase 3：单策略模块化实现（第 2–4 周）

**目标**：每种优化独立开关、独立对比，代码结构便于组合。

建议目录结构：

```
mixup/
├── __init__.py
├── registry.py          # 策略注册与组合
├── cppo.py              # CPPO 剪枝（迁移自实验）
├── gfpo.py              # GFPO Top-k 掩码
├── ngrpo.py             # NGRPO 负信号
├── dagrpo.py            # DaGRPO（可选）
├── project_baseline.py  # 难度感知 + 长度 reward（策略 B）
└── trainer_mixup.py     # 统一 Trainer 或 monkey-patch 入口
```

| 策略 | 对应前期文件 | 实现任务 | 优先级 |
|------|-------------|----------|--------|
| **CPPO** | `tinyllava_trainer_reason（CPPO实现）.py` | 迁移 + 参数化 `cppo_pruning_rate` + 单测 | P0 |
| **项目基线增强** | `tinyllava_trainer_reason（策略B-对比测试）.py` | 难度权重 + 长度 shaping 解耦为模块 | P0 |
| **GFPO** | 无 | 按调研报告式(10)–(13)实现 Top-k 掩码 | P1 |
| **NGRPO** | 无 | 全错 group 优势校准 | P1 |
| **DaGRPO** | 无 | 梯度冲突样本掩码（简化版） | P2 |

**每个策略的实验命名规范**：

```
outputs/{strategy}_{profile}_{data}_{seed}/
  ├── trainer_state.json
  ├── run_config.yaml      # 完整超参快照
  ├── git_commit.txt
  └── metrics_summary.json
```

**单策略验收**（在 10% 子集、相同 steps 下）：

1. 训练能完整跑完且无 OOM  
2. 相对 baseline：`reward` 不显著下降（±5% 以内）  
3. 至少一项效率或稳定性指标有 measurable 提升（如 `steps_per_second`、loss 方差）

---

### Phase 4：MixUp 混合优化实验（第 4–5 周）

**目标**：组合 2–3 个互补模块，做消融实验（Ablation），选出最优配方。

| 实验 ID | 组合 | 假设 |
|---------|------|------|
| M0 | Baseline（原始 GRPO） | 对照 |
| M1 | 项目基线增强（B） | 质量提升 |
| M2 | B + CPPO | 质量保持 + 加速 |
| M3 | B + CPPO + NGRPO | 早期训练更稳定 |
| M4 | B + CPPO + GFPO | 推理更简洁、token 更少 |
| M5 | B + CPPO + NGRPO + GFPO（全 MixUp） | 最终候选 |

| # | 任务 | 交付物 |
|---|------|--------|
| 4.1 | 实现 `mixup/registry.py` 策略开关组合 | `--enable cppo,ngrpo,gfpo,project` |
| 4.2 | 在 10% 子集完成 M0–M5 消融 | `outputs/ablation_10p/` 对比表 |
| 4.3 | 撰写中期对比分析 | `docs/ABLATION_REPORT.md` |
| 4.4 | 选定 1–2 个配方进入全量 | 决策记录 |

---

### Phase 5：全量训练与评估（第 5–7 周）

**目标**：在 **5,496 条**完整数据上训练最优 MixUp 配方，并与 baseline 对比。

| # | 任务 | 说明 | 交付物 |
|---|------|------|--------|
| 5.1 | 全量 Baseline GRPO | 时间预算：参考上游 ~30h/8×A100；单卡 4080 需预估 **数倍** | `outputs/full_baseline/` |
| 5.2 | 全量 MixUp 最佳配方 | 与 baseline 相同 epoch/steps | `outputs/full_mixup/` |
| 5.3 | 快速 benchmark | 优先 MVBench 或 Video-MME short split | `results/benchmark/` |
| 5.4 | 训练曲线与资源统计 | wandb/tensorboard 或本地 JSON | `results/training_curves/` |
| 5.5 | 案例分析 | 「aha moment」推理样例对比 | `docs/CASE_STUDY.md` |

**全量训练风险与应对**：

- **时间**：单卡远慢于 8 卡；可考虑夜间连续跑、或减少 `max_steps` 做「等步数」公平对比  
- **磁盘**：checkpoint 体积大；`save_total_limit=2`，大文件不入 Git  
- **中断恢复**：开启 `--resume_from_checkpoint`

---

### Phase 6：开源发布（第 7–8 周）

| # | 任务 | 交付物 |
|---|------|--------|
| 6.1 | 完善 `README.md`（安装、数据、训练、MixUp 开关说明） | 中英双语摘要 |
| 6.2 | 添加 `LICENSE`（继承上游 Apache-2.0） | LICENSE |
| 6.3 | 发布 `scripts/train/train_mixup_local.sh` 一键示例 | 可复现脚本 |
| 6.4 | 整理 `docs/`：调研摘要 + 实验结论 + 与 CPPO 前期对比 | 文档集 |
| 6.5 | GitHub Release + 结果表格 | Release v0.1.0 |
| 6.6 | （可选）Hugging Face 模型卡 | 权重链接 |

---

## 3. 近期执行清单（接下来 3 天）

按顺序执行，**完成一项勾一项**：

- [ ] **D1 上午**：解压数据集与 checkpoint，克隆上游代码，创建 conda 环境  
- [ ] **D1 下午**：Git 初始化并 push 骨架；迁移 CPPO / 策略 B 代码至 `mixup/`  
- [ ] **D2**：编写 `baseline_grpo_local.sh`，完成 50 条 smoke test  
- [ ] **D3**：完成显存档位 L0/L1 标定，生成 10% 子集 `nextqa_0-30s_10p_seed42.jsonl`  
- [ ] **D3 晚**：启动 baseline 10% 训练（500 steps），记录首份 metrics  

---

## 4. 实验管理规范

### 4.1 数据子集文件

| 文件 | 规模 | 用途 |
|------|------|------|
| `nextqa_smoke50_seed42.jsonl` | 50 | 冒烟测试 |
| `nextqa_0-30s_10p_seed42.jsonl` | ~550 | 消融与开发 |
| `nextqa_0-30s.jsonl` | 5496 | 全量训练 |

### 4.2 随机种子

所有实验固定 `seed=42`（Python、NumPy、PyTorch、采样）。

### 4.3 不可提交 Git 的大文件

通过 `.gitignore` 排除：`checkpoints/`, `Dataset/`, `outputs/`, `*.zip`, `*.crdownload`, `wandb/`

### 4.4 对比公平性原则

- 同一数据文件、同一 cold-start 权重、同一 `memory_profile`  
- 仅改变优化策略相关开关  
- 记录完整 `run_config.yaml` 与 git commit hash  

---

## 5. 参考资源

| 资源 | 路径/链接 |
|------|-----------|
| 调研报告 | `doc/基于Video-R1的视频交通异常行为检测项目中的GRPO优化调研报告.pdf` |
| 原始论文 | `doc/2504.09641v1.pdf` |
| 上游仓库 | https://github.com/ZhangXJ199/TinyLLaVA-Video-R1 |
| 前期 Colab 实验 | `实验/TinyLLaVA-Video-R1 优化测试/GRPO测试优化-基于tiny video R1的GRPO优化.ipynb` |
| 前期工程文件 | `实验/工程文件/` |
| Google Drive 备份 | https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m |
| 实习复盘（Week 5–8） | `张一鸣-第一段实习复盘报告.xlsx` |

---

## 6. 进度跟踪

| 阶段 | 计划开始 | 计划完成 | 实际完成 | 备注 |
|------|----------|----------|----------|------|
| Phase 0 | 2026-07-16 | 2026-07-18 | | |
| Phase 1 | 2026-07-18 | 2026-07-25 | | |
| Phase 2 | 2026-07-20 | 2026-07-23 | | 可与 Phase 1 并行 |
| Phase 3 | 2026-07-25 | 2026-08-15 | | |
| Phase 4 | 2026-08-10 | 2026-08-22 | | |
| Phase 5 | 2026-08-22 | 2026-09-10 | | 视全量训练耗时调整 |
| Phase 6 | 2026-09-10 | 2026-09-20 | | |

---

*最后更新：2026-07-16 · 维护者：张一鸣*
