# MixUpLLaVA-Video-R1 项目计划表

> **项目目标**：在 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) 基线上，将调研报告第 3 节「优化方法论」中的多种 GRPO 改进方案进行**模块化实现 + 混合组合 + 评估**，形成可复现的开源项目 **MixUpLLaVA-Video-R1**。
>
> **GitHub 仓库**：https://github.com/SanXue-YG/MixUpLLaVA-Video-R1  
> **上游基线**：https://github.com/ZhangXJ199/TinyLLaVA-Video-R1  
> **方法论依据**：`doc/基于Video-R1的视频交通异常行为检测项目中的GRPO优化调研报告.pdf` 第 3 节  
> **主运行路径**：Google Colab（A100）+ Google Drive（仿照已成功路径）  
> **Drive 根目录**：`/content/drive/MyDrive/MixUpLLaVA-video-r1`（结构见 [README.md](./README.md)）  
> **Drive 共享链接**：https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing

---

## 0. 项目背景与变更说明

| 维度 | 前期 CPPO（已完成） | Windows 本地尝试（已放弃） | **当前（正式路径）** |
|------|---------------------|----------------------------|----------------------|
| 运行环境 | Mac + Colab + Drive | Windows + RTX 4080 12GB | **Colab + Drive（A100）** |
| DeepSpeed | ✅ 可跑 | ❌ 原生 Windows 不兼容 / 不稳定 | ✅ 沿用已验证 ZeRO-3 + 参数 offload |
| 数据规模 | 50 条子集 | 计划全量 5496 | 先 50 / 10%，再视预算扩规模 |
| 优化范围 | 仅 CPPO（+ 策略 B 草稿） | MixUp 全模块规划 | **MixUp：CPPO + GFPO/NGRPO 等** |
| 显存策略 | g4, frames=2, queries=32 | 12GB 分档 L0–L2 | **优先 A100：g8 + frames=2**；快验可用 g4 |
| 交付形态 | Drive outputs + 本地笔记 | Git 骨架 | **Git 文档/模块 + Drive 训练资产** |

### 为何回到 Colab + Drive

1. 上游 TinyLLaVA-Video-R1 的 GRPO 训练依赖 **DeepSpeed**，Windows 本机难以稳定复现。  
2. 已有 **A100-80GB + Drive 路径** 成功跑通基线 A / CPPO C（见前期实验）。  
3. 数据、权重、`outputs/` 放 Drive，便于断连后续跑；代码放 GitHub，便于 Dev 编辑与他人复现结构。

### 双环境分工（成本控制）

| 环境 | 何时用 | 做什么 |
|------|--------|--------|
| **Dev（实惠）** | 平时默认 | 改 README/schedule、`mixup/`、Notebook 逻辑；**不跑**完整 deepspeed |
| **Train（A100）** | 仅训练与验证 | 挂载 Drive → 装依赖 → 训练 → 写 `outputs/` → 断开 |

原则：Dev 改完 → push GitHub / 同步 Drive → A100 会话按清单一次性跑完必要实验。

### 混合优化（MixUp）策略组合（拟采用）

根据调研报告互补性，「MixUp」指以下**可叠加模块**的组合评估：

| 层级 | 模块 | 来源 | 主要作用 | 优先级 |
|------|------|------|----------|--------|
| **A. 项目基线增强** | Difficulty-aware + 自适应长度 reward | 调研报告 §2.3 / 策略 B | 难度样本学习、控长度 | P0 |
| **B. 效率** | CPPO（\|advantage\| 剪枝） | 调研报告 §3.1 | 降计算量、加速 | P0（已复现，待 g8 放大加速） |
| **C. 效率/简洁** | GFPO（Top-k 掩码） | 调研报告 §3.1 | 抑冗余推理 | P1 |
| **D. 稳定性** | NGRPO（负信号增强） | 调研报告 §3.3 | 全错 group 无梯度 | P1 |
| **E. 稳定性** | DaGRPO | 调研报告 §3.4 | 梯度冲突掩码 | P2 |
| **F / G** | GMPO/MAPO、MO-GRPO | §3.4–3.5 | 数值稳定 / 多奖励 | P2 可选 |

**最终配方（待消融）**：`基线增强 + CPPO + NGRPO` 或 `基线增强 + CPPO + GFPO` 等 2–3 模块组合。

**暂不纳入首期**：StepGRPO、GRPO-A、GFPO 动态难度采样（二期）。

---

## 1. Drive 路径约定（与 README 一致）

```
PROJECT_DIR = /content/drive/MyDrive/MixUpLLaVA-video-r1
REPO        = {PROJECT_DIR}/repo/TinyLLaVA-Video-R1
MIXUP_REPO  = {PROJECT_DIR}/repo/MixUpLLaVA-Video-R1   # 本仓库
DATA_ROOT   = {PROJECT_DIR}/data/dataset/
CKPT        = {PROJECT_DIR}/checkpoints/coldstart
OUT_BASE    = {PROJECT_DIR}/outputs
```

完整子目录与复现步骤见 **[README.md · Google Drive 目录结构](./README.md)**。

### 已确认训练环境（2026-07-16）

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA A100-SXM4-80GB（~85 GB） |
| Python / PyTorch | 3.12.x / 2.11+cu128（以当次 Colab 为准） |
| DeepSpeed | ZeRO-3 + **参数** CPU offload（优化器不 offload） |
| Drive | `MixUpLLaVA-video-r1` 下 repo / data / checkpoints / outputs 均存在 |

### v1 对照结果（历史，勿覆盖）

| 策略 | 目录 | runtime(s) | steps/s | reward | 备注 |
|------|------|------------|---------|--------|------|
| A | `grpo_A_baseline_small` | 1760 | 0.028 | 0.493 | g4, 50 step |
| C | `grpo_CPPO_small` | 1748 | 0.029 | 0.493 | CPPO 加速 ~0.7% |

---

## 2. 里程碑总览

```
Phase 0  Drive/仓库对齐 + 文档     ████░░░░░░  Week 1
Phase 1  Colab 基线 / CPPO(g8)    ██████░░░░  Week 1–2
Phase 2  A100 显存档位标定         ██████░░░░  Week 2
Phase 3  单策略模块化（GFPO/NGRPO）████████░░  Week 2–4
Phase 4  MixUp 消融                ████████░░  Week 4–5
Phase 5  扩规模训练与评估          ██████████  Week 5–7
Phase 6  开源文档与 Release        ██████████  Week 7–8
```

---

## 3. 分阶段详细计划

### Phase 0：Drive + GitHub 对齐（第 1 周 · 1–2 天）

**目标**：文档、路径、代码入口与已成功 Colab 流程一致；他人可按 README 重建 Drive。

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 0.1 | 更新 README / schedule：Colab+Drive 主路径 + Drive 目录结构 | 本文件、README | ✅ |
| 0.2 | Drive 上确认 `repo/TinyLLaVA-Video-R1`、`data/`、`checkpoints/coldstart`、`outputs/` | `docs/PHASE0_DRIVE_SYNC.md`（共享页已见四目录；Colab 勾选） | ⬜ Colab |
| 0.3 | 将本仓库 clone / 同步到 `repo/MixUpLLaVA-Video-R1` | Drive 上可读；步骤见 `docs/PHASE0_DRIVE_SYNC.md` | ⬜ Colab |
| 0.4 | 迁移调研 PDF 至 `doc/` | `doc/基于Video-R1的…调研报告.pdf` + CPPO 论文 PDF | ✅ |
| 0.5 | 整理 Colab Notebook 入口 | `notebooks/mixup-GRPO优化.ipynb` | ✅ |
| 0.6 | 从前期工程迁移 CPPO / 策略补丁到 `mixup/` | `mixup/cppo.py`、`mixup/patches/` | ✅ |

**验收**：README 中的 Drive 树与真实 Drive 一致；Colab 能找到 `REPO`、`CKPT`；Drive 上存在本仓库副本。  
**本地进度**：0.1 / 0.4 / 0.5 / 0.6 已在本机 Git 仓库完成；0.2 / 0.3 需打开 Colab 按 `docs/PHASE0_DRIVE_SYNC.md` 勾选。

---

### Phase 1：Colab 基线与 CPPO 放大（第 1–2 周）

**目标**：在 **同一 Drive 路径** 上跑通 g8 对比，争取比 v1 更明显的 CPPO 加速；建立 MixUp 前的效率基线。

| # | 任务 | 说明 | 交付物 |
|---|------|------|--------|
| 1.1 | 固定 Notebook 超参 | `NUM_GENERATIONS=8`；快验 `MAX_STEPS=50`，正式可 100 | 路径配置单元 |
| 1.2 | **A2**：GRPO baseline（无剪枝） | 同参、同数据、同 CKPT | `outputs/grpo_A_g8/` |
| 1.3 | **C2**：CPPO `pruning_rate=0.5` | 仅改剪枝率 | `outputs/grpo_CPPO_g8_p50/` |
| 1.4 | （可选）**C3**：`pruning_rate=0.75` | 加速上限 | `outputs/grpo_CPPO_g8_p75/` |
| 1.5 | 对比汇总 | runtime / steps/s / reward / speedup% | schedule §8 或 docs |

**对比指标（每实验必记）**：`loss`, `reward`, `reward_std`, `kl`, `train_runtime`, `steps_per_second`。

**成本提示**：A100 贵，优先 A2+C2；C3 / g16 视结果再开。

---

### Phase 2：A100 显存档位标定（第 2 周）

**目标**：在 A100-80GB 上标定可稳定档位（替代 Windows 12GB 规划）。

| 档位 | num_gen | num_frames | num_queries | max_length | 用途 |
|------|---------|------------|-------------|------------|------|
| C0 快验 | 4 | 2 | 32 | 256 | 冒烟 / 调试 |
| C1 推荐 | **8** | 2 | 32 | 256 | CPPO 放大（当前主档） |
| C2 挑战 | 8–16 | 2–4 | 32–64 | 512 | 显存有余时 |

| # | 任务 | 交付物 |
|---|------|--------|
| 2.1 | 记录各档 OOM 边界与 steps/s | `docs/MEMORY_BENCHMARK_COLAB.md` |
| 2.2 | 配置写入 `configs/`（yaml 或 Notebook 常量） | `configs/colab_c1.yaml` |
| 2.3 | 约定：Train 会话只用 C1，除非明确升级 C2 | schedule 勾选 |

---

### Phase 3：单策略模块化（第 2–4 周）

**目标**：每种优化独立开关；代码在 GitHub `mixup/`，训练仍打补丁到 Drive 上 `REPO/tinyllava/train/` 或统一入口。

```
mixup/
├── __init__.py
├── registry.py
├── cppo.py
├── gfpo.py
├── ngrpo.py
├── project_baseline.py
└── trainer_mixup.py
```

| 策略 | 前期资产 | 任务 | 优先级 |
|------|----------|------|--------|
| CPPO | `tinyllava_trainer_reason.py`（工程文件） | 参数化 `cppo_pruning_rate` | P0 |
| 项目基线增强 | 策略 B 补丁 | 难度 + 长度 reward 解耦 | P0 |
| GFPO | 调研报告式 | Top-k 掩码 | P1 |
| NGRPO | 调研报告 | 全错 group 校准 | P1 |

**输出命名**：

```
outputs/{strategy}_{profile}_{data}_{seed}/
  ├── trainer_state.json
  ├── run_config.yaml
  └── metrics_summary.json
```

---

### Phase 4：MixUp 消融（第 4–5 周）

| 实验 ID | 组合 | 假设 |
|---------|------|------|
| M0 | Baseline GRPO | 对照 |
| M1 | 项目基线增强（B） | 质量↑ |
| M2 | B + CPPO | 质量保持 + 加速 |
| M3 | B + CPPO + NGRPO | 更稳 |
| M4 | B + CPPO + GFPO | 更简洁 |
| M5 | B + CPPO + NGRPO + GFPO | 最终候选 |

在 **同一 Drive 数据 / CKPT / 档位** 下完成；优先 50 或 10% 子集，再扩规模。

---

### Phase 5：扩规模训练与评估（第 5–7 周）

| # | 任务 | 说明 |
|---|------|------|
| 5.1 | 10% / 全量 baseline | 视 A100 预算；可等步数公平对比 |
| 5.2 | 最优 MixUp 配方 | 与 baseline 同数据同档位 |
| 5.3 | 轻量 eval / 案例分析 | parse rate、样例对比 |
| 5.4 | 曲线与资源统计 | 写入 `outputs/` 与 `docs/` |

风险：会话中断 → 输出落盘 Drive；`save_strategy` / resume 按预算开启。

---

### Phase 6：开源发布（第 7–8 周）

| # | 任务 |
|---|------|
| 6.1 | README 完善（Drive 结构 + Colab 步骤 + MixUp 开关） |
| 6.2 | LICENSE（Apache-2.0） |
| 6.3 | 示例 Notebook / 一键复现说明 |
| 6.4 | 消融报告 + 与 v1 CPPO 对比 |
| 6.5 | GitHub Release v0.1.0 |

---

## 4. 近期执行清单（接下来 3 天）

- [x] **D0**：放弃 Windows 正式训练路径；确认回到 Colab + Drive  
- [x] **D0**：更新本仓库 README / schedule（Drive 结构 + 双环境）  
- [x] **D1（Dev）**：Phase 0 本地交付 — `doc/`、`notebooks/`、`mixup/`、`docs/PHASE0_DRIVE_SYNC.md`  
- [ ] **D1（Colab）**：按 `docs/PHASE0_DRIVE_SYNC.md` 完成 0.2 路径确认 + 0.3 `git clone/pull` 到 Drive  
- [ ] **D2（A100）**：挂载 Drive → 跑 A2（g8 baseline）  
- [ ] **D3（A100）**：跑 C2（CPPO p50）→ §对比 → 填 §8 结果表 → 断开  

---

## 5. 实验管理规范

### 5.1 数据子集

| 文件（位于 `data/dataset/`） | 规模 | 用途 |
|------------------------------|------|------|
| `nextqa_small50.jsonl` | 50 | 冒烟 / 快验 |
| `nextqa_0-30s_10p_seed42.jsonl` | ~550 | 消融 |
| `nextqa_0-30s.jsonl` | ~5496 | 全量 |

### 5.2 公平对比

- 同一 `DATA_ROOT` / jsonl / `CKPT` / 档位（C0/C1）  
- A vs CPPO：**仅**差 `cppo_pruning_rate` 与输出目录  
- 策略 B / GFPO / NGRPO 改目标函数时，单独成组，不与「纯加速」混谈  

### 5.3 Git 与 Drive 分工

| 内容 | 存放 |
|------|------|
| 文档、`mixup/`、Notebook 模板 | GitHub |
| 视频、权重、`outputs/` | Google Drive only |
| 上游可运行 `TinyLLaVA-Video-R1` | Drive `repo/`（或 submodule，按体积决定） |

### 5.4 A100 会话检查清单

```
[ ] GPU 仍为 A100-80GB
[ ] PROJECT_DIR / REPO / CKPT / SMALL_JSONL 存在
[ ] NUM_GENERATIONS 与档位一致（推荐 8）
[ ] Trainer 已重置为本次策略（无残留错误补丁）
[ ] 本次只跑清单内实验（如 A2+C2）
[ ] 结束后确认 outputs 已在 Drive
```

---

## 6. 参考资源

| 资源 | 路径/链接 |
|------|-----------|
| 本仓库 README（含 Drive 树） | [README.md](./README.md) |
| 调研报告 | `doc/…GRPO优化调研报告.pdf` / 实验目录 `优化方案/` |
| 上游仓库 | https://github.com/ZhangXJ199/TinyLLaVA-Video-R1 |
| 前期 Colab Notebook | 实验 `工程文件/GRPO测试优化-….ipynb`、`mixup-GRPO优化.ipynb` |
| Drive 项目根 | `MyDrive/MixUpLLaVA-video-r1` |
| Drive 共享包（官方） | https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing |

---

## 7. 进度跟踪

| 阶段 | 计划开始 | 计划完成 | 实际完成 | 备注 |
|------|----------|----------|----------|------|
| Phase 0 | 2026-07-16 | 2026-07-18 | | 文档已切 Colab+Drive |
| Phase 1 | 2026-07-17 | 2026-07-25 | | A2/C2 g8 |
| Phase 2 | 2026-07-20 | 2026-07-23 | | 可与 Phase 1 并行 |
| Phase 3 | 2026-07-25 | 2026-08-15 | | GFPO/NGRPO |
| Phase 4 | 2026-08-10 | 2026-08-22 | | 消融 |
| Phase 5 | 2026-08-22 | 2026-09-10 | | 视预算 |
| Phase 6 | 2026-09-10 | 2026-09-20 | | Release |

---

## 8. v2 实验结果（待填）

| 策略 | 输出目录 | max_steps | train_runtime(s) | steps/s | reward | 相对 A2 加速 |
|------|----------|-----------|------------------|---------|--------|--------------|
| A2 | `grpo_A_g8` | | | | | — |
| C2 | `grpo_CPPO_g8_p50` | | | | | |
| C3 | `grpo_CPPO_g8_p75` | | | | | |

---

*最后更新：2026-07-17 · 维护者：张一鸣 · 主路径：Colab + Google Drive*
