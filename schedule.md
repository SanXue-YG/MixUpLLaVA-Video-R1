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
| B. 效率 | CPPO（\|advantage\| 剪枝） | 调研报告 §3.1 | 降计算量、加速 | **P0 ✅ Phase 1 g8 已验证** |
| **C. 效率/简洁** | GFPO（Top-k 掩码） | 调研报告 §3.1 | 抑冗余推理 | P1 |
| **D. 稳定性** | NGRPO（负信号增强） | 调研报告 §3.3 | 全错 group 无梯度 | P1 |
| **E. 稳定性** | DaGRPO | 调研报告 §3.4 | 梯度冲突掩码 | P2 |
| **F / G** | GMPO/MAPO、MO-GRPO | §3.4–3.5 | 数值稳定 / 多奖励 | P2 可选 |

**最终配方（修订）**：不再做全矩阵消融。Phase 4 只训 **M1（项目基线增强 B）** 作组内基线，再训 **M5（B+CPPO+NGRPO+GFPO）** 作为个人最终方案；可与 Phase 1 的 A2/C2 对照。正式下游评测放在 Phase 5。

**暂不纳入首期**：StepGRPO、GRPO-A、GFPO 动态难度采样（二期）；单模块独立训练评测（Phase 3 只实现代码）。

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
Phase 0  Drive/仓库对齐 + 文档     ████████░░  Week 1（文档 ✅；Drive 勾选可补）
Phase 1  Colab 基线 / CPPO(g8)    ██████████  Week 1–2 ✅ 见 docs/PHASE1_REPORT.md
Phase 2  锁定 Phase1 档位（C1）    ████████░░  Week 2（复用 A2/C2 超参，少做新标定）
Phase 3  单策略代码模块化          ████████░░  Week 2–4（实现开关；不单训单测）
Phase 4  M1 基线 + M5 最终方案     ████████░░  Week 4–5
Phase 5  上游四基准总评估          ██████████  Week 5–7（评 M1 vs M5；可对照 Phase1）
Phase 6  开源文档与 Release        ██████████  Week 7–8
```

### 跨阶段对比约定（重要）

为复用 Phase 1 已有产出，**Phase 2 起默认沿用 Phase 1 训练超参档位（C1）**，除非显存不够才降级。下表同时列出上游 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) 官方 GRPO 脚本默认值，便于报告中写清「相对原文精简了多少」。

**上游默认来源**：`scripts/train/train_qwen2_reason_nextqa.sh`（及 README：约 **8×A100-40G、~30h**；数据为 NextQA 0–30s 全量约 **5496** 条）。

| 项 | TinyLLaVA-Video-R1（官方默认） | 本项目 C1 / Phase1+（后续默认） | 相对精简（约） |
|----|-------------------------------|----------------------------------|----------------|
| GPU 规模 | **8** GPU（脚本 `localhost:0..7`） | **1×A100-80GB**（Colab） | 多卡 → 单卡快验 |
| DeepSpeed | `zero3.json` | ZeRO-3 + **参数** CPU offload | 适配单卡显存 |
| 数据 | `nextqa_0-30s.jsonl`（~5496） | `nextqa_small50.jsonl`（**50**） | 样本约 **1/110** |
| `num_frames`（脚本 / 数据参数） | **16** | **2** | **1/8** |
| `num_frame`（trainer 内采样） | **16** | **8**（Phase1 `NUM_FRAME_TRAINER`） | **1/2** |
| `num_queries` | **512** | **32** | **1/16** |
| `model_max_length` | **3072** | **256** | **1/12** |
| `learning_rate` | **1e-6** | **5e-6**（Phase1 沿用前期 Colab） | 更高 lr（快验） |
| 训练长度 | `num_train_epochs=1`（全量 epoch，无 `max_steps`） | `MAX_STEPS=**50**`（快验） | 步数大幅截断 |
| `NUM_GENERATIONS`（GRPO） | **8**（`tinyllava_trainer_reason.py`） | **8** | **相同**（未精简） |
| `SAVE_STEPS` / limit | 50000 / 1 | **10** / **1** | 快验更密存盘（Drive 成本高） |
| CKPT | ColdStart（官方路径） | 同一 `checkpoints/coldstart` | 对齐冷启动 |
| 训练时长（官方自称） | ~**30 h** / 8×A100-40G | Phase1 A2 约 **2 h** 量级（50 step 快验） | 设定与规模均大幅缩小 |

> 官方默认摘自 [`train_qwen2_reason_nextqa.sh`](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1/blob/main/scripts/train/train_qwen2_reason_nextqa.sh) 与上游 trainer；本项目 C1 以 Phase1 Notebook 为准。写报告时建议单列「相对 TinyLLaVA-Video-R1 的设定精简」，避免把 C1 快验表述为全量复现。

| 项 | Phase 1 已用值（后续默认，同上 C1） |
|----|---------------------------|
| `NUM_GENERATIONS` | **8** |
| `NUM_FRAMES` / `NUM_QUERIES` / `MODEL_MAX_LENGTH` | **2 / 32 / 256** |
| `MAX_STEPS`（快验） | **50**（扩规模时再另开实验，需在报告中单独标注） |
| 数据 | `nextqa_small50.jsonl`（扩规模换 10%/全量时单独成组） |
| `SAVE_STEPS` / `SAVE_TOTAL_LIMIT` | **10 / 1** |
| CKPT | 同一 `checkpoints/coldstart` |

这样 **A2 / C2 / M1 / M5** 可在同一档位下并表对照（训练期 reward + 效率；Phase 5 再对 M1/M5 做官方四基准）。写报告时建议单独一小节「相对 TinyLLaVA-Video-R1 的设定精简」，引用上表，避免把 C1 快验结果直接表述为「复现了官方全量训练」。

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
| 0.5 | 整理 Colab Notebook 入口 | `notebooks/phase1-cppo-g8.ipynb`（别名 `mixup-GRPO优化.ipynb`） | ✅ |
| 0.6 | 从前期工程迁移 CPPO / 策略补丁到 `mixup/` | `mixup/cppo.py`、`mixup/patches/` | ✅ |

**验收**：README 中的 Drive 树与真实 Drive 一致；Colab 能找到 `REPO`、`CKPT`；Drive 上存在本仓库副本。  
**本地进度**：0.1 / 0.4 / 0.5 / 0.6 已在本机 Git 仓库完成；0.2 / 0.3 需打开 Colab 按 `docs/PHASE0_DRIVE_SYNC.md` 勾选。

---

### Phase 1：Colab 基线与 CPPO 放大（第 1–2 周）

**目标**：在 **同一 Drive 路径** 上跑通 g8 对比，争取比 v1 更明显的 CPPO 加速；建立 MixUp 前的效率基线。

**Notebook 入口**：`notebooks/phase1-cppo-g8.ipynb`（同内容别名：`notebooks/mixup-GRPO优化.ipynb`）

| # | 任务 | 说明 | 交付物 | Notebook 章节 | 状态 |
|---|------|------|--------|----------------|------|
| 1.1 | 固定超参 | `NUM_GENERATIONS=8`；快验 `MAX_STEPS=50` | 路径配置单元 | §2 | ✅ |
| 1.2 | **A2** GRPO baseline | 无剪枝 | `outputs/grpo_A_g8/` | §5 + §7 | ✅ |
| 1.3 | **C2** CPPO p=0.5 | 仅改剪枝率 | `outputs/grpo_CPPO_g8_p50/` | §8 | ✅ |
| 1.4 | （可选）**C3** p=0.75 | 加速上限 | `outputs/grpo_CPPO_g8_p75/` | §9 | ⬜ 可选 |
| 1.5 | 对比汇总 | runtime / steps/s / reward | `docs/PHASE1_REPORT.md` + §8 | §10 | ✅ |
| 1.6 | 进度条 + 断点续训 | `SAVE_STEPS=10` + resume | Notebook §2/§6 | §6 | ✅ |

**评估口径（Phase 1–4 训练期）**：用训练日志 `reward` / `accuracy_reward` / `format_reward` / `loss` / `kl` 判断质量是否保持；用 `train_runtime` / `train_steps_per_second`（及去存盘校正）判断效率。  
**正式下游能力**：留到 **Phase 5**，按上游 TinyLLaVA-Video-R1 的 Video-MME / MVBench / MLVU / MMVU 做总评估。

**判定（主）**：相对 A2 的 steps/s 提升，目标 **≥10%**，且 reward 不明显下降。  
**结果**：无中断全程重跑后达成（runtime **−27.4%**，steps/s **+37.5%**；去存盘校正时长 **−31.9%**）。详见 [`docs/PHASE1_REPORT.md`](./docs/PHASE1_REPORT.md)。

**断点 / 云盘**：`SAVE_STEPS=10`、`SAVE_TOTAL_LIMIT=1`；删除的 checkpoint 进回收站仍占配额 → 须清空回收站。

**本地进度**：**Phase 1（A2+C2）已完成**；C3 按需。

---

### Phase 2：锁定档位以复用 Phase 1（第 2 周）

**目标**：确认后续实验继续使用 Phase 1 的 **C1 档**，使 A2/C2 结果可直接参与后续对照；**不再另开大规模档位扫参**（除非 OOM）。

| 档位 | num_gen | num_frames | num_queries | max_length | 用途 |
|------|---------|------------|-------------|------------|------|
| C0 快验 | 4 | 2 | 32 | 256 | 仅调试 / OOM 兜底 |
| **C1（锁定）** | **8** | **2** | **32** | **256** | **Phase 1 已用；后续默认** |
| C2 挑战 | 8–16 | 2–4 | 32–64 | 512 | 显存有余且明确升级时才用（须单独成表，不与 Phase1 混比） |

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 2.1 | 将 C1 超参写入 `configs/colab_c1.yaml`（与 Phase1 Notebook 常量对齐） | `configs/colab_c1.yaml` | ⬜ |
| 2.2 | 约定：Train 默认 C1；换档须在报告声明「不可与 Phase1 直接比」 | 本文件 | ⬜ |
| 2.3 | （可选）文档化 Phase1 实测显存 / steps/s | `docs/MEMORY_BENCHMARK_COLAB.md` | ⬜ 可选 |

---

### Phase 3：单策略模块化——只实现、不单训（第 2–4 周）

**目标**：在 GitHub `mixup/` 实现可开关模块；训练时打补丁到 Drive `REPO`。**本阶段不额外开跑「单模块性能」训练**（避免算力分散）；正确性靠代码审查 + 可选 1–2 step 冒烟。

```
mixup/
├── __init__.py
├── registry.py
├── cppo.py              # ✅ Phase1 已用
├── gfpo.py              # 待实现
├── ngrpo.py             # 待实现
├── project_baseline.py  # 策略 B / 难度+长度（待完善）
└── trainer_mixup.py     # 统一入口（待实现）
```

| 策略 | 任务 | 优先级 | 本阶段训练？ |
|------|------|--------|--------------|
| CPPO | 已参数化 `cppo_pruning_rate` | P0 | ✅ 已在 Phase1 |
| 项目基线增强（B） | 难度 + 长度 reward 解耦 | P0 | ❌ → Phase4 训 **M1** |
| GFPO | Top-k 掩码 | P1 | ❌ → 仅随 **M5** |
| NGRPO | 全错 group 校准 | P1 | ❌ → 仅随 **M5** |

**输出命名（Phase4 起）**：

```
outputs/{run_id}_c1_small50/     # 例：m1_baseline_b_c1_small50 / m5_mixup_c1_small50
  ├── trainer_state.json
  └── ...
```

---

### Phase 4：精简训练——M1 基线 + M5 最终方案（第 4–5 周）

**修订说明**：取消 M0/M2/M3/M4 全矩阵消融。只跑两组正式训练（同 Phase1 **C1** 档 + 同数据设定，便于与 A2/C2 对照）：

| 实验 ID | 组合 | 角色 | 是否训练 |
|---------|------|------|----------|
| （对照）A2 / C2 | GRPO / CPPO p=0.5 | Phase1 已有 | ✅ 已完成，直接引用 |
| **M1** | 项目基线增强（B） | **组内基线**（对应项目组原始设计方向） | ✅ 本阶段训练 |
| ~~M2–M4~~ | 中间组合 | — | ❌ **不跑** |
| **M5** | B + CPPO + NGRPO + GFPO | **个人最终优化方案 / 开源主成果** | ✅ 本阶段训练 |

| # | 任务 | 说明 |
|---|------|------|
| 4.1 | 训 **M1** | 同 C1 超参；输出如 `outputs/m1_baseline_b_c1_small50/` |
| 4.2 | 训 **M5** | 打开 B+CPPO+NGRPO+GFPO；输出如 `outputs/m5_mixup_c1_small50/` |
| 4.3 | 训练期对照表 | M1 / M5 vs Phase1 A2、C2：`reward` + runtime/steps/s（去存盘校正可选） |
| 4.4 | 报告 | `docs/PHASE4_REPORT.md`（说明跳过中间组合、算力集中在 M5） |

在 **同一 Drive 数据 / CKPT / C1 档位** 下完成；若日后扩 10%/全量，须新开实验 ID，勿覆盖 small50 目录。

---

### Phase 5：上游标准总评估（第 5–7 周）

**评估分层（约定）**：

| 层级 | 指标 | 用途 |
|------|------|------|
| 训练期 | `reward` / `accuracy_reward` / `format_reward` / `loss` / `kl` + 效率 | Phase 1–4；含 A2/C2/M1/M5 并表 |
| **总评估（对齐上游）** | **Video-MME、MVBench、MLVU、MMVU**（[TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) `scripts/eval/*.sh`） | **主要评 Phase 4 的 M1（基线）与 M5（最终方案）权重** |

| # | 任务 | 说明 |
|---|------|------|
| 5.1 | 准备评测数据与脚本路径 | 按上游 README 放置 Video-MME / MVBench / MLVU / MMVU |
| 5.2 | 评测 **M1** 权重 | 四基准（可先 1–2 项冒烟再全开） |
| 5.3 | 评测 **M5** 权重 | 同设置、同脚本，与 M1 公平对比 |
| 5.4 | （可选）引用 Phase1 | A2/C2 若保留最终权重且同协议，可附录对照；否则只报告训练期指标 |
| 5.5 | 总评报告 | `docs/PHASE5_EVAL_REPORT.md` + 更新 README 结果表 |

风险：评测数据体积大，需预留 Drive；会话中断时权重留在 `outputs/`；勿与换档实验混比。

---

### Phase 6：开源发布（第 7–8 周）

| # | 任务 |
|---|------|
| 6.1 | README 完善（Drive 结构 + Colab 步骤 + MixUp 开关：M1 / M5） |
| 6.2 | LICENSE（Apache-2.0） |
| 6.3 | 示例 Notebook / 一键复现说明 |
| 6.4 | Phase1+4 训练对照 + Phase5 四基准结果 |
| 6.5 | GitHub Release v0.1.0（主打 **M5** 配方） |

---

## 4. 近期执行清单

- [x] Phase 1：A2 / C2（g8）无中断对比 + `docs/PHASE1_REPORT.md`
- [ ] **Phase 2**：锁定 C1 超参到 `configs/colab_c1.yaml`（与 Phase1 对齐）
- [ ] **Phase 3**：实现 `gfpo.py` / `ngrpo.py` / 完善 `project_baseline.py`（**不单训**）
- [ ] **Phase 4**：训 **M1** → 训 **M5** → 与 A2/C2 训练期对照
- [ ] **Phase 5**：对 **M1 / M5** 跑 Video-MME、MVBench、MLVU、MMVU

---

## 5. 实验管理规范

### 5.1 数据子集

| 文件（位于 `data/dataset/`） | 规模 | 用途 |
|------------------------------|------|------|
| `nextqa_small50.jsonl` | 50 | 冒烟 / 快验 |
| `nextqa_0-30s_10p_seed42.jsonl` | ~550 | 消融 |
| `nextqa_0-30s.jsonl` | ~5496 | 全量 |

### 5.2 公平对比

- 默认锁定 **C1**（与 Phase1 相同）：同一 `DATA_ROOT` / jsonl / `CKPT` / g=8 / frames=2  
- Phase1 **A2 / C2** 可与 Phase4 **M1 / M5** 做训练期并表；换档或换数据规模须单独标注  
- 正式下游能力对比：Phase5 只对 **M1 vs M5** 跑上游四基准（主结论）  

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
[ ] 本次只跑清单内实验（如 M1 或 M5；勿顺手开 M2–M4）
[ ] 结束后确认 outputs 已在 Drive
```

---

## 6. 参考资源

| 资源 | 路径/链接 |
|------|-----------|
| 本仓库 README（含 Drive 树） | [README.md](./README.md) |
| 调研报告 | `doc/…GRPO优化调研报告.pdf` / 实验目录 `优化方案/` |
| 上游仓库 | https://github.com/ZhangXJ199/TinyLLaVA-Video-R1 |
| Phase 1 报告 | [`docs/PHASE1_REPORT.md`](./docs/PHASE1_REPORT.md) |
| Phase 1 原始日志 | `docs/phase1运行记录.ipynb` |
| 上游评测脚本 | TinyLLaVA-Video-R1 `scripts/eval/{videomme,mvbench,mlvu,mmvu}.sh` |

| Drive 项目根 | `MyDrive/MixUpLLaVA-video-r1` |
| Drive 共享包（官方） | https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing |

---

## 7. 进度跟踪

| 阶段 | 计划开始 | 计划完成 | 实际完成 | 备注 |
|------|----------|----------|----------|------|
| Phase 0 | 2026-07-16 | 2026-07-18 | 文档侧 ✅ | Drive 勾选可补 |
| Phase 1 | 2026-07-17 | 2026-07-25 | **2026-07-17** | A2/C2 g8 ✅；见 `docs/PHASE1_REPORT.md` |
| Phase 2 | 2026-07-18 | 2026-07-23 | | 锁定 C1，复用 Phase1 |
| Phase 3 | 2026-07-20 | 2026-08-05 | | 代码模块化，不单训 |
| Phase 4 | 2026-08-05 | 2026-08-20 | | **仅 M1 + M5** |
| Phase 5 | 2026-08-20 | 2026-09-10 | | 四基准评 **M1 vs M5** |
| Phase 6 | 2026-09-10 | 2026-09-20 | | Release（主打 M5） |

---

## 8. v2 实验结果（Phase 1 已填）

详见 **`docs/PHASE1_REPORT.md`**。主列：`steps/s` 与相对 A2 加速；另给出去存盘校正值。

| 策略 | 输出目录 | max_steps | steps/s | 相对A2加速(steps/s) | train_runtime(s) | Cell墙钟 | reward均值 |
|------|----------|-----------|---------|---------------------|------------------|----------|------------|
| A2 | `grpo_A_g8` | 50 | 0.008 | — | 6357.3 | 113.0 min | 0.190 |
| C2 | `grpo_CPPO_g8_p50` | 50 | 0.011 | **+37.5%**（校正吞吐 **+46.9%**） | 4616.4（**−27.4%**） | 83.4 min（**−26.2%**） | 0.321 |

去存盘校正：将步 11/21/31/41/50 耗时替换为非存盘步均值后再比 → 时长加速 **+31.9%**。

---

*最后更新：2026-07-18 · Phase 1 ✅ · 后续：锁定 C1 → 模块化(不单训) → M1+M5 → 四基准评 M1/M5*
