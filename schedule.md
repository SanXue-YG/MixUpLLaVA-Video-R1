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
| **F / G** | GMPO/MAPO、**MO-GRPO** | §3.4–3.5 | 数值稳定 / 多奖励公平 | P1–P2（MO-GRPO 优先备选） |

**交付形态（再修订 · 2026-07-18）**：最终「唯一最优组合」暂不确定。改为：

| 阶段 | 做什么 |
|------|--------|
| **Phase 3** | 模块库：尽量复现调研报告可嵌入方法，统一注册、可扩展 |
| **Phase 4** | **优化方案选择器** + 默认基线 **M1（仅 B）**；用户按目标勾选模块组合训练（流程复用 Phase1 CPPO 打通路径） |
| **Phase 5** | 对**用户所选**（及推荐对照）组合做训练期指标 + 上游四基准评测 |

配方备选与准确率向建议见 [`docs/ACCURACY_ORIENTED_OPTIONS.md`](./docs/ACCURACY_ORIENTED_OPTIONS.md)（如曾议的 B+CPPO+NGRPO+GFPO / M5q 等，均为**可选预设**，非强制最终方案）。

**暂缓但预留接口**：StepGRPO、GRPO-A、GFPO 动态难度采样、结构性跳出 Video-R1 的范式（调研 §4）——Phase 3 可先 stub / 文档化，实现优先级靠后。

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
Phase 1  Colab 基线 / CPPO(g8)    ██████████  Week 1–2 ✅ 打通可复用训练流程
Phase 2  默认档位 + 环境可调参   ████████░░  Week 2（默认=Phase1；部署可改）
Phase 3  MixUp 模块库（可扩展）    ████████░░  Week 2–4 复现调研可嵌入方案
Phase 4  优化方案选择器 + M1 基线  ████████░░  Week 4–5 用户自选组合训练
Phase 5  评测所选方案              ██████████  Week 5–7 四基准 + 训练期指标
Phase 6  开源文档与 Release        ██████████  Week 7–8 主打「可组合 GRPO 实验台」
```

### 跨阶段对比约定（重要）

**Phase 1** 已在 Colab A100 上用一组参数成功复现 CPPO 相对 GRPO 的优化策略（吞吐↑、reward 未见掉）。**Phase 2 起将该组参数记为默认档（C1）**，便于复用 A2/C2；**其它部署环境（显存、卡型、数据规模不同）可灵活改参试其它设置**，不强制锁死唯一档位。

换档原则：

1. **同表可比**：与 Phase1 A2/C2 直接并表时，须保持同一档位（默认 C1）。  
2. **环境试参**：换机器 / OOM / 扩规模时，可改 `num_gen`、frames、queries、`max_length`、`max_steps`、数据量等；**须在报告中单独成组**，不与 Phase1 默认档混比。  
3. **推荐路径**：新环境先跑默认 C1 冒烟 → 再按需试 C0 / C2 或自定义档。

下表同时列出上游 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) 官方 GRPO 脚本默认值，便于报告中写清「相对原文精简了多少」。

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

在**默认 C1**下，**A2 / C2 / M1 / 任意自选组合** 可并表对照（训练期 reward + 效率；Phase 5 对所选组合做官方四基准）。其它环境改参后的结果单独列表。写报告时建议单列「相对 TinyLLaVA-Video-R1 的设定精简」，避免把 C1 快验表述为官方全量复现。

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

### Phase 2：默认档位 + 部署环境可调参（第 2 周）

**背景**：Phase 1 已在 Colab A100 上验证——在当前参数设置下，CPPO（C2）相对 GRPO（A2）可复现预期优化策略（吞吐提升、reward 未见系统性下降）。该组参数可作为**后续默认值**。

**目标**：

1. 把 Phase1 成功档位固化为 **默认配置（C1）**，方便复用 A2/C2 与后续模块对比。  
2. 明确：**其它部署环境不必死锁 C1**——可按显存 / 卡型 / 时间预算灵活改参；试其它设置时单独记录，不与默认档混比。  
3. 提供可选档位与「如何改参」文档，而不是做穷尽网格搜索。

| 档位 | num_gen | num_frames | num_queries | max_length | 用途 |
|------|---------|------------|-------------|------------|------|
| C0 | 4 | 2 | 32 | 256 | 调试 / 小显存兜底 |
| **C1（默认）** | **8** | **2** | **32** | **256** | **Phase1 已验证；推荐起点** |
| C2 | 8–16 | 2–4 | 32–64 | 512 | 显存有余时升级（单独成表） |
| 自定义 | 用户自定 | … | … | … | 新环境试参；报告注明环境与改动项 |

**可调参数建议**（按部署环境灵活尝试，非强制扫全表）：

| 参数 | Phase1 默认 | 何时改 |
|------|-------------|--------|
| `NUM_GENERATIONS` | 8 | OOM → 降到 4；显存充足可试 16（耗时↑） |
| `NUM_FRAMES` / `NUM_QUERIES` / `MODEL_MAX_LENGTH` | 2 / 32 / 256 | 显存紧再降；要更接近官方再升（须单独成组） |
| `MAX_STEPS` | 50 | 冒烟可 10–20；严肃对比可加长 |
| 数据 | `nextqa_small50` | 扩到 10%/全量须单独成组 |
| DeepSpeed / offload | ZeRO-3 + 参数 offload | 多卡 / 更大显存可关掉 offload 提速 |
| `cppo_pruning_rate` | 0.5（Phase1 C2） | 新环境可再试 0.3/0.7 等（与模块选择分开记） |

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 2.1 | 将 Phase1 默认超参写入 `configs/colab_c1.yaml`（与 Notebook 常量对齐） | `configs/colab_c1.yaml` | ⬜ |
| 2.2 | 文档约定：默认 = C1；其它环境可改参；改档须单独成组、不可与 Phase1 默认档混比 | 本文件 + README | ⬜ |
| 2.3 | （可选）简短「换环境改参清单」+ Phase1 实测显存 / steps/s | `docs/MEMORY_BENCHMARK_COLAB.md` | ⬜ 可选 |
| 2.4 | （可选）Notebook / 选择器入口暴露可覆盖字段（覆盖 yaml 默认） | Notebook 或 `configs/` 注释 | ⬜ 可选 |

---

### Phase 3：MixUp 模块库——尽量复现调研方案、可扩展（第 2–4 周）

**目标**：在 `mixup/` 建成**可注册、可开关、可继续扩展**的 GRPO 改进模块库，尽可能覆盖调研报告 §3 中「可直接嵌入 Video-R1 / TinyLLaVA-Video-R1」的方法。本阶段以**实现 + 单元/冒烟**为主，**不强制**为每个模块单独开完整 50-step 评测训练（完整训练留给 Phase 4 选择器）。

```
mixup/
├── __init__.py
├── registry.py              # 模块注册表：name → apply(config)
├── config.py                # MixUpConfig / 开关与超参 schema
├── project_baseline.py      # B：难度感知 + 长度 shaping（M1 默认开）
├── cppo.py                  # ✅ Phase1 已通
├── gfpo.py
├── ngrpo.py
├── mo_grpo.py               # 多目标 normalize-then-sum（准确率向备选）
├── dagrpo.py / gmpo.py / mapo.py / grpo_a.py  # P1–P2，可先 stub
├── trainer_mixup.py         # 将开关打到上游 trainer
└── presets/                 # 可选预设 JSON/YAML（非强制最终方案）
    ├── m1_baseline_b.yaml
    ├── speed_cppo.yaml      # 对齐 Phase1 C2
    └── ...                  # 用户或文档推荐组合
```

| 优先级 | 模块 | 调研位置 | Phase3 交付 |
|--------|------|----------|-------------|
| P0 | **B** project_baseline、**CPPO** | §2.3 / §3.1.1 | 完整可用（CPPO 已验证） |
| P1 | **GFPO、NGRPO、MO-GRPO** | §3.1.2 / §3.3.2 / §3.5 | 完整可用 + 注册进选择器 |
| P2 | DaGRPO、GMPO、MAPO、GRPO-A | §3.3–3.4 | 实现或清晰 stub + 接口 |
| 预留 | StepGRPO 等 | §3.2 / §4 | 文档 + 接口占位，不强求首期完整 |

| # | 任务 | 说明 |
|---|------|------|
| 3.1 | `registry` + `MixUpConfig` | 布尔开关 / 超参；未知模块报错清晰 |
| 3.2 | 各模块 `apply()` | 尽量局部改 advantage / mask / prune / reward，贴合调研「修改位置」 |
| 3.3 | 打补丁到 Drive `REPO` | 复用 Phase1「覆盖 trainer → deepspeed 训练」流程 |
| 3.4 | 扩展性 | 新方法 = 新文件 + `register()`，不必改 Notebook 训练主循环 |
| 3.5 | 文档 | `docs/MIXUP_MODULES.md`：模块列表、开关名、与调研章节对照 |

**验收**：至少 B + CPPO +（GFPO 或 NGRPO 或 MO-GRPO 之一）可通过配置启用；新增模块有注册示例。

---

### Phase 4：优化方案选择器 + M1 基线训练（第 4–5 周）

**定位变更**：不再锁定「唯一最终配方 M5」。开源主成果改为 **可组合的 GRPO 优化实验台**：用户按目标自选模块组合，沿 **Phase1 已打通的 Colab+Drive 流程**（默认 C1，部署环境可改参）快速试其它算法。

**默认基线**：**M1 = 仅开启项目基线增强 B**（对应项目组原始设计方向）。Phase1 的 A2（纯 GRPO）/ C2（仅 CPPO）继续作为历史对照，不必重跑。

| # | 任务 | 说明 |
|---|------|------|
| 4.1 | **选择器 UI/配置** | Notebook 或 YAML：勾选模块 + 填超参（如 `cppo_pruning_rate`、GFPO `k`）；生成 `run_id` 与 `outputs/{run_id}_c1_*/` |
| 4.2 | **预设（可选，非强制）** | 如 `speed`（B?+CPPO）、`stable`（B+NGRPO）、`quality`（B+NGRPO+MO-GRPO）等，见 `ACCURACY_ORIENTED_OPTIONS.md`；用户可改 |
| 4.3 | 训 **M1** | 选择器默认配置跑通一遍，作为组内基线权重 |
| 4.4 | 训 **≥1 组自选组合** | 演示「换开关即可训练」；组合由实习目标当时决定，写入当次报告 |
| 4.5 | 报告模板 | `docs/PHASE4_REPORT.md`：记录所选开关、与 A2/C2/M1 的训练期对照 |

**选择器示意（Notebook）**：

```python
MIXUP = dict(
    project_baseline=True,   # M1 默认 True
    cppo=False, cppo_pruning_rate=0.5,
    gfpo=False, gfpo_top_k=4,
    ngrpo=False,
    mo_grpo=False,
    # dagrpo=False, gmpo=False, ...
)
# → apply_mixup(MIXUP) → run_training(...)  # 同 Phase1 流程
```

#### 预计成果（Phase 4）

1. **可交付物**：`mixup` 选择器（Notebook/配置）+ **M1** 基线 run + 至少一套「自选组合」run；配置快照随 `outputs/` 落盘。  
2. **产品叙事**：他人 / 未来的自己无需重写训练脚本，即可在默认流程上开关调研中的算法做对比；换部署环境时可改超参再试。  
3. **成功标准**：从改 YAML/单元开关到开训 ≤ 既有 Phase1 复杂度；错误开关有明确报错；M1 可复现。

---

### Phase 5：评测所选优化方案（第 5–7 周）

**评估分层**：

| 层级 | 指标 | 用途 |
|------|------|------|
| 训练期 | `reward` / `accuracy_*` / `format_*` / `loss` / `kl` + runtime / steps/s | 快筛组合；对照 A2/C2/M1 |
| **总评估** | **Video-MME、MVBench、MLVU、MMVU** | 对 **M1** 与 **Phase4 选定的目标组合**（可多组）正式对比 |

| # | 任务 | 说明 |
|---|------|------|
| 5.1 | 准备评测数据与上游 `scripts/eval/*.sh` | 同 TinyLLaVA-Video-R1 流程 |
| 5.2 | 评测 **M1** | 组内基线分数 |
| 5.3 | 评测 **所选组合** | 与 M1 同协议；若多组自选则分别评并制表 |
| 5.4 | （可选）附录 Phase1 | A2/C2 有最终权重则可附；否则仅训练期对照 |
| 5.5 | 总评报告 | `docs/PHASE5_EVAL_REPORT.md`：写清「本次评了哪些开关组合、为何选」 |

#### 预计成果（Phase 5）

1. **可交付物**：所选方案 vs M1 的四基准表 + 训练期效率/reward 表。  
2. **成功标准**：结论可复述为「在选择器中启用 {模块列表} 后，相对 M1 …」；并注明所用超参档位（默认 C1 或环境自定义）。  
3. **不预设**必须打败某一固定 M5；以**当次选定组合**为准。

风险：评测数据体积大；每多一组权重多一轮评测成本——优先评 M1 + 1～2 个候选。

---

### Phase 6：开源发布（第 7–8 周）

| # | 任务 |
|---|------|
| 6.1 | README：选择器用法 + 模块对照调研章节 + Drive/Colab 流程（Phase1 传承） |
| 6.2 | LICENSE（Apache-2.0） |
| 6.3 | 示例：仅 M1、仅 CPPO、以及一两个推荐预设 |
| 6.4 | Phase1 效率结论 + Phase4/5 当次组合结果 |
| 6.5 | Release v0.1.0（主打 **可组合 GRPO 实验台**，而非单一神秘配方） |

---

## 4. 近期执行清单

- [x] Phase 1：A2 / C2（g8）无中断对比 + `docs/PHASE1_REPORT.md`（**流程已打通**）
- [ ] **Phase 2**：默认 C1（=Phase1）→ `configs/colab_c1.yaml`；文档写清「部署环境可改参」
- [ ] **Phase 3**：MixUp **模块库**（B/CPPO/GFPO/NGRPO/MO-GRPO… + registry，可扩展）
- [ ] **Phase 4**：**优化方案选择器** + 默认训 **M1**；再跑至少 1 组自选组合
- [ ] **Phase 5**：评测 **M1 + 所选组合**（四基准）

---

## 5. 实验管理规范

### 5.1 数据子集

| 文件（位于 `data/dataset/`） | 规模 | 用途 |
|------------------------------|------|------|
| `nextqa_small50.jsonl` | 50 | 冒烟 / 快验 |
| `nextqa_0-30s_10p_seed42.jsonl` | ~550 | 消融 |
| `nextqa_0-30s.jsonl` | ~5496 | 全量 |

### 5.2 公平对比

- 默认 **C1**（与 Phase1 相同）：同一 `DATA_ROOT` / jsonl / `CKPT` / g=8 / frames=2；换环境可覆盖 yaml  
- 与 Phase1 并表对比时保持同档；改参结果单独成组
- Phase1 **A2 / C2** 可与 Phase4 **M1 / 自选组合** 做训练期并表；换档或换数据规模须单独标注  
- 正式下游：Phase5 对 **M1 vs 当次所选组合** 跑上游四基准（可多组，但须同协议）  
- 每次 run 必须保存 **MixUp 开关快照**（否则无法复现「选了什么」）  

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
[ ] 本次开关快照已记录；只跑计划中的 M1 / 自选组合
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
| **准确率向备选方案** | [`docs/ACCURACY_ORIENTED_OPTIONS.md`](./docs/ACCURACY_ORIENTED_OPTIONS.md) |
| 上游评测脚本 | TinyLLaVA-Video-R1 `scripts/eval/{videomme,mvbench,mlvu,mmvu}.sh` |

| Drive 项目根 | `MyDrive/MixUpLLaVA-video-r1` |
| Drive 共享包（官方） | https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing |

---

## 7. 进度跟踪

| 阶段 | 计划开始 | 计划完成 | 实际完成 | 备注 |
|------|----------|----------|----------|------|
| Phase 0 | 2026-07-16 | 2026-07-18 | 文档侧 ✅ | Drive 勾选可补 |
| Phase 1 | 2026-07-17 | 2026-07-25 | **2026-07-17** | A2/C2 g8 ✅；见 `docs/PHASE1_REPORT.md` |
| Phase 2 | 2026-07-18 | 2026-07-23 | | 默认 C1 + 环境可调 |
| Phase 3 | 2026-07-20 | 2026-08-10 | | **模块库**（可扩展复现调研方案） |
| Phase 4 | 2026-08-05 | 2026-08-22 | | **选择器** + M1 + 自选组合训练 |
| Phase 5 | 2026-08-20 | 2026-09-10 | | 评测 **M1 + 所选方案** |
| Phase 6 | 2026-09-10 | 2026-09-20 | | Release（可组合实验台） |

---

## 8. v2 实验结果（Phase 1 已填）

详见 **`docs/PHASE1_REPORT.md`**。主列：`steps/s` 与相对 A2 加速；另给出去存盘校正值。

| 策略 | 输出目录 | max_steps | steps/s | 相对A2加速(steps/s) | train_runtime(s) | Cell墙钟 | reward均值 |
|------|----------|-----------|---------|---------------------|------------------|----------|------------|
| A2 | `grpo_A_g8` | 50 | 0.008 | — | 6357.3 | 113.0 min | 0.190 |
| C2 | `grpo_CPPO_g8_p50` | 50 | 0.011 | **+37.5%**（校正吞吐 **+46.9%**） | 4616.4（**−27.4%**） | 83.4 min（**−26.2%**） | 0.321 |

去存盘校正：将步 11/21/31/41/50 耗时替换为非存盘步均值后再比 → 时长加速 **+31.9%**。

---

*最后更新：2026-07-18 · Phase 1 ✅ · 后续：默认 C1（可调）→ 模块库 → **选择器+M1** → 评测所选方案*
