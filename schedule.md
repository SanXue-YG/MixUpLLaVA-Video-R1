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
| **Phase 4** | **优化方案选择器**（能力交付）：默认勾选 **B+CPPO**；再叠其它模块；**不强制立刻开训** |
| **Phase 5** | **评测流水线**（脚本 + 报告模板）：协议对齐上游四基准；**实跑可延后** |
| **Phase 6** | **总控制台 Notebook**：汇总前期成果；用户直观调参 + 选策略 → 先训基线出报告 → 再训优化方案并对比 |

**执行节奏（重要）**：当前优先**跑通并完成整条项目流程与交付物**（配置、模块库、选择器、评测模板、总控制台）。**大规模 / 正式训练不卡进度**——需要训时，借助 Phase 4 选择能力 + Phase 6 总控制台按需启动。

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
Phase 2  默认档位 + 环境可调参   ██████████  Week 2 ✅ 默认=C1；可覆盖加载
Phase 3  MixUp 模块库（可扩展）    ██████████  Week 2–4 ✅ 调研可嵌入方案已注册
Phase 4  优化方案选择器（能力）    ██████████  Week 4–5 ✅ 可勾选；开训可延后
Phase 5  评测流水线（训练期为主） ██████████  Week 5–7 ✅ 总评估可选预留
Phase 6  总控制台 Notebook         ██████████  Week 7–8 调参→基线→优化对比→报告
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
| 2.1 | 将 Phase1 默认超参写入 `configs/colab_c1.yaml`（与 Notebook 常量对齐） | `configs/colab_c1.yaml`（另附 `colab_c0.yaml`） | ✅ |
| 2.2 | 文档约定：默认 = C1；其它环境可改参；改档须单独成组、不可与 Phase1 默认档混比 | 本文件 + README + `configs/README.md` | ✅ |
| 2.3 | 简短「换环境改参清单」+ Phase1 实测吞吐 | `docs/MEMORY_BENCHMARK_COLAB.md` | ✅ |
| 2.4 | Notebook / 选择器入口暴露可覆盖字段（覆盖 yaml 默认） | `mixup/config_loader.py` | ✅ |

**本地进度**：**Phase 2 已完成**（默认档固化 + 可覆盖加载；无需再开训）。

---

### Phase 3：MixUp 模块库——尽量复现调研方案、可扩展（第 2–4 周）

**目标**：在 `mixup/` 建成**可注册、可开关、可继续扩展**的 GRPO 改进模块库，尽可能覆盖调研报告 §3 中「可直接嵌入 Video-R1 / TinyLLaVA-Video-R1」的方法。本阶段以**实现 + 单元/冒烟**为主，**不强制**为每个模块单独开完整 50-step 评测训练（完整训练留给 Phase 4 选择器）。

```
mixup/
├── __init__.py
├── registry.py              # 模块注册表：name → apply(config)
├── config.py                # MixUpConfig / 开关与超参 schema
├── project_baseline.py      # B：难度感知 + 长度 shaping（**默认开**）
├── cppo.py                  # ✅ Phase1 已通（**默认开**）
├── gfpo.py
├── ngrpo.py
├── mo_grpo.py               # 多目标 normalize-then-sum（准确率向备选）
├── dagrpo.py / gmpo.py / mapo.py / grpo_a.py  # P1–P2，可先 stub
├── trainer_mixup.py         # 将开关打到上游 trainer
└── presets/                 # 可选预设 JSON/YAML（非强制最终方案）
    ├── m1_baseline_b.yaml   # 默认 = B+CPPO
    ├── speed_cppo.yaml      # 消融：仅 CPPO（显式关 B）
    └── ...
```

> **项目默认保留**：B 与 CPPO 在后续策略选择中**默认包含**；仅配置/选择器里显式关闭时才去掉。

| 优先级 | 模块 | 调研位置 | Phase3 交付 |
|--------|------|----------|-------------|
| P0 | **B** project_baseline、**CPPO** | §2.3 / §3.1.1 | 完整可用（CPPO 已验证） |
| P1 | **GFPO、NGRPO、MO-GRPO** | §3.1.2 / §3.3.2 / §3.5 | 完整可用 + 注册进选择器 |
| P2 | DaGRPO、GMPO、MAPO、GRPO-A | §3.3–3.4 | 实现或清晰 stub + 接口 |
| 预留 | StepGRPO 等 | §3.2 / §4 | 文档 + 接口占位，不强求首期完整 |

| # | 任务 | 说明 | 状态 |
|---|------|------|------|
| 3.1 | `registry` + `MixUpConfig` | 布尔开关 / 超参；未知模块报错清晰 | ✅ |
| 3.2 | 各模块 `apply()` | 局部改 advantage / mask / prune / reward | ✅ P0–P1 完整；P2 可用/stub |
| 3.3 | 打补丁到 Drive `REPO` | `apply_mixup_to_repo` + `patches/*_mixup.py` | ✅ |
| 3.4 | 扩展性 | 新方法 = 新文件 + `@register()` | ✅ |
| 3.5 | 文档 | `docs/MIXUP_MODULES.md` + `doc/` 原论文 | ✅ |

**验收**：B + CPPO + GFPO/NGRPO/MO-GRPO 均可配置启用；`python -m mixup.tests_smoke` 通过。  
**本地进度**：**Phase 3 已完成**（模块库 + 论文入库；无需开训）。

---

### Phase 4：优化方案选择器（第 4–5 周）——能力优先，开训可延后

**定位变更**：不再锁定「唯一最终配方 M5」。开源主成果改为 **可组合的 GRPO 优化实验台**：用户按目标自选模块组合，沿 **Phase1 已打通的 Colab+Drive 流程**（默认 C1，部署环境可改参）快速试其它算法。

**默认基线定义**：**M1 = 项目默认栈 = B（自研质量向）+ CPPO（原定效率目标）**。  
加开 GFPO / NGRPO / MO-GRPO 等时，**除非显式写 `false`，否则 B 与 CPPO 保持开启**。  
Phase1 的 A2（纯 GRPO）/ C2（仅 CPPO）作历史/消融对照，见 `presets/ablation_*`、`speed_cppo`。

**本阶段重点**：把「选什么、怎么配」做成可调用能力（供 Phase 6 总控制台接入）。**正式训 M1 / 自选组合不强制在本阶段完成**——需要时再通过总控制台启动。

| # | 任务 | 说明 | 状态 |
|---|------|------|------|
| 4.1 | **选择器 UI/配置** | `mixup/selector.py` + `notebooks/phase4-selector.ipynb`；默认 B+CPPO；`run_id` + 快照 | ✅ |
| 4.2 | **预设（可选）** | 别名 `m1`/`speed`/`m5`/`m5q`/… → `mixup/presets/` | ✅ |
| 4.3 | **训练入口封装** | `mixup/training_entry.py`：`prepare_training` / `apply_mixup`；开训可延后 | ✅ |
| 4.4 | 报告模板 | `docs/PHASE4_REPORT.md` + 每次 run 的 stub | ✅ |
| 4.5 | （按需）训 M1 / 自选 | **非门禁**；有项目需要或验收演示时再跑 | ⬜ 按需 |

**验收**：改开关即可生成合法 `RunPlan` 与 `outputs/{run_id}/`；未知开关报错清晰；`python -m mixup.tests_smoke` 含选择器用例。  
**本地进度**：**Phase 4 能力已完成**（开训非门禁）。

---

### Phase 5：评测流水线（第 5–7 周）——训练期为主，总评估可选

**评估分层**：

| 层级 | 指标 | 用途 | Phase 5 态度 |
|------|------|------|----------------|
| **训练期（主路径）** | `reward` / `accuracy_*` / `format_*` / `loss` / `kl` + runtime / steps/s | 快筛；对照 A2/C2/M1 | ✅ **必做交付** |
| **总评估（可选）** | Video-MME、MVBench、MLVU、MMVU | 与上游同协议的正式对比 | ⬜ **仅预留框架**；数据约 **~600GB**，小规模不强制 |

**背景**：按 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) README 下载四基准体积巨大，不适合默认走 Drive/Colab 快验。故 Phase 5 与「能力优先、实跑可延后」一致：**先完成训练期评估流水线**；总评估由实验者在空间充足时自行下载再跑。

| # | 任务 | 交付物 | 状态 |
|---|------|--------|------|
| 5.1 | **训练期评估入口** | `mixup/eval_training.py`：读 `trainer_state`/日志 → JSON + MD | ✅ |
| 5.2 | 训练期报告模板 | `docs/PHASE5_EVAL_REPORT.md`（A 节为主） | ✅ |
| 5.3 | **总评估框架（可选）** | `scripts/eval/*` + `mixup/eval_benchmarks.py` + `docs/PHASE5_BENCHMARKS.md` | ✅ 预留 |
| 5.4 | 上游脚本对齐说明 | 同 TinyLLaVA `scripts/eval/*.sh`；下载需人工 | ✅ |
| 5.5 | （按需）实跑四基准 | **非门禁**；确认磁盘后再 `confirm_large_download` | ⬜ 按需 |
| 5.6 | （可选）附录 Phase1 | A2/C2 训练期效率对照 | ✅ 模板内 |

**验收**：对任意 `outputs/{run_id}/` 可生成训练期报告；总评估脚本可 dry-run / 检查数据是否就位；**不要求**本机已有四基准数据。

**本地进度**：**Phase 5 已完成（训练期流水线 + 总评估预留）**。

---

### Phase 6：总控制台 Notebook + 开源交付（第 7–8 周）

**定位**：汇总 Phase 0–5 成果的**使用者入口**。用 **Jupyter Notebook** 让人直观调参、勾选优化策略，并按固定实验节奏产出报告。开源叙事 = **可组合 GRPO 实验台**，总控制台是其主界面。

**当前节奏提醒**：本阶段以**交付可运行的总控制台与文档**为主；单元格内的「训练 / 评测」在需要时再执行（Colab A100），不要求立刻把所有模型训完。

#### 总控制台工作流（固定顺序）

```
① 环境与路径（Drive / REPO / 默认 C1，可改）
        ↓
② 调参 + 选择优化策略（调用 Phase4 选择器）
        ↓
③ 【基线】按当前参数训 M1（或指定基线）→ 训练期指标
        ↓
④ 【基线评测】**默认训练期报告**；四基准总评估仅当数据已下载时可选
        ↓
⑤ 【优化方案】勾选其它模块组合 → 同参再训
        ↓
⑥ 【对比】相对基线：**训练期**效率/reward（+ 可选四基准 Δ）
```

| # | 任务 | 交付物 | 说明 |
|---|------|--------|------|
| 6.1 | **总控制台 Notebook** | `notebooks/mixup_console.ipynb` | 分节：环境 → 参数 → 策略 → 基线训/评 → 方案训/评 → 对比报告 |
| 6.2 | 接入前期能力 | 同 Notebook | Phase2 默认 yaml；Phase3 registry；Phase4 选择器；Phase5 评测入口；Phase1 流程 |
| 6.3 | **基线报告生成** | `outputs/{run}/baseline_report.md`（或 docs） | 参数快照 + 训练期表 +（可选）四基准 |
| 6.4 | **对比报告生成** | `outputs/{run}/compare_vs_baseline.md` | 优化方案 vs 基线：steps/s、reward、评测 Δ |
| 6.5 | 成果汇总页 | Notebook 首节或 `docs/CONSOLE_GUIDE.md` | Phase1 效率结论摘要、模块列表、档位约定、使用步骤 |
| 6.6 | README / LICENSE / Release | README、Apache-2.0、v0.1.0 | 主推总控制台用法；预设示例（M1、CPPO、推荐组合） |

**Notebook 分区建议**：

| 分区 | 内容 |
|------|------|
| A. 总览 | 项目目标、Phase1 结论、模块一览、默认 C1 |
| B. 环境 | 挂载 Drive、路径检查、依赖 |
| C. 训练超参 | 覆盖 `colab_c1.yaml`（g / frames / steps / 数据等） |
| D. 策略选择 | Phase4 开关 + 预设一键填入 |
| E. 基线实验 | Train baseline → Eval → 写 `baseline_report` |
| F. 优化实验 | 改 MIXUP → Train → Eval → 写 `compare_vs_baseline` |
| G. 历史对照 | （可选）挂载 Phase1 A2/C2 训练期数字 |

#### 预计成果（Phase 6）

1. **可交付物**：一个可打开即用的总控制台 + 报告自动落盘约定 + 开源 README。  
2. **成功标准**：使用者无需翻多个阶段 Notebook，即可完成「调参 → 基线 → 优化 → 对比报告」；未开训时也可浏览参数与策略说明。  
3. **与「先流程、后训练」一致**：流程交付完成即 Phase 6 可验收；训练单元格保留，按需在 A100 上执行。

---

## 4. 近期执行清单

- [x] Phase 1：A2 / C2（g8）无中断对比 + `docs/PHASE1_REPORT.md`（**流程已打通**）
- [x] **Phase 2**：默认 C1 → `configs/colab_c1.yaml` + `mixup.config_loader` + `docs/MEMORY_BENCHMARK_COLAB.md`
- [x] **Phase 3**：MixUp **模块库**（B/CPPO/GFPO/NGRPO/MO-GRPO… + registry；见 `docs/MIXUP_MODULES.md`）
- [x] **Phase 4**：**优化方案选择器**（`mixup/selector.py` + `notebooks/phase4-selector.ipynb`）；开训可延后
- [x] **Phase 5**：**训练期评估流水线** + 四基准框架预留（~600GB，不强制）
- [ ] **Phase 6**：**总控制台 Notebook**（调参 → 基线报告 → 优化对比报告）+ Release

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
- 正式下游：经总控制台对 **基线 vs 当次优化方案** 跑上游四基准（可多组，须同协议）  
- 每次 run 必须保存 **MixUp 开关快照**（否则无法复现「选了什么」）  
- **先流程后训练**：Phase 2–6 以交付物就位为准；A100 开训按项目需要触发，不作为阶段门禁  

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
[ ] 本次开关快照已记录；按总控制台流程：先基线再优化方案
[ ] 结束后确认 outputs / 报告已在 Drive
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
| **Phase 2 默认档 / 改参** | [`configs/colab_c1.yaml`](./configs/colab_c1.yaml) · [`docs/MEMORY_BENCHMARK_COLAB.md`](./docs/MEMORY_BENCHMARK_COLAB.md) |
| **Phase 3 模块库** | [`docs/MIXUP_MODULES.md`](./docs/MIXUP_MODULES.md) · [`doc/`](./doc/README.md) 原论文 |
| **Phase 4 选择器** | [`notebooks/phase4-selector.ipynb`](./notebooks/phase4-selector.ipynb) · [`docs/PHASE4_REPORT.md`](./docs/PHASE4_REPORT.md) |
| **Phase 5 评测** | [`docs/PHASE5_EVAL_REPORT.md`](./docs/PHASE5_EVAL_REPORT.md) · [`docs/PHASE5_BENCHMARKS.md`](./docs/PHASE5_BENCHMARKS.md)（总评估可选） |
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
| Phase 2 | 2026-07-18 | 2026-07-23 | **2026-07-18** | 默认 C1 + loader + 换环境清单 |
| Phase 3 | 2026-07-20 | 2026-08-10 | **2026-07-18** | 模块库 + doc 原论文 + smoke ✅ |
| Phase 4 | 2026-08-05 | 2026-08-22 | **2026-07-18** | 选择器能力 ✅；开训非门禁 |
| Phase 5 | 2026-08-20 | 2026-09-10 | **2026-07-18** | 训练期评估 ✅；四基准可选预留 |
| Phase 6 | 2026-09-10 | 2026-09-20 | | **总控制台** + Release |

---

## 8. v2 实验结果（Phase 1 已填）

详见 **`docs/PHASE1_REPORT.md`**。主列：`steps/s` 与相对 A2 加速；另给出去存盘校正值。

| 策略 | 输出目录 | max_steps | steps/s | 相对A2加速(steps/s) | train_runtime(s) | Cell墙钟 | reward均值 |
|------|----------|-----------|---------|---------------------|------------------|----------|------------|
| A2 | `grpo_A_g8` | 50 | 0.008 | — | 6357.3 | 113.0 min | 0.190 |
| C2 | `grpo_CPPO_g8_p50` | 50 | 0.011 | **+37.5%**（校正吞吐 **+46.9%**） | 4616.4（**−27.4%**） | 83.4 min（**−26.2%**） | 0.321 |

去存盘校正：将步 11/21/31/41/50 耗时替换为非存盘步均值后再比 → 时长加速 **+31.9%**。

---

*最后更新：2026-07-18 · Phase 1–5 ✅ · 后续：**总控制台***
