# MixUpLLaVA-Video-R1

**MixUpLLaVA-Video-R1** 是在 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) 基线上进行的 **GRPO 混合优化**升级版本。项目在保持 SFT + GRPO 总体框架不变的前提下，将多种可互补的 GRPO 改进方案（如 CPPO、GFPO、NGRPO 及项目内 difficulty-aware reward shaping）进行模块化实现与组合评估。

**主运行环境**：Google Colab（推荐 A100）+ Google Drive（数据 / 权重 / 输出持久化）。  
Windows 本机原生 DeepSpeed 训练流程不稳定，**不作为正式训练路径**；日常可在本地编辑代码与 Notebook，训练与验证在 Colab 上完成。

## 背景

- 项目基础：[TinyLLaVA-Video-R1](https://arxiv.org/abs/2504.09641)（[GitHub](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)）
- 优化方法论：`doc/基于Video-R1的视频交通异常行为检测项目中的GRPO优化调研报告.pdf`
- 前期 CPPO 验证：Colab + Drive，50 条子集复现（加速约 0.7%，reward 与 baseline 一致）
- GitHub：https://github.com/SanXue-YG/MixUpLLaVA-Video-R1
- Google Drive 共享包（供参考）：[MixUpLLaVA-video-r1](https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing)（含 `repo` / `data` / `checkpoints` / `outputs`）

## MixUp 优化思路

将调研中的 GRPO 改进做成**可开关模块**，经 **选择器** 组合；最终由 **Phase 6 总控制台 Notebook** 统一调参、训基线、训优化方案并出对比报告。正式训练按需触发，不阻塞流程建设。推荐组合见 [`docs/ACCURACY_ORIENTED_OPTIONS.md`](./docs/ACCURACY_ORIENTED_OPTIONS.md)。

| 模块 | 作用 |
|------|------|
| 项目基线增强（B） | Difficulty-aware advantage + 自适应长度 reward（**M1 默认基线**） |
| CPPO | \|advantage\| completion 剪枝，训练加速（Phase1 ✅） |
| GFPO | Top-k 优势掩码，抑冗余推理 |
| NGRPO | 负信号增强，缓解全错 group 无梯度 |
| MO-GRPO 等 | 多目标公平 / 其它调研可嵌入方案（模块库扩展） |

详细计划见 **[schedule.md](./schedule.md)**。

## 项目状态

🚧 **开发中** — **Phase 1–2 已完成**（CPPO 流程 + 默认档 C1）。当前节奏：**先完成整条项目流程与交付物**；正式训练延后，需要时经 **Phase 6 总控制台**（调用 Phase 4 选择器）按需开训。最终单一「M5 配方」暂不锁定。

- [x] 项目目录与文档（GitHub 骨架）
- [x] 实验计划表（对齐 Drive 成功路径 + 共享链接）
- [x] `doc/` 调研报告与 CPPO 论文
- [x] `notebooks/phase1-cppo-g8.ipynb`（断点续训 + 进度条）
- [x] `mixup/` CPPO 模块与 trainer patches
- [x] Phase 1：A2 vs C2（g=8）效率对比 — 见 [`docs/PHASE1_REPORT.md`](./docs/PHASE1_REPORT.md)
- [x] Phase 2：默认 C1（`configs/colab_c1.yaml`）+ 可覆盖加载（`mixup.config_loader`）+ [`docs/MEMORY_BENCHMARK_COLAB.md`](./docs/MEMORY_BENCHMARK_COLAB.md)
- [ ] Phase 3：**MixUp 模块库**（尽量复现调研可嵌入方案，可扩展）
- [ ] Phase 4：**优化方案选择器**（能力交付；完整开训可延后）
- [ ] Phase 5：**评测流水线**（四基准脚本/模板；实跑可延后）
- [ ] Phase 6：**总控制台 Notebook** — 调参/选策略 → 基线训练与报告 → 优化方案训练与对比报告

### 评估口径（约定）

| 阶段 | 用什么判断「好不好」 |
|------|----------------------|
| **训练期** | `reward`、`accuracy_reward`、`format_reward`、`loss`、`kl` + 效率；可与 Phase1 A2/C2 并表 |
| **总评估** | 经总控制台对 **基线（如 M1）** 与 **当次优化方案** 跑上游四基准 |

**开源主叙事**：可组合的 GRPO 实验台；**Phase 6 总控制台**为使用者主入口（汇总前期成果 + 直观调参选策略）。推荐组合见 [`docs/ACCURACY_ORIENTED_OPTIONS.md`](./docs/ACCURACY_ORIENTED_OPTIONS.md)，非强制唯一终点。

Phase 1 结论摘要：C2 相对 A2，runtime **约 −27%**、steps/s **约 +37%**（去存盘校正时长约 **−32%**）；训练 reward 未见下降。详见报告。

---

## Google Drive 目录结构（复现必读）

### 共享资源包（推荐直接使用）

实验资产（代码副本、数据、权重、训练输出）已整理到共享文件夹，可直接「添加到云端硬盘」后在 Colab 中挂载使用：

| 项 | 内容 |
|----|------|
| **共享链接** | [MixUpLLaVA-video-r1（Google Drive）](https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing) |
| 文件夹名 | `MixUpLLaVA-video-r1` |
| 顶层目录（与共享页一致） | `repo/` · `data/` · `checkpoints/` · `outputs/` |

**他人复现最短路径：**

1. 打开上方共享链接 → 右键 / 菜单选择 **「添加到云端硬盘」**（或「建立快捷方式」到「我的云端硬盘」）。
2. 打开 Colab → `drive.mount('/content/drive')`。
3. 确认路径为 `/content/drive/MyDrive/MixUpLLaVA-video-r1`（若快捷方式落在子目录，请相应修改 `PROJECT_DIR`）。
4. 按下方目录树核对四个顶层文件夹齐全后，再跑 Notebook / 训练。

> 注意：共享夹体积较大（含视频与模型权重）。仅浏览结构可打开链接；完整训练请确保自己的 Drive 有足够空间，并遵守数据/权重的使用许可。

### 根目录约定

| 项 | 值 |
|----|-----|
| Drive 根目录名 | `MixUpLLaVA-video-r1` |
| Colab 挂载后路径 | `/content/drive/MyDrive/MixUpLLaVA-video-r1` |
| Notebook 变量 | `PROJECT_DIR = "/content/drive/MyDrive/MixUpLLaVA-video-r1"` |
| 官方共享包 | https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing |

> **命名说明**：早期实验曾使用 `tiny-video-r1-GRPO`，已统一更名为 `MixUpLLaVA-video-r1`。若你仍保留旧文件夹，请重命名或全局替换 `PROJECT_DIR`。  
> 子目录相对结构请保持不变。

### 当前 Drive 完整结构（与共享夹对齐）

共享根目录下现有四个一级文件夹：`checkpoints`、`data`、`outputs`、`repo`。推荐/实际布局如下：

```
MixUpLLaVA-video-r1/                       # 共享根 / PROJECT_DIR
├── repo/                                  # 可训练代码（DeepSpeed 入口）
│   ├── TinyLLaVA-Video-R1/                # 上游基线（正式训练用）
│   │   ├── tinyllava/train/
│   │   │   ├── train.py
│   │   │   └── tinyllava_trainer_reason.py
│   │   ├── scripts/zero3_offload.json     # 可由 Notebook 生成
│   │   └── ...
│   ├── TinyLLaVA-Video-R1-CPPO/           # （可选）CPPO 对照副本
│   └── MixUpLLaVA-Video-R1/               # （推荐）本 GitHub 仓库同步副本
│       ├── README.md / schedule.md
│       ├── mixup/                         # 策略模块（开发中）
│       ├── notebooks/
│       └── doc/
│
├── data/
│   └── dataset/                           # DATA_ROOT
│       ├── NextQA/                        # 视频文件
│       ├── nextqa_0-30s.jsonl             # 全量标注
│       ├── nextqa_0-30s_10p_seed42.jsonl  # （可选）10% 子集
│       └── nextqa_small50.jsonl           # 冒烟 / 最小实验
│
├── checkpoints/
│   └── coldstart/                         # Cold-Start 权重（CKPT）
│
└── outputs/                               # 训练日志与 trainer_state.json
    ├── grpo_A_baseline_small/             # v1 基线（历史）
    ├── grpo_B_optimized_small/            # v1 策略 B（历史）
    ├── grpo_CPPO_small/                   # v1 CPPO（历史）
    ├── grpo_A_g8/                         # v2：g8 GRPO baseline（Phase 1 ✅）
    ├── grpo_CPPO_g8_p50/                  # v2：CPPO p=0.5（Phase 1 ✅）
    └── grpo_CPPO_g8_p75/                  # v2：CPPO p=0.75（可选）
```

### Notebook 中的路径变量对照

与已跑通 Colab Notebook 一致，建议固定如下：

```python
PROJECT_DIR = "/content/drive/MyDrive/MixUpLLaVA-video-r1"
REPO       = f"{PROJECT_DIR}/repo/TinyLLaVA-Video-R1"
DATA_ROOT  = f"{PROJECT_DIR}/data/dataset/"          # 注意末尾 /
CKPT       = f"{PROJECT_DIR}/checkpoints/coldstart"
OUT_BASE   = f"{PROJECT_DIR}/outputs"
SMALL_JSONL = f"{PROJECT_DIR}/data/dataset/nextqa_small50.jsonl"
```

训练入口示例（与 v1/v2 成功命令一致）：

- 脚本：`{REPO}/tinyllava/train/train.py`
- DeepSpeed：`{REPO}/scripts/zero3_offload.json`（ZeRO-3 + **参数** CPU offload，优化器不 offload）
- 数据：`--video_data_path DATA_ROOT` + `--video_folder SMALL_JSONL`（或其它 jsonl）
- 权重：`--pretrained_model_path CKPT`

### 各目录放什么 / 不放什么

| 目录 | 放入 | 不要放入 GitHub |
|------|------|-----------------|
| `repo/` | 上游源码 + MixUp 仓库代码 | —（大权重勿提交） |
| `data/dataset/` | jsonl + NextQA 视频 | 视频与全量 jsonl |
| `checkpoints/coldstart/` | Cold-Start 模型文件 | 全部权重文件 |
| `outputs/` | `trainer_state.json`、日志 | 大体量 checkpoint（可选本地保留） |

数据与权重**不进入本 Git 仓库**（见 `.gitignore`），仅通过 Drive 在 Colab 间共享。

### 他人获取数据的两种方式

**方式 A（推荐）**：使用 [共享文件夹](https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing)「添加到云端硬盘」，直接得到 `repo/`、`data/`、`checkpoints/`、`outputs/`。

**方式 B（自建）**：

1. 在 Google Drive 创建 `MixUpLLaVA-video-r1/{repo,data/dataset,checkpoints/coldstart,outputs}`。
2. 将上游 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) clone / 解压到 `repo/TinyLLaVA-Video-R1`。
3. 将本仓库 clone 到 `repo/MixUpLLaVA-Video-R1`。
4. 按上游 README 准备 NextQA 与 `nextqa_0-30s.jsonl` → `data/dataset/`。
5. 下载 / 自训 Cold-Start 权重 → `checkpoints/coldstart/`。
6. Colab 挂载 Drive → 运行 Notebook → 按 `schedule.md` 执行。

---

## 双环境工作方式

| 环境 | 用途 | 说明 |
|------|------|------|
| **Dev（实惠）** | 改代码、改 schedule/README、逻辑审查 | 本地 Cursor / 普通 Colab CPU；**不跑**完整 DeepSpeed 训练 |
| **Train（A100）** | `pip install -e .`、打补丁、deepspeed 训练、读 `trainer_state.json` | Colab Pro+ **A100-SXM4-80GB**（已验证可用） |

原则：Dev 改完 → 同步 GitHub / Drive → A100 会话只跑必要训练单元，避免空转计费。

## 环境要求（训练）

- Google Colab + Drive 挂载
- 推荐 GPU：**NVIDIA A100-SXM4-80GB**（约 80GB 显存）
- 软件：PyTorch（CUDA）、DeepSpeed、trl、flash-attn（失败可改 `eager`）
- 安装：在 `REPO` 下执行 `pip install -e .`，并安装 `trl`、`deepspeed`、`math_verify` 等

已验证参考配置（2026-07）：Python 3.12 + PyTorch 2.11+cu128 + A100-80GB。

## GitHub 仓库目录

```
MixUpLLaVA-Video-R1/
├── README.md
├── schedule.md
├── configs/                  # Phase 2 默认档位（C1 / C0）
│   ├── colab_c1.yaml         # 推荐默认（= Phase1）
│   ├── colab_c0.yaml         # OOM / 小显存
│   └── README.md
├── doc/                      # 调研报告 + CPPO 论文 PDF
├── notebooks/                # Colab 实验 Notebook
├── mixup/                    # 策略模块
│   ├── config_loader.py      # 加载 yaml + Notebook 覆盖
│   ├── cppo.py
│   ├── registry.py
│   ├── project_baseline.py
│   └── patches/              # trainer 参考副本与应用说明
├── docs/                     # Phase 清单与实验报告
│   ├── PHASE0_DRIVE_SYNC.md
│   ├── PHASE1_REPORT.md      # Phase 1 A2/C2 报告
│   ├── MEMORY_BENCHMARK_COLAB.md  # 换环境改参 + Phase1 吞吐
│   ├── ACCURACY_ORIENTED_OPTIONS.md  # 相对 M1 的准确率向备选配方
│   └── phase1运行记录.ipynb  # 原始 cell 日志备份
└── .gitignore
```

## 数据与权重来源

1. **训练数据**：NextQA（0–30s）jsonl + 视频目录，结构见上游 README  
2. **Cold-Start 权重**：TinyLLaVA-Video-ColdStart 或自行 cold-start  
3. **不在本仓库内分发**；请放到 Drive 对应目录后再在 Colab 引用

## 引用

```bibtex
@article{zhang2025tinyllava,
  title={TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning},
  author={Zhang, Xingjian and Wen, Siwei and Wu, Wenjun and Huang, Lei},
  journal={arXiv preprint arXiv:2504.09641},
  year={2025}
}
```

## License

Apache-2.0（与上游 TinyLLaVA-Video-R1 保持一致）
