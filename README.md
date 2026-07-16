# MixUpLLaVA-Video-R1

**MixUpLLaVA-Video-R1** 是在 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) 基线上进行的 **GRPO 混合优化**升级版本。项目在保持 SFT + GRPO 总体框架不变的前提下，将多种可互补的 GRPO 改进方案（如 CPPO、GFPO、NGRPO 及项目内 difficulty-aware reward shaping）进行模块化实现与组合评估。

**主运行环境**：Google Colab（推荐 A100）+ Google Drive（数据 / 权重 / 输出持久化）。  
Windows 本机原生 DeepSpeed 训练流程不稳定，**不作为正式训练路径**；日常可在本地编辑代码与 Notebook，训练与验证在 Colab 上完成。

## 背景

- 原始工作：[TinyLLaVA-Video-R1](https://arxiv.org/abs/2504.09641)（[GitHub](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)）
- 优化方法论：`doc/基于Video-R1的视频交通异常行为检测项目中的GRPO优化调研报告.pdf`
- 前期 CPPO 验证：Colab + Drive，50 条子集复现（加速约 0.7%，reward 与 baseline 一致）
- GitHub：https://github.com/SanXue-YG/MixUpLLaVA-Video-R1

## MixUp 优化思路

| 模块 | 作用 |
|------|------|
| 项目基线增强 | Difficulty-aware advantage + 自适应长度 reward |
| CPPO | 基于 \|advantage\| 的 completion 剪枝，训练加速 |
| GFPO | Top-k 优势掩码，抑制冗余推理 |
| NGRPO | 负信号增强，缓解全错 group 无梯度 |

详细实验计划与里程碑见 **[schedule.md](./schedule.md)**。

## 项目状态

🚧 **开发中** — Phase 0（环境与仓库）已切回 **Colab + Google Drive**。

- [x] 项目目录与文档（GitHub 骨架）
- [x] 实验计划表（对齐 Drive 成功路径）
- [ ] Drive 目录规范化 + 上游代码挂载说明
- [ ] Colab 基线 / CPPO（g8）对比跑通
- [ ] MixUp 模块实现与消融
- [ ] 全量训练与评估

---

## Google Drive 目录结构（复现必读）

本项目沿用已验证可跑通的 Drive 布局。他人复现时，请在 Google Drive 下创建同名根目录，并按下列结构放置资产。

### 根目录约定

| 项 | 值 |
|----|-----|
| Drive 根目录名 | `tiny-video-r1-GRPO` |
| Colab 挂载后路径 | `/content/drive/MyDrive/tiny-video-r1-GRPO` |
| Notebook 变量 | `PROJECT_DIR = "/content/drive/MyDrive/tiny-video-r1-GRPO"` |

> 若你使用不同文件夹名，只需全局替换 `PROJECT_DIR`；**子目录相对结构请保持不变**。

### 推荐完整结构

```
MyDrive/
└── tiny-video-r1-GRPO/                    # PROJECT_DIR
    ├── repo/                              # 可训练代码（DeepSpeed 入口在此）
    │   ├── TinyLLaVA-Video-R1/            # 上游基线（正式训练用）
    │   │   ├── tinyllava/
    │   │   │   └── train/
    │   │   │       ├── train.py
    │   │   │       └── tinyllava_trainer_reason.py
    │   │   ├── scripts/
    │   │   │   └── zero3_offload.json     # 训练时可由 Notebook 生成
    │   │   ├── pyproject.toml / setup.py
    │   │   └── ...
    │   ├── TinyLLaVA-Video-R1-CPPO/       # （可选）CPPO 对照副本，只读参考
    │   └── MixUpLLaVA-Video-R1/           # （推荐）本仓库 clone，放文档/mixup/notebooks
    │       ├── README.md
    │       ├── schedule.md
    │       ├── mixup/                     # 策略模块（开发中）
    │       ├── notebooks/                 # Colab 实验 Notebook
    │       └── doc/                       # 调研报告 PDF 等
    │
    ├── data/
    │   └── dataset/                       # DATA_ROOT / DATA_ROOT_STRIP
    │       ├── NextQA/                    # 视频文件（按上游 README 组织）
    │       ├── nextqa_0-30s.jsonl         # 全量标注（约 5496 条）
    │       ├── nextqa_0-30s_10p_seed42.jsonl   # （可选）10% 子集
    │       └── nextqa_small50.jsonl       # 冒烟 / 最小实验（Notebook 可自动生成）
    │
    ├── checkpoints/
    │   └── coldstart/                     # CKPT：Cold-Start 权重目录
    │       ├── config.json
    │       ├── model*.safetensors / *.bin
    │       └── ...
    │
    └── outputs/                           # OUT_BASE：训练日志与 trainer_state.json
        ├── grpo_A_baseline_small/         # v1 基线（历史）
        ├── grpo_CPPO_small/               # v1 CPPO（历史）
        ├── grpo_A_g8/                     # v2：g8 GRPO 基线
        ├── grpo_CPPO_g8_p50/              # v2：CPPO pruning=0.5
        └── grpo_CPPO_g8_p75/              # v2：CPPO pruning=0.75（可选）
```

### Notebook 中的路径变量对照

与已跑通 Colab Notebook 一致，建议固定如下：

```python
PROJECT_DIR = "/content/drive/MyDrive/tiny-video-r1-GRPO"
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

### 他人从零重建 Drive（最短步骤）

1. 在 Google Drive 创建 `tiny-video-r1-GRPO/{repo,data/dataset,checkpoints/coldstart,outputs}`。
2. 将上游 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) clone / 解压到 `repo/TinyLLaVA-Video-R1`。
3. 将本仓库 clone 到 `repo/MixUpLLaVA-Video-R1`（或本地改完后推送，再在 Colab 拉取）。
4. 按上游 README 准备 NextQA 数据与 `nextqa_0-30s.jsonl`，放入 `data/dataset/`。
5. 下载 / 自训 Cold-Start 权重到 `checkpoints/coldstart/`。
6. 打开 Colab → 挂载 Drive → 运行仓库内 Notebook（见 `notebooks/` 或实验目录说明）→ 按 `schedule.md` 执行。

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

## GitHub 仓库目录（规划）

```
MixUpLLaVA-Video-R1/
├── README.md                 # 本文件（含 Drive 复现结构）
├── schedule.md               # 实验与里程碑计划
├── doc/                      # 调研报告与论文 PDF
├── mixup/                    # 混合优化策略模块（待建）
├── notebooks/                # Colab 实验 Notebook
├── configs/                  # 显存档位与实验配置（待建）
├── docs/                     # 实验记录与消融报告（待建）
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
