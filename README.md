# MixUpLLaVA-Video-R1

**MixUpLLaVA-Video-R1** 是在 [TinyLLaVA-Video-R1](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1) 基线上进行的 **GRPO 混合优化**升级版本。项目在保持 SFT + GRPO 总体框架不变的前提下，将多种可互补的 GRPO 改进方案（如 CPPO、GFPO、NGRPO 及项目内 difficulty-aware reward shaping）进行模块化实现与组合评估，面向本地有限算力环境（单卡消费级 GPU）提供可复现的训练与对比实验流程。

## 背景

- 原始工作：[TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning](https://arxiv.org/abs/2504.09641)（[GitHub](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)）
- 优化方法论：见 `doc/基于Video-R1的视频交通异常行为检测项目中的GRPO优化调研报告.pdf`
- 前期 CPPO 验证：在 Colab 50 条子集上完成 CPPO 复现（训练加速约 0.7%，reward 与 baseline 一致）

## MixUp 优化思路

| 模块 | 作用 |
|------|------|
| 项目基线增强 | Difficulty-aware advantage + 自适应长度 reward |
| CPPO | 基于 \|advantage\| 的 completion 剪枝，训练加速 |
| GFPO | Top-k 优势掩码，抑制冗余推理 |
| NGRPO | 负信号增强，缓解全错 group 无梯度 |

详细实验计划与里程碑见 **[schedule.md](./schedule.md)**。

## 项目状态

🚧 **开发中** — 当前处于 Phase 0（环境与仓库初始化）。

- [x] 项目目录与文档
- [x] 实验计划表
- [ ] 上游代码集成
- [ ] 本地基线跑通
- [ ] MixUp 模块实现与消融
- [ ] 全量训练与评估

## 目录结构（规划）

```
MixUpLLaVA-Video-R1/
├── doc/                  # 论文与调研报告
├── mixup/                # 混合优化策略模块（待建）
├── scripts/train/        # 本地训练脚本（待建）
├── configs/              # 显存档位与实验配置（待建）
├── docs/                 # 实验记录与报告（待建）
├── schedule.md           # 项目计划表
└── README.md
```

## 环境要求

- Python 3.10
- CUDA GPU（开发环境：RTX 4080 Laptop 12GB）
- 参考上游安装：`pip install -e .` + `flash-attn`

## 数据与权重

数据与 checkpoint **不包含在仓库中**（见 `.gitignore`），请自行准备：

1. **训练数据**：NextQA 子集（0–30s，5496 条），目录结构见上游 README
2. **Cold-Start 权重**：[TinyLLaVA-Video-ColdStart](https://huggingface.co) 或自行 cold-start 训练

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
