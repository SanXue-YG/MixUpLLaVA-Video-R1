# Phase 5 可选总评估：四基准说明

上游：[TinyLLaVA-Video-R1 · Evaluation](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)

## 为什么默认不做总评估？

按上游流程评 **Video-MME、MVBench、MLVU、MMVU**，需要各自下载完整数据。四套合计体量约在 **~600GB** 量级，对 Colab / 个人 Drive 的小规模 MixUp 实验**不实惠**。

因此本项目 Phase 5：

| 路径 | 是否默认 | 说明 |
|------|----------|------|
| **训练期评估** | ✅ 主路径 | `reward` / `accuracy_*` / `format_*` / `loss` / `kl` + runtime / steps/s |
| **四基准总评估** | ⬜ 可选 | 预留目录与 `scripts/eval/*`；实验者自行下载后再跑 |

与「能力优先、实跑可延后」一致；总控制台默认也应先出训练期报告。

## 何时启用总评估？

- 已有足够云盘/本地空间，并接受长时间推理；  
- 需要与上游论文同协议的正式分数；  
- 通常只评 **M1（或基线）+ 1～2 个候选**，不要一次开满四基准×多权重。

## 准备步骤

```bash
# 1) 只建空目录（不下载）
export PROJECT_DIR=/content/drive/MyDrive/MixUpLLaVA-video-r1
export EVAL_ROOT=$PROJECT_DIR/data/eval
bash scripts/eval/download_benchmarks.sh --init-only

# 2) 按上游 README 手动下载并解压到：
#    $EVAL_ROOT/Video-MME
#    $EVAL_ROOT/MVBench
#    $EVAL_ROOT/MLVU
#    $EVAL_ROOT/MMVU

# 3) 确认后（本仓库脚本仍不会自动拉 600GB）
bash scripts/eval/download_benchmarks.sh --confirm-large
```

上游各基准放置与命令见 README「3. Evaluation」小节（`scripts/eval/videomme.sh` 等）。

## 运行（单项）

```bash
export TINYLLAVA_REPO=$PROJECT_DIR/repo/TinyLLaVA-Video-R1
export MODEL_PATH=$PROJECT_DIR/outputs/<run_id>   # 或最终权重路径
export EVAL_DIR=$EVAL_ROOT/Video-MME
bash scripts/eval/run_videomme.sh
```

Python 侧可先检查就绪度（默认 dry-run）：

```python
from mixup.eval_benchmarks import build_benchmark_plan, run_benchmark
plan = build_benchmark_plan(
    eval_root=EVAL_ROOT,
    model_path=MODEL_PATH,
    tinyllava_repo=TINYLLAVA_REPO,
)
print(plan.to_dict())
# run_benchmark("videomme", ..., dry_run=True)  # 默认不真跑
# 真跑需：dry_run=False, confirm_large_download=True，且数据已就位
```

## 与训练期评估的关系

```text
训完一个 run
  → evaluate_run(...)           # 必做 / 默认
  → （可选）四基准 run_*.sh     # 仅当磁盘与时间允许
```
