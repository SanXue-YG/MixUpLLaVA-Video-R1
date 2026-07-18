# scripts/eval/ — Phase 5 评测脚本

## 主路径：训练期评估（推荐，小规模）

不需要下载四基准。在 Python 中：

```python
from mixup.eval_training import evaluate_run
evaluate_run("outputs/your_run_id", baseline_dir="outputs/m1_run_id")
# → training_eval_report.md / .json
```

## 可选：总评估（Video-MME / MVBench / MLVU / MMVU）

对齐上游 [TinyLLaVA-Video-R1 Evaluation](https://github.com/ZhangXJ199/TinyLLaVA-Video-R1)。  
**四基准合计约 ~600GB**，小规模实验不建议默认下载。

1. 建目录：`EVAL_ROOT=... bash scripts/eval/download_benchmarks.sh --init-only`
2. 按上游 README 自行下载各数据集到对应子目录  
3. 确认后：`bash scripts/eval/download_benchmarks.sh --confirm-large`  
4. 跑单项（示例）：

```bash
export TINYLLAVA_REPO=/path/to/TinyLLaVA-Video-R1
export MODEL_PATH=/path/to/checkpoint
export EVAL_DIR=$EVAL_ROOT/Video-MME
bash scripts/eval/run_videomme.sh
```

说明见 [`docs/PHASE5_BENCHMARKS.md`](../../docs/PHASE5_BENCHMARKS.md)。
