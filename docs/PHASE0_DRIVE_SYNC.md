# Phase 0 — Drive + GitHub 对齐清单

> 本地仓库已完成文档与资产迁入。以下项需在 **Google Drive / Colab** 上勾选完成。

共享包：https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing  
本地路径：`MixUpLLaVA-Video-R1/`（本 GitHub 仓库）

---

## 0.2 Drive 顶层确认（共享页已见四个一级目录）

在 Colab 运行：

```python
from google.colab import drive
drive.mount('/content/drive')
import os
PROJECT_DIR = "/content/drive/MyDrive/MixUpLLaVA-video-r1"
for name in ["repo", "data", "checkpoints", "outputs"]:
    p = os.path.join(PROJECT_DIR, name)
    print(name, "->", "OK" if os.path.isdir(p) else "MISSING", p)

repo = os.path.join(PROJECT_DIR, "repo/TinyLLaVA-Video-R1")
ckpt = os.path.join(PROJECT_DIR, "checkpoints/coldstart")
print("REPO", "OK" if os.path.isdir(repo) else "MISSING", repo)
print("CKPT", "OK" if os.path.isdir(ckpt) else "MISSING", ckpt)
```

- [ ] `repo/` 存在且含 `TinyLLaVA-Video-R1`
- [ ] `data/dataset/` 存在（含 jsonl / NextQA）
- [ ] `checkpoints/coldstart/` 存在
- [ ] `outputs/` 存在

---

## 0.3 将本仓库同步到 Drive

目标路径：

```text
/content/drive/MyDrive/MixUpLLaVA-video-r1/repo/MixUpLLaVA-Video-R1/
```

任选其一：

**A. Colab 直接 clone（推荐）**

```bash
!mkdir -p /content/drive/MyDrive/MixUpLLaVA-video-r1/repo
%cd /content/drive/MyDrive/MixUpLLaVA-video-r1/repo
!git clone https://github.com/SanXue-YG/MixUpLLaVA-Video-R1.git
# 若目录已存在：
%cd MixUpLLaVA-Video-R1
!git pull
```

**B. 本机推送后，在 Drive 网页打开 GitHub 同步 / 手动上传**

先 push 本仓库最新 Phase 0 提交，再在 Drive `repo/` 下 clone/pull。

- [ ] Drive 上可读 `README.md`、`schedule.md`、`mixup/`、`notebooks/`、`doc/`

---

## 0.4–0.6 本地已完成（本仓库）

| 任务 | 状态 | 位置 |
|------|------|------|
| 0.4 调研 PDF → `doc/` | ✅ | `doc/基于Video-R1的…调研报告.pdf` |
| 0.4 CPPO 论文 PDF | ✅ | `doc/CPPO-Accelerating-GRPO-Reasoning-Models.pdf` |
| 0.5 Notebook 入口 | ✅ | `notebooks/mixup-GRPO优化.ipynb` |
| 0.6 CPPO / baseline / B 补丁 | ✅ | `mixup/patches/` + `mixup/cppo.py` |

---

## Phase 0 验收标准

- [ ] README Drive 树与真实共享夹一致（一级：repo/data/checkpoints/outputs）
- [ ] Colab 能定位 `REPO`、`CKPT`
- [ ] Drive 上存在本仓库副本，且含 `mixup/`、`notebooks/`、`doc/`
- [ ] （可选）`python -c "from mixup.cppo import build_keep_mask"` 在装好 torch 的环境中通过

完成后将 `schedule.md` Phase 0 中 0.2 / 0.3 勾为 ✅，进入 Phase 1（A2/C2 训练）。
