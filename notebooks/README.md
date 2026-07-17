# notebooks/ — Phase 1 Colab 入口

| Notebook | 用途 |
|----------|------|
| **`phase1-cppo-g8.ipynb`** | **Phase 1 主入口**：A2 / C2 /（可选）C3，g=8，默认快验 50 step |
| `mixup-GRPO优化.ipynb` | 与上相同内容（兼容旧文件名） |

## Phase 1 流程摘要

1. 挂载 Drive → 确认 A100  
2. `PROJECT_DIR=/content/drive/MyDrive/MixUpLLaVA-video-r1`  
3. Phase 0 验收 + `git clone/pull` MixUp 仓库  
4. 小数据集 → 装依赖 → 从 `mixup/patches` 覆盖 trainer  
5. 重跑 §6（进度条 + 断点）→ 训练 **A2** → **C2** → 对比（可粘贴 schedule §8）  

## 断点续训 / 进度

- `SAVE_STEPS=10`：约每 10 step 写入 `checkpoint-*`（`SAVE_TOTAL_LIMIT=1` 只留最新）  
- `ENABLE_RESUME=True`：有 checkpoint 时自动续训（如 A2 从 `checkpoint-40` → 50）  
- 日志流式输出 + 简易进度条 / ETA  
- 断连后：按重连顺序跑完 §0–§6，再跑对应训练单元；**清空 Drive 回收站** 避免占配额  

共享包：https://drive.google.com/drive/folders/1Qjh19WSLGIeu1NX-UYRYAh4oyFQjCp2m?usp=sharing  

计划详见仓库根目录 `schedule.md` Phase 1。
