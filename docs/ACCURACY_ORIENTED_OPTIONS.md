# 准确率向优化备选方案（相对 Phase 4 · M1）

- **日期**：2026-07-18（计划已调整：M5 等为**可选预设**，非强制唯一最终方案）  
- **依据**：`doc/基于Video-R1的视频交通异常行为检测项目中的GRPO优化调研报告.pdf`（尤其 §2.3、§3、表 1）及所引原论文  
- **对照基线**：Phase 4 **M1** = 仅项目基线增强 B（选择器默认）  
- **当前主交付**：Phase 3 **模块库** + Phase 4 **优化方案选择器**（用户自选组合；流程复用 Phase1）  
- **本文角色**：给出若干**推荐/备选预设**（如曾议的 B+CPPO+NGRPO+GFPO、M5q 等），供选择器一键加载或手工勾选；**不代替**用户最终选哪套。  
- **用途**：后续目标（加速 / 准确率 / 简洁）明确后，按决策树选用预设，并在 Phase 5 评测该次所选方案。

---

## 1. 核心结论（先读）

1. **调研报告表 1 未单独设立「准确率↑」目标列**；多数方法按效率 / 简洁 / 推理结构 / 稳定 / 鲁棒 / 多目标分类。报告本身**几乎没有**「相对策略 B，Video/NextQA 准确率 +N」的现成数字。  
2. **M1（B）已是组内「质量向」设计**（难样本加权 + 长度 shaping）；再叠模块时，不要默认「模块越多越准」。  
3. 对照原论文，**相对纯 GRPO / 相对「只有 B」最可能抬升准确率类指标的是**：  
   - **NGRPO**（已在 M5）：数学基准上相对 GRPO 有明确 Pass@k 等提升叙事；  
   - **MO-GRPO**（M5 未用）：多奖励场景下抑制 format/长度 hacking，**保住 accuracy**；  
   - **StepGRPO**（暂缓）：过程级正确性奖励，机制上最「准」，工程最重。  
4. **CPPO / GFPO** 按报告与论文主叙事是 **加速 / 简洁**，不宜写成准确率卖点。  
5. 因此：**M5 里真正支撑「比 M1 更准」叙事的，应是 B+NGRPO**；CPPO/GFPO 负责效率与简洁。若 Phase 5 显示 M5 准不动，优先试 **MO-GRPO**，而不是再加 GFPO。

---

## 2. 调研报告中的目标归类（摘要）

| 方法 | 报告中的优化目标 | 对准确率的直接性 | 是否已在 M5 |
|------|------------------|------------------|------------|
| **B（难度感知 + 长度）** | 难样本学习强度、长度分布稳定 | 质量底盘（M1） | ✅（M1/M5） |
| **CPPO** | 训练加速（少算低 \|A\| completion） | 弱 | ✅ |
| **GFPO** | 推理简洁为主；吞吐为辅（Top-k 掩码） | 弱 | ✅ |

> **CPPO vs GFPO（是否都要进 M5）**：见下文 §2.1。二者**不是同一类「加速插件」**；加速主靠 CPPO，GFPO 的独特卖点是简洁/隐式 shaping。
| **NGRPO** | 全错 group 稳定；负信号 | **中→强（间接→任务表现）** | ✅ |
| **StepGRPO** | 过程奖励 / 推理结构 | **强（机制）** | ❌（二期） |
| **MO-GRPO** | 多目标公平、抑 reward hacking | **中→强（保 accuracy）** | ❌ |
| DaGRPO / GMPO / MAPO / GRPO-A | 冲突 / 数值 / 鲁棒 / 引导稳定 | 间接 | ❌（P2） |

---

## 2.1 CPPO vs GFPO：主要区别与 M5 是否需同时使用

二者在调研表 1 里都沾「效率」，但**改的位置、省的是什么、主叙事**不同：

| 维度 | **CPPO** | **GFPO** |
|------|----------|----------|
| 改哪里 | 策略损失求和集合：丢掉低 \|advantage\| 的 completion，**少做** logps / KL / loss | 优势层：按属性（常为长度/token 效率）Top-k，**其余 A=0**；batch 前向常仍保留全组 |
| 主收益 | **墙钟 / steps/s↑**（Phase1 C2 已验证量级） | **更短、更「像样」的轨迹被强化**（简洁 / 隐式 reward shaping） |
| 辅收益 | 略减噪声更新 | 掩码后有效梯度更少，吞吐可能略升，但通常**弱于 CPPO** |
| 筛选依据 | \|A\|（与当前 reward 相对比较绑定） | `metric(·)`（可与准确率无关，如长度） |
| 与「准」 | 弱；不改奖励定义 | 弱；若 metric≠准确率，**不保证准** |

示意：

```text
同一 group G 条 completion
  ├─ CPPO：按 |A| 剪掉一部分 → 后面算 logps/loss 的条数变少 → 主要省算力
  └─ GFPO：仍可生成/前向 G 条 → 只对 Top-k 给非零 A → 主要改「学谁」，省算力是副产品
```

### 有没有必要在 M5 里同时用？

- **若目标是加速**：**不必要**。CPPO（± 合适 `pruning_rate`）已足够；再叠 GFPO 对速度的边际收益通常有限，还多一套超参（k、metric、是否难度自适应）。  
- **若目标是「加速 + 抑冗长推理」**：可以同时用，但应写成 **CPPO=效率、GFPO=简洁**，不要写成「双重加速」。  
- **叠加风险**：两者都在「少学一部分样本」——CPPO 按 \|A\|，GFPO 按长度等；可能过度过滤、或与 NGRPO/B 的难样本学习打架。  
- **务实建议**：  
  - 主路径 M5 若算力紧 / 要减变量：优先 **B + CPPO + NGRPO**（即文档中的 **M5-lite±CPPO**），GFPO 作可选；  
  - 若案例分析里冗余推理仍重，再开 GFPO。

---

## 3. 原论文中与「准」相关的证据

### 3.1 NGRPO（已纳入 M5）

- **论文**：[NGRPO: Negative-enhanced Group Relative Policy Optimization](https://arxiv.org/abs/2509.18851)  
- **机制**：虚拟最大奖励 → 全错 group 不再 advantage=0；非对称裁剪稳定负向更新。  
- **证据**：Qwen2.5-Math-7B 上 MATH500 / AMC23 / AIME2025 等 **相对 GRPO、PPO、DAPO 等有提升**（Pass@k / AUC 叙事）。  
- **对本项目**：领域是数学推理，迁到视频需 Phase 5 验证；但在可嵌入方法中，这是**最接近「准确率/推理表现↑」的定量叙事**。  
- **相对 M1**：M1 无负信号校准；难样本上全错 group 仍可能白更新 → **M5 相对 M1 的「准」预期应主要归功于 NGRPO，而非 CPPO/GFPO**。

### 3.2 MO-GRPO（强烈建议作准确率向备选）

- **论文**：[MO-GRPO: Mitigating Reward Hacking of GRPO on Multi-Objective Problems](https://arxiv.org/abs/2509.22047)  
- **机制**：对各奖励分量 **先组内标准化再聚合**（normalize-then-sum），避免高方差项主导。  
- **与本项目契合点**：训练已有 `accuracy_reward` + `format_reward`，M1/M5 还有长度 shaping → **典型多目标**；Phase1 中曾出现 format 与 reward 走势不完全一致的现象，MO-GRPO 更对症。  
- **预期**：不一定总分暴涨，更可能 **accuracy 不被 format/长度劫持**，下游「选对答案」更稳。  
- **工程**：只改优势构造，符合调研「可嵌入」标准；建议 Phase 3 实现开关。

### 3.3 StepGRPO / R1-VL（质量向上限，二期）

- **调研**：§3.2；StepRAR（中间正确步骤）+ StepRVR（结构有效性）。  
- **预期**：过程信号更密，缓解「只靠终态奖励」的推理偏差 → **机制上最直接对准正确性**。  
- **成本**：步骤切分、匹配、奖励重写 → 改动面大于 NGRPO/MO-GRPO；schedule 已列为暂不纳入首期单训。

### 3.4 CPPO / GFPO（勿当准确率组合核心）

- **CPPO**：不改奖励/优势定义，只剪低 \|advantage\| completion → 主收益是算力。  
- **GFPO**：按属性 Top-k 掩码，报告强调简洁与效率；若 metric 选长度，**不保证准确率↑**。  
- **写法建议**：报告中 CPPO/GFPO 写效率/简洁；准确率对比写 B / NGRPO /（可选）MO-GRPO。

### 3.5 B（M1）本身

- **调研 §2.3**：difficulty-aware advantage reweighting + 自适应长度。  
- **预期**：提升难样本学习强度与长度稳定性；**报告未给出相对标准 GRPO 的准确率表**。  
- **定位**：组内质量基线，不是「已经证明准了多少点」的天花板。

---

## 4. 备选配方（相对 M1）

均以 **C1（Phase1 已验证档）为默认**，以便与 A2/C2/M1 等训练期并表；换部署环境可改参，换档或换数据规模须新开实验 ID、单独成组。

| ID | 组合 | 相对 M1 的准确率叙事 | 效率 | 建议时机 |
|----|------|----------------------|------|----------|
| **M5**（主路径） | B + CPPO + NGRPO + GFPO | 准靠 **B+NGRPO**；速度 CPPO；简洁 GFPO | 高（对齐 Phase1 C2 量级预期） | Phase 4 必跑 |
| **M5-lite** | B + NGRPO（± CPPO） | 论文证据最干净的「准」向叠法 | 中高 | M5 准不动且怀疑 GFPO 干扰时 |
| **M5q**（质量优先备选） | **B + NGRPO + MO-GRPO**（± CPPO） | 负信号学难样本 + 多目标保 accuracy | 中（+CPPO 可补速） | **Phase5 显示 M5≤M1 准或 format 劫持时优先** |
| **M5-step**（二期） | B + StepGRPO（± NGRPO） | 过程正确性 | 低–中（实现重） | 资源允许再开 |

### 推荐决策树

```text
Phase4 先跑 M1、M5
    │
    ├─ 训练期 accuracy_reward / Phase5 四基准：M5 ≥ M1
    │      → 保持 M5 为最终方案；报告写清准来自 B+NGRPO
    │
    └─ M5 准 ≤ M1，或 format 高、acc 低（疑似 hacking）
           → 实现 MO-GRPO，试 M5q = B+NGRPO+MO-GRPO±CPPO
           → 仍不足再考虑 StepGRPO（二期）
```

---

## 5. 可检验指标（避免再写空）

### 5.1 训练期（相对 M1）

| 指标 | 「准」向成功 | 备注 |
|------|--------------|------|
| `rewards/accuracy_reward` 全程均值 / 末 10 step 均值 | **≥ M1**（或差距写入可接受阈值，如相对降幅 &lt; X%） | 主代理 |
| 合成 `reward` | 不显著低于 M1 | 受 format/长度影响，次主 |
| `format_reward` | 允许略降，但若远高于 acc 而 acc 掉 → 怀疑 hacking → 试 MO-GRPO | |
| `train_steps_per_second` / 去存盘校正时长 | M5/M5q 仍应 **明显快于 A2/M1**（若含 CPPO） | 效率叙事与准分离 |

### 5.2 Phase 5 下游（相对 M1）

| 指标 | 「准」向成功 |
|------|--------------|
| Video-MME / MVBench / MLVU / **MMVU** | **M5 或 M5q ≥ M1**（主看相对增益，不追论文绝对分） |
| 案例分析 | 难样本少「全组乱猜」；少冗长胡言（GFPO/B 长度） |

**必写限定**：C1 快验 ≠ 官方 8 卡全量；精简幅度见 `schedule.md`「跨阶段对比约定」。NGRPO 数学基准增益 **不能直接等同** Video-MME 分数。

---

## 6. 与当前 schedule 的关系

| 文档/阶段 | 关系 |
|-----------|------|
| `schedule.md` Phase 3 | **模块库**：尽量实现调研可嵌入方法并注册 |
| `schedule.md` Phase 4 | **选择器** + 默认 **M1**；本文中的 M5/M5q/M5-lite 等 = **可选预设** |
| `schedule.md` Phase 5 | 评测 **M1 + 当次所选组合**（不必固定评旧「M5」） |
| 本文 | 目标导向选开关的参考；可做成 `mixup/presets/*.yaml` |

---

## 7. 参考文献（文中已链）

- 调研报告 PDF（仓库 `doc/`）  
- NGRPO: https://arxiv.org/abs/2509.18851  
- MO-GRPO: https://arxiv.org/abs/2509.22047  
- TinyLLaVA-Video-R1: https://github.com/ZhangXJ199/TinyLLaVA-Video-R1  
- Phase 1 效率报告：`docs/PHASE1_REPORT.md`

---

*维护：随 Phase 3/4 实现进度更新「是否已实现」列；Phase 5 出数后在本文末追加实测结论。*
