# SALT-Track 论文大纲与训练排期

## 1. 论文主线定位

本文聚焦视觉语言跟踪中的一个核心问题：

> 现有视觉语言跟踪器通常把文本作为输入条件使用，但训练监督仍主要来自边界框。我们希望进一步探索：文本能否作为训练阶段的语义监督信号，帮助跟踪器学习更稳定、更符合目标描述的实例表征。

因此，SALT-Track 的主线不是重新设计一个新的文本输入分支，而是在已有视觉语言跟踪框架上补充一条文本驱动的语义对齐监督。

建议论文叙述重点：

- 跟踪任务中的传统监督主要回答“框得准不准”。
- 视觉语言跟踪还应进一步回答“预测区域是不是文本描述的那个目标”。
- 我们将文本从输入条件扩展为训练监督信号，形成感知监督与语义监督共同约束的训练框架。

## 2. 核心贡献

### 2.1 文本语义对齐损失

这是论文的第一核心贡献。

我们在 SALTTrack的视觉语言特征空间内构造实例级语义监督，不额外引入外部 CLIP 对齐空间。具体做法是：

1. 从融合后的搜索特征图中提取预测框对应的实例特征。
2. 从同一特征图中提取 GT 框对应的实例特征。
3. 使用 SALTTrack 文本分支增强后的文本特征作为语义锚点。
4. 构造视觉一致性约束与文本语义约束。
5. 使用可靠性门控和阶段权重调度控制语义监督强度。

总损失形式：

```text
L_total = L_giou + L_l1 + L_focal + L_confidence + lambda(t) * L_semantic
```

语义监督形式：

```text
L_semantic = w_visual * L_visual_consistency + w_text * g * L_text
```

其中：

- `L_visual_consistency`：预测框实例特征与 GT 框实例特征的一致性约束。
- `L_text`：实例特征与文本语义锚点之间的语义约束。
- `g`：可靠性门控，当 GT 特征比预测特征更接近文本时增强文本监督。
- `lambda(t)`：训练阶段权重，用于避免语义监督过度干扰定位主任务。

### 2.2 方向式文本监督

这是第一贡献中的关键技术点。

直接距离监督会强行拉近预测实例特征和文本特征，容易导致特征相似度过高、实例表征坍塌。因此当前更推荐把主方法写成方向式语义监督：

```text
L_text_direction = 1 - cos(normalize(text - pred), normalize(text - gt))
```

它不强制所有预测特征贴近文本锚点，而是要求预测实例向文本语义靠近的方向与 GT 实例一致，从而保留实例特征的区分性。

### 2.3 语义引导的 LoRA 高效微调

这是论文的第二创新点。

在全量微调之外，我们引入 LoRA 进行参数高效适配。更进一步，我们不是简单使用普通 LoRA，而是探索语义监督如何指导 LoRA 更新：

- 仅在指定视觉语言交互模块中注入 LoRA。
- 冻结原模型参数，只训练低秩增量参数。
- 利用语义 gate 对 LoRA residual energy 进行约束。
- 在文本监督不可靠的样本上抑制 LoRA 过强更新，降低噪声文本对小参数适配的影响。

建议表述：

> We further introduce a semantic-guided LoRA adaptation strategy to achieve parameter-efficient fine-tuning while preserving the semantic supervision behavior of the full model.

## 3. 论文结构建议

### 3.1 Introduction

需要讲清楚三个问题：

1. 视觉语言跟踪中，文本通常只作为输入条件。
2. 边界框监督只能提供感知层面的定位信号。
3. 文本描述天然包含目标语义，应当进一步作为训练监督使用。

建议引出贡献：

- 提出文本语义对齐监督，让文本参与训练目标。
- 提出方向式文本监督，缓解距离对齐导致的表征坍塌。
- 引入语义引导 LoRA，实现参数高效微调。

### 3.2 Related Work

建议分三节：

1. Visual Language Tracking
   - 重点对比已有方法如何使用文本输入。
   - 强调我们关注的是文本作为监督信号。

2. Semantic Alignment and Representation Learning
   - 讨论视觉语义对齐、实例语义约束、特征一致性。
   - 说明我们不依赖外部 CLIP 空间，而是在跟踪器内部特征空间构造监督。

3. Parameter-Efficient Fine-Tuning
   - 简述 Adapter、Prompt Tuning、LoRA。
   - 引出我们在 VLT 中探索语义引导 LoRA。

### 3.3 Method

建议小节：

1. Overview
   - 给整体框架图。
   - 主体为 SALTTrack，新增训练阶段语义监督分支。

2. Instance-Level Feature Extraction
   - 预测框 RoI 特征。
   - GT 框 RoI 特征。
   - 文本语义锚点。

3. Text-Driven Semantic Alignment Loss
   - 视觉一致性损失。
   - 文本语义损失。
   - 可靠性门控。
   - 两阶段权重调度。

4. Directional Semantic Supervision
   - 对比距离式监督和方向式监督。
   - 解释为什么方向式监督更稳。

5. Semantic-Guided LoRA Adaptation
   - LoRA 注入位置。
   - 只训练低秩参数。
   - 语义 gate 约束 LoRA residual energy。

### 3.4 Experiments

建议实验分组：

1. Main Comparison
   - SALTTrack baseline。
   - SALT-Track final semantic-guided LoRA。
   - 其他公开方法。
   - 在代表性 VLT 数据集上统一报告 AUC、P、NP。

**表1 主实验结果**

| Method | LaSOT AUC | LaSOT P | LaSOT NP | TNL2K AUC | TNL2K P | TNL2K NP | OTB AUC | OTB P | OTB NP | VastTrack AUC | VastTrack P | VastTrack NP | MGIT AUC | MGIT P | MGIT NP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| SALT-Track | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

说明：

- 表1只放最终版 SALT-Track 与代表性已有方法/基线的横向对比。
- 最终版 SALT-Track 对应 `salttrack_base`，即 Semantic LoRA-Direction-Backbone。
- `distance`、`direction`、`both`、`no gate` 等训练配置只进入消融表，不进入主表。
- LaSOT 当前对应语言跟踪评测时优先使用 LaSOT-Language，最终表名可按投稿规范调整为 `LaSOT` 或 `LaSOT-Lang`。
- OTB 当前优先使用 OTB99-Lang，最终表名可按实验设置调整为 `OTB` 或 `OTB99`。
- 如果 VastTrack 或 MGIT 的官方指标口径与 AUC/P/NP 不完全一致，需要在实验设置中说明统一计算方式。

2. Ablation on Semantic Supervision
   - 无语义监督。
   - 仅视觉一致性。
   - 仅文本语义。
   - 视觉 + 文本。

**表2 语义监督消融实验**

| Variant | Visual Consistency | Text Semantic Loss | Text Loss Type | Gate | Schedule | TNL2K AUC | TNL2K P | TNL2K NP |
|---|:---:|:---:|---|---|---|---:|---:|---:|
| Baseline | - | - | - | - | - | - | - | - |
| Visual Only | yes | no | - | - | stage | - | - | - |
| Text Only | no | yes | direction | sigmoid | stage | - | - | - |
| Distance Supervision | yes | yes | distance | sigmoid | stage | - | - | - |
| Direction Supervision | yes | yes | direction | sigmoid | stage | - | - | - |
| Distance + Direction | yes | yes | both | sigmoid | stage | - | - | - |
| Direction w/o Gate | yes | yes | direction | none | stage | - | - | - |
| Direction w/o Schedule | yes | yes | direction | sigmoid | none | - | - | - |
| Direction w/o Gate & Schedule | yes | yes | direction | none | none | - | - | - |

说明：

- 表2只服务于内部消融，重点证明语义监督各组件的必要性。
- 消融实验统一只在 TNL2K 上做，避免评估成本过高。
- 泛化性由表1的多数据集主结果体现，不在每个消融项上重复验证。

3. Ablation on Text Loss Design
   - distance。
   - direction。
   - both。

4. Ablation on Gate and Schedule
   - sigmoid gate。
   - no gate。
   - no schedule。
   - no gate + no schedule。

5. LoRA Experiments
   - base LoRA，不加语义监督。
   - semantic LoRA distance。
   - semantic LoRA direction。
   - semantic LoRA direction + backbone。
   - 对比 full fine-tuning 的性能、可训练参数量和训练成本。

**表3 LoRA 高效微调实验**

| Variant | Fine-tuning Type | Semantic Loss | Text Loss Type | LoRA Target Modules | Trainable Params | Trainable Ratio | TNL2K AUC | TNL2K P | TNL2K NP |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| Full Fine-tuning | full | yes | final | all trainable modules | - | - | - | - | - |
| LoRA | peft_lora | no | - | language/fusion/confidence | - | - | - | - | - |
| Semantic LoRA-Distance | peft_lora | yes | distance | language/fusion/confidence | - | - | - | - | - |
| Semantic LoRA-Direction | peft_lora | yes | direction | language/fusion/confidence | - | - | - | - | - |
| Semantic LoRA-Direction-Backbone | peft_lora | yes | direction | backbone + language/fusion/confidence | - | - | - | - | - |

说明：

- 表3用于支撑第二创新点：语义监督与参数高效微调结合。
- 重点不是只比较性能，还要报告可训练参数量和可训练参数占比。
- `Full Fine-tuning` 使用同等语义监督设计，作为 LoRA 性能上界和训练成本参考。
- `LoRA` 是不加语义监督的普通 LoRA baseline。
- `Semantic LoRA-Direction-Backbone` 是当前最终版主方法，对应 `experiments/salttrack/salttrack_base.yaml`。

6. Generalization Evaluation
   - TNL2K。
   - LaSOT-Language。
   - 可选 OTB99-Lang 或其他语言跟踪集。

7. Visualization and Analysis
   - 文本-视觉相似度曲线。
   - 特征 t-SNE。
   - pred-text similarity 分布。
   - 困难样例可视化。

## 4. 当前实验矩阵

### 4.0 最终版配置入口

论文最终版算法统一放在：

- `experiments/salttrack/salttrack_base.yaml`

推荐训练入口：

```bash
python lib/train/run_training.py \
  --script salttrack \
  --config salttrack_base \
  --save_dir output \
  --use_lmdb 0 \
  --use_wandb 0
```

推荐测试入口：

```bash
python tracking/test.py \
  --tracker_name salttrack \
  --tracker_param salttrack_base \
  --dataset_name tnl2k \
  --threads 8 \
  --num_gpus 1 \
  --ckpt_path output/checkpoints/train/salttrack/salttrack_base/SALTTrack_ep0080.pth.tar
```

说明：

- `salttrack` 是当前项目唯一正式入口，配置、checkpoint 和结果目录统一使用 SALTTrack 命名。
- 后续论文最终结果、主表和正式推理优先使用 `salttrack_base`，避免继续混用消融配置名。

### 4.1 语义监督消融

| 配置 | 目的 | 备注 |
|---|---|---|
| `salttrack_baseline` | 原始 baseline | 不加语义监督 |
| `salttrack_distance` | 距离式语义监督 | 当前 v3 baseline |
| `salttrack_direction_full` | 方向式语义监督 | full fine-tuning 参考配置 |
| `salttrack_both` | 距离 + 方向 | 验证组合监督 |

### 4.2 语义组件消融

| 配置 | 目的 |
|---|---|
| `salttrack_visual` | 只保留视觉一致性 |
| `salttrack_text` | 只保留文本语义约束 |
| `salttrack_nogate` | 去掉 gate |
| `salttrack_noschedule` | 去掉 schedule |
| `salttrack_nogate_noschedule` | gate 与 schedule 同时去掉 |

### 4.3 LoRA 高效微调

| 配置 | 语义监督 | LoRA 注入位置 | 目的 |
|---|---|---|---|
| `salttrack_lora` | 否 | language/fusion/confidence | 普通 LoRA baseline |
| `salttrack_lora_distance` | 是，distance | language/fusion/confidence | LoRA + 距离监督 |
| `salttrack_lora_direction` | 是，direction | language/fusion/confidence | LoRA + 方向监督 |
| `salttrack_base` | 是，direction | backbone + language/fusion/confidence | 最终版主方法 |

## 5. 训练排期建议

### 阶段一：确定主方法结果

目标：先把最终版 Semantic LoRA-Direction-Backbone 结果站稳。

优先级：

1. `salttrack_base`
2. `salttrack_baseline`
3. 代表性公开方法结果整理

评估：

- LaSOT-Language。
- TNL2K。
- OTB99-Lang。
- VastTrack。
- MGIT。

产出：

- 主结果表。
- 最终方法相对 baseline 和公开方法的多数据集对比。

### 阶段二：完成语义监督消融

目标：证明语义监督设计不是偶然有效。

优先级：

1. distance。
2. visual-only。
3. text-only。
4. no gate。
5. no schedule。
6. no gate + no schedule。

评估：

- 优先 TNL2K。
- 关键配置补 LaSOT-Language。

产出：

- 消融表。
- gate 和 schedule 是否保留的证据。

### 阶段三：LoRA 第二创新点

目标：证明参数高效微调路线成立，并且语义监督对 LoRA 仍然有效。

优先级：

1. `salttrack_lora`
2. `salttrack_lora_distance`
3. `salttrack_lora_direction`
4. `salttrack_base`

评估：

- TNL2K 全量评估。
- 最优两组补 LaSOT-Language。

必须补充：

- 总参数量。
- 可训练参数量。
- 可训练参数占比。
- 训练显存或训练时间，如果方便记录。

产出：

- LoRA 结果表。
- full fine-tuning vs LoRA 性能/参数效率对比表。

### 阶段四：可视化和论文分析

目标：解释为什么方法有效。

优先级：

1. pred-text similarity 分布。
2. distance vs direction 的特征坍塌分析。
3. 跟踪过程中文本-视觉相似度曲线。
4. 困难场景定性图。

产出：

- 一张方法图。
- 一张主结果表。
- 一张消融表。
- 一张 LoRA 参数效率表。
- 一到两张可视化分析图。

## 6. 近期行动清单

### 立即需要做

- [ ] 确认主方法最终采用 `direction` 还是 `both`。
- [ ] 汇总 TNL2K 已有结果，填入主结果表。
- [ ] 补跑缺失的 LaSOT-Language 关键配置。
- [ ] 跑完 LoRA 四组从 epoch 60 到 80 的评估。
- [ ] 增加参数统计脚本或训练启动日志。

### 写作前必须明确

- [ ] 主方法名称：建议使用 `SALT-Track`。
- [ ] 第一贡献名称：建议使用 `Text-Driven Semantic Alignment`。
- [ ] 方向监督名称：建议使用 `Directional Semantic Supervision`。
- [ ] 第二贡献名称：建议使用 `Semantic-Guided LoRA Adaptation`。
- [ ] 是否把 LoRA 放在 Method 主体，还是作为单独 subsection。

## 7. 当前建议结论

论文主贡献应当按以下顺序强调：

1. 文本语义对齐损失是核心贡献。
2. 方向式文本监督是语义损失中的关键改进。
3. LoRA 是第二创新点，用来证明该语义监督框架可以支持参数高效适配。

LoRA 不建议写成“我们用了 LoRA”这么弱的表述，而应写成：

> 我们进一步探索了语义监督与参数高效微调的结合，提出语义引导的 LoRA 适配策略，使模型在只更新少量低秩参数的情况下仍能获得文本语义监督带来的收益。
