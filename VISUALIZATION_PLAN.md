# SALT-Track 可视化分析规划

## 核心目标

通过可视化和可解释性分析，证明语义监督loss的有效性：
1. 文本被有效利用，提升了模型的语义认知能力
2. 方向监督相比距离监督能防止特征坍塌
3. 语义对齐确实改善了跟踪性能

---

## 优先级1：必做可视化（论文主图）

### 1.1 跟踪过程中的文本-视觉对齐度变化

**目的**：证明语义监督让模型更好地利用了文本信息

**方案**：
- 选择典型序列（5-10个），从第1帧到最后一帧
- 绘制两条曲线：
  - Baseline（无语义监督）：`cos_sim(pred_feat, text_feat)`
  - Ours（语义监督）：`cos_sim(pred_feat, text_feat)`
- 横轴：帧数，纵轴：余弦相似度

**预期结果**：
- Baseline：相似度较低且波动大（文本未被有效利用）
- Ours：相似度更高且更稳定（文本有效引导跟踪）

**实现位置**：
- 在 `lib/test/tracker/salttrack.py` 的 `track()` 方法中记录每帧的相似度
- 保存为 JSON 文件，后处理绘图

---

### 1.2 特征空间t-SNE可视化

**目的**：证明方向监督防止特征坍塌

**方案**：
- 在测试集上提取所有预测框的特征（1000-2000个样本）
- 用t-SNE降维到2D
- 对比三种方法：
  - Baseline（无语义监督）
  - 距离监督
  - 方向监督（Ours）
- 按目标类别着色

**预期结果**：
- 距离监督：特征聚成一团（坍塌）
- 方向监督：特征分散，不同类别有明显边界

**实现**：
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# features: [N, D], labels: [N]
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features)

plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', s=5, alpha=0.6)
plt.title('Feature Space (Direction Supervision)')
```

---

### 1.3 pred_text_similarity分布直方图

**目的**：量化特征坍塌问题

**方案**：
- 统计测试集上所有帧的 `pred_text_similarity`
- 绘制分布直方图，对比：
  - 距离监督：集中在0.85-0.95（坍塌）
  - 方向监督：分散在0.60-0.80（健康）

**实现**：
```python
plt.hist(sim_distance, bins=50, alpha=0.5, label='Distance Supervision')
plt.hist(sim_direction, bins=50, alpha=0.5, label='Direction Supervision')
plt.xlabel('Cosine Similarity (pred, text)')
plt.ylabel('Frequency')
plt.legend()
```

---

### 1.4 训练曲线对比

**目的**：展示训练动态差异

**方案**：
- 从TensorBoard日志中提取：
  - `Loss/semantic_text`
  - `pred_text_similarity`
  - `gt_text_similarity`
  - `semantic_gate`
- 绘制4×1子图，对比距离监督 vs 方向监督

**预期结果**：
- 距离监督：`pred_text_similarity` 快速飙升到0.9+
- 方向监督：`pred_text_similarity` 保持在合理范围

---

## 优先级2：强烈建议

### 2.1 方向一致性可视化

**目的**：直观展示方向监督的核心思想

**方案**：
- 在2D特征空间中（PCA降维）
- 画出：
  - `pred_feat` 点
  - `gt_feat` 点
  - `text_feat` 点
  - `pred → text` 箭头
  - `gt → text` 箭头
- 展示方向监督下两个箭头更平行

**实现**：
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pred_2d = pca.transform(pred_feat)
gt_2d = pca.transform(gt_feat)
text_2d = pca.transform(text_feat)

plt.scatter(pred_2d[:, 0], pred_2d[:, 1], c='red', label='Pred')
plt.scatter(gt_2d[:, 0], gt_2d[:, 1], c='green', label='GT')
plt.scatter(text_2d[:, 0], text_2d[:, 1], c='blue', label='Text')
plt.quiver(pred_2d[:, 0], pred_2d[:, 1], 
           text_2d[:, 0]-pred_2d[:, 0], text_2d[:, 1]-pred_2d[:, 1],
           color='red', alpha=0.5)
```

---

### 2.2 困难场景案例分析

**目的**：定性展示方法优势

**选择场景**：
- 遮挡场景（occlusion）
- 相似物干扰（similar objects）
- 外观剧变（appearance change）
- 快速运动（fast motion）

**展示内容**：
- 每个场景选2-3个关键帧
- 对比Baseline vs Ours：
  - 预测框位置
  - `cos_sim(pred, text)` 数值
  - 成功/失败标注

---

## 优先级3：加分项

### 3.1 特征有效秩分析

**目的**：量化特征坍塌程度

**方案**：
```python
def effective_rank(features):
    # features: [N, D]
    U, S, V = torch.svd(features)
    return (S.sum() ** 2) / (S ** 2).sum()

rank_distance = effective_rank(features_distance)
rank_direction = effective_rank(features_direction)
```

**预期**：方向监督的有效秩更高（特征更丰富）

---

### 3.2 类内/类间距离分析

**目的**：评估特征判别性

**方案**：
```python
# 同类目标特征的平均距离
intra_class_dist = mean_distance_within_class(features, labels)
# 不同类目标特征的平均距离
inter_class_dist = mean_distance_between_class(features, labels)
# 判别性指标
discriminability = inter_class_dist / intra_class_dist
```

---

### 3.3 消融实验特征空间对比

**目的**：全面展示不同监督策略的效果

**方案**：
- 3×2子图矩阵，每个子图是t-SNE结果：
  - 无语义监督
  - 纯视觉一致性
  - 纯文本对齐（距离）
  - 纯文本对齐（方向）
  - 视觉+文本（距离）
  - 视觉+文本（方向）

---

## 实现计划

### 阶段1：数据收集（训练完成后）
- [ ] 修改 `lib/test/tracker/salttrack.py`，记录每帧的特征和相似度
- [ ] 在测试集上运行，保存特征到 `.pkl` 文件
- [ ] 提取TensorBoard训练日志

### 阶段2：核心可视化（优先级1）
- [ ] 1.1 跟踪过程对齐度曲线
- [ ] 1.2 t-SNE特征空间
- [ ] 1.3 相似度分布直方图
- [ ] 1.4 训练曲线对比

### 阶段3：补充分析（优先级2-3）
- [ ] 2.1 方向一致性可视化
- [ ] 2.2 困难场景案例
- [ ] 3.1 有效秩分析
- [ ] 其他加分项

---

## 代码组织

建议创建：
```
tracking/
  visualization/
    extract_features.py      # 提取特征和相似度
    plot_alignment_curve.py  # 1.1 对齐度曲线
    plot_tsne.py             # 1.2 t-SNE
    plot_similarity_dist.py  # 1.3 分布直方图
    plot_training_curves.py  # 1.4 训练曲线
    plot_direction.py        # 2.1 方向可视化
    analyze_cases.py         # 2.2 案例分析
    compute_metrics.py       # 3.1-3.3 量化指标
```

---

## 论文图表规划

**Figure 1**: 方法框架图（已有 `asset/framework.png`）

**Figure 2**: 训练曲线对比（1.4）
- 4个子图：semantic_text_loss, pred_text_similarity, gt_text_similarity, semantic_gate

**Figure 3**: 特征空间可视化（1.2）
- 3个子图：Baseline, 距离监督, 方向监督（Ours）

**Figure 4**: 跟踪过程对齐度变化（1.1）
- 选择3-4个典型序列，每个序列一条曲线对比

**Figure 5**: 相似度分布对比（1.3）
- 直方图叠加显示

**Figure 6**: 困难场景案例（2.2）
- 4个场景 × 3帧 = 12个子图

**Table**: 量化指标对比
- 有效秩、判别性、平均相似度等

---

## 当前状态

- [x] 方向监督代码实现
- [x] v3_direction配置创建
- [ ] 等待训练完成
- [ ] 开始可视化分析

**下一步**：优先拿到最佳实验结果，训练完成后再开展可视化工作。
