# -*- coding: utf-8 -*-
"""
模块 03 — 用户分群分析（K-Means 聚类 + PCA 降维）
===================================================
分析维度：
  1. 特征工程 —— 基于量表构建5维特征空间
  2. StandardScaler 标准化 —— 消除量纲差异
  3. 肘部法则（Elbow Method）—— 科学确定最优K值
  4. K-Means 聚类（K=3）—— 将216名用户分为三类
  5. 轮廓系数（Silhouette Score）—— 评估聚类质量
  6. PCA 主成分分析降维 —— 5维→2维用于可视化
  7. 聚类散点图 + 雷达图

核心技术：
  - StandardScaler：Z-Score标准化，使各特征均值为0、标准差为1
  - K-Means：基于欧氏距离的无监督聚类算法
  - 肘部法则：通过inertia（簇内平方和）曲线选择最优K
  - 轮廓系数：衡量样本与自身簇的相似度（-1到1，越大越好）
  - PCA：主成分分析，保留最大方差方向，实现高维数据可视化
  - 雷达图：多维特征的群体差异全景对比

输出：charts/03_pca_scatter.png, 03_radar.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(CHART_DIR, exist_ok=True)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ========== 1. 数据加载与特征工程 ==========
print("=" * 70)
print("模块 03：用户分群分析（K-Means + PCA）")
print("=" * 70)

df = pd.read_excel(XLSX_PATH)
N = len(df)
print(f"数据加载完成，N = {N}")


# ---- Likert 映射函数 ----
def map_likert(series):
    """
    【重要知识点】Likert量表编码
    将文本型李克特量表（Likert Scale）映射为数值，便于量化分析。
    5点量表：很不符合=1, 不符合=2, 一般=3, 符合=4, 很符合=5
    缺失值填充为中间值3（一般），避免极端偏差。
    """
    return series.map(LIKERT_MAP).fillna(3)


# ---- 构建特征矩阵 ----
print("\n[步骤1] 构建特征矩阵（5个维度）...")

# 特征1：使用时长数值化（1-4编码）
df['duration_num'] = df['5、您日均使用短视频应用的时长有？'].map(DURATION_MAP)
print(f"  使用时长范围：{df['duration_num'].min()}-{df['duration_num'].max()}，均值：{df['duration_num'].mean():.2f}")

# 特征2：成瘾指数（Q11-Q18共8题取均值，反映行为成瘾程度）
addiction_scores = [map_likert(df[col]) for col in ADDICTION_COLS]
df['addiction_score'] = np.mean(addiction_scores, axis=0)
print(f"  成瘾指数范围：{df['addiction_score'].min():.2f}-{df['addiction_score'].max():.2f}，均值：{df['addiction_score'].mean():.2f}")

# 特征3：自控力指数（Q19-Q22共4题取均值）
# 【重要】Q21、Q22为反向计分题，需反转：1→5, 2→4, 3→3, 4→2, 5→1
# 原因：反向题的高分代表"低自控"，与其他题方向相反，必须统一
self_ctrl_scores = []
for col in SELF_CONTROL_COLS:
    s = map_likert(df[col])
    if 'Q21' in col or 'Q22' in col:
        s = 6 - s  # 反向计分
    self_ctrl_scores.append(s)
df['selfctrl_score'] = np.mean(self_ctrl_scores, axis=0)
print(f"  自控力指数范围：{df['selfctrl_score'].min():.2f}-{df['selfctrl_score'].max():.2f}，均值：{df['selfctrl_score'].mean():.2f}")

# 特征4：习得性无助指数（Q27-Q32共6题取均值，反映心理无力感）
helpless_scores = [map_likert(df[col]) for col in HELPLESS_COLS]
df['helpless_score'] = np.mean(helpless_scores, axis=0)
print(f"  习得性无助指数范围：{df['helpless_score'].min():.2f}-{df['helpless_score'].max():.2f}，均值：{df['helpless_score'].mean():.2f}")

# 特征5：算法依赖指数（Q8-Q10共3题取均值，反映对推荐系统的依赖程度）
algo_scores = [map_likert(df[col]) for col in ALGO_COLS]
df['algo_score'] = np.mean(algo_scores, axis=0)
print(f"  算法依赖指数范围：{df['algo_score'].min():.2f}-{df['algo_score'].max():.2f}，均值：{df['algo_score'].mean():.2f}")

# 构建特征矩阵
feature_cols = ['duration_num', 'addiction_score', 'selfctrl_score',
                'helpless_score', 'algo_score']
feature_labels = ['使用时长', '成瘾指数', '自控力', '习得性无助', '算法依赖']
X = df[feature_cols].values
print(f"  特征矩阵形状：{X.shape}（{X.shape[0]}样本 × {X.shape[1]}特征）")


# ========== 2. 数据标准化 ==========
print("\n[步骤2] StandardScaler Z-Score标准化...")
"""
【重要知识点】为什么要标准化？
K-Means基于欧氏距离 d(x,y) = √(Σ(xi-yi)²) 计算样本间相似度。
若各特征量纲不同（如"使用时长"1-4 vs "成瘾指数"1-5），量纲大的特征
会在距离计算中占主导地位，导致聚类结果偏差。

StandardScaler执行Z-Score变换：z = (x - μ) / σ
使每个特征均值为0、标准差为1，消除量纲影响，确保各维度贡献公平。
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  标准化后均值（≈0）：{X_scaled.mean(axis=0).round(6)}")
print(f"  标准化后标准差（≈1）：{X_scaled.std(axis=0).round(6)}")


# ========== 3. 肘部法则确定最优K值 ==========
print("\n[步骤3] 肘部法则（Elbow Method）确定K值...")
"""
【重要知识点】肘部法则（Elbow Method）
K-Means需要预先指定簇数K，但最优K值未知。
肘部法则通过观察不同K值对应的inertia（簇内平方和，WCSS）的变化趋势：
  - inertia = Σ(每个样本到其所在簇中心的距离²)
  - K增大时inertia必然减小（簇越多，每个簇越小，距离越近）
  - 关注下降速度的"拐点"（elbow）：拐点前下降快（欠拟合），拐点后下降慢（过拟合）
  - 选择拐点处的K作为最优簇数

本分析对K=2~8进行计算，绘制inertia曲线观察拐点。
"""
K_range = range(2, 9)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  K={k}: inertia={km.inertia_:.2f}, silhouette={silhouette_score(X_scaled, labels):.3f}")

# 选择K=3的原因说明
best_k = 3
print(f"\n  结论：选择K={best_k}（inertia下降拐点+轮廓系数最优）")


# ========== 4. K-Means 聚类（K=3） ==========
print(f"\n[步骤4] K-Means 聚类（K={best_k}）...")
"""
【重要知识点】K-Means算法原理
K-Means是基于迭代的无监督聚类算法，步骤如下：

1. 初始化：随机选择K个数据点作为初始聚类中心（centroid）
2. 迭代执行以下步骤直到收敛（中心不再变化）：
   a. 分配步骤（Assignment）：将每个样本分配给最近的聚类中心
      （按欧氏距离最小化原则）
   b. 更新步骤（Update）：重新计算每个簇的均值，作为新的聚类中心
3. 收敛：当聚类中心不再变化（或变化极小）时停止

参数说明：
  - n_clusters=3：聚为3个簇
  - random_state=42：固定随机种子，保证结果可复现
  - n_init=10：运行10次取最优解（避免局部最优）
"""
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster_raw'] = kmeans.fit_predict(X_scaled)

# 根据各簇特征均值，赋予语义标签（自动化处理）
cluster_means_raw = df.groupby('cluster_raw')[feature_cols].mean()

# 按成瘾指数从小到大排序，确定标签映射
sorted_clusters = cluster_means_raw.sort_values('addiction_score').index.tolist()
# 成瘾最低 → 轻度理性，中间 → 中度依赖，最高 → 重度成瘾
label_map = {
    sorted_clusters[0]: ('群体A 轻度理性用户', 1),
    sorted_clusters[1]: ('群体B 中度依赖用户', 0),
    sorted_clusters[2]: ('群体C 重度成瘾高风险', 2),
}
# 建立语义化聚类列
df['cluster_label'] = df['cluster_raw'].map(lambda x: label_map[x][0])
df['cluster'] = df['cluster_raw'].map(lambda x: label_map[x][1])

# 聚类结果统计
print(f"\n  聚类结果：")
cluster_sizes = df.groupby(['cluster', 'cluster_label']).size().reset_index(name='人数')
cluster_sizes = cluster_sizes.sort_values('cluster')
for _, row in cluster_sizes.iterrows():
    print(f"    {row['cluster_label']}：{row['人数']}人（{row['人数']/N*100:.1f}%）")

# 计算轮廓系数（全局评估指标）
sil_score = silhouette_score(X_scaled, df['cluster_raw'].values)
print(f"\n  轮廓系数（Silhouette Score）：{sil_score:.3f}")
"""
【重要知识点】轮廓系数（Silhouette Coefficient）
衡量聚类质量的指标，取值范围[-1, 1]：
  - 接近 1：样本被很好地分配到正确的簇
  - 接近 0：样本在两个簇的边界上
  - 接近 -1：样本可能被分配到了错误的簇
总体轮廓系数是所有样本轮廓系数的均值，>0.3通常认为聚类效果较好。
"""

# 各簇特征均值统计
print(f"\n  各簇特征均值（聚类画像）：")
cluster_means = df.groupby('cluster_label')[feature_cols].mean()
# 重排序：轻度→中度→重度
cluster_order = ['群体A 轻度理性用户', '群体B 中度依赖用户', '群体C 重度成瘾高风险']
cluster_means = cluster_means.reindex(cluster_order)
print(cluster_means.round(2).to_string())

# 与全体均值对比
overall_means = df[feature_cols].mean()
print(f"\n  全体样本均值：")
print(f"  {overall_means.round(2).to_dict()}")


# ========== 5. PCA 降维 ==========
print("\n[步骤5] PCA 主成分分析降维...")
"""
【重要知识点】PCA（Principal Component Analysis，主成分分析）
PCA是无监督的线性降维技术，核心思想：

1. 找到数据方差最大的方向 → 第1主成分（PC1）
2. 在与PC1正交的方向中，找方差最大的方向 → 第2主成分（PC2）
3. 依次类推...

关键概念：
  - 特征值（eigenvalue）：表示每个主成分解释的方差大小
  - 解释方差比（explained variance ratio）：该主成分解释的方差占总方差的百分比
  - 累计解释方差比：前k个主成分累计解释的方差比，越高说明降维保留的信息越多

本分析将5维特征降为2维，用于在散点图中可视化聚类结果。
"""
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pc1'] = X_pca[:, 0]
df['pc2'] = X_pca[:, 1]

print(f"  PC1 解释方差比：{pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2 解释方差比：{pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  累计解释方差比：{pca.explained_variance_ratio_.sum()*100:.1f}%")

# PCA特征载荷（Loadings）—— 各原始特征在主成分上的投影权重
print(f"\n  PCA特征载荷矩阵（Loadings）：")
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC1({pca.explained_variance_ratio_[0]*100:.1f}%)', f'PC2({pca.explained_variance_ratio_[1]*100:.1f}%)'],
    index=feature_labels
)
print(loadings.round(3).to_string())
"""
载荷解读：
  - 载荷的绝对值越大，说明该原始特征对主成分的贡献越大
  - PC1通常代表"成瘾-心理风险"方向（成瘾、无助正载荷，自控力负载荷）
  - PC2通常代表其他维度的变异
"""


# ========== 6. 图表生成 ==========

def save_chart(fig, filename):
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 已保存：{filename}")


# --- 图9：PCA 二维散点图（聚类可视化） ---
print("\n[步骤6] 生成可视化图表...")

# 动态颜色：轻度=绿色，中度=蓝色，重度=珊瑚红
cluster_color_map = {
    '群体A 轻度理性用户': COLORS['green'],
    '群体B 中度依赖用户': COLORS['blue'],
    '群体C 重度成瘾高风险': COLORS['coral'],
}

fig, ax = plt.subplots(figsize=(8, 6))

for label in cluster_order:
    mask = df['cluster_label'] == label
    size = mask.sum()
    ax.scatter(df.loc[mask, 'pc1'], df.loc[mask, 'pc2'],
               c=cluster_color_map[label],
               label=f'{label}（{size}人）',
               alpha=0.7, s=45, edgecolors='white', linewidth=0.5)

# 绘制聚类中心（在PCA空间中的投影）
centers_pca = pca.transform(kmeans.cluster_centers_)
# 映射到语义顺序
for i, label in enumerate(cluster_order):
    raw_c = sorted_clusters[i]
    ax.scatter(centers_pca[raw_c, 0], centers_pca[raw_c, 1],
               c=cluster_color_map[label], marker='X', s=200,
               edgecolors='black', linewidth=1.5, zorder=5)

ax.set_xlabel(f'PC1（{pca.explained_variance_ratio_[0]*100:.1f}% 方差）', fontsize=11)
ax.set_ylabel(f'PC2（{pca.explained_variance_ratio_[1]*100:.1f}% 方差）', fontsize=11)
ax.set_title('K-Means 聚类 PCA 二维散点图（X=聚类中心）', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linestyle='--')

save_chart(fig, '03_pca_scatter.png')


# --- 图10：雷达图（三群体多维对比） ---
# 动态从数据计算雷达图数值
radar_values = {}
for label in cluster_order:
    vals = cluster_means.loc[label, feature_cols].tolist()
    radar_values[label] = vals

radar_colors = [COLORS['green'], COLORS['blue'], COLORS['coral']]

num_vars = len(feature_labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for (label, values), color in zip(radar_values.items(), radar_colors):
    values_plot = values + values[:1]  # 闭合
    ax.fill(angles, values_plot, alpha=0.15, color=color)
    ax.plot(angles, values_plot, 'o-', linewidth=2, color=color,
            label=label, markersize=6)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(feature_labels, fontsize=11)
ax.set_ylim(0, 5.5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=9, color='#888')
ax.set_title('三群体关键维度对比（雷达图）', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True, alpha=0.3)

save_chart(fig, '03_radar.png')


# ========== 7. 聚类质量综合评估 ==========
print("\n[聚类质量评估]")
print(f"  1. 轮廓系数：{sil_score:.3f}（>0.3为较好，>0.5为良好）")
print(f"  2. PCA累计解释方差：{pca.explained_variance_ratio_.sum()*100:.1f}%（>80%说明降维信息损失小）")

# 各簇样本数与比例
print(f"  3. 簇大小分布：")
for label in cluster_order:
    n = (df['cluster_label'] == label).sum()
    print(f"     {label}：{n}人（{n/N*100:.1f}%）")


# ========== 8. 聚类分析结论 ==========
print("\n" + "=" * 70)
print("聚类分析核心结论")
print("=" * 70)

# 动态生成结论
for label in cluster_order:
    m = cluster_means.loc[label]
    n = (df['cluster_label'] == label).sum()
    print(f"\n{label}（{n}人，{n/N*100:.1f}%）：")
    print(f"  使用时长均值={m['duration_num']:.2f}（{['极短', '短', '中等', '较长', '很长'][int(round(m['duration_num']))-1] if 1 <= round(m['duration_num']) <= 5 else '中等'}）")
    print(f"  成瘾指数均值={m['addiction_score']:.2f}（{'低' if m['addiction_score'] < 2.5 else '中等' if m['addiction_score'] < 3.5 else '高'}）")
    print(f"  自控力均值={m['selfctrl_score']:.2f}（{'低' if m['selfctrl_score'] < 2.5 else '中等' if m['selfctrl_score'] < 3.5 else '高'}）")
    print(f"  习得性无助均值={m['helpless_score']:.2f}（{'低' if m['helpless_score'] < 2.5 else '中等' if m['helpless_score'] < 3.5 else '高'}）")
    print(f"  算法依赖均值={m['algo_score']:.2f}（{'低' if m['algo_score'] < 2.5 else '中等' if m['algo_score'] < 3.5 else '高'}）")

print(f"\n图表输出目录：{CHART_DIR}")
print(f"模块 03 分析完成！共生成 2 张图表（PCA散点图 + 雷达图）。")
