# -*- coding: utf-8 -*-
"""
模块 05 — 决策树分析（成瘾风险预测）
=======================================
目标：基于人口统计、行为习惯、心理因素等特征，构建决策树分类器
      预测用户的短视频成瘾风险等级（低/中/高）。

分析维度：
  1. 特征工程 —— 构建可解释的特征集（不含成瘾量表题目，避免循环逻辑）
  2. 成瘾风险三分类 —— 基于成瘾指数分位数划分低/中/高风险
  3. 决策树建模 —— CART 算法，信息增益（基尼系数）作为分裂准则
  4. 决策树可视化 —— 绘制树结构图，展示分裂规则
  5. 特征重要性 —— 识别影响成瘾风险的关键因素
  6. 模型评估 —— 准确率、混淆矩阵、分类报告、交叉验证

核心技术：
  - DecisionTreeClassifier：基于 CART 算法的决策树分类器
  - 基尼系数（Gini Impurity）：衡量节点纯度的分裂准则
  - 特征重要性（Feature Importance）：基于不纯度减少量的特征贡献度
  - 交叉验证（Cross-Validation）：评估模型泛化能力
  - 混淆矩阵（Confusion Matrix）：分类结果的详细评估

输出：charts/05_decision_tree.png, 05_feature_importance.png, 05_confusion_matrix.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(CHART_DIR, exist_ok=True)

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

# ========== 1. 数据加载 ==========
print("=" * 70)
print("模块 05：决策树分析（成瘾风险预测）")
print("=" * 70)

df = pd.read_excel(XLSX_PATH)
N = len(df)
print(f"数据加载完成，N = {N}")


# ========== 2. 特征工程 ==========
print("\n[步骤1] 特征工程...")
"""
【重要知识点】特征工程 (Feature Engineering)
将原始问卷数据转换为可用于机器学习的数值特征。
注意：不将成瘾量表各题作为特征（否则会产生循环论证），
     而是使用人口统计、行为习惯和其他心理量表。
"""

def map_likert(series):
    """Likert量表文本→数值映射"""
    return series.map(LIKERT_MAP).fillna(3)


# ---- 特征1：性别（二值编码） ----
df['gender'] = (df['1、您的性别？'] == 'B.男').astype(int)  # 0=女, 1=男

# ---- 特征2：年龄（有序编码） ----
age_map = {'A.18岁以下': 1, 'B.18-22岁': 2, 'C.23-30岁': 3, 'D.31-40岁': 4, 'E.41岁以上': 5}
df['age_num'] = df['2、您的年龄？'].map(age_map).fillna(2)

# ---- 特征3：职业（有序编码） ----
job_map = {'A.在校学生': 1, 'B.上班族': 2, 'C.自由职业者': 3}
df['job_num'] = df['3、您的职业或身份？'].map(job_map).fillna(1)

# ---- 特征4：使用时长（有序编码） ----
df['duration_num'] = df['5、您日均使用短视频应用的时长有？'].map(DURATION_MAP).fillna(2)

# ---- 特征5：自控力指数 ----
self_ctrl_scores = []
for col in SELF_CONTROL_COLS:
    s = map_likert(df[col])
    if 'Q21' in col or 'Q22' in col:
        s = 6 - s  # 反向计分
    self_ctrl_scores.append(s)
df['selfctrl_score'] = np.mean(self_ctrl_scores, axis=0)

# ---- 特征6：习得性无助指数 ----
df['helpless_score'] = np.mean([map_likert(df[c]) for c in HELPLESS_COLS], axis=0)

# ---- 特征7：算法依赖指数 ----
df['algo_score'] = np.mean([map_likert(df[c]) for c in ALGO_COLS], axis=0)

# ---- 特征8：是否熬夜（有序编码） ----
late_map = {
    'A.从不': 1, 'B.偶尔': 2, 'C.经常': 3, 'D.几乎每天': 4,
}
late_col = '27、短视频是否曾导致您熬夜（晚于计划时间1小时以上入睡）？ '
df['late_night'] = df[late_col].map(late_map).fillna(2)

# ---- 特征9：影响日常生活（Likert编码） ----
life_col = '28、短视频是否影响我的日常生活习惯'
df['life_impact'] = map_likert(df[life_col])

# ---- 特征10：评论区行为（有序编码） ----
comment_map = {
    'A.直接划走': 1, 'B.看评论但不发言': 2,
    'C.参与讨论，但只支持自己认同的观点': 3,
    'D.愿意理性交流，尊重不同意见': 4,
}
comment_col = '10、您在评论区看到不同观点时，通常会?'
df['comment_behavior'] = df[comment_col].map(comment_map).fillna(2)

# ---- 特征11：刷完视频后的感受（有序编码） ----
after_col = '23、在长时间刷完短视频后，您通常的感觉是?'
after_map = {}
for val in df[after_col].dropna().unique():
    s = str(val).strip()
    if '满足' in s or '充实' in s:
        after_map[val] = 1  # 正面
    elif '空虚' in s or '浪费' in s or '后悔' in s:
        after_map[val] = 3  # 负面
    else:
        after_map[val] = 2  # 中性
df['after_feeling'] = df[after_col].map(after_map).fillna(2)

# ---- 特征12：审美认知提升（Likert编码） ----
aesthetic_col = '29、短视频中优质内容是否提高我的审美认知'
df['aesthetic'] = map_likert(df[aesthetic_col])

# ---- 特征13：推荐内容消费（单选编码） ----
rec_map = {
    'A.完全不受影响': 1, 'B.影响较小': 2,
    'C.有一定影响': 3, 'D.影响很大': 4,
}
rec_col = '17、您认为短视频营销对您的消费决策影响程度是？'
df['consume_influence'] = df[rec_col].map(rec_map).fillna(2)

print("  构建13个特征：性别、年龄、职业、使用时长、自控力、习得性无助、")
print("  算法依赖、熬夜频率、日常影响、评论行为、刷后感受、审美认知、消费影响")

# ========== 3. 目标变量构建 ==========
print("\n[步骤2] 构建成瘾风险三分类目标变量...")
"""
【重要知识点】目标变量构建
基于成瘾指数（8题均值）将用户分为三个风险等级：
  - 低风险：成瘾指数 < 2.5（约1-2分，"很不符合~不符合"区间）
  - 中风险：2.5 ≤ 成瘾指数 < 3.5（约3分，"一般"区间）
  - 高风险：成瘾指数 ≥ 3.5（约4-5分，"符合~很符合"区间）
这是有序多分类问题。
"""

# 计算成瘾指数
addiction_scores = np.mean([map_likert(df[col]) for col in ADDICTION_COLS], axis=0)
df['addiction_score'] = addiction_scores

def classify_risk(score):
    """将成瘾指数映射为风险等级"""
    if score < 2.5:
        return 0  # 低风险
    elif score < 3.5:
        return 1  # 中风险
    else:
        return 2  # 高风险

df['risk_level'] = df['addiction_score'].apply(classify_risk)

risk_labels = {0: '低风险', 1: '中风险', 2: '高风险'}
risk_colors = {0: COLORS['green'], 1: COLORS['amber'], 2: COLORS['coral']}

print(f"  成瘾指数分布：均值={df['addiction_score'].mean():.2f}, "
      f"标准差={df['addiction_score'].std():.2f}")
print(f"  风险等级分布：")
for level, label in risk_labels.items():
    n = (df['risk_level'] == level).sum()
    print(f"    {label}：{n}人（{n/N*100:.1f}%）")


# ========== 4. 特征矩阵与数据集划分 ==========
print("\n[步骤3] 准备特征矩阵与数据集划分...")
"""
【重要知识点】训练集/测试集划分
- 训练集（Train Set）：用于训练模型（本题80%=173人）
- 测试集（Test Set）：用于评估模型泛化能力（本题20%=43人）
- 随机种子 random_state=42：保证每次运行结果一致（可复现）
"""

feature_names = ['性别', '年龄', '职业', '使用时长', '自控力',
                 '习得性无助', '算法依赖', '熬夜频率', '日常影响',
                 '评论行为', '刷后感受', '审美认知', '消费影响']
feature_cols_dt = ['gender', 'age_num', 'job_num', 'duration_num', 'selfctrl_score',
                   'helpless_score', 'algo_score', 'late_night', 'life_impact',
                   'comment_behavior', 'after_feeling', 'aesthetic', 'consume_influence']

X = df[feature_cols_dt].values
y = df['risk_level'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  特征数：{X.shape[1]}")
print(f"  训练集：{X_train.shape[0]}人，测试集：{X_test.shape[0]}人")
print(f"  训练集风险分布：低风险{(y_train==0).sum()}人, "
      f"中风险{(y_train==1).sum()}人, 高风险{(y_train==2).sum()}人")


# ========== 5. 决策树建模 ==========
print("\n[步骤4] 构建决策树分类器...")
"""
【重要知识点】决策树 (Decision Tree) 原理
决策树是一种基于树结构的监督学习分类/回归算法，本题使用CART算法。

核心概念：
1. 节点分裂：每个内部节点根据某个特征的阈值将数据分为两部分
   - 例如：自控力 ≤ 3.0 → 左子树，自控力 > 3.0 → 右子树
2. 分裂准则：基尼系数（Gini Impurity）
   - Gini = 1 - Σ(p_i²)，p_i是第i类样本的比例
   - Gini=0 表示纯节点（只含一类样本），Gini越大越不纯
   - 选择使分裂后Gini加权减少量最大的特征和阈值
3. 叶节点：最终分类结果，取该节点多数类
4. 剪枝：限制树的最大深度（max_depth），防止过拟合

参数说明：
  - max_depth：树的最大深度，越大越容易过拟合
  - min_samples_split：分裂节点所需的最少样本数
  - min_samples_leaf：叶节点所需的最少样本数
  - random_state=42：保证结果可复现
"""

# 通过交叉验证选择最优树深度
print("  交叉验证选择最优超参数...")
best_depth = 3
best_cv_score = 0
for depth in range(2, 8):
    dt_temp = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5,
                                      random_state=42)
    cv_scores = cross_val_score(dt_temp, X_train, y_train, cv=5, scoring='accuracy')
    mean_cv = cv_scores.mean()
    print(f"    max_depth={depth}: CV准确率={mean_cv:.3f} (±{cv_scores.std():.3f})")
    if mean_cv > best_cv_score:
        best_cv_score = mean_cv
        best_depth = depth

print(f"\n  最优深度：{best_depth}（CV准确率={best_cv_score:.3f}）")

# 用最优参数训练最终模型
dt = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=42,
    class_weight='balanced'  # 处理类别不平衡
)
dt.fit(X_train, y_train)

# 预测
y_pred = dt.predict(X_test)
train_acc = accuracy_score(y_train, dt.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

print(f"\n  训练集准确率：{train_acc:.3f}")
print(f"  测试集准确率：{test_acc:.3f}")


# ========== 6. 模型评估 ==========
print("\n[步骤5] 模型评估...")
"""
【重要知识点】分类评估指标
- 准确率（Accuracy）：预测正确的比例 = (TP+TN)/(TP+TN+FP+FN)
- 精确率（Precision）：预测为正的样本中实际为正的比例 = TP/(TP+FP)
- 召回率（Recall）：实际为正的样本中被预测为正的比例 = TP/(TP+FN)
- F1分数：精确率和召回率的调和平均 = 2×P×R/(P+R)
- 混淆矩阵：展示各类别的详细预测情况
"""

print("\n  分类报告：")
print(classification_report(y_test, y_pred, target_names=list(risk_labels.values())))

# 交叉验证最终评估
cv_final = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
print(f"  5折交叉验证准确率：{cv_final.mean():.3f} (±{cv_final.std():.3f})")


# ========== 7. 特征重要性 ==========
print("\n[步骤6] 特征重要性排名...")
"""
【重要知识点】特征重要性 (Feature Importance)
决策树的特征重要性衡量每个特征对分裂决策的贡献程度：
  - 计算方式：该特征被用于分裂时，基尼系数减少量的加权和
  - 重要性越高，说明该特征在区分不同类别时越关键
  - 所有特征重要性之和 = 1
"""
importances = dt.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("  特征重要性排名：")
for i in sorted_idx:
    if importances[i] > 0:
        print(f"    {feature_names[i]}: {importances[i]:.4f} ({importances[i]*100:.1f}%)")


# ========== 8. 决策规则提取 ==========
print("\n[步骤7] 核心决策规则...")
"""
【重要知识点】决策树规则提取
决策树的一大优势是可解释性强——每条从根到叶的路径
就是一条IF-THEN规则，例如：
  IF 自控力 ≤ 3.0 AND 习得性无助 > 2.5 THEN → 高风险
"""
tree_text = export_text(dt, feature_names=feature_names, decimals=2)
# 只打印前30行规则
lines = tree_text.split('\n')
for line in lines[:35]:
    print(f"  {line}")
if len(lines) > 35:
    print(f"  ... (共{len(lines)}行规则)")


# ========== 9. 图表生成 ==========

def save_chart(fig, filename):
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 已保存：{filename}")

print("\n[步骤8] 生成可视化图表...")

# --- 图A：决策树结构图 ---
fig, ax = plt.subplots(figsize=(16, 10))

# 颜色映射
class_colors = [COLORS['green'], COLORS['amber'], COLORS['coral']]

plot_tree(dt,
          feature_names=feature_names,
          class_names=list(risk_labels.values()),
          filled=True,
          rounded=True,
          fontsize=9,
          ax=ax,
          impurity=True,
          node_ids=False,
          proportion=False,
          max_depth=best_depth)

ax.set_title('决策树结构图（成瘾风险预测）\n'
             f'准确率={test_acc:.1%}，深度={best_depth}，基尼系数分裂',
             fontsize=14, fontweight='bold', pad=15)
save_chart(fig, '05_decision_tree.png')


# --- 图B：特征重要性水平柱状图 ---
fig, ax = plt.subplots(figsize=(7, 5))

# 只展示有贡献的特征
mask = importances[sorted_idx] > 0
names_sorted = [feature_names[i] for i in sorted_idx if importances[i] > 0]
vals_sorted = [importances[i] for i in sorted_idx if importances[i] > 0]

# 升序排列（最大的在上面）
names_sorted = names_sorted[::-1]
vals_sorted = vals_sorted[::-1]

# 颜色：最重要的特征用醒目颜色
bar_colors = [COLORS['coral'] if v > 0.15 else
               COLORS['amber'] if v > 0.08 else
               COLORS['blue'] for v in vals_sorted]

bars = ax.barh(names_sorted, vals_sorted, color=bar_colors,
               edgecolor='white', height=0.6)

for bar, v in zip(bars, vals_sorted):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{v*100:.1f}%', va='center', fontsize=10, fontweight='bold', color='#333')

ax.set_title('决策树特征重要性排名', fontsize=14, fontweight='bold')
ax.set_xlabel('重要性（基尼系数减少量占比）', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, max(vals_sorted) * 1.3)
save_chart(fig, '05_feature_importance.png')


# --- 图C：混淆矩阵 ---
fig, ax = plt.subplots(figsize=(5.5, 4.5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(risk_labels.values()))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('混淆矩阵（测试集）', fontsize=14, fontweight='bold')
ax.set_xlabel('预测标签', fontsize=11)
ax.set_ylabel('真实标签', fontsize=11)

# 在格子中标注数值
for i in range(3):
    for j in range(3):
        val = cm[i, j]
        color = 'white' if val > cm.max()/2 else 'black'
        ax.text(j, i, str(val), ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)

save_chart(fig, '05_confusion_matrix.png')


# ========== 10. 总结 ==========
print("\n" + "=" * 70)
print("决策树分析总结")
print("=" * 70)
print(f"""
1. 模型概况
   - 目标：预测用户成瘾风险等级（低/中/高）
   - 特征数：{len(feature_cols_dt)}个（人口统计+行为习惯+心理量表）
   - 最优树深度：{best_depth}（通过5折交叉验证选定）
   - 测试集准确率：{test_acc:.1%}
   - 5折交叉验证准确率：{cv_final.mean():.1%} ± {cv_final.std():.1%}

2. 风险等级分布
   - 低风险：{(df['risk_level']==0).sum()}人（{(df['risk_level']==0).sum()/N*100:.1f}%）
   - 中风险：{(df['risk_level']==1).sum()}人（{(df['risk_level']==1).sum()/N*100:.1f}%）
   - 高风险：{(df['risk_level']==2).sum()}人（{(df['risk_level']==2).sum()/N*100:.1f}%）

3. 关键发现
   - 最重要的预测特征：{feature_names[sorted_idx[0]]}（{importances[sorted_idx[0]]*100:.1f}%）
   - 其次：{feature_names[sorted_idx[1]]}（{importances[sorted_idx[1]]*100:.1f}%）
   - 决策树可解释性强，每条路径即为一条IF-THEN预测规则
   - 模型使用 balanced class_weight 处理了类别不平衡问题

4. 图表输出
   - 05_decision_tree.png：决策树结构可视化
   - 05_feature_importance.png：特征重要性排名
   - 05_confusion_matrix.png：混淆矩阵评估
""")
print(f"模块 05 分析完成！共生成 3 张图表。")
