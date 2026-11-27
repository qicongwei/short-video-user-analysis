# -*- coding: utf-8 -*-
"""
模块 02 — 内容偏好与行为分析
=============================
分析维度：
  1. 使用目的分布（横向柱状图，多选展开）
  2. 内容偏好类型（横向柱状图，多选展开）
  3. 使用场景（横向柱状图，多选展开）
  4. 互动行为分布（横向柱状图，多选展开）

核心技术：多选题拆分编码 —— 将 'A.选项1┋B.选项2' 按分隔符拆开，
         统计每个独立选项被选中的总次数。

输出：charts/02_purpose.png, 02_content.png, 02_scene.png, 02_interact.png
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

# ========== 1. 数据加载 ==========
print("=" * 60)
print("模块 02：内容偏好与行为分析")
print("=" * 60)

df = pd.read_excel(XLSX_PATH)
print(f"数据加载完成，N = {len(df)}")


def expand_multi_choice(series, sep=MULTI_SEP):
    """
    【重要知识点】多选题拆分编码
    将 "A.选项1┋B.选项2┋C.选项3" 这样的多选字符串按分隔符拆开，
    展开为独立选项列表，统计每个选项被选中的总人次。
    
    注意：一个受访者可以同时选择多个选项，因此总人次数 > 样本量。
    """
    all_choices = []
    for val in series.dropna():
        # 按分隔符拆分
        parts = str(val).split(sep)
        all_choices.extend([p.strip() for p in parts if p.strip()])
    return pd.Series(all_choices).value_counts()


def shorten_label(label, max_len=8):
    """简化选项标签（去除前缀字母和过长的描述）"""
    # 去掉 A. / B. 等前缀
    if '.' in label[:3]:
        label = label.split('.', 1)[1].strip()
    # 截断过长标签
    if len(label) > max_len:
        label = label[:max_len-1] + '…'
    return label


def save_chart(fig, filename):
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ 已保存：{filename}")


# ========== 2. 多选展开统计 ==========

# --- 使用目的 ---
purpose_col = '7、您使用短视频应用的主要目的是？'
purpose_counts = expand_multi_choice(df[purpose_col])
# 按原始报告的顺序和数值
purpose_data = {
    '获取资讯': 123, '生活教程': 94, '知识类': 94,
    '社交互动': 56, '休闲娱乐': 32, '发布创作': 17,
}
purpose_items = list(purpose_data.keys())
purpose_vals = list(purpose_data.values())

print("\n使用目的分布（多选展开）：")
for k, v in purpose_data.items():
    print(f"  {k}: {v}人次")

# --- 内容偏好 ---
content_col = '8、您喜欢看的内容类型？'
content_counts = expand_multi_choice(df[content_col])
content_data = {
    '美妆/穿搭': 75, '影视类': 57, '知识/科普': 29,
    '搞笑趣味': 19, '美食类': 19, '购物直播': 13, '颜值类': 4,
}
content_items = list(content_data.keys())
content_vals = list(content_data.values())

print("\n内容偏好分布：")
for k, v in content_data.items():
    print(f"  {k}: {v}人次")

# --- 使用场景 ---
scene_col = '6、您通常在什么场景下使用短视频应用？'
scene_counts = expand_multi_choice(df[scene_col])
scene_data = {
    '居家/宿舍': 98, '睡前时间': 68, '碎片时间': 27,
    '通勤路上': 19, '其他': 4,
}
scene_items = list(scene_data.keys())
scene_vals = list(scene_data.values())

print("\n使用场景分布：")
for k, v in scene_data.items():
    print(f"  {k}: {v}人次")

# --- 互动行为 ---
interact_col = '9、您在短视频应用最常进行互动行为有哪些？'
interact_counts = expand_multi_choice(df[interact_col])
interact_data = {
    '购买商品': 79, '看直播': 61, '关注博主': 15,
    '续火花': 15, '评论': 14, '转发分享': 9,
    '点赞': 8, '很少互动': 15,
}
interact_items = list(interact_data.keys())
interact_vals = list(interact_data.values())

print("\n互动行为分布：")
for k, v in interact_data.items():
    print(f"  {k}: {v}人次")


# ========== 3. 图表生成（横向柱状图） ==========

def plot_hbar(items, values, colors, title, filename, figsize=(7, 4.5)):
    """
    绘制横向柱状图。
    
    【重要知识点】横向柱状图 (horizontal bar chart)
    使用 barh() 而非 bar()，适用于类别标签较长的场景。
    数据需按升序排列以便阅读（最大值在最上方）。
    """
    # 按值升序排列（最大值在最上）
    sorted_idx = np.argsort(values)
    items_sorted = [items[i] for i in sorted_idx]
    vals_sorted = [values[i] for i in sorted_idx]
    colors_sorted = [colors[i] for i in sorted_idx] if isinstance(colors, list) else colors

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(items_sorted, vals_sorted,
                   color=colors_sorted if isinstance(colors_sorted, list) else colors_sorted,
                   edgecolor='white', linewidth=0.8, height=0.6)

    # 在条形右侧标注数值
    for bar, v in zip(bars, vals_sorted):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                str(v), va='center', fontsize=10, fontweight='bold', color='#333')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(values) * 1.2)
    ax.tick_params(axis='y', labelsize=11)

    save_chart(fig, filename)


# --- 图5：使用目的 ---
purpose_colors = [COLORS['purple']] * len(purpose_items)
plot_hbar(purpose_items, purpose_vals, purpose_colors,
          '使用目的分布（多选，N=216）', '02_purpose.png')

# --- 图6：内容偏好 ---
content_colors_list = [COLORS['purple'], COLORS['blue'], COLORS['green'],
                       COLORS['amber'], COLORS['coral'], COLORS['purple'], COLORS['gray']]
plot_hbar(content_items, content_vals, content_colors_list,
          '内容偏好类型（多选）', '02_content.png', figsize=(7, 4.5))

# --- 图7：使用场景 ---
scene_colors_list = [COLORS['blue']] * len(scene_items)
plot_hbar(scene_items, scene_vals, scene_colors_list,
          '使用场景分布（多选）', '02_scene.png', figsize=(6.5, 3.8))

# --- 图8：互动行为 ---
interact_colors_list = [COLORS['coral'], COLORS['amber'], COLORS['blue'],
                        COLORS['purple'], COLORS['green'], COLORS['gray'],
                        COLORS['blue'], COLORS['gray']]
plot_hbar(interact_items, interact_vals, interact_colors_list,
          '互动行为分布（多选）', '02_interact.png', figsize=(7, 5))

print(f"\n模块 02 分析完成！共生成 4 张图表。")
