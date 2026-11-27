# -*- coding: utf-8 -*-
"""
模块 01 — 用户基础画像分析
============================
分析维度：
  1. 性别分布（饼图）
  2. 年龄分布（柱状图）
  3. 职业构成（饼图）
  4. 日均使用时长分布（柱状图）

输出：charts/01_gender.png, 01_age.png, 01_job.png, 01_duration.png
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import sys

# 添加父目录到 path 以导入 config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# ========== 中文字体设置 ==========
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(CHART_DIR, exist_ok=True)

# ========== 1. 数据加载与预处理 ==========
print("=" * 60)
print("模块 01：用户基础画像分析")
print("=" * 60)

df = pd.read_excel(XLSX_PATH)
print(f"数据加载完成，样本量 N = {len(df)}")

# ---- 性别 ----
gender_col = '1、您的性别？'
gender_counts = df[gender_col].value_counts()
male = gender_counts.get('B.男', 0)
female = gender_counts.get('A.女', 0)
print(f"\n性别分布：男 {male}人（{male/216*100:.1f}%），女 {female}人（{female/216*100:.1f}%）")

# ---- 年龄 ----
age_col = '2、您的年龄？'
age_map = {
    'A.18岁以下': '18岁以下',
    'B.18-22岁': '18-22岁',
    'C.23-30岁': '23-30岁',
    'D.31-40岁': '31-40岁',
    'E.41岁以上': '41岁以上',
}
age_counts = df[age_col].map(age_map).value_counts()
age_order = ['18岁以下', '18-22岁', '23-30岁', '31-40岁', '41岁以上']
age_vals = [age_counts.get(k, 0) for k in age_order]
print(f"年龄分布：{dict(zip(age_order, age_vals))}")

# ---- 职业 ----
job_col = '3、您的职业或身份？'
job_map = {
    'A.在校学生': '在校学生',
    'B.上班族': '上班族',
    'C.自由职业者': '自由职业者',
}
job_counts = df[job_col].map(job_map).value_counts()
job_labels = ['上班族', '自由职业者', '在校学生']
job_vals = [job_counts.get(k, 0) for k in job_labels]
print(f"职业构成：{dict(zip(job_labels, job_vals))}")

# ---- 使用时长 ----
dur_col = '5、您日均使用短视频应用的时长有？'
dur_map = {
    'A.半小时以内': '半小时以内',
    'B.1-2小时': '1-2小时',
    'C.2-3小时': '2-3小时',
    'D.3小时以上': '3小时以上',
}
dur_counts = df[dur_col].map(dur_map).value_counts()
dur_order = ['半小时以内', '1-2小时', '2-3小时', '3小时以上']
dur_vals = [dur_counts.get(k, 0) for k in dur_order]
print(f"日均使用时长：{dict(zip(dur_order, dur_vals))}")

# ========== 2. 图表生成 ==========

def save_chart(fig, filename):
    """保存图表到 charts/ 目录"""
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ 已保存：{filename}")

# --- 图1：性别分布饼图 ---
fig, ax = plt.subplots(figsize=(5, 5))
wedges, texts, autotexts = ax.pie(
    [male, female],
    labels=['男', '女'],
    colors=[COLORS['blue'], COLORS['purple']],
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.6,
    wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 12},
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
ax.set_title('性别分布（N=216）', fontsize=14, fontweight='bold', pad=15)
save_chart(fig, '01_gender.png')

# --- 图2：年龄分布柱状图 ---
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(age_order, age_vals, color=[COLORS['blue_light'], COLORS['blue'],
              COLORS['purple'], COLORS['blue'], COLORS['blue_light']],
              edgecolor='white', linewidth=1, width=0.6)
for bar, v in zip(bars, age_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{v}人\n({v/216*100:.1f}%)', ha='center', fontsize=10, color='#333')
ax.set_title('年龄分布', fontsize=14, fontweight='bold')
ax.set_ylabel('人数', fontsize=11)
ax.set_ylim(0, max(age_vals) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', rotation=15)
save_chart(fig, '01_age.png')

# --- 图3：职业构成饼图 ---
fig, ax = plt.subplots(figsize=(5, 5))
wedges, texts, autotexts = ax.pie(
    job_vals,
    labels=job_labels,
    colors=[COLORS['blue'], COLORS['purple'], COLORS['green']],
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.6,
    wedgeprops={'width': 0.4, 'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 12},
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
ax.set_title('职业构成（N=216）', fontsize=14, fontweight='bold', pad=15)
save_chart(fig, '01_job.png')

# --- 图4：使用时长分布柱状图 ---
fig, ax = plt.subplots(figsize=(6, 4.5))
bar_colors = [COLORS['green'], COLORS['blue'], COLORS['amber'], COLORS['coral']]
bars = ax.bar(dur_order, dur_vals, color=bar_colors, edgecolor='white', linewidth=1, width=0.6)
for bar, v in zip(bars, dur_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{v}人\n({v/216*100:.1f}%)', ha='center', fontsize=10, color='#333')
ax.set_title('日均使用时长分布', fontsize=14, fontweight='bold')
ax.set_ylabel('人数', fontsize=11)
ax.set_ylim(0, max(dur_vals) * 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
save_chart(fig, '01_duration.png')

print("\n模块 01 分析完成！共生成 4 张图表。")
print(f"图表输出目录：{CHART_DIR}")
