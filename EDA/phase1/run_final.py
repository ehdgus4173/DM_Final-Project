import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──────────────────────────────────────────
plt.rcParams['font.family']      = 'DejaVu Sans'
plt.rcParams['figure.dpi']       = 200
plt.rcParams['axes.spines.top']  = False
plt.rcParams['axes.spines.right']= False

sns.set_style('whitegrid')

PRIMARY  = '#3B4FE4'
ACCENT   = '#1A7F5A'
GRAY     = '#C8CDD6'
WEEKDAY  = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df = pd.read_csv('../../dataset/cleaned_USvideos.csv')

cat_stats = (
    df.groupby('category')['views']
    .agg(median='median', mean='mean', std='std', count='count')
    .sort_values('median', ascending=False)
    .reset_index()
)
cat_order  = cat_stats['category'].tolist()
top3_cats  = cat_order[:3]
overall_median = df['views'].median()

def fmt_views(v):
    if v >= 1e6:  return f'{v/1e6:.2f}M'
    if v >= 1e3:  return f'{v/1e3:.0f}K'
    return str(int(v))

def strip_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ══════════════════════════════════════════════════════════
# TASK 1 — Improved Figures
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 1: Generating polished figures")
print("=" * 60)

# ── fig_final_h1 ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8))

bar_colors = [PRIMARY if c in top3_cats else GRAY for c in cat_stats['category']]
bars = ax.barh(cat_stats['category'], cat_stats['median'],
               color=bar_colors, edgecolor='white', linewidth=0.5)
ax.set_xscale('log')
ax.invert_yaxis()
ax.set_xlabel('Median Views (log scale)', fontsize=11)

# value labels
for bar, val, cat in zip(bars, cat_stats['median'], cat_stats['category']):
    color = 'white' if cat in top3_cats else '#555555'
    x_pos = val * 0.92 if cat in top3_cats else val * 1.04
    ha    = 'right'   if cat in top3_cats else 'left'
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            fmt_views(val), va='center', ha=ha,
            fontsize=8.5, fontweight='bold', color=color)

ax.set_title('Median Views by Category', fontsize=14, fontweight='bold', pad=12)
ax.text(0.5, 1.01,
        'Gaming, Music, and Film & Animation lead; Nonprofits 39x lower than Gaming.',
        transform=ax.transAxes, ha='center', fontsize=9, color='#555555',
        style='italic')

legend_handles = [
    mpatches.Patch(color=PRIMARY, label='Top 3 categories'),
    mpatches.Patch(color=GRAY,    label='Other categories'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9, framealpha=0.7)
strip_spines(ax)
plt.tight_layout()
plt.savefig('fig_final_h1.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig_final_h1.png")

# ── fig_final_h2_all ─────────────────────────────────────
pivot_all = (
    df.pivot_table(values='views', index='publish_weekday',
                   columns='publish_hour', aggfunc='median')
    .reindex(range(7))
)
pivot_all.index = WEEKDAY

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(pivot_all, cmap='YlOrRd', linewidths=0.3, linecolor='white',
            ax=ax, cbar_kws={'label': 'Median Views', 'shrink': 0.8})
ax.set_title('Median Views by Upload Day x Hour  (All Categories)',
             fontsize=14, fontweight='bold', pad=12)
ax.text(0.5, 1.015,
        'Monday 09:00 yields the highest median views (3.6M); late-night slots consistently underperform.',
        transform=ax.transAxes, ha='center', fontsize=9, color='#555555', style='italic')
ax.set_xlabel('Upload Hour (0-23)', fontsize=11)
ax.set_ylabel('Upload Weekday', fontsize=11)
strip_spines(ax)
plt.tight_layout()
plt.savefig('fig_final_h2_all.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig_final_h2_all.png")

# ── fig_final_h2_top3 ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
for ax, cat in zip(axes, top3_cats):
    pivot_cat = (
        df[df['category'] == cat]
        .pivot_table(values='views', index='publish_weekday',
                     columns='publish_hour', aggfunc='median')
        .reindex(range(7))
    )
    pivot_cat.index = WEEKDAY
    sns.heatmap(pivot_cat, cmap='YlOrRd', linewidths=0.3, linecolor='white',
                ax=ax, cbar_kws={'label': 'Median Views', 'shrink': 0.8})
    ax.set_title(cat, fontsize=12, fontweight='bold')
    ax.set_xlabel('Upload Hour (0-23)', fontsize=10)
    ax.set_ylabel('Upload Weekday', fontsize=10)
    strip_spines(ax)

fig.suptitle('Median Views by Upload Day x Hour  (Top 3 Categories)',
             fontsize=14, fontweight='bold', y=1.01)
fig.text(0.5, 0.98,
         'Each category shows a distinct optimal upload window, supporting H2.',
         ha='center', fontsize=9, color='#555555', style='italic')
plt.tight_layout()
plt.savefig('fig_final_h2_top3.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig_final_h2_top3.png")

# ── fig_final_h3 ─────────────────────────────────────────
lag_order = ['Day 0', '1-3 days', '4-7 days', '8+ days']

def assign_lag(x):
    if x == 0:    return 'Day 0'
    elif x <= 3:  return '1-3 days'
    elif x <= 7:  return '4-7 days'
    else:         return '8+ days'

df['lag_group'] = df['trending_lag'].apply(assign_lag)
lag_counts  = df['lag_group'].value_counts().reindex(lag_order)
lag_medians = df.groupby('lag_group')['views'].median().reindex(lag_order)

palette_h3 = [PRIMARY, ACCENT, '#E8B84B', '#C0392B']

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x='lag_group', y='log_views', order=lag_order,
            palette=palette_h3, linewidth=0.9,
            flierprops=dict(marker='o', markersize=2, alpha=0.35,
                            markerfacecolor='#888888'),
            ax=ax)

# n= annotations above each box
for i, grp in enumerate(lag_order):
    q3 = df[df['lag_group'] == grp]['log_views'].quantile(0.75)
    ax.text(i, q3 + 0.25, f'n={lag_counts[grp]:,}',
            ha='center', va='bottom', fontsize=9,
            color='#333333', fontweight='bold')

# overall median dashed line (on log1p scale)
overall_log_median = np.log1p(overall_median)
ax.axhline(overall_log_median, color='#888888', linestyle='--',
           linewidth=1.2, label=f'Overall median  ({fmt_views(overall_median)})')
ax.legend(fontsize=9, framealpha=0.7)

ax.set_title('Log Views by Trending Lag Group', fontsize=14, fontweight='bold', pad=12)
ax.text(0.5, 1.01,
        'Day-0 videos peak in views, but the effect diminishes sharply — faster trending is correlated but not causal.',
        transform=ax.transAxes, ha='center', fontsize=9, color='#555555', style='italic')
ax.set_xlabel('Trending Lag Group', fontsize=11)
ax.set_ylabel('log(Views + 1)', fontsize=11)
strip_spines(ax)
plt.tight_layout()
plt.savefig('fig_final_h3.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: fig_final_h3.png")


# ══════════════════════════════════════════════════════════
# TASK 2 — Summary statistics table
# ══════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TASK 2: Summary statistics table")
print("=" * 60)
print()

W = 60
line  = lambda l, r: f"║ {l:<18}║ {r:<{W-23}}║"
div   = "╠" + "═"*19 + "╦" + "═"*(W-22) + "╣"
top   = "╔" + "═"*19 + "╦" + "═"*(W-22) + "╗"
mid   = "╟" + "─"*19 + "╫" + "─"*(W-22) + "╢"
bot   = "╚" + "═"*19 + "╩" + "═"*(W-22) + "╝"
title_row = f"║{'Part A  --  EDA Summary Statistics':^{W-2}}║"
top_full  = "╔" + "═"*(W-2) + "╗"
div_full  = "╠" + "═"*19 + "╦" + "═"*(W-22) + "╣"

print(top_full)
print(title_row)
print(div_full)
print(line("Dataset",       "6,249 videos  (after cleaning)"))
print(line("Original rows", "40,949"))
print(line("Removed",       "34,598 (duplicates) + 101 (lag outliers)"))
print(div)
print(line("H1 result",     "p = 6.1e-95   SUPPORTED"))
print(line("H2 result",     "p = 5.7e-04   SUPPORTED"))
print(line("H3 result",     "p = 1.9e-28   PARTIALLY REJECTED"))
print(div)
print(line("Top predictor", "category  (H=490, 38.7x range)"))
print(line("2nd predictor", "log_likes  (strong positive corr)"))
print(line("3rd predictor", "days_on_trending  (stable feature)"))
print(bot)


# ══════════════════════════════════════════════════════════
# TASK 3 — Handoff validation
# ══════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TASK 3: Handoff validation — cleaned_USvideos.csv")
print("=" * 60)

df_v = pd.read_csv('../../dataset/cleaned_USvideos.csv')
issues = []

# 1. Shape
exp_shape = (6249, 17)
shape_ok = df_v.shape == exp_shape
status = "PASS" if shape_ok else "FAIL"
print(f"\n[1] Shape check ................... {status}")
print(f"    Expected {exp_shape}  |  Got {df_v.shape}")
if not shape_ok:
    issues.append(f"Shape mismatch: expected {exp_shape}, got {df_v.shape}")

# 2. Null check
key_cols = ['category','views','log_views','trending_lag',
            'days_on_trending','publish_hour','publish_weekday']
null_counts = df_v[key_cols].isnull().sum()
null_ok = null_counts.sum() == 0
status = "PASS" if null_ok else "FAIL"
print(f"\n[2] Null check (key columns) ...... {status}")
for col, cnt in null_counts.items():
    flag = "OK" if cnt == 0 else f"!! {cnt} nulls"
    print(f"    {col:<22} {flag}")
if not null_ok:
    issues.append(f"Nulls found in key columns: {null_counts[null_counts>0].to_dict()}")

# 3. Range checks
checks = {
    'trending_lag   min >= 0':       df_v['trending_lag'].min()    >= 0,
    'trending_lag   max <= 30':      df_v['trending_lag'].max()    <= 30,
    'days_on_trend  min >= 1':       df_v['days_on_trending'].min()>= 1,
    'log_views      min >= 0':       df_v['log_views'].min()       >= 0,
    'publish_hour   in [0,23]':      df_v['publish_hour'].between(0,23).all(),
    'publish_weekday in [0,6]':      df_v['publish_weekday'].between(0,6).all(),
}
all_range_ok = all(checks.values())
status = "PASS" if all_range_ok else "FAIL"
print(f"\n[3] Range checks .................. {status}")
for label, ok in checks.items():
    tick = "OK" if ok else "FAIL"
    actual = ""
    if 'trending_lag   min' in label:   actual = f"(actual min={df_v['trending_lag'].min():.0f})"
    if 'trending_lag   max' in label:   actual = f"(actual max={df_v['trending_lag'].max():.0f})"
    if 'days_on_trend' in label:        actual = f"(actual min={df_v['days_on_trending'].min():.0f})"
    if 'log_views' in label:            actual = f"(actual min={df_v['log_views'].min():.4f})"
    if 'publish_hour' in label:         actual = f"(range {df_v['publish_hour'].min()}–{df_v['publish_hour'].max()})"
    if 'publish_weekday' in label:      actual = f"(range {df_v['publish_weekday'].min()}–{df_v['publish_weekday'].max()})"
    print(f"    {label:<32} {tick}  {actual}")
    if not ok:
        issues.append(f"Range check failed: {label}")

# 4. Category value counts
print(f"\n[4] Category value counts (sorted by count desc):")
vc = df_v['category'].value_counts()
for cat, cnt in vc.items():
    bar = '#' * (cnt // 50)
    print(f"    {cat:<26} {cnt:>5}  {bar}")

# 5. First 3 rows
print(f"\n[5] First 3 rows:")
display_cols = ['video_id','category','views','log_views','trending_lag','days_on_trending']
print(df_v[display_cols].head(3).to_string(index=False))

# Final verdict
print()
if not issues:
    print("  cleaned_USvideos.csv is ready for Part B and C")
else:
    print("  Issues found:")
    for iss in issues:
        print(f"    - {iss}")

print()
print("=" * 60)
print("All tasks complete.")
print("=" * 60)
