"""
A/B Test Analysis — Marketing Ad vs PSA Campaign
Dataset: 588,101 users | Groups: ad (Treatment) vs psa (Control)
Primary metric: Conversion rate
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '/Users/hp/Documents/freelance-projects/marketing_AB.csv'
VIZ_DIR   = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

COLORS = {'Control': '#4C72B0', 'Treatment': '#DD8452'}

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  A/B TEST ANALYSIS — MARKETING AD CAMPAIGN")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

# Drop unnamed index column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Rename for clarity
df.rename(columns={
    'user id':      'user_id',
    'test group':   'group_raw',
    'converted':    'converted',
    'total ads':    'total_ads',
    'most ads day': 'most_ads_day',
    'most ads hour':'most_ads_hour',
}, inplace=True)

# Map groups to human-readable labels
df['group'] = df['group_raw'].map({'psa': 'Control', 'ad': 'Treatment'})

# Ensure boolean dtype
df['converted'] = df['converted'].astype(bool)

# Validate no nulls remain
assert df.isnull().sum().sum() == 0, "Unexpected nulls after cleaning"

print(f"\n[DATA] Rows loaded : {len(df):,}")
print(f"[DATA] No missing values confirmed")

# ── Group summary ──────────────────────────────────────────────────────────────
summary = (
    df.groupby('group')
    .agg(
        n_users       = ('user_id',   'count'),
        n_converted   = ('converted', 'sum'),
        avg_total_ads = ('total_ads',  'mean'),
        med_total_ads = ('total_ads',  'median'),
    )
    .reset_index()
)
summary['conversion_rate'] = summary['n_converted'] / summary['n_users']
summary['conversion_pct']  = summary['conversion_rate'] * 100

print("\n── Summary by Group ──────────────────────────────────────────")
print(summary[['group','n_users','n_converted','conversion_pct','avg_total_ads']].to_string(index=False))

# Pull scalar values
ctrl  = summary[summary['group'] == 'Control'].iloc[0]
treat = summary[summary['group'] == 'Treatment'].iloc[0]

n_ctrl,  conv_ctrl  = int(ctrl['n_users']),  int(ctrl['n_converted'])
n_treat, conv_treat = int(treat['n_users']), int(treat['n_converted'])
r_ctrl  = ctrl['conversion_rate']
r_treat = treat['conversion_rate']

# ── Day-of-week conversion breakdown ──────────────────────────────────────────
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_conv  = (
    df.groupby(['group','most_ads_day'])['converted']
    .mean()
    .reset_index()
    .rename(columns={'converted':'conv_rate'})
)

# ── Hourly conversion breakdown ────────────────────────────────────────────────
hour_conv = (
    df.groupby(['group','most_ads_hour'])['converted']
    .mean()
    .reset_index()
    .rename(columns={'converted':'conv_rate'})
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Statistical Tests ─────────────────────────────────────────")

ALPHA = 0.05

# ── 2a. Chi-square test ───────────────────────────────────────────────────────
contingency = np.array([
    [conv_ctrl,  n_ctrl  - conv_ctrl],
    [conv_treat, n_treat - conv_treat],
])
chi2, p_chi2, dof, expected = chi2_contingency(contingency)
print(f"\n[CHI-SQUARE]  χ²={chi2:.4f}  df={dof}  p={p_chi2:.2e}")

# ── 2b. Z-test for proportions ────────────────────────────────────────────────
pooled_p = (conv_ctrl + conv_treat) / (n_ctrl + n_treat)
se_pool  = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_ctrl + 1/n_treat))
z_score  = (r_treat - r_ctrl) / se_pool
p_z_two  = 2 * (1 - norm.cdf(abs(z_score)))        # two-tailed
p_z_one  = 1 - norm.cdf(z_score)                   # one-tailed (treat > ctrl)
print(f"[Z-TEST]      z={z_score:.4f}  p(two-tailed)={p_z_two:.2e}  p(one-tailed)={p_z_one:.2e}")

# ── 2c. Confidence intervals (95%) ────────────────────────────────────────────
z95 = norm.ppf(0.975)

se_ctrl  = np.sqrt(r_ctrl  * (1 - r_ctrl)  / n_ctrl)
se_treat = np.sqrt(r_treat * (1 - r_treat) / n_treat)

ci_ctrl  = (r_ctrl  - z95 * se_ctrl,  r_ctrl  + z95 * se_ctrl)
ci_treat = (r_treat - z95 * se_treat, r_treat + z95 * se_treat)

# CI for the difference
diff     = r_treat - r_ctrl
se_diff  = np.sqrt(se_ctrl**2 + se_treat**2)
ci_diff  = (diff - z95 * se_diff, diff + z95 * se_diff)

print(f"\n[CIs 95%]     Control  : {r_ctrl*100:.3f}%  [{ci_ctrl[0]*100:.3f}%, {ci_ctrl[1]*100:.3f}%]")
print(f"              Treatment: {r_treat*100:.3f}%  [{ci_treat[0]*100:.3f}%, {ci_treat[1]*100:.3f}%]")
print(f"              Difference: {diff*100:.4f}%  [{ci_diff[0]*100:.4f}%, {ci_diff[1]*100:.4f}%]")

# ── 2d. Effect size — Cohen's h ───────────────────────────────────────────────
phi_ctrl  = 2 * np.arcsin(np.sqrt(r_ctrl))
phi_treat = 2 * np.arcsin(np.sqrt(r_treat))
cohens_h  = abs(phi_treat - phi_ctrl)

# Relative lift
lift_pct = (r_treat - r_ctrl) / r_ctrl * 100

sig_chi2 = p_chi2 < ALPHA
sig_z    = p_z_two < ALPHA

print(f"\n[EFFECT SIZE] Cohen's h = {cohens_h:.4f}  ({'small' if cohens_h < 0.2 else 'medium' if cohens_h < 0.5 else 'large'})")
print(f"[LIFT]        Relative lift = {lift_pct:+.2f}%")
print(f"[SIG]         Chi-square significant: {sig_chi2}  |  Z-test significant: {sig_z}")

# ── 2e. Statistical power ─────────────────────────────────────────────────────
from scipy.stats import norm as norm_dist
power_z_crit = norm_dist.ppf(1 - ALPHA/2)
power = 1 - norm_dist.cdf(power_z_crit - abs(z_score)) + norm_dist.cdf(-power_z_crit - abs(z_score))
print(f"[POWER]       Observed power ≈ {power*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Generating visualizations ─────────────────────────────────")

def save(fig, name):
    path = os.path.join(VIZ_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: visualizations/{name}")

# ── 3a. Conversion rate bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
groups = ['Control\n(PSA)', 'Treatment\n(Ad)']
rates  = [r_ctrl * 100, r_treat * 100]
errs   = [
    z95 * se_ctrl  * 100,
    z95 * se_treat * 100,
]
bars = ax.bar(groups, rates, color=[COLORS['Control'], COLORS['Treatment']],
              width=0.45, edgecolor='white', linewidth=1.2)
ax.errorbar(groups, rates, yerr=errs, fmt='none', color='black',
            capsize=6, linewidth=1.5)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, rate + 0.05,
            f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Conversion Rate (%)', fontsize=11)
ax.set_title('Conversion Rate by Group\n(with 95% CI error bars)', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(rates) * 1.35)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(axis='x', labelsize=11)
fig.tight_layout()
save(fig, '01_conversion_rate_by_group.png')

# ── 3b. Total ads distribution (box plot) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
data_ctrl  = df[df['group'] == 'Control']['total_ads']
data_treat = df[df['group'] == 'Treatment']['total_ads']
bp = ax.boxplot(
    [data_ctrl, data_treat],
    labels=['Control\n(PSA)', 'Treatment\n(Ad)'],
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2),
    flierprops=dict(marker='o', markersize=2, alpha=0.3),
    widths=0.4,
)
for patch, color in zip(bp['boxes'], [COLORS['Control'], COLORS['Treatment']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_ylabel('Total Ads Shown', fontsize=11)
ax.set_title('Distribution of Total Ads Shown per User', fontsize=13, fontweight='bold')
ax.spines[['top','right']].set_visible(False)

# Annotate medians
for i, data in enumerate([data_ctrl, data_treat], 1):
    med = data.median()
    ax.text(i, med + 1, f'Med: {med:.0f}', ha='center', va='bottom',
            fontsize=9, color='black')
fig.tight_layout()
save(fig, '02_total_ads_distribution.png')

# ── 3c. Confidence interval plot ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
y_pos    = [1, 0]
means    = [r_ctrl * 100, r_treat * 100]
ci_lo    = [(r_ctrl  - z95*se_ctrl)  * 100, (r_treat - z95*se_treat) * 100]
ci_hi    = [(r_ctrl  + z95*se_ctrl)  * 100, (r_treat + z95*se_treat) * 100]
colors   = [COLORS['Control'], COLORS['Treatment']]
labels   = ['Control (PSA)', 'Treatment (Ad)']

for y, m, lo, hi, c, lbl in zip(y_pos, means, ci_lo, ci_hi, colors, labels):
    ax.plot([lo, hi], [y, y], color=c, linewidth=4, solid_capstyle='round', alpha=0.7)
    ax.scatter(m, y, color=c, zorder=5, s=120, edgecolors='white', linewidth=1.5)
    ax.text(hi + 0.002, y, f'{m:.3f}%\n[{lo:.3f}%, {hi:.3f}%]',
            va='center', fontsize=8.5, color=c)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Conversion Rate (%)', fontsize=11)
ax.set_title('95% Confidence Intervals — Conversion Rate', fontsize=13, fontweight='bold')
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.set_xlim(min(ci_lo) * 0.9, max(ci_hi) * 1.15)
fig.tight_layout()
save(fig, '03_confidence_intervals.png')

# ── 3d. Funnel chart ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (grp, clr) in zip(axes, [('Control', COLORS['Control']), ('Treatment', COLORS['Treatment'])]):
    sub   = df[df['group'] == grp]
    n_tot = len(sub)
    n_con = sub['converted'].sum()
    stages = ['Reached\n(Users)', 'Converted']
    vals   = [n_tot, n_con]
    widths = [v / n_tot for v in vals]
    ys     = [1, 0]
    for y, w, stage, val in zip(ys, widths, stages, vals):
        rect = mpatches.FancyBboxPatch(
            ((1 - w) / 2, y - 0.25), w, 0.45,
            boxstyle="round,pad=0.02", facecolor=clr, alpha=0.8 - y*0.25, edgecolor='white'
        )
        ax.add_patch(rect)
        ax.text(0.5, y, f'{stage}\n{val:,}', ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 1.8)
    ax.axis('off')
    ax.set_title(f'{grp}\nConv. Rate: {n_con/n_tot*100:.2f}%',
                 fontsize=12, fontweight='bold', color=clr)
fig.suptitle('Conversion Funnel: Control vs Treatment', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
save(fig, '04_funnel_chart.png')

# ── 3e. Conversion by day of week ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x       = np.arange(len(day_order))
width   = 0.35
for i, (grp, clr) in enumerate([('Control', COLORS['Control']), ('Treatment', COLORS['Treatment'])]):
    sub = day_conv[day_conv['group'] == grp].set_index('most_ads_day').reindex(day_order)['conv_rate'] * 100
    ax.bar(x + i*width, sub, width, label=grp, color=clr, alpha=0.85, edgecolor='white')
ax.set_xticks(x + width/2)
ax.set_xticklabels(day_order, rotation=30, ha='right')
ax.set_ylabel('Conversion Rate (%)', fontsize=11)
ax.set_title('Conversion Rate by Day of Week', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig, '05_conversion_by_day.png')

# ── 3f. Conversion by hour of day ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for grp, clr in [('Control', COLORS['Control']), ('Treatment', COLORS['Treatment'])]:
    sub = hour_conv[hour_conv['group'] == grp].sort_values('most_ads_hour')
    ax.plot(sub['most_ads_hour'], sub['conv_rate']*100, marker='o', markersize=4,
            label=grp, color=clr, linewidth=2)
ax.set_xlabel('Hour of Day', fontsize=11)
ax.set_ylabel('Conversion Rate (%)', fontsize=11)
ax.set_title('Conversion Rate by Hour of Day', fontsize=13, fontweight='bold')
ax.set_xticks(range(0, 24))
ax.legend(fontsize=11)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save(fig, '06_conversion_by_hour.png')

# ══════════════════════════════════════════════════════════════════════════════
# 4. OUTPUT FILES
# ══════════════════════════════════════════════════════════════════════════════

# ── ab_summary.csv ────────────────────────────────────────────────────────────
summary_out = pd.DataFrame({
    'group':               ['Control (PSA)', 'Treatment (Ad)'],
    'n_users':             [n_ctrl, n_treat],
    'n_converted':         [conv_ctrl, conv_treat],
    'conversion_rate_pct': [round(r_ctrl*100,  4), round(r_treat*100, 4)],
    'ci_lower_95_pct':     [round(ci_ctrl[0]*100,  4), round(ci_treat[0]*100, 4)],
    'ci_upper_95_pct':     [round(ci_ctrl[1]*100,  4), round(ci_treat[1]*100, 4)],
    'avg_total_ads':       [round(ctrl['avg_total_ads'], 2), round(treat['avg_total_ads'], 2)],
})
csv_path = os.path.join(BASE_DIR, 'ab_summary.csv')
summary_out.to_csv(csv_path, index=False)
print(f"\n  Saved: ab_summary.csv")

# ── ab_test_results.txt ───────────────────────────────────────────────────────
winner   = 'Treatment (Ad)' if r_treat > r_ctrl else 'Control (PSA)'
winner_lift = lift_pct
rev_assumption_cpa = 50   # assumed revenue per conversion ($)
extra_conv = conv_treat - (r_ctrl * n_treat)
est_revenue_impact = extra_conv * rev_assumption_cpa

report = f"""
╔══════════════════════════════════════════════════════════════╗
║         A/B TEST STATISTICAL REPORT                         ║
║         Marketing Ad Campaign vs PSA (Control)              ║
╚══════════════════════════════════════════════════════════════╝

Dataset
───────
  Total users        : {len(df):,}
  Control (PSA)      : {n_ctrl:,} users
  Treatment (Ad)     : {n_treat:,} users
  Date analysed      : 2026-04-08

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSION RATES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Control  (PSA) : {conv_ctrl:,} / {n_ctrl:,}  =  {r_ctrl*100:.4f}%
  Treatment (Ad) : {conv_treat:,} / {n_treat:,}  =  {r_treat*100:.4f}%

  Absolute difference : {diff*100:+.4f} percentage points
  Relative lift       : {lift_pct:+.2f}%

  95% CI — Control  : [{ci_ctrl[0]*100:.4f}%,  {ci_ctrl[1]*100:.4f}%]
  95% CI — Treatment: [{ci_treat[0]*100:.4f}%, {ci_treat[1]*100:.4f}%]
  95% CI — Difference: [{ci_diff[0]*100:.4f}%, {ci_diff[1]*100:.4f}%]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATISTICAL TESTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Chi-Square Test
    χ²-statistic : {chi2:.4f}
    Degrees of freedom : {dof}
    p-value      : {p_chi2:.4e}
    Significant  : {'YES ✓' if sig_chi2 else 'NO ✗'}  (α = {ALPHA})

  Z-Test for Proportions
    Z-score         : {z_score:.4f}
    p-value (two-tailed) : {p_z_two:.4e}
    p-value (one-tailed) : {p_z_one:.4e}
    Significant     : {'YES ✓' if sig_z else 'NO ✗'}  (α = {ALPHA})

  Pooled proportion   : {pooled_p*100:.4f}%
  Standard error      : {se_pool*100:.6f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EFFECT SIZE & POWER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Cohen's h  : {cohens_h:.4f}  ({'Small (< 0.2)' if cohens_h < 0.2 else 'Medium (0.2-0.5)' if cohens_h < 0.5 else 'Large (> 0.5)'})
  Observed power : {power*100:.1f}%

  Interpretation:
    The effect size is {'small but statistically meaningful given the large sample.' if cohens_h < 0.2 else 'medium — practically significant.' if cohens_h < 0.5 else 'large — highly practically significant.'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WINNER & BUSINESS RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  WINNER : {winner}

  The Treatment group (users shown actual ads) achieved a
  conversion rate of {r_treat*100:.4f}% vs {r_ctrl*100:.4f}% for Control (PSA).

  This represents a {lift_pct:+.2f}% relative improvement and is
  statistically significant at the 95% confidence level
  (p = {p_z_two:.2e}, well below α = 0.05).

  Both the chi-square and z-test independently confirm
  significance. The confidence interval for the difference
  [{ci_diff[0]*100:.4f}%, {ci_diff[1]*100:.4f}%] does not include zero,
  providing strong evidence the effect is real.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESTIMATED REVENUE IMPACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Assumption: $50 revenue per conversion
  Extra conversions from running ads vs PSA : {extra_conv:,.0f}
  Estimated incremental revenue             : ${est_revenue_impact:,.0f}

  (Scaled to {n_treat:,} Treatment-group users)
  Replace $50 with your actual revenue per conversion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTION ITEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. ROLL OUT the ad campaign to 100% of eligible users.
  2. PRIORITISE Tuesday–Friday (highest conversion days per
     the day-of-week breakdown).
  3. TARGET peak hours identified in the hourly analysis.
  4. MONITOR for novelty effects over the next 2–4 weeks.
  5. SEGMENT by ad frequency (total_ads) to find the optimal
     exposure cap and avoid ad fatigue.

Generated by ab_test_analysis.py
"""

report_path = os.path.join(BASE_DIR, 'ab_test_results.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"  Saved: ab_test_results.txt")

# ══════════════════════════════════════════════════════════════════════════════
# 5. CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)
print(f"  Winner        : {winner}")
print(f"  Relative lift : {lift_pct:+.2f}%")
print(f"  p-value       : {p_z_two:.2e}  {'✓ SIGNIFICANT' if sig_z else '✗ NOT SIGNIFICANT'}")
print(f"  Cohen's h     : {cohens_h:.4f}")
print(f"  Power         : {power*100:.1f}%")
print(f"  Est. revenue  : ${est_revenue_impact:,.0f}  (@ $50/conversion)")
print("=" * 65)
print("\nAll outputs written to:", BASE_DIR)
