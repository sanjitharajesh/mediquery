from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent / "evaluation_reports"
OUTPUT_PATH = REPORTS_DIR / "eval_charts.png"

# (latency_s, coverage_pct, difficulty, passed, label)
QUERY_DATA = [
    (6.3,  75,  'easy',   True,  'side effects'),
    (2.7,  75,  'easy',   True,  'pregnancy safety'),
    (2.6,  50,  'easy',   True,  'indications'),
    (2.6,  100, 'medium', True,  'drug interactions'),
    (2.6,  50,  'medium', True,  'black box warning'),
    (2.3,  75,  'medium', True,  'dosage'),
    (11.0, 75,  'medium', True,  'contraindications'),
    (11.7, 75,  'medium', True,  'clinical monitoring'),
    (8.1,  50,  'medium', True,  'comparative'),
    (10.4, 75,  'medium', True,  'precautions'),
    (10.6, 100, 'medium', True,  'organ toxicity'),
    (13.0, 25,  'hard',   False, 'patient counseling'),
    (6.5,  60,  'hard',   True,  'discontinuation protocol'),
    (10.8, 40,  'hard',   False, 'mechanism interaction'),
    (10.2, 20,  'hard',   False, 'specific lab values'),
]

DIFF_COLORS = {'easy': '#10B981', 'medium': '#2563EB', 'hard': '#EF4444'}


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'figure.facecolor': '#FAFAFA',
        'axes.facecolor': '#FAFAFA',
    })

    fig = plt.figure(figsize=(18, 7), facecolor='#FAFAFA')
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 3, 1.2], figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ------------------------------------------------------------------ #
    # PANEL 1 — Scatter: Retrieval Complexity Envelope
    # ------------------------------------------------------------------ #
    latencies = [d[0] for d in QUERY_DATA]
    coverages = [d[1] for d in QUERY_DATA]
    difficulties = [d[2] for d in QUERY_DATA]
    passed = [d[3] for d in QUERY_DATA]
    labels = [d[4] for d in QUERY_DATA]
    colors = [DIFF_COLORS[diff] for diff in difficulties]

    for i, (x, y, passed_i, color) in enumerate(zip(latencies, coverages, passed, colors)):
        if passed_i:
            ax1.scatter(x, y, s=120, color=color, zorder=3)
        else:
            ax1.scatter(x, y, s=120, facecolors='none', edgecolors=color,
                        linewidths=2, zorder=3)

    # Trend line
    z = np.polyfit(latencies, coverages, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(latencies) - 0.5, max(latencies) + 0.5, 100)
    ax1.plot(x_line, p(x_line), '--', color='#9CA3AF', linewidth=1.2, zorder=1)

    # Annotations with simple offset to reduce overlap
    offsets = [
        (5, 5), (-5, 8), (5, -10), (5, 5), (-5, -12), (5, 5),
        (5, 5), (-5, -12), (5, 8), (-5, 8), (5, -10), (5, 5),
        (5, -10), (-5, 8), (-5, -12),
    ]
    for i, (x, y, lbl) in enumerate(zip(latencies, coverages, labels)):
        ox, oy = offsets[i]
        ax1.annotate(lbl, (x, y), xytext=(ox, oy), textcoords='offset points',
                     fontsize=7, color='#374151', ha='left')

    # Legend
    legend_handles = [
        mpatches.Patch(color=DIFF_COLORS['easy'],   label='Easy'),
        mpatches.Patch(color=DIFF_COLORS['medium'], label='Medium'),
        mpatches.Patch(color=DIFF_COLORS['hard'],   label='Hard'),
        plt.scatter([], [], s=80, color='#6B7280', label='Pass'),
        plt.scatter([], [], s=80, facecolors='none', edgecolors='#6B7280',
                    linewidths=2, label='Fail'),
    ]
    ax1.legend(handles=legend_handles, fontsize=7, frameon=False, loc='upper right')

    ax1.set_xlabel('Avg Latency (s)', fontsize=9)
    ax1.set_ylabel('Coverage (%)', fontsize=9)
    ax1.set_title('Retrieval Complexity Envelope', fontweight='bold', fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(color='#E5E7EB', linewidth=0.6, alpha=0.8)
    ax1.set_axisbelow(True)

    # ------------------------------------------------------------------ #
    # PANEL 2 — Heatmap: Failure Mode Analysis
    # ------------------------------------------------------------------ #
    cat_names = [d[4] for d in QUERY_DATA]
    pass_vals = [1 if d[3] else 0 for d in QUERY_DATA]
    cov_vals = [d[1] for d in QUERY_DATA]
    lat_vals = [d[0] for d in QUERY_DATA]

    df_actual = pd.DataFrame({
        'Pass': pass_vals,
        'Coverage (%)': cov_vals,
        'Latency (s)': lat_vals,
    }, index=cat_names)

    df_norm = df_actual.copy().astype(float)
    for col in df_norm.columns:
        col_min, col_max = df_norm[col].min(), df_norm[col].max()
        if col_max > col_min:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.5

    # Build annotation strings from actual values
    annot = pd.DataFrame(index=cat_names, columns=df_actual.columns)
    for idx in cat_names:
        annot.loc[idx, 'Pass'] = str(int(df_actual.loc[idx, 'Pass']))
        annot.loc[idx, 'Coverage (%)'] = f"{int(df_actual.loc[idx, 'Coverage (%)'])}%"
        annot.loc[idx, 'Latency (s)'] = f"{df_actual.loc[idx, 'Latency (s)']:.1f}s"

    sns.heatmap(
        df_norm,
        ax=ax2,
        cmap='RdYlGn',
        annot=annot,
        fmt='',
        linewidths=0.4,
        linecolor='#E5E7EB',
        cbar=False,
        annot_kws={'fontsize': 7},
    )
    ax2.set_title('Failure Mode Analysis', fontweight='bold', fontsize=10)
    ax2.tick_params(axis='x', labelsize=9, rotation=0)
    ax2.tick_params(axis='y', labelsize=8, rotation=0)
    ax2.set_ylabel('')

    # ------------------------------------------------------------------ #
    # PANEL 3 — Stat cards
    # ------------------------------------------------------------------ #
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    cards = [
        ('80%',  'Query Accuracy',    '12 of 15 queries'),
        ('100%', 'Factual Retrieval', 'Easy + Medium difficulty'),
        ('7.4s', 'Avg Latency',       'Under 5s for simple queries'),
    ]
    card_ys = [0.72, 0.40, 0.08]
    card_h = 0.24

    for (big, title, subtitle), y0 in zip(cards, card_ys):
        fancy = mpatches.FancyBboxPatch(
            (0.05, y0), 0.90, card_h,
            boxstyle='round,pad=0.02',
            facecolor='#2563EB',
            edgecolor='none',
            transform=ax3.transAxes,
            clip_on=False,
        )
        ax3.add_patch(fancy)
        cx = 0.50
        ax3.text(cx, y0 + card_h * 0.62, big,
                 transform=ax3.transAxes, ha='center', va='center',
                 fontsize=22, fontweight='bold', color='white')
        ax3.text(cx, y0 + card_h * 0.35, title,
                 transform=ax3.transAxes, ha='center', va='center',
                 fontsize=9, fontweight='bold', color='white')
        ax3.text(cx, y0 + card_h * 0.12, subtitle,
                 transform=ax3.transAxes, ha='center', va='center',
                 fontsize=7, color='#BFDBFE')

    # ------------------------------------------------------------------ #
    # Figure title + layout
    # ------------------------------------------------------------------ #
    fig.suptitle('MediQuery — RAG System Evaluation Report',
                 fontsize=15, fontweight='bold', y=1.01)

    fig.tight_layout(pad=1.5)
    fig.savefig(OUTPUT_PATH, dpi=200, facecolor='#FAFAFA', bbox_inches='tight')
    print(f'Saved chart to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
