"""
Research-grade visualization module for graph coloring experiments.
Generates publication-quality plots for academic papers.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Publication styling
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

sns.set_palette("husl")
sns.set_style("whitegrid")


def load_data(csv_file):
    """Load raw experiment data."""
    path = os.path.join(RESULTS_DIR, csv_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def plot_colors_by_family(df, output_prefix):
    """Bar plot: average colors used per algorithm by puzzle family."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    summary = df.groupby(['family', 'algorithm'])['colors_used'].mean().reset_index()
    
    sns.barplot(
        data=summary,
        x='family',
        y='colors_used',
        hue='algorithm',
        ax=ax,
        palette='Set2',
        ci=None
    )
    
    ax.set_xlabel('Puzzle Family', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Colors Used', fontsize=12, fontweight='bold')
    ax.set_title('Graph Coloring Quality: Colors Used by Algorithm and Puzzle Family', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(title='Algorithm', title_fontsize=11, fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.4)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, f"{output_prefix}_01_colors_by_family.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def plot_runtime_by_family(df, output_prefix):
    """Bar plot: average runtime per algorithm by puzzle family."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    summary = df.groupby(['family', 'algorithm'])['execution_time'].mean().reset_index()
    # Convert to milliseconds for readability
    summary['execution_time'] = summary['execution_time'] * 1000
    
    sns.barplot(
        data=summary,
        x='family',
        y='execution_time',
        hue='algorithm',
        ax=ax,
        palette='Set2',
        ci=None
    )
    
    ax.set_xlabel('Puzzle Family', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Speed: Execution Time by Puzzle Family', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(title='Algorithm', title_fontsize=11, fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.4)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, f"{output_prefix}_02_runtime_by_family.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def plot_efficiency_scatter(df, output_prefix):
    """Scatter plot: runtime vs colors (efficiency frontier)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors_map = {'greedy': '#FF6B6B', 'dsatur': '#4ECDC4', 'welsh_powell': '#45B7D1'}
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        # Convert time to ms
        x = algo_data['execution_time'] * 1000
        y = algo_data['colors_used']
        ax.scatter(x, y, label=algo, s=80, alpha=0.6, 
                  color=colors_map.get(algo, 'gray'), edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Colors Used', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency Frontier: Quality vs Speed Trade-off', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(title='Algorithm', fontsize=11, title_fontsize=11, loc='best')
    ax.grid(True, alpha=0.4)
    ax.set_xscale('log')  # Log scale for time (spans orders of magnitude)
    
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, f"{output_prefix}_03_efficiency_scatter.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def plot_heatmap_colors(df, output_prefix):
    """Heatmap: colors used across families and algorithms."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    pivot_data = df.pivot_table(
        values='colors_used',
        index='family',
        columns='algorithm',
        aggfunc='mean'
    )
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Colors Used'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Puzzle Family', fontsize=12, fontweight='bold')
    ax.set_title('Color Usage Heatmap: Algorithm Performance Across Families', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, f"{output_prefix}_04_heatmap_colors.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def plot_heatmap_runtime(df, output_prefix):
    """Heatmap: execution time across families and algorithms."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert to milliseconds
    df_ms = df.copy()
    df_ms['execution_time'] = df_ms['execution_time'] * 1000
    
    pivot_data = df_ms.pivot_table(
        values='execution_time',
        index='family',
        columns='algorithm',
        aggfunc='mean'
    )
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Time (ms)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Puzzle Family', fontsize=12, fontweight='bold')
    ax.set_title('Runtime Heatmap: Execution Time Across Families (ms)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, f"{output_prefix}_05_heatmap_runtime.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def plot_box_colors(df, output_prefix):
    """Box plot: distribution of colors used per algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(
        data=df,
        x='algorithm',
        y='colors_used',
        palette='Set2',
        ax=ax
    )
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Colors Used', fontsize=12, fontweight='bold')
    ax.set_title('Color Usage Distribution: Statistical Comparison', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.4)
    
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, f"{output_prefix}_06_box_colors.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def plot_box_runtime(df, output_prefix):
    """Box plot: distribution of runtimes per algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_ms = df.copy()
    df_ms['execution_time'] = df_ms['execution_time'] * 1000
    
    sns.boxplot(
        data=df_ms,
        x='algorithm',
        y='execution_time',
        palette='Set2',
        ax=ax
    )
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime Distribution: Statistical Comparison', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_yscale('log')  # Log scale due to wide variance
    ax.grid(axis='y', alpha=0.4)
    
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, f"{output_prefix}_07_box_runtime.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def plot_level_performance(df, output_prefix):
    """Line plot: algorithm performance across difficulty levels by family."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    families = df['family'].unique()
    
    for idx, family in enumerate(sorted(families)):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        family_data = df[df['family'] == family]
        
        # Group by level and algorithm
        summary = family_data.groupby(['level', 'algorithm'])['colors_used'].mean().reset_index()
        
        for algo in summary['algorithm'].unique():
            algo_data = summary[summary['algorithm'] == algo]
            ax.plot(algo_data['level'], algo_data['colors_used'], 
                   marker='o', label=algo, linewidth=2, markersize=6)
        
        ax.set_title(f'{family.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Difficulty Level', fontsize=10)
        ax.set_ylabel('Colors Used', fontsize=10)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(len(families), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Algorithm Performance Across Difficulty Levels by Puzzle Family', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, f"{output_prefix}_08_level_performance.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def generate_summary_table(df, output_prefix):
    """Generate a summary statistics table."""
    summary = df.groupby('algorithm').agg({
        'colors_used': ['mean', 'std', 'min', 'max'],
        'execution_time': ['mean', 'std', 'min', 'max'],
        'valid': 'mean'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    output_path = os.path.join(RESULTS_DIR, f"{output_prefix}_summary_stats.csv")
    summary.to_csv(output_path)
    print(f"✓ Saved: {output_path}")
    
    return summary


def main(csv_file='research_20251228_192858_raw.csv'):
    """Generate all research-grade plots."""
    print(f"\n📊 Generating research-quality plots from: {csv_file}\n")
    
    # Extract prefix from csv filename
    prefix = csv_file.replace('_raw.csv', '').replace('.csv', '')
    
    # Load data
    df = load_data(csv_file)
    
    print(f"Data loaded: {len(df)} records across {df['family'].nunique()} families\n")
    
    # Generate plots
    print("Generating plots...")
    plot_colors_by_family(df, prefix)
    plot_runtime_by_family(df, prefix)
    plot_efficiency_scatter(df, prefix)
    plot_heatmap_colors(df, prefix)
    plot_heatmap_runtime(df, prefix)
    plot_box_colors(df, prefix)
    plot_box_runtime(df, prefix)
    plot_level_performance(df, prefix)
    
    print("\nGenerating summary statistics...")
    stats = generate_summary_table(df, prefix)
    print("\nSummary Statistics:")
    print(stats)
    
    print(f"\n✅ All plots saved to: {RESULTS_DIR}/")
    print("\nGenerated files:")
    print("  01_colors_by_family.png      — Bar chart: quality by family")
    print("  02_runtime_by_family.png     — Bar chart: speed by family")
    print("  03_efficiency_scatter.png    — Scatter: quality vs speed trade-off")
    print("  04_heatmap_colors.png        — Heatmap: color usage matrix")
    print("  05_heatmap_runtime.png       — Heatmap: runtime matrix")
    print("  06_box_colors.png            — Box plot: color distribution")
    print("  07_box_runtime.png           — Box plot: runtime distribution")
    print("  08_level_performance.png     — Line plot: performance by difficulty")
    print("  summary_stats.csv            — Statistical summary table\n")


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'research_20251228_192858_raw.csv'
    main(csv_file)
