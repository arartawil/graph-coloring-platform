"""
Statistical analysis of graph coloring algorithm performance.
Includes ANOVA, pairwise comparisons, effect sizes, and significance tests.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"

# Publication styling
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
})

sns.set_style("whitegrid")


def load_data(csv_file):
    """Load raw experiment data."""
    path = os.path.join(RESULTS_DIR, csv_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def cohen_d(group1, group2):
    """Calculate Cohen's d effect size."""
    mean_diff = group1.mean() - group2.mean()
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    if pooled_std == 0:
        return 0
    return mean_diff / pooled_std


def hedge_g(group1, group2):
    """Calculate Hedge's g (unbiased Cohen's d)."""
    n1, n2 = len(group1), len(group2)
    d = cohen_d(group1, group2)
    correction = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
    return d * correction


def perform_anova(df, metric='colors_used'):
    """Perform one-way ANOVA across all algorithms."""
    print("\n" + "="*70)
    print("1. ONE-WAY ANOVA TEST")
    print("="*70)
    print(f"Hypothesis: Algorithm differences in {metric} are statistically significant")
    print()
    
    algorithms = df['algorithm'].unique()
    groups = [df[df['algorithm'] == algo][metric].values for algo in algorithms]
    
    f_stat, p_value = f_oneway(*groups)
    
    print(f"Number of algorithms: {len(algorithms)}")
    print(f"F-statistic: {f_stat:.6f}")
    print(f"P-value: {p_value:.2e}")
    
    if p_value < 0.001:
        print("✓ HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        print("✓ VERY SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        print("✓ SIGNIFICANT (p < 0.05)")
    else:
        print("✗ NOT SIGNIFICANT (p ≥ 0.05)")
    
    return f_stat, p_value


def perform_pairwise_comparisons(df, metric='colors_used'):
    """Perform pairwise t-tests and Mann-Whitney U tests."""
    print("\n" + "="*70)
    print("2. PAIRWISE COMPARISONS (T-TESTS & MANN-WHITNEY U)")
    print("="*70)
    print()
    
    algorithms = sorted(df['algorithm'].unique())
    results = []
    
    for i, algo1 in enumerate(algorithms):
        for algo2 in algorithms[i+1:]:
            group1 = df[df['algorithm'] == algo1][metric]
            group2 = df[df['algorithm'] == algo2][metric]
            
            # T-test
            t_stat, t_pval = ttest_ind(group1, group2)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pval = mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Effect size
            d = cohen_d(group1, group2)
            g = hedge_g(group1, group2)
            
            # Mean difference
            mean_diff = group1.mean() - group2.mean()
            
            results.append({
                'Comparison': f"{algo1} vs {algo2}",
                'Mean Diff': mean_diff,
                't-statistic': t_stat,
                't-pvalue': t_pval,
                'U-statistic': u_stat,
                'U-pvalue': u_pval,
                "Cohen's d": d,
                "Hedge's g": g,
            })
            
            # Print results
            sig_symbol = "✓" if t_pval < 0.05 else "✗"
            print(f"{sig_symbol} {algo1:20s} vs {algo2:20s}")
            print(f"   Mean Diff: {mean_diff:8.3f}  |  t={t_stat:7.3f}  |  p={t_pval:.4f}")
            print(f"   Cohen's d: {d:8.3f}  |  Hedge's g: {g:8.3f}")
            if t_pval < 0.001:
                print(f"   *** HIGHLY SIGNIFICANT ***")
            elif t_pval < 0.05:
                print(f"   ** SIGNIFICANT **")
            print()
    
    return pd.DataFrame(results)


def descriptive_statistics(df, metric='colors_used'):
    """Generate descriptive statistics by algorithm."""
    print("\n" + "="*70)
    print("3. DESCRIPTIVE STATISTICS BY ALGORITHM")
    print("="*70)
    print()
    
    stats_df = df.groupby('algorithm')[metric].agg([
        ('Count', 'count'),
        ('Mean', 'mean'),
        ('Std Dev', 'std'),
        ('Min', 'min'),
        ('Q1', lambda x: x.quantile(0.25)),
        ('Median', 'median'),
        ('Q3', lambda x: x.quantile(0.75)),
        ('Max', 'max'),
    ]).round(4)
    
    stats_df = stats_df.sort_values('Mean')
    print(stats_df.to_string())
    
    return stats_df


def confidence_intervals(df, metric='colors_used', confidence=0.95):
    """Calculate confidence intervals for each algorithm."""
    print("\n" + "="*70)
    print(f"4. {int(confidence*100)}% CONFIDENCE INTERVALS")
    print("="*70)
    print()
    
    algorithms = sorted(df['algorithm'].unique())
    ci_results = []
    
    alpha = 1 - confidence
    
    for algo in algorithms:
        data = df[df['algorithm'] == algo][metric]
        mean = data.mean()
        sem = data.sem()  # Standard error of mean
        margin = sem * stats.t.ppf(1 - alpha/2, len(data) - 1)
        ci_lower = mean - margin
        ci_upper = mean + margin
        
        ci_results.append({
            'Algorithm': algo,
            'Mean': mean,
            'Std Error': sem,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            'Margin': margin,
        })
        
        print(f"{algo:20s}: {mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    return pd.DataFrame(ci_results)


def normality_test(df, metric='colors_used'):
    """Test normality of distributions (Shapiro-Wilk)."""
    print("\n" + "="*70)
    print("5. NORMALITY TESTS (SHAPIRO-WILK)")
    print("="*70)
    print("H0: Data is normally distributed")
    print()
    
    algorithms = sorted(df['algorithm'].unique())
    
    for algo in algorithms:
        data = df[df['algorithm'] == algo][metric]
        if len(data) > 3:  # Shapiro-Wilk requires n >= 3
            stat, p = stats.shapiro(data)
            sig = "✗ NOT NORMAL" if p < 0.05 else "✓ NORMAL"
            print(f"{algo:20s}: W={stat:.4f}, p={p:.4f}  [{sig}]")


def homogeneity_of_variance_test(df, metric='colors_used'):
    """Test homogeneity of variance (Levene's test)."""
    print("\n" + "="*70)
    print("6. HOMOGENEITY OF VARIANCE TEST (LEVENE'S)")
    print("="*70)
    print("H0: All groups have equal variance")
    print()
    
    algorithms = df['algorithm'].unique()
    groups = [df[df['algorithm'] == algo][metric].values for algo in algorithms]
    
    stat, p = stats.levene(*groups)
    
    print(f"Levene's statistic: {stat:.6f}")
    print(f"P-value: {p:.4f}")
    
    if p < 0.05:
        print("✗ VARIANCES ARE NOT EQUAL (heteroscedastic)")
    else:
        print("✓ VARIANCES ARE EQUAL (homoscedastic)")


def plot_confidence_intervals(df, metric='colors_used', ci_df=None):
    """Visualize confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if ci_df is None:
        ci_df = confidence_intervals(df, metric)
    
    ci_df = ci_df.sort_values('Mean')
    
    ax.errorbar(
        range(len(ci_df)),
        ci_df['Mean'],
        yerr=ci_df['Margin'],
        fmt='o',
        markersize=8,
        capsize=5,
        capthick=2,
        linewidth=2,
        label='95% CI'
    )
    
    ax.set_xticks(range(len(ci_df)))
    ax.set_xticklabels(ci_df['Algorithm'], rotation=45, ha='right')
    ax.set_ylabel('Mean Colors Used', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Intervals for Algorithm Performance', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.4)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, "statistical_confidence_intervals.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close(fig)


def plot_distributions(df, metric='colors_used'):
    """Visualize distributions with violin plots."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Violin plot
    sns.violinplot(data=df, x='algorithm', y=metric, ax=axes[0], palette='Set2')
    axes[0].set_title(f'Distribution of {metric.replace("_", " ").title()}', 
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Algorithm', fontsize=11, fontweight='bold')
    axes[0].set_ylabel(metric.replace("_", " ").title(), fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Strip plot with error bars
    sns.stripplot(data=df, x='algorithm', y=metric, ax=axes[1], 
                  alpha=0.5, size=6, palette='Set2')
    
    # Add mean points
    means = df.groupby('algorithm')[metric].mean().sort_index()
    algo_order = sorted(df['algorithm'].unique())
    for i, algo in enumerate(algo_order):
        axes[1].plot(i, means[algo], marker='D', markersize=10, 
                    color='red', zorder=5, label='Mean' if i == 0 else '')
    
    axes[1].set_title(f'{metric.replace("_", " ").title()} by Algorithm', 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Algorithm', fontsize=11, fontweight='bold')
    axes[1].set_ylabel(metric.replace("_", " ").title(), fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, "statistical_distributions.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def main(csv_file='research_20251228_193652_raw.csv'):
    """Run complete statistical analysis."""
    print("\n" + "="*70)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("Graph Coloring Algorithm Performance")
    print("="*70)
    
    # Load data
    df = load_data(csv_file)
    print(f"\nData loaded: {len(df)} records, {df['algorithm'].nunique()} algorithms")
    
    metric = 'colors_used'
    
    # Run analyses
    f_stat, p_value = perform_anova(df, metric)
    pairwise_df = perform_pairwise_comparisons(df, metric)
    stats_df = descriptive_statistics(df, metric)
    ci_df = confidence_intervals(df, metric)
    normality_test(df, metric)
    homogeneity_of_variance_test(df, metric)
    
    # Generate visualizations
    plot_confidence_intervals(df, metric, ci_df)
    plot_distributions(df, metric)
    
    # Save results
    pairwise_df.to_csv(os.path.join(RESULTS_DIR, "statistical_pairwise_comparisons.csv"), index=False)
    stats_df.to_csv(os.path.join(RESULTS_DIR, "statistical_descriptive.csv"))
    ci_df.to_csv(os.path.join(RESULTS_DIR, "statistical_confidence_intervals.csv"), index=False)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"ANOVA: F={f_stat:.4f}, p={p_value:.2e}")
    if p_value < 0.05:
        print("✓ Algorithms differ SIGNIFICANTLY in performance")
    else:
        print("✗ No significant difference between algorithms")
    
    print(f"\nBest Algorithm: {stats_df.index[0]} ({stats_df.iloc[0]['Mean']:.3f} colors)")
    print(f"Worst Algorithm: {stats_df.index[-1]} ({stats_df.iloc[-1]['Mean']:.3f} colors)")
    
    print("\n✅ Statistical analysis complete!")
    print("\nFiles saved:")
    print("  statistical_pairwise_comparisons.csv")
    print("  statistical_descriptive.csv")
    print("  statistical_confidence_intervals.csv")
    print("  statistical_confidence_intervals.png")
    print("  statistical_distributions.png")


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'research_20251228_193652_raw.csv'
    main(csv_file)
