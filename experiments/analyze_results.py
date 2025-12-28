# Analyze Experiment Results
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_results(filename='experiment_results.csv'):
    """Load experiment results from CSV."""
    if not os.path.exists(filename):
        print(f"Results file {filename} not found!")
        return None
    
    return pd.read_csv(filename)

def plot_colors_used(df):
    """Plot colors used by algorithm and puzzle."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='puzzle', y='colors_used', hue='algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.title('Colors Used by Algorithm and Puzzle')
    plt.tight_layout()
    plt.savefig('colors_used_comparison.png')
    print("Saved: colors_used_comparison.png")

def plot_execution_time(df):
    """Plot execution time by algorithm and puzzle."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='puzzle', y='time', hue='algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.title('Execution Time by Algorithm and Puzzle')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('execution_time_comparison.png')
    print("Saved: execution_time_comparison.png")

def plot_efficiency(df):
    """Plot efficiency (colors vs time) scatter plot."""
    plt.figure(figsize=(10, 6))
    
    for algo in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algo]
        plt.scatter(algo_data['time'], algo_data['colors_used'], 
                   label=algo, s=100, alpha=0.6)
    
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Colors Used')
    plt.title('Algorithm Efficiency: Colors vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('efficiency_comparison.png')
    print("Saved: efficiency_comparison.png")

def generate_summary_stats(df):
    """Generate and save summary statistics."""
    summary = df.groupby('algorithm').agg({
        'colors_used': ['mean', 'std', 'min', 'max'],
        'time': ['mean', 'std', 'min', 'max'],
        'valid': 'sum'
    }).round(4)
    
    print("\nSummary Statistics:")
    print(summary)
    
    summary.to_csv('summary_statistics.csv')
    print("\nSaved: summary_statistics.csv")
    
    return summary

def analyze_best_algorithm(df):
    """Determine best algorithm for each puzzle."""
    best_by_colors = df.loc[df.groupby('puzzle')['colors_used'].idxmin()]
    best_by_time = df.loc[df.groupby('puzzle')['time'].idxmin()]
    
    print("\nBest Algorithm by Colors Used:")
    print(best_by_colors[['puzzle', 'algorithm', 'colors_used']])
    
    print("\nFastest Algorithm:")
    print(best_by_time[['puzzle', 'algorithm', 'time']])

def main():
    """Main analysis function."""
    print("Analyzing experiment results...")
    
    df = load_results()
    if df is None:
        return
    
    print(f"\nLoaded {len(df)} results")
    
    # Generate plots
    plot_colors_used(df)
    plot_execution_time(df)
    plot_efficiency(df)
    
    # Generate statistics
    generate_summary_stats(df)
    analyze_best_algorithm(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
