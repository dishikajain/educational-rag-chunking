import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def clean_strategy_name(strategy):
    mapping = {
        "baseline_fixed_size": "Fixed-size",
        "eluc_section_aware": "ELUC / Section-aware",
        "recursive_character": "Recursive character",
        "paragraph_based": "Paragraph-based",
        "heading_based": "Heading-based"
    }
    return mapping.get(strategy, strategy)

STRATEGY_ORDER = [
    "baseline_fixed_size",
    "eluc_section_aware",
    "recursive_character",
    "paragraph_based",
    "heading_based"
]

def make_dirs():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

def generate_table1():
    with open("data/experiments/phase16_summary.json", "r") as f:
        data = json.load(f)
        
    records = []
    for s in STRATEGY_ORDER:
        records.append({
            "Strategy": clean_strategy_name(s),
            "Mean Recall@1": f"{data[s]['mean recall@1']:.3f}",
            "Mean Recall@3": f"{data[s]['mean recall@3']:.3f}",
            "Mean Recall@5": f"{data[s]['mean recall@5']:.3f}",
            "Mean MRR@5": f"{data[s]['mean MRR@5']:.3f}"
        })
        
    df = pd.DataFrame(records)
    df.to_csv("results/tables/table1_overall_performance.csv", index=False)
    with open("results/tables/table1_overall_performance.md", "w") as f:
        f.write(df.to_markdown(index=False))

def generate_table2():
    with open("data/statistics/phase17_statistical_results.json", "r") as f:
        data = json.load(f)
        
    records = []
    for m in ["recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5"]:
        d = data[m]
        records.append({
            "Metric": m,
            "Friedman Statistic": f"{d['statistic']:.3f}",
            "p-value": f"{d['p_value']:.2e}",
            "Kendall's W": f"{d['effect_size']:.3f}",
            "Sample Size": d['sample_size'],
            "Significant at alpha=0.05": "Yes" if d['significant_at_05'] else "No"
        })
        
    df = pd.DataFrame(records)
    df.to_csv("results/tables/table2_omnibus_statistics.csv", index=False)
    with open("results/tables/table2_omnibus_statistics.md", "w") as f:
        f.write(df.to_markdown(index=False))

def generate_table3():
    df = pd.read_csv("data/statistics/phase17_pairwise_results.csv")
    records = []
    for _, row in df.iterrows():
        records.append({
            "Metric": row["metric"],
            "Strategy A": clean_strategy_name(row["strategy_a"]) if row["strategy_a"] != "SKIPPED" else "SKIPPED",
            "Strategy B": clean_strategy_name(row["strategy_b"]) if row["strategy_b"] != "SKIPPED" else "SKIPPED",
            "Median Difference": f"{row['median_difference']:.3f}" if pd.notnull(row['median_difference']) else "",
            "CI Lower": f"{row['ci_lower']:.3f}" if pd.notnull(row['ci_lower']) else "",
            "CI Upper": f"{row['ci_upper']:.3f}" if pd.notnull(row['ci_upper']) else "",
            "Wilcoxon Statistic": f"{row['wilcoxon_statistic']:.3f}" if pd.notnull(row['wilcoxon_statistic']) else "",
            "Unadjusted p-value": f"{row['unadjusted_p_value']:.2e}" if pd.notnull(row['unadjusted_p_value']) else "",
            "Holm-adjusted p-value": f"{row['holm_adjusted_p_value']:.2e}" if pd.notnull(row['holm_adjusted_p_value']) else "",
            "Rank-Biserial Correlation": f"{row['effect_size_rbc']:.3f}" if pd.notnull(row['effect_size_rbc']) else ""
        })
        
    out_df = pd.DataFrame(records)
    out_df.to_csv("results/tables/table3_pairwise_comparisons.csv", index=False)
    with open("results/tables/table3_pairwise_comparisons.md", "w") as f:
        f.write(out_df.to_markdown(index=False))

def count_chunks():
    counts = {}
    pattern = "data/chunks/formal/**/*.jsonl"
    for fpath in glob.glob(pattern, recursive=True):
        if "baseline" in fpath: s = "baseline_fixed_size"
        elif "eluc" in fpath: s = "eluc_section_aware"
        elif "recursive" in fpath: s = "recursive_character"
        elif "paragraph" in fpath: s = "paragraph_based"
        elif "heading" in fpath: s = "heading_based"
        else: continue
        
        c = sum(1 for _ in open(fpath, "r", encoding="utf-8"))
        counts[s] = c
    return counts

def generate_table4():
    # Authoritative from phase14a_chunking_methodology.md
    configs = {
        "baseline_fixed_size": ("Character position", "chunk_size=350, overlap=40"),
        "eluc_section_aware": ("Heading labels + is_section_break", "max_chars=1400"),
        "recursive_character": ("Text separators", "chunk_size=350, overlap=40"),
        "paragraph_based": ("is_paragraph_break gap", "max_chars=1400"),
        "heading_based": ("Heading labels", "max_chars=1400")
    }
    
    counts = count_chunks()
    
    records = []
    for s in STRATEGY_ORDER:
        sig, p = configs[s]
        records.append({
            "Strategy": clean_strategy_name(s),
            "Primary Boundary Signal": sig,
            "Relevant Size/Overlap Parameter": p,
            "Total Formal Chunks Generated": counts.get(s, "N/A")
        })
        
    df = pd.DataFrame(records)
    df.to_csv("results/tables/table4_chunking_configurations.csv", index=False)
    with open("results/tables/table4_chunking_configurations.md", "w") as f:
        f.write(df.to_markdown(index=False))

def save_fig(fig, path_prefix):
    fig.savefig(path_prefix + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(path_prefix + ".pdf", bbox_inches="tight")
    plt.close(fig)

def plot_distribution(metric, df):
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [df[df["strategy"] == s][metric].values for s in STRATEGY_ORDER]
    labels = [clean_strategy_name(s) for s in STRATEGY_ORDER]
    
    # Boxplot
    bp = ax.boxplot(data, positions=np.arange(len(STRATEGY_ORDER)), showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#e0e0e0')
        
    # Jitter
    rng = np.random.default_rng(42)
    for i, d in enumerate(data):
        x = np.random.normal(i, 0.05, size=len(d))
        ax.scatter(x, d, alpha=0.5, zorder=3)
        
    ax.set_xticks(np.arange(len(STRATEGY_ORDER)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Distribution of {metric}")
    
    save_fig(fig, f"results/figures/{metric}_distribution")

def plot_heatmap(df):
    import numpy as np
    # Pivot to questions rows, strategies cols
    # Sort questions Q001..Q050
    q_order = sorted(df["question_id"].unique())
    pivot = df.pivot(index="question_id", columns="strategy", values="mrr_at_5")
    pivot = pivot.reindex(index=q_order, columns=STRATEGY_ORDER)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto", interpolation='nearest')
    
    ax.set_xticks(np.arange(len(STRATEGY_ORDER)))
    ax.set_xticklabels([clean_strategy_name(s) for s in STRATEGY_ORDER], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(q_order)))
    ax.set_yticklabels(q_order, fontsize=6)
    
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Question ID")
    ax.set_title("MRR@5 Query Performance Heatmap")
    
    fig.colorbar(im, ax=ax, label="MRR@5")
    
    save_fig(fig, "results/figures/mrr_at_5_query_heatmap")

def plot_pairwise_forest(metric, df_pairs):
    df_m = df_pairs[df_pairs["metric"] == metric]
    if df_m.empty or (df_m.iloc[0]["strategy_a"] == "SKIPPED"):
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y = 0
    y_ticks = []
    y_labels = []
    
    # We want ALL 10 pairs.
    for i, row in df_m.iterrows():
        sa = clean_strategy_name(row["strategy_a"])
        sb = clean_strategy_name(row["strategy_b"])
        lbl = f"{sa} vs\n{sb}"
        
        md = float(row["median_difference"])
        cl = float(row["ci_lower"])
        cu = float(row["ci_upper"])
        
        ax.plot([cl, cu], [y, y], color="black")
        ax.scatter([md], [y], color="blue", zorder=3)
        
        y_ticks.append(y)
        y_labels.append(lbl)
        y -= 1
        
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.axvline(0, color="red", linestyle="--")
    
    ax.set_xlabel("Median Paired Difference")
    ax.set_title(f"Pairwise Comparisons (95% CI): {metric}")
    
    save_fig(fig, f"results/figures/pairwise_{metric}")

def plot_overall_comparison():
    import numpy as np
    with open("data/experiments/phase16_summary.json", "r") as f:
        data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ["mean recall@1", "mean recall@3", "mean recall@5", "mean MRR@5"]
    metric_labels = ["Recall@1", "Recall@3", "Recall@5", "MRR@5"]
    
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, s in enumerate(STRATEGY_ORDER):
        vals = [data[s][m] for m in metrics]
        offset = (i - 2) * width
        ax.bar(x + offset, vals, width, label=clean_strategy_name(s))
        
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Mean Value")
    ax.set_title("Overall Metric Comparison by Strategy")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    
    save_fig(fig, "results/figures/overall_metric_comparison")

def main():
    make_dirs()
    generate_table1()
    generate_table2()
    generate_table3()
    generate_table4()
    
    df16 = pd.read_csv("data/experiments/phase16_results.csv")
    for m in ["recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5"]:
        plot_distribution(m, df16)
        
    plot_heatmap(df16)
    
    df_pairs = pd.read_csv("data/statistics/phase17_pairwise_results.csv")
    for m in ["recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5"]:
        plot_pairwise_forest(m, df_pairs)
        
    plot_overall_comparison()

if __name__ == "__main__":
    main()
