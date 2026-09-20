import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import os
import itertools

def holm_correction(p_values):
    """Simple Holm-Bonferroni correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted_p = np.zeros(n)
    
    for i, idx in enumerate(sorted_indices):
        adjusted_p[idx] = min(1.0, p_values[idx] * (n - i))
        if i > 0:
            adjusted_p[idx] = max(adjusted_p[idx], adjusted_p[sorted_indices[i-1]])
            
    return adjusted_p

def bootstrap_median_diff(a, b, n_resamples=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = a - b
    n = len(diffs)
    boot_medians = np.zeros(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        boot_diffs = diffs[indices]
        boot_medians[i] = np.median(boot_diffs)
    return np.percentile(boot_medians, 2.5), np.percentile(boot_medians, 97.5)

def rank_biserial_correlation(x, y):
    """Rank-biserial correlation for Wilcoxon signed-rank test.
    Formula: (W_pos - W_neg) / (W_pos + W_neg)
    Uses scipy wilcoxon which drops zero differences by default ('wilcox').
    """
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    
    ranks = stats.rankdata(np.abs(diff))
    w_pos = np.sum(ranks[diff > 0])
    w_neg = np.sum(ranks[diff < 0])
    if w_pos + w_neg == 0:
        return 0.0
    return (w_pos - w_neg) / (w_pos + w_neg)

def main():
    os.makedirs("data/statistics", exist_ok=True)
    
    df = pd.read_csv("data/experiments/phase16_results.csv")
    metrics = ["recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5"]
    strategies = df["strategy"].unique()
    
    omnibus_results = {}
    pairwise_records = []
    question_stats = []
    
    # Question level stats
    for qid in df["question_id"].unique():
        qdf = df[df["question_id"] == qid]
        q_stat = {"question_id": qid}
        for metric in metrics:
            vals = qdf[metric].values
            q_stat[f"{metric}_median"] = np.median(vals)
            q_stat[f"{metric}_mean"] = np.mean(vals)
            q_stat[f"{metric}_max"] = np.max(vals)
        question_stats.append(q_stat)
    
    pd.DataFrame(question_stats).to_csv("data/statistics/phase17_question_level_statistics.csv", index=False)
    
    for metric in metrics:
        pivot = df.pivot(index="question_id", columns="strategy", values=metric)
        
        # 1. Friedman Test
        k = len(strategies)
        N = len(pivot)
        data_arrays = [pivot[col].values for col in strategies]
        
        stat, p_val = stats.friedmanchisquare(*data_arrays)
        
        # Kendall's W
        w = stat / (N * (k - 1)) if k > 1 else 0
        
        is_significant = bool(p_val < 0.05)
        omnibus_results[metric] = {
            "metric": metric,
            "test": "Friedman",
            "sample_size": int(N),
            "statistic": float(stat),
            "p_value": float(p_val),
            "effect_size_name": "Kendall_W",
            "effect_size": float(w),
            "significant_at_05": is_significant,
            "post_hoc_performed": is_significant
        }
        
        if is_significant:
            comparisons = list(itertools.combinations(strategies, 2))
            unadjusted_p = []
            
            pairs_data = []
            for s1, s2 in comparisons:
                d1 = pivot[s1].values
                d2 = pivot[s2].values
                
                # Check for all ties
                if np.all(d1 == d2):
                    w_stat, w_p = 0.0, 1.0
                else:
                    res = stats.wilcoxon(d1, d2, zero_method="wilcox", alternative="two-sided")
                    w_stat = res.statistic
                    w_p = res.pvalue
                    
                unadjusted_p.append(w_p)
                
                med_diff = np.median(d1 - d2)
                ci_low, ci_high = bootstrap_median_diff(d1, d2)
                r_bc = rank_biserial_correlation(d1, d2)
                
                pairs_data.append({
                    "metric": metric,
                    "strategy_a": s1,
                    "strategy_b": s2,
                    "median_difference": med_diff,
                    "wilcoxon_statistic": w_stat,
                    "unadjusted_p_value": w_p,
                    "effect_size_rbc": r_bc,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high
                })
                
            adjusted_p = holm_correction(unadjusted_p)
            for i, p_data in enumerate(pairs_data):
                p_data["holm_adjusted_p_value"] = adjusted_p[i]
                pairwise_records.append(p_data)
        else:
            # If not significant, we could skip pairwise records or output them as skipped.
            # The prompt requested: "Pairwise results should clearly indicate whether post-hoc testing was performed or skipped"
            # We just add a dummy record indicating skipped.
            pairwise_records.append({
                "metric": metric,
                "strategy_a": "SKIPPED",
                "strategy_b": "SKIPPED",
                "median_difference": None,
                "wilcoxon_statistic": None,
                "unadjusted_p_value": None,
                "effect_size_rbc": None,
                "ci_lower": None,
                "ci_upper": None,
                "holm_adjusted_p_value": None
            })
            
    with open("data/statistics/phase17_statistical_results.json", "w") as f:
        json.dump(omnibus_results, f, indent=4)
        
    pd.DataFrame(pairwise_records).to_csv("data/statistics/phase17_pairwise_results.csv", index=False)
    
if __name__ == "__main__":
    main()
