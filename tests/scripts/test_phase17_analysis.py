import pytest
import json
import csv
import pandas as pd

def test_phase17_omnibus_results():
    with open("data/statistics/phase17_statistical_results.json", "r") as f:
        results = json.load(f)
        
    metrics = ["recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5"]
    assert set(results.keys()) == set(metrics), "Omnibus results must contain all 4 metrics"
    
    for metric, data in results.items():
        assert data["sample_size"] == 50
        assert data["test"] == "Friedman"
        assert "statistic" in data
        assert "p_value" in data
        assert "effect_size" in data
        assert 0.0 <= data["p_value"] <= 1.0

def test_phase17_pairwise_results():
    df = pd.read_csv("data/statistics/phase17_pairwise_results.csv")
    assert not df.empty
    
    # Check that skipped metrics are marked appropriately and performed metrics have 10 rows
    for metric in df["metric"].unique():
        metric_df = df[df["metric"] == metric]
        
        with open("data/statistics/phase17_statistical_results.json", "r") as f:
            omnibus = json.load(f)
            
        is_sig = omnibus[metric]["significant_at_05"]
        if is_sig:
            assert len(metric_df) == 10, f"Expected 10 pairwise comparisons for {metric}"
            for col in ["ci_lower", "ci_upper", "effect_size_rbc", "holm_adjusted_p_value", "unadjusted_p_value"]:
                assert not metric_df[col].isnull().all()
                
            # Verify CI ordering
            assert (metric_df["ci_lower"] <= metric_df["ci_upper"]).all(), "CI lower must be <= CI upper"
            assert (metric_df["unadjusted_p_value"] <= metric_df["holm_adjusted_p_value"]).all(), "Adjusted p >= unadjusted p"
        else:
            assert len(metric_df) == 1
            assert metric_df.iloc[0]["strategy_a"] == "SKIPPED"
            
def test_phase17_question_stats():
    df = pd.read_csv("data/statistics/phase17_question_level_statistics.csv")
    assert len(df) == 50
    assert "question_id" in df.columns

