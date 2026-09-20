import pytest
import os
import glob
import pandas as pd

def test_phase18_table_existence_and_contents():
    # Table 1
    t1_md = "results/tables/table1_overall_performance.md"
    t1_csv = "results/tables/table1_overall_performance.csv"
    assert os.path.exists(t1_md)
    assert os.path.exists(t1_csv)
    df1 = pd.read_csv(t1_csv)
    assert len(df1) == 5, "Table 1 must have exactly 5 strategies"
    
    # Table 2
    t2_md = "results/tables/table2_omnibus_statistics.md"
    t2_csv = "results/tables/table2_omnibus_statistics.csv"
    assert os.path.exists(t2_md)
    assert os.path.exists(t2_csv)
    df2 = pd.read_csv(t2_csv)
    assert len(df2) == 4, "Table 2 must have exactly 4 metrics"
    
    # Table 3
    t3_md = "results/tables/table3_pairwise_comparisons.md"
    t3_csv = "results/tables/table3_pairwise_comparisons.csv"
    assert os.path.exists(t3_md)
    assert os.path.exists(t3_csv)
    df3 = pd.read_csv(t3_csv)
    assert len(df3) == 40, "Table 3 must have exactly 40 comparisons"
    
    # Table 4
    t4_md = "results/tables/table4_chunking_configurations.md"
    t4_csv = "results/tables/table4_chunking_configurations.csv"
    assert os.path.exists(t4_md)
    assert os.path.exists(t4_csv)
    df4 = pd.read_csv(t4_csv)
    assert len(df4) == 5, "Table 4 must have exactly 5 strategies"
    
def test_phase18_figure_existence():
    expected_figures = [
        "recall_at_1_distribution",
        "recall_at_3_distribution",
        "recall_at_5_distribution",
        "mrr_at_5_distribution",
        "mrr_at_5_query_heatmap",
        "pairwise_recall_at_1",
        "pairwise_recall_at_3",
        "pairwise_recall_at_5",
        "pairwise_mrr_at_5",
        "overall_metric_comparison"
    ]
    
    for fig in expected_figures:
        assert os.path.exists(f"results/figures/{fig}.png"), f"Missing {fig}.png"
        assert os.path.exists(f"results/figures/{fig}.pdf"), f"Missing {fig}.pdf"

def test_phase16_17_inputs_unchanged():
    # Basic check that the file sizes or shapes haven't fundamentally changed.
    # The generation script just reads them.
    df16 = pd.read_csv("data/experiments/phase16_results.csv")
    assert len(df16) == 250
    df17 = pd.read_csv("data/statistics/phase17_pairwise_results.csv")
    assert len(df17) == 40
