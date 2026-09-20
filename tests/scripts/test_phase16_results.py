import pytest
import json
import csv
import math

def test_phase16_jsonl_records():
    with open("data/experiments/phase16_results.jsonl", "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    assert len(records) == 250, "Exactly 250 records required"
    
    strategies = {"baseline_fixed_size", "eluc_section_aware", "recursive_character", "paragraph_based", "heading_based"}
    strat_counts = {s: 0 for s in strategies}
    pairs = set()
    
    for r in records:
        strat = r["strategy"]
        qid = r["question_id"]
        assert strat in strategies, f"Unknown strategy {strat}"
        strat_counts[strat] += 1
        
        pair = (qid, strat)
        assert pair not in pairs, f"Duplicate pair {pair}"
        pairs.add(pair)
        
        # Check metrics
        for metric in ["recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5"]:
            assert metric in r, f"Missing metric {metric}"
            val = r[metric]
            assert 0.0 <= val <= 1.0, f"{metric} out of bounds: {val}"
            
    for s, c in strat_counts.items():
        assert c == 50, f"Strategy {s} has {c} records, expected 50"
        
def test_phase16_csv_records():
    with open("data/experiments/phase16_results.csv", "r") as f:
        reader = csv.DictReader(f)
        records = list(reader)
        
    assert len(records) == 250, "Exactly 250 CSV records required"
    for r in records:
        for metric in ["recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5"]:
            val = float(r[metric])
            assert 0.0 <= val <= 1.0

def test_phase16_aggregate_match():
    # Load Phase 15 metrics
    with open("data/evaluation/retrieval_metrics.json", "r") as f:
        p15_metrics = json.load(f)
        
    # Load Phase 16 summary
    with open("data/experiments/phase16_summary.json", "r") as f:
        p16_summary = json.load(f)
        
    for strat in p15_metrics:
        p15 = p15_metrics[strat]
        p16 = p16_summary[strat]
        assert p16["evaluated_questions"] == 50
        assert math.isclose(p16["mean recall@1"], p15["recall@1"])
        assert math.isclose(p16["mean recall@3"], p15["recall@3"])
        assert math.isclose(p16["mean recall@5"], p15["recall@5"])
        assert math.isclose(p16["mean MRR@5"], p15["mrr@5"])

