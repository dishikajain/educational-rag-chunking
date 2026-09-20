import pytest
from scripts.evaluate_retrieval import calculate_recall_at_k, calculate_mrr_at_k

def test_calculate_recall():
    R = ["c1", "c2", "c3"]
    # Empty retrieved
    assert calculate_recall_at_k(R, [], 5) == 0.0
    
    # Empty ground truth (should raise ValueError as instructed by user: "treat this as an invalid ground-truth condition and FAIL clearly")
    with pytest.raises(ValueError):
        calculate_recall_at_k([], ["c1"], 5)
        
    # Full overlap
    assert calculate_recall_at_k(R, ["c1", "c2", "c3", "c4", "c5"], 5) == 1.0
    
    # Partial overlap
    assert calculate_recall_at_k(R, ["c1", "c4", "c5"], 3) == 1/3
    assert calculate_recall_at_k(R, ["c4", "c1", "c2", "c5", "c6"], 5) == 2/3
    
    # Cutoff by K
    assert calculate_recall_at_k(R, ["c4", "c5", "c6", "c1", "c2"], 3) == 0.0
    assert calculate_recall_at_k(R, ["c1", "c2", "c3"], 1) == 1/3

def test_calculate_mrr():
    R = ["c1", "c2"]
    
    with pytest.raises(ValueError):
        calculate_mrr_at_k([], ["c1"], 5)
        
    assert calculate_mrr_at_k(R, [], 5) == 0.0
    
    # First relevant at rank 1
    assert calculate_mrr_at_k(R, ["c1", "c3", "c4", "c5", "c6"], 5) == 1.0
    assert calculate_mrr_at_k(R, ["c2", "c1", "c3"], 5) == 1.0
    
    # First relevant at rank 2
    assert calculate_mrr_at_k(R, ["c3", "c1", "c4", "c5", "c6"], 5) == 0.5
    
    # First relevant at rank 5
    assert calculate_mrr_at_k(R, ["c3", "c4", "c5", "c6", "c1"], 5) == 0.2
    
    # Cutoff by K (rank 6, so mrr@5 = 0.0)
    assert calculate_mrr_at_k(R, ["c3", "c4", "c5", "c6", "c7", "c1"], 5) == 0.0

