import json
import os
from collections import defaultdict
from typing import List
from src.retrieval.config import load_config
from src.retrieval.retriever import Retriever

def calculate_recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant_chunk_ids cannot be empty")
    k_retrieved = retrieved[:k]
    overlap = set(relevant).intersection(set(k_retrieved))
    return len(overlap) / len(relevant)

def calculate_mrr_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant_chunk_ids cannot be empty")
    
    for i, chunk_id in enumerate(retrieved[:k]):
        if chunk_id in relevant:
            return 1.0 / (i + 1)
            
    return 0.0

def main():
    questions_file = "data/questions/questions.jsonl"
    ground_truth_file = "data/ground_truth/retrieval_ground_truth.jsonl"
    metrics_out = "data/evaluation/retrieval_metrics.json"
    detailed_out = "data/evaluation/retrieval_detailed_results.jsonl"
    
    os.makedirs("data/evaluation", exist_ok=True)
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = {json.loads(l)["question_id"]: json.loads(l) for l in f}
        
    ground_truth = defaultdict(dict)
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            ground_truth[record["question_id"]][record["strategy"]] = record["relevant_chunk_ids"]
            
    config = load_config("config/retrieval_config.yaml")
    # Ensure top_k is at least 5 so we can compute up to @5
    if config.top_k < 5:
        config.top_k = 5
        
    retriever = Retriever(config=config)
    
    strategies = [
        "baseline_fixed_size",
        "eluc_section_aware",
        "recursive_character",
        "paragraph_based",
        "heading_based"
    ]
    
    aggregate_metrics = defaultdict(lambda: {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0, "mrr@5": 0.0, "count": 0})
    detailed_results = []
    
    for strat in strategies:
        print(f"Evaluating strategy: {strat}")
        for q_id, q in questions.items():
            relevant_chunks = ground_truth.get(q_id, {}).get(strat, [])
            if not relevant_chunks:
                raise ValueError(f"Missing or empty ground truth for question {q_id} in strategy {strat}")
                
            retrieved = retriever.retrieve(q["question"], strat)
            retrieved_ids = [c.chunk_id for c in retrieved]
            
            r1 = calculate_recall_at_k(relevant_chunks, retrieved_ids, 1)
            r3 = calculate_recall_at_k(relevant_chunks, retrieved_ids, 3)
            r5 = calculate_recall_at_k(relevant_chunks, retrieved_ids, 5)
            mrr5 = calculate_mrr_at_k(relevant_chunks, retrieved_ids, 5)
            
            aggregate_metrics[strat]["recall@1"] += r1
            aggregate_metrics[strat]["recall@3"] += r3
            aggregate_metrics[strat]["recall@5"] += r5
            aggregate_metrics[strat]["mrr@5"] += mrr5
            aggregate_metrics[strat]["count"] += 1
            
            detailed_results.append({
                "question_id": q_id,
                "strategy": strat,
                "retrieved_chunk_ids": retrieved_ids,
                "relevant_chunk_ids": relevant_chunks,
                "retrieved_relevant_chunk_ids": list(set(relevant_chunks).intersection(set(retrieved_ids))),
                "recall_at_1": r1,
                "recall_at_3": r3,
                "recall_at_5": r5,
                "mrr_at_5": mrr5
            })
            
    # Calculate averages
    final_metrics = {}
    for strat, mets in aggregate_metrics.items():
        count = mets["count"]
        final_metrics[strat] = {
            "recall@1": mets["recall@1"] / count,
            "recall@3": mets["recall@3"] / count,
            "recall@5": mets["recall@5"] / count,
            "mrr@5": mets["mrr@5"] / count,
            "evaluated_questions": count
        }
        
    with open(metrics_out, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)
        
    with open(detailed_out, 'w', encoding='utf-8') as f:
        for r in detailed_results:
            f.write(json.dumps(r) + "\n")
            
    print(json.dumps(final_metrics, indent=4))
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()

