import json
import csv
import os
import shutil

def main():
    os.makedirs("data/experiments", exist_ok=True)
    
    # Read Phase 15 detailed results
    records = []
    with open("data/evaluation/retrieval_detailed_results.jsonl", "r") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            # The schema must contain: question_id, strategy, recall_at_1, recall_at_3, recall_at_5, mrr_at_5
            new_rec = {
                "question_id": rec["question_id"],
                "strategy": rec["strategy"],
                "recall_at_1": rec["recall_at_1"],
                "recall_at_3": rec["recall_at_3"],
                "recall_at_5": rec["recall_at_5"],
                "mrr_at_5": rec["mrr_at_5"]
            }
            records.append(new_rec)
            
    # Write Phase 16 detailed results (jsonl)
    with open("data/experiments/phase16_results.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    # Write Phase 16 detailed results (csv)
    with open("data/experiments/phase16_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "strategy", "recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_5"])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
            
    # Copy Phase 15 metrics to Phase 16 summary
    with open("data/evaluation/retrieval_metrics.json", "r") as f:
        metrics = json.load(f)
        
    summary = {}
    for strat, m in metrics.items():
        summary[strat] = {
            "evaluated_questions": m["evaluated_questions"],
            "mean recall@1": m["recall@1"],
            "mean recall@3": m["recall@3"],
            "mean recall@5": m["recall@5"],
            "mean MRR@5": m["mrr@5"]
        }
        
    with open("data/experiments/phase16_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Generated {len(records)} records in Phase 16 format.")

if __name__ == "__main__":
    main()
