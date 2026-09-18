import os
import json
import argparse
import random
import yaml
import sys
from collections import defaultdict
import itertools

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def validate_candidate(c):
    if "chapter" not in c or not isinstance(c["chapter"], int):
        return False, "missing or invalid chapter"
    if "question_type" not in c or not isinstance(c["question_type"], str):
        return False, "missing or invalid question_type"
    if "difficulty" not in c or not isinstance(c["difficulty"], str):
        return False, "missing or invalid difficulty"

    for k in c.keys():
        if k.startswith("relevant_chunk_ids_"):
            return False, f"contains Phase 14 field: {k}"

    return True, ""

def solve(groups, group_idx, current_selection, chapter_counts, chapter_targets, max_backtracks=[100000]):
    if max_backtracks[0] <= 0:
        return None

    if group_idx == len(groups):
        # Check if chapter targets are met exactly
        for chap, target in chapter_targets.items():
            if chapter_counts[chap] != target["target"]:
                return None
        return current_selection

    group_key, k, pool = groups[group_idx]

    # If pool is smaller than required k, this path is invalid
    if len(pool) < k:
        return None

    # Generate combinations and shuffle them for randomness
    combs = list(itertools.combinations(pool, k))
    random.shuffle(combs)

    for comb in combs:
        max_backtracks[0] -= 1
        if max_backtracks[0] <= 0:
            return None

        new_chapter_counts = chapter_counts.copy()
        valid = True
        for c in comb:
            chap = c["chapter"]
            new_chapter_counts[chap] += 1
            if new_chapter_counts[chap] > chapter_targets[chap]["range"][1]:
                valid = False
                break

        if valid:
            res = solve(groups, group_idx + 1, current_selection + list(comb), new_chapter_counts, chapter_targets, max_backtracks)
            if res is not None:
                return res

    return None

def solve_with_ranges(groups, group_idx, current_selection, chapter_counts, chapter_targets, max_backtracks=[100000]):
    if max_backtracks[0] <= 0:
        return None

    if group_idx == len(groups):
        # Check if chapter targets are within ranges
        for chap, target in chapter_targets.items():
            if chapter_counts[chap] < target["range"][0] or chapter_counts[chap] > target["range"][1]:
                return None
        return current_selection

    group_key, k, pool = groups[group_idx]

    if len(pool) < k:
        return None

    combs = list(itertools.combinations(pool, k))
    random.shuffle(combs)

    for comb in combs:
        max_backtracks[0] -= 1
        if max_backtracks[0] <= 0:
            return None

        new_chapter_counts = chapter_counts.copy()
        valid = True
        for c in comb:
            chap = c["chapter"]
            new_chapter_counts[chap] += 1
            if new_chapter_counts[chap] > chapter_targets[chap]["range"][1]:
                valid = False
                break

        if valid:
            res = solve_with_ranges(groups, group_idx + 1, current_selection + list(comb), new_chapter_counts, chapter_targets, max_backtracks)
            if res is not None:
                return res

    return None

def select_questions(candidates_path, output_path, config_path, seed):
    random.seed(seed)

    config = load_config(config_path)
    target_count = config["target_count"]
    chapter_targets = config["chapter_targets"]
    td_targets = config["type_difficulty_targets"]

    if not os.path.exists(candidates_path):
        print(f"Candidates file not found: {candidates_path}")
        sys.exit(1)

    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                c = json.loads(line)
                valid, msg = validate_candidate(c)
                if not valid:
                    print(f"Malformed candidate at line {i+1}: {msg}")
                    sys.exit(1)
                candidates.append(c)

    if not candidates:
        print(f"No candidates found in {candidates_path}.")
        sys.exit(1)

    # Group candidates
    pool_by_group = defaultdict(list)
    for c in candidates:
        pool_by_group[(c["question_type"], c["difficulty"])].append(c)

    # Check if we have enough candidates per group for exact targets
    base_groups = []
    base_ks = []
    keys = []

    # We only include cells where target k > 0 or where we might want to allow k=1
    # Actually, the requirement says "+-1 deviation PER type x difficulty cell"
    # This means a cell with target 0 can become 1.
    for typ, diffs in td_targets.items():
        for diff, k in diffs.items():
            pool = pool_by_group.get((typ, diff), [])
            keys.append((typ, diff, pool))
            base_ks.append(k)

    # Step 1 & 2: Exact type x difficulty targets
    groups = []
    exact_td_valid = True
    for i, (typ, diff, pool) in enumerate(keys):
        k = base_ks[i]
        if k > 0:
            if len(pool) < k:
                exact_td_valid = False
                break
            groups.append(((typ, diff), k, pool))

    selected = None
    chapter_counts = defaultdict(int)

    if exact_td_valid:
        print("Attempting to satisfy exact targets (exact chapter, exact type x diff)...")
        selected = solve(groups, 0, [], chapter_counts, chapter_targets, max_backtracks=[500000])

        if selected is None:
            print("Could not satisfy exact chapter targets. Falling back to acceptable ranges (exact type x diff)...")
            selected = solve_with_ranges(groups, 0, [], chapter_counts, chapter_targets, max_backtracks=[500000])

    if selected is None:
        print("Exact type x diff failed. Attempting to relax type x diff targets by +-1...")
        # Step 3: Relaxed type x diff targets
        def generate_configs(idx, current_sum, current_config):
            if idx == len(base_ks):
                if current_sum == target_count:
                    # Validate aggregate type and difficulty constraints
                    type_counts = defaultdict(int)
                    diff_counts = defaultdict(int)
                    for i, k_prime in enumerate(current_config):
                        if k_prime > 0:
                            typ, diff, _ = keys[i]
                            type_counts[typ] += k_prime
                            diff_counts[diff] += k_prime

                    valid_aggregates = True
                    for typ, t_range in config["type_targets"].items():
                        if not (t_range["range"][0] <= type_counts[typ] <= t_range["range"][1]):
                            valid_aggregates = False
                            break

                    if valid_aggregates:
                        for diff, d_range in config["difficulty_targets"].items():
                            if not (d_range["range"][0] <= diff_counts[diff] <= d_range["range"][1]):
                                valid_aggregates = False
                                break

                    if valid_aggregates:
                        yield tuple(current_config)
                return

            base_val = base_ks[idx]
            possible_vals = []
            if base_val > 0:
                possible_vals.append(base_val - 1)
            possible_vals.append(base_val)
            possible_vals.append(base_val + 1)

            for v in possible_vals:
                max_remaining = sum(base_ks[idx+1:]) + len(base_ks) - idx - 1
                if current_sum + v + max_remaining < target_count:
                    continue
                min_remaining = sum(max(0, x - 1) for x in base_ks[idx+1:])
                if current_sum + v + min_remaining > target_count:
                    continue

                current_config.append(v)
                yield from generate_configs(idx + 1, current_sum + v, current_config)
                current_config.pop()

        configs = list(generate_configs(0, 0, []))
        random.shuffle(configs)

        global_backtracks = [2000000]

        for cfg in configs:
            if global_backtracks[0] <= 0:
                break

            valid_cfg = True
            relaxed_groups = []
            for i, k_prime in enumerate(cfg):
                if k_prime > 0:
                    typ, diff, pool = keys[i]
                    if len(pool) < k_prime:
                        valid_cfg = False
                        break
                    relaxed_groups.append(((typ, diff), k_prime, pool))

            if not valid_cfg:
                continue

            selected = solve_with_ranges(relaxed_groups, 0, [], defaultdict(int), chapter_targets, max_backtracks=global_backtracks)
            if selected is not None:
                print("Found valid selection using relaxed type x diff constraints.")
                break

    if selected is None:
        print("Constraint failure: Could not find a combination satisfying the exact or relaxed constraints.")
        sys.exit(1)

    # Sort by chapter and assign IDs
    selected.sort(key=lambda x: (x["chapter"], x.get("section", "")))

    for i, q in enumerate(selected):
        q["question_id"] = f"Q{i+1:03d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for q in selected:
            f.write(json.dumps(q) + "\n")

    print(f"Successfully selected {len(selected)} questions to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/questions/candidates.jsonl")
    parser.add_argument("--output", default="data/questions/questions.jsonl")
    parser.add_argument("--config", default="config/question_dataset_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    select_questions(args.candidates, args.output, args.config, args.seed)
