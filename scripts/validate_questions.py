import os
import json
import jsonschema
import sys
import argparse
import yaml
from collections import defaultdict

VALID_SECTIONS = {
    1: {
        "1.1": "The Science of Biology",
        "1.2": "Themes and Concepts of Biology"
    },
    2: {
        "2.1": "Atoms, Isotopes, Ions, and Molecules: The Building Blocks",
        "2.2": "Water",
        "2.3": "Carbon"
    },
    3: {
        "3.1": "Synthesis of Biological Macromolecules",
        "3.2": "Carbohydrates",
        "3.3": "Lipids",
        "3.4": "Proteins",
        "3.5": "Nucleic Acids"
    },
    4: {
        "4.1": "Studying Cells",
        "4.2": "Prokaryotic Cells",
        "4.3": "Eukaryotic Cells",
        "4.4": "The Endomembrane System and Proteins",
        "4.5": "The Cytoskeleton",
        "4.6": "Connections between Cells and Cellular Activities"
    },
    5: {
        "5.1": "Components and Structure",
        "5.2": "Passive Transport",
        "5.3": "Active Transport",
        "5.4": "Bulk Transport"
    }
}

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def validate_dataset(jsonl_path, schema_path, config_path, allow_empty=False):
    if not os.path.exists(jsonl_path):
        print(f"File not found: {jsonl_path}")
        return False
        
    if not os.path.exists(schema_path):
        print(f"Schema not found: {schema_path}")
        return False

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    questions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if not questions:
        if allow_empty:
            print(f"Warning: {jsonl_path} is empty. Passing because --allow-empty is set.")
            return True
        else:
            print(f"VALIDATION ERROR: {jsonl_path} is empty. Use --allow-empty to permit empty datasets.")
            return False

    config = load_config(config_path)

    errors = []
    
    # 1. Schema Validation & Field constraints
    q_ids = set()
    chapter_counts = defaultdict(int)
    type_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    chapter_sections = defaultdict(set)
    td_counts = defaultdict(lambda: defaultdict(int))

    for i, q in enumerate(questions):
        try:
            jsonschema.validate(instance=q, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            errors.append(f"Row {i} failed schema validation: {e.message}")
            continue

        q_id = q["question_id"]
        if q_id in q_ids:
            errors.append(f"Duplicate question_id: {q_id}")
        q_ids.add(q_id)

        # Disallowed fields
        for k in q.keys():
            if k.startswith("relevant_chunk_ids_"):
                errors.append(f"{q_id}: found prohibited field {k}")

        if q.get("requires_outside_knowledge") is not False:
            errors.append(f"{q_id}: requires_outside_knowledge must be false")

        if q.get("created_by") != "manual_annotation":
            errors.append(f"{q_id}: created_by must be manual_annotation")

        if not (10 <= len(q["question"]) <= 250):
            errors.append(f"{q_id}: question length {len(q['question'])} out of bounds [10, 250]")

        if len(q["reference_answer"]) < 50:
            errors.append(f"{q_id}: reference_answer length {len(q['reference_answer'])} < 50")

        if not q["source_passage"].strip():
            errors.append(f"{q_id}: source_passage is empty")

        # Section validation
        chap = q["chapter"]
        sec = q["section"]
        title = q["section_title"]
        
        if chap not in VALID_SECTIONS:
            errors.append(f"{q_id}: invalid chapter {chap}")
        elif sec not in VALID_SECTIONS[chap]:
            errors.append(f"{q_id}: invalid section {sec} for chapter {chap}")
        elif VALID_SECTIONS[chap][sec] != title:
            errors.append(f"{q_id}: invalid section_title '{title}' for section {sec}")

        # Update stats
        chapter_counts[chap] += 1
        type_counts[q["question_type"]] += 1
        difficulty_counts[q["difficulty"]] += 1
        chapter_sections[chap].add(sec)
        td_counts[q["question_type"]][q["difficulty"]] += 1

    # 2. Distribution Validation
    acceptable_range = config.get("acceptable_range", [45, 60])
    if len(questions) < acceptable_range[0] or len(questions) > acceptable_range[1]:
        errors.append(f"Dataset size {len(questions)} outside acceptable range {acceptable_range}")

    min_sections = config.get("minimum_section_coverage_per_chapter", 3)
    chap_targets = config.get("chapter_targets", {})
    
    for chap in range(1, 6):
        count = chapter_counts[chap]
        target_range = chap_targets[chap]["range"]
        if count < target_range[0] or count > target_range[1]:
            errors.append(f"Chapter {chap} count {count} not in {target_range}")
        actual_min = min(min_sections, len(VALID_SECTIONS[chap]))
        if len(chapter_sections[chap]) < actual_min:
            errors.append(f"Chapter {chap} has only {len(chapter_sections[chap])} distinct sections represented, need at least {actual_min}")

    type_targets = config.get("type_targets", {})
    for qt, targets in type_targets.items():
        t_range = targets["range"]
        if type_counts[qt] < t_range[0] or type_counts[qt] > t_range[1]:
            errors.append(f"Question type {qt} count {type_counts[qt]} not in {t_range}")

    diff_targets = config.get("difficulty_targets", {})
    for diff, targets in diff_targets.items():
        d_range = targets["range"]
        if difficulty_counts[diff] < d_range[0] or difficulty_counts[diff] > d_range[1]:
            errors.append(f"Difficulty {diff} count {difficulty_counts[diff]} not in {d_range}")

    td_targets = config.get("type_difficulty_targets", {})
    for typ, diffs in td_targets.items():
        for diff, target in diffs.items():
            lower_bound = max(0, target - 1)
            upper_bound = target + 1
            actual = td_counts[typ][diff]
            if actual < lower_bound or actual > upper_bound:
                errors.append(f"Type x Difficulty '{typ}' x '{diff}' count {actual} not in [{lower_bound}, {upper_bound}]")

    if errors:
        for e in errors:
            print(f"VALIDATION ERROR: {e}")
        return False
        
    print(f"Successfully validated {len(questions)} questions.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/questions/questions.jsonl")
    parser.add_argument("--schema", default="data/questions/questions_schema.json")
    parser.add_argument("--config", default="config/question_dataset_config.yaml")
    parser.add_argument("--allow-empty", action="store_true", help="Permit empty datasets")
    args = parser.parse_args()
    
    success = validate_dataset(args.input, args.schema, args.config, args.allow_empty)
    sys.exit(0 if success else 1)
