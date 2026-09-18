import pytest
import os
import json
import yaml
import jsonschema
from collections import defaultdict
from scripts.validate_questions import validate_dataset, VALID_SECTIONS
from scripts.select_questions import select_questions

def get_valid_question():
    return {
        "question_id": "Q001",
        "question": "What is the primary function of mitochondria in eukaryotic cells?",
        "chapter": 4,
        "section": "4.3",
        "section_title": "Eukaryotic Cells",
        "question_type": "factual_lookup",
        "difficulty": "easy",
        "reference_answer": "Mitochondria are the sites of cellular respiration, producing ATP to power cellular activities.",
        "source_passage": "Mitochondria are organelles found in the cytoplasm of eukaryotic cells. They are the site of cellular respiration, a process that produces ATP—the cell's primary energy currency.",
        "source_page_range": "68-69",
        "answer_grounding": "Directly stated in Section 4.3.",
        "requires_outside_knowledge": False,
        "created_by": "manual_annotation",
        "creation_date": "2026-09-18"
    }

@pytest.fixture
def schema():
    schema_path = "data/questions/questions_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_schema_validates_correct_question(schema):
    q = get_valid_question()
    jsonschema.validate(instance=q, schema=schema)

def test_invalid_chapter(schema):
    q = get_valid_question()
    q["chapter"] = 6
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=q, schema=schema)

def test_invalid_question_type(schema):
    q = get_valid_question()
    q["question_type"] = "invalid_type"
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=q, schema=schema)

def test_invalid_difficulty(schema):
    q = get_valid_question()
    q["difficulty"] = "extreme"
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=q, schema=schema)

def test_outside_knowledge_rejection(schema):
    q = get_valid_question()
    q["requires_outside_knowledge"] = True
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=q, schema=schema)

def test_phase_14_field_rejection(schema):
    q = get_valid_question()
    q["relevant_chunk_ids_baseline"] = ["chunk1"]
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=q, schema=schema)

def test_section_title_mapping(tmp_path):
    q = get_valid_question()
    q["section_title"] = "Incorrect Title"
    
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(json.dumps(q) + "\n")
    
    assert not validate_dataset(str(jsonl), "data/questions/questions_schema.json", "config/question_dataset_config.yaml")

def test_invalid_section(tmp_path):
    q = get_valid_question()
    q["section"] = "4.99"
    
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(json.dumps(q) + "\n")
    
    assert not validate_dataset(str(jsonl), "data/questions/questions_schema.json", "config/question_dataset_config.yaml")

def test_duplicate_question_ids(tmp_path):
    q1 = get_valid_question()
    q2 = get_valid_question()
    
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text(json.dumps(q1) + "\n" + json.dumps(q2) + "\n")
    
    assert not validate_dataset(str(jsonl), "data/questions/questions_schema.json", "config/question_dataset_config.yaml")

def test_distribution_validation(tmp_path):
    # Create 50 questions that violate distribution
    questions = []
    for i in range(50):
        q = get_valid_question()
        q["question_id"] = f"Q{i+1:03d}"
        q["chapter"] = 1 # all chapter 1
        q["section"] = "1.1"
        q["section_title"] = "The Science of Biology"
        questions.append(q)
        
    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
            
    assert not validate_dataset(str(jsonl), "data/questions/questions_schema.json", "config/question_dataset_config.yaml")

def test_empty_dataset_fails_by_default(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text("")
    assert not validate_dataset(str(jsonl), "data/questions/questions_schema.json", "config/question_dataset_config.yaml")

def test_empty_dataset_passes_with_allow_empty(tmp_path):
    jsonl = tmp_path / "test.jsonl"
    jsonl.write_text("")
    assert validate_dataset(str(jsonl), "data/questions/questions_schema.json", "config/question_dataset_config.yaml", allow_empty=True)

def test_insufficient_candidate_pool(tmp_path):
    candidates = []
    for i in range(10): # Only 10 candidates
        q = get_valid_question()
        q["question_id"] = f"C{i+1:03d}"
        candidates.append(q)
        
    cand_file = tmp_path / "candidates.jsonl"
    out_file = tmp_path / "out.jsonl"
    with open(cand_file, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    with pytest.raises(SystemExit):
        select_questions(str(cand_file), str(out_file), "config/question_dataset_config.yaml", seed=42)

def test_malformed_candidate(tmp_path):
    q = get_valid_question()
    del q["chapter"] # malformed
    
    cand_file = tmp_path / "candidates.jsonl"
    out_file = tmp_path / "out.jsonl"
    cand_file.write_text(json.dumps(q) + "\n")
    
    with pytest.raises(SystemExit):
        select_questions(str(cand_file), str(out_file), "config/question_dataset_config.yaml", seed=42)

def test_phase14_field_rejection_in_selector(tmp_path):
    q = get_valid_question()
    q["relevant_chunk_ids_baseline"] = ["c1"]
    
    cand_file = tmp_path / "candidates.jsonl"
    out_file = tmp_path / "out.jsonl"
    cand_file.write_text(json.dumps(q) + "\n")
    
    with pytest.raises(SystemExit):
        select_questions(str(cand_file), str(out_file), "config/question_dataset_config.yaml", seed=42)

def test_impossible_distribution(tmp_path):
    # Give it 60 candidates but all are of one type (factual_lookup), which violates the type_targets for other types
    candidates = []
    for i in range(60):
        q = get_valid_question()
        q["question_id"] = f"C{i+1:03d}"
        q["question_type"] = "factual_lookup"
        q["difficulty"] = "easy"
        candidates.append(q)
        
    cand_file = tmp_path / "candidates.jsonl"
    out_file = tmp_path / "out.jsonl"
    with open(cand_file, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    with pytest.raises(SystemExit):
        select_questions(str(cand_file), str(out_file), "config/question_dataset_config.yaml", seed=42)

def test_successful_deterministic_selection(tmp_path):
    config_path = "config/question_dataset_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Generate a perfect candidate pool that exactly matches the type x difficulty targets
    # We will generate slightly more than needed per cell
    candidates = []
    c_idx = 1
    
    # We need chapters to sum exactly to targets too, so we cycle chapters evenly
    chap_cycle = [1,2,3,4,5]
    chap_idx = 0
    
    for typ, diffs in config["type_difficulty_targets"].items():
        for diff, k in diffs.items():
            # generate k + 1 candidates for this bucket
            for i in range(k + 1):
                q = get_valid_question()
                q["question_id"] = f"C{c_idx:03d}"
                q["question_type"] = typ
                q["difficulty"] = diff
                q["chapter"] = chap_cycle[chap_idx % len(chap_cycle)]
                
                # Assign valid section
                chap_sections = list(VALID_SECTIONS[q["chapter"]].keys())
                sec = chap_sections[0]
                q["section"] = sec
                q["section_title"] = VALID_SECTIONS[q["chapter"]][sec]
                
                candidates.append(q)
                c_idx += 1
                chap_idx += 1
                
    cand_file = tmp_path / "candidates.jsonl"
    out_file1 = tmp_path / "out1.jsonl"
    out_file2 = tmp_path / "out2.jsonl"
    
    with open(cand_file, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    mtime_before = os.path.getmtime(cand_file)
            
    select_questions(str(cand_file), str(out_file1), config_path, seed=42)
    select_questions(str(cand_file), str(out_file2), config_path, seed=42)
    
    assert os.path.getmtime(cand_file) == mtime_before
    
    out1_text = out_file1.read_text()
    out2_text = out_file2.read_text()
    assert out1_text == out2_text # Determinism
    
    selected = [json.loads(line) for line in out1_text.strip().split("\n")]
    assert len(selected) == 50
    
    # Check Q assignment and sorting
    for i, q in enumerate(selected):
        assert q["question_id"] == f"Q{i+1:03d}"
        if i > 0:
            assert q["chapter"] >= selected[i-1]["chapter"]
            
    # Check type x difficulty targets were met
    td_counts = defaultdict(lambda: defaultdict(int))
    for q in selected:
        td_counts[q["question_type"]][q["difficulty"]] += 1
        
    for typ, diffs in config["type_difficulty_targets"].items():
        for diff, k in diffs.items():
            assert td_counts[typ][diff] == k
            
    # Check chapter ranges were met
    chap_counts = defaultdict(int)
    for q in selected:
        chap_counts[q["chapter"]] += 1
        
    for chap, target in config["chapter_targets"].items():
        assert target["range"][0] <= chap_counts[chap] <= target["range"][1]

def test_successful_relaxed_selection(tmp_path):
    config_path = "config/question_dataset_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    candidates = []
    c_idx = 1
    chap_cycle = [1,2,3,4,5]
    chap_idx = 0
    
    cell_keys = []
    for typ, diffs in config["type_difficulty_targets"].items():
        for diff, k in diffs.items():
            if k > 0:
                cell_keys.append((typ, diff, k))
                
    # Modify targets for the candidate pool: cell 0 gets k-1, cell 1 gets k+1
    typ0, diff0, k0 = cell_keys[0]
    typ1, diff1, k1 = cell_keys[1]
    
    for typ, diffs in config["type_difficulty_targets"].items():
        for diff, k in diffs.items():
            actual_k = k
            if (typ, diff) == (typ0, diff0):
                actual_k -= 1
            elif (typ, diff) == (typ1, diff1):
                actual_k += 1
                
            for i in range(actual_k):
                q = get_valid_question()
                q["question_id"] = f"C{c_idx:03d}"
                q["question_type"] = typ
                q["difficulty"] = diff
                q["chapter"] = chap_cycle[chap_idx % len(chap_cycle)]
                
                chap_sections = list(VALID_SECTIONS[q["chapter"]].keys())
                sec = chap_sections[0]
                q["section"] = sec
                q["section_title"] = VALID_SECTIONS[q["chapter"]][sec]
                
                candidates.append(q)
                c_idx += 1
                chap_idx += 1
                
    cand_file = tmp_path / "candidates.jsonl"
    out_file = tmp_path / "out.jsonl"
    
    with open(cand_file, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    # Selection should succeed using the fallback
    select_questions(str(cand_file), str(out_file), config_path, seed=42)
    
    out_text = out_file.read_text()
    selected = [json.loads(line) for line in out_text.strip().split("\n")]
    assert len(selected) == 50
    
    td_counts = defaultdict(lambda: defaultdict(int))
    for q in selected:
        td_counts[q["question_type"]][q["difficulty"]] += 1
        
    for typ, diffs in config["type_difficulty_targets"].items():
        for diff, k in diffs.items():
            assert abs(td_counts[typ][diff] - k) <= 1

def test_relaxed_selection_fails_if_type_aggregate_violated(tmp_path):
    config_path = "config/question_dataset_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    candidates = []
    c_idx = 1
    
    # We will modify the pool so that factual_lookup is missing 3 candidates (falling below 8)
    # and another type gets +3. Since relaxed selection only allows +-1 per cell, it will try
    # to pull from other factual_lookup cells to make up the difference, but it won't be able
    # to reach the aggregate minimum of 8 if it's strictly enforced.
    
    for typ, diffs in config["type_difficulty_targets"].items():
        for diff, k in diffs.items():
            actual_k = k
            if typ == "factual_lookup" and diff == "easy":
                actual_k -= 3
            elif typ == "definition" and diff == "easy":
                actual_k += 3
                
            for i in range(actual_k):
                q = get_valid_question()
                q["question_id"] = f"C{c_idx:03d}"
                q["question_type"] = typ
                q["difficulty"] = diff
                q["chapter"] = 1
                q["section"] = "1.1"
                q["section_title"] = VALID_SECTIONS[1]["1.1"]
                candidates.append(q)
                c_idx += 1
                
    cand_file = tmp_path / "candidates.jsonl"
    out_file = tmp_path / "out.jsonl"
    
    with open(cand_file, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    with pytest.raises(SystemExit):
        select_questions(str(cand_file), str(out_file), config_path, seed=42)

def test_relaxed_selection_fails_if_difficulty_aggregate_violated(tmp_path):
    config_path = "config/question_dataset_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    candidates = []
    c_idx = 1
    
    # Modify pool so that "easy" is missing 3 candidates
    for typ, diffs in config["type_difficulty_targets"].items():
        for diff, k in diffs.items():
            actual_k = k
            if diff == "easy" and k > 0:
                actual_k -= 1
            if diff == "medium" and k > 0:
                actual_k += 1
                
            for i in range(actual_k):
                q = get_valid_question()
                q["question_id"] = f"C{c_idx:03d}"
                q["question_type"] = typ
                q["difficulty"] = diff
                q["chapter"] = 1
                q["section"] = "1.1"
                q["section_title"] = VALID_SECTIONS[1]["1.1"]
                candidates.append(q)
                c_idx += 1
                
    cand_file = tmp_path / "candidates.jsonl"
    out_file = tmp_path / "out.jsonl"
    
    with open(cand_file, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    with pytest.raises(SystemExit):
        select_questions(str(cand_file), str(out_file), config_path, seed=42)

def test_target_count_not_hardcoded(tmp_path):
    config_path = "config/question_dataset_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config["target_count"] = 2
    config["chapter_targets"] = {
        1: {"target": 2, "range": [2, 2]},
        2: {"target": 0, "range": [0, 0]},
        3: {"target": 0, "range": [0, 0]},
        4: {"target": 0, "range": [0, 0]},
        5: {"target": 0, "range": [0, 0]}
    }
    config["type_targets"] = {
        "factual_lookup": {"target": 2, "range": [2, 2]},
        "definition": {"target": 0, "range": [0, 0]},
        "explanation_conceptual": {"target": 0, "range": [0, 0]},
        "relationship_comparison": {"target": 0, "range": [0, 0]},
        "application_inference": {"target": 0, "range": [0, 0]}
    }
    config["difficulty_targets"] = {
        "easy": {"target": 2, "range": [2, 2]},
        "medium": {"target": 0, "range": [0, 0]},
        "hard": {"target": 0, "range": [0, 0]}
    }
    config["type_difficulty_targets"] = {
        "factual_lookup": {"easy": 2, "medium": 0, "hard": 0},
        "definition": {"easy": 0, "medium": 0, "hard": 0},
        "explanation_conceptual": {"easy": 0, "medium": 0, "hard": 0},
        "relationship_comparison": {"easy": 0, "medium": 0, "hard": 0},
        "application_inference": {"easy": 0, "medium": 0, "hard": 0}
    }
    
    custom_config_path = tmp_path / "custom_config.yaml"
    with open(custom_config_path, "w") as f:
        yaml.dump(config, f)
        
    candidates = []
    for i in range(2):
        q = get_valid_question()
        q["question_id"] = f"C{i+1:03d}"
        q["chapter"] = 1
        q["section"] = "1.1"
        q["section_title"] = VALID_SECTIONS[1]["1.1"]
        q["question_type"] = "factual_lookup"
        q["difficulty"] = "easy"
        candidates.append(q)
        
    cand_file = tmp_path / "candidates.jsonl"
    out_file = tmp_path / "out.jsonl"
    
    with open(cand_file, "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    select_questions(str(cand_file), str(out_file), str(custom_config_path), seed=42)
    
    out_text = out_file.read_text()
    selected = [json.loads(line) for line in out_text.strip().split("\n")]
    assert len(selected) == 2

def generate_perfect_dataset(tmp_path):
    config_path = "config/question_dataset_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config["minimum_section_coverage_per_chapter"] = 1
    custom_config = tmp_path / "custom_config.yaml"
    with open(custom_config, "w") as f:
        yaml.dump(config, f)
    
    questions = []
    c_idx = 1
    chap_cycle = [1,2,3,4,5]
    chap_idx = 0
    
    for typ, diffs in config["type_difficulty_targets"].items():
        for diff, k in diffs.items():
            for i in range(k):
                q = get_valid_question()
                q["question_id"] = f"Q{c_idx:03d}"
                q["question_type"] = typ
                q["difficulty"] = diff
                q["chapter"] = chap_cycle[chap_idx % len(chap_cycle)]
                
                chap_sections = list(VALID_SECTIONS[q["chapter"]].keys())
                sec_idx = c_idx % len(chap_sections)
                sec = chap_sections[sec_idx]
                q["section"] = sec
                q["section_title"] = VALID_SECTIONS[q["chapter"]][sec]
                
                questions.append(q)
                c_idx += 1
                chap_idx += 1
    return questions, str(custom_config)

def test_type_count_above_maximum_fails(tmp_path):
    questions, custom_config_path = generate_perfect_dataset(tmp_path)
    # factual_lookup target is 10, max is 12.
    # We change one question to factual_lookup/easy (target 8 -> 9)
    # one to factual_lookup/medium (target 2 -> 3)
    # one to factual_lookup/hard (target 0 -> 1)
    # Now factual_lookup is 13 (fails aggregate max 12).
    # All cell changes are +1, which is valid individually.
    # We take from 'definition/easy' (5->4), 'explanation_conceptual/medium' (8->7), 'relationship_comparison/hard' (6->5)
    # These cell changes are -1, which is valid.
    
    changes_needed = [
        ("definition", "easy", "factual_lookup", "easy"),
        ("explanation_conceptual", "medium", "factual_lookup", "medium"),
        ("relationship_comparison", "hard", "factual_lookup", "hard")
    ]
    
    for (src_typ, src_diff, dst_typ, dst_diff) in changes_needed:
        for q in questions:
            if q["question_type"] == src_typ and q["difficulty"] == src_diff:
                q["question_type"] = dst_typ
                q["difficulty"] = dst_diff
                break
                
    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
            
    assert not validate_dataset(str(jsonl), "data/questions/questions_schema.json", custom_config_path)

def test_difficulty_count_above_maximum_fails(tmp_path):
    questions, custom_config_path = generate_perfect_dataset(tmp_path)
    # 'easy' target is 15, max is 17. We want to increase it by 3 to 18.
    # We change:
    # definition/medium -> definition/easy (easy 5 -> 6, medium 5 -> 4)
    # explanation_conceptual/medium -> explanation_conceptual/easy (easy 2 -> 3, medium 8 -> 7)
    # relationship_comparison/medium -> relationship_comparison/easy (easy 0 -> 1, medium 4 -> 3)
    # Each cell changes by +-1 (valid).
    # easy aggregate becomes 18 (fails max 17).
    # medium aggregate becomes 20-3=17 (valid, range 18-22? Wait, min medium is 18, so 17 fails medium too.
    # We just need it to fail validation).
    
    changes_needed = [
        ("definition", "medium", "definition", "easy"),
        ("explanation_conceptual", "medium", "explanation_conceptual", "easy"),
        ("relationship_comparison", "medium", "relationship_comparison", "easy")
    ]
    
    for (src_typ, src_diff, dst_typ, dst_diff) in changes_needed:
        for q in questions:
            if q["question_type"] == src_typ and q["difficulty"] == src_diff:
                q["question_type"] = dst_typ
                q["difficulty"] = dst_diff
                break
                
    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
            
    assert not validate_dataset(str(jsonl), "data/questions/questions_schema.json", custom_config_path)

def test_invalid_type_difficulty_cell_fails(tmp_path):
    questions, custom_config_path = generate_perfect_dataset(tmp_path)
    # factual_lookup/hard target is 0. A change of +1 is valid (lower_bound=0, upper_bound=1).
    # A change of +2 makes it 2, which is invalid.
    changed = 0
    for q in questions:
        if q["question_type"] == "factual_lookup" and q["difficulty"] == "easy":
            q["difficulty"] = "hard"
            changed += 1
            if changed == 2:
                break
                
    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
            
    assert not validate_dataset(str(jsonl), "data/questions/questions_schema.json", custom_config_path)

def test_valid_plus_minus_one_type_difficulty_cell_passes(tmp_path):
    questions, custom_config_path = generate_perfect_dataset(tmp_path)
    # Change one factual_lookup/easy (target 8 -> 7) to factual_lookup/hard (target 0 -> 1)
    # The aggregate factual_lookup is unchanged.
    # The aggregate easy drops by 1 (15 -> 14, min 13), hard increases by 1 (15 -> 16, max 17).
    changed = 0
    for q in questions:
        if q["question_type"] == "factual_lookup" and q["difficulty"] == "easy":
            q["difficulty"] = "hard"
            changed += 1
            if changed == 1:
                break
                
    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
            
    assert validate_dataset(str(jsonl), "data/questions/questions_schema.json", custom_config_path)

def test_dataset_below_acceptable_range_fails(tmp_path):
    questions, custom_config_path = generate_perfect_dataset(tmp_path)
    # Remove 6 questions, leaving 44 (below 45)
    questions = questions[:-6]
    
    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
            
    assert not validate_dataset(str(jsonl), "data/questions/questions_schema.json", custom_config_path)

def test_dataset_above_acceptable_range_fails(tmp_path):
    questions, custom_config_path = generate_perfect_dataset(tmp_path)
    # Duplicate 11 questions, making 61 (above 60)
    for i in range(11):
        q = questions[i].copy()
        q["question_id"] = f"Q9{i:02d}"
        questions.append(q)
        
    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
            
    assert not validate_dataset(str(jsonl), "data/questions/questions_schema.json", custom_config_path)
