import os
import yaml
from dataclasses import dataclass
from typing import List, Set, Union

@dataclass
class PreprocessingConfig:
    corpus_name: str
    corpus_chapters: List[int]
    min_font_size_for_baseline: float
    baseline_method: str
    body_size_fallback: float
    min_blocks_for_page_baseline: int
    bold_indicator_strings: List[str]
    ratio_thresholds: List[float]
    chapter_number_ratio_threshold: float
    page_width: float
    page_height: float
    left_margin: float
    right_margin: float
    column_split_x: float
    left_col_x0: float
    left_col_x1: float
    right_col_x0: float
    right_col_x1: float
    full_width_threshold: float
    single_col_block_threshold: int
    sidebar_labels: Set[str]
    header_zone_height: float
    footer_zone_height: float
    tolerance_pt: float
    page_number_pattern: str
    gap_line_pt: float
    gap_para_pt: float
    gap_section_pt: float
    line_continuation_factor: float
    para_break_factor: float


def load_config(config_path: str) -> PreprocessingConfig:
    """Load PreprocessingConfig from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sidebar_labels = set(data.get("sidebar_labels", []))

    return PreprocessingConfig(
        corpus_name=str(data["corpus_name"]),
        corpus_chapters=list(data["corpus_chapters"]),
        min_font_size_for_baseline=float(data["min_font_size_for_baseline"]),
        baseline_method=str(data["baseline_method"]),
        body_size_fallback=float(data["body_size_fallback"]),
        min_blocks_for_page_baseline=int(data["min_blocks_for_page_baseline"]),
        bold_indicator_strings=list(data["bold_indicator_strings"]),
        ratio_thresholds=[float(r) for r in data["ratio_thresholds"]],
        chapter_number_ratio_threshold=float(data["chapter_number_ratio_threshold"]),
        page_width=float(data["page_width"]),
        page_height=float(data["page_height"]),
        left_margin=float(data["left_margin"]),
        right_margin=float(data["right_margin"]),
        column_split_x=float(data["column_split_x"]),
        left_col_x0=float(data["left_col_x0"]),
        left_col_x1=float(data["left_col_x1"]),
        right_col_x0=float(data["right_col_x0"]),
        right_col_x1=float(data["right_col_x1"]),
        full_width_threshold=float(data["full_width_threshold"]),
        single_col_block_threshold=int(data["single_col_block_threshold"]),
        sidebar_labels=sidebar_labels,
        header_zone_height=float(data["header_zone_height"]),
        footer_zone_height=float(data["footer_zone_height"]),
        tolerance_pt=float(data["tolerance_pt"]),
        page_number_pattern=str(data["page_number_pattern"]),
        gap_line_pt=float(data["gap_line_pt"]),
        gap_para_pt=float(data["gap_para_pt"]),
        gap_section_pt=float(data["gap_section_pt"]),
        line_continuation_factor=float(data["line_continuation_factor"]),
        para_break_factor=float(data["para_break_factor"]),
    )
