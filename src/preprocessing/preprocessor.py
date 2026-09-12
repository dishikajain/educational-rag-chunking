import re
import uuid
import statistics
from typing import List, Tuple, Set, Optional
from src.extraction.models import ExtractedDocument, ExtractedPage, TextBlock
from src.preprocessing.config import PreprocessingConfig
from src.preprocessing.models import (
    PreprocessedDocument,
    PreprocessedPage,
    PreprocessedBlock,
    ProvenanceSpan,
)


def _establish_baselines(doc: ExtractedDocument, config: PreprocessingConfig) -> Tuple[float, dict]:
    """Compute document-level body size baseline and per-page baselines."""
    candidate_sizes = [
        b.font_size
        for page in doc.pages
        for b in page.blocks
        if b.font_size is not None and b.font_size > config.min_font_size_for_baseline
    ]
    if candidate_sizes:
        try:
            doc_body_size = float(statistics.mode(candidate_sizes))
        except statistics.StatisticsError:
            doc_body_size = float(statistics.median(candidate_sizes))
    else:
        doc_body_size = config.body_size_fallback

    page_baselines = {}
    for page in doc.pages:
        page_candidates = [
            b.font_size
            for b in page.blocks
            if b.font_size is not None and b.font_size > config.min_font_size_for_baseline
        ]
        if len(page_candidates) >= config.min_blocks_for_page_baseline:
            try:
                p_size = float(statistics.mode(page_candidates))
            except statistics.StatisticsError:
                p_size = float(statistics.median(page_candidates))
            page_baselines[page.page_number] = p_size
        else:
            page_baselines[page.page_number] = doc_body_size

    return doc_body_size, page_baselines


def _classify_block(block: TextBlock, body_size: float, config: PreprocessingConfig) -> Tuple[str, bool]:
    """Classify block typography label and determine bold status."""
    font_name = block.font_name or ""
    is_bold = any(ind.lower() in font_name.lower() for ind in config.bold_indicator_strings)

    if block.font_size is None:
        return ("BODY_BOLD" if is_bold else "BODY"), is_bold

    ratio = block.font_size / body_size
    r_thresh = config.ratio_thresholds  # [0.5, 0.9, 1.1, 1.35, 1.6, 2.0, 3.5]

    if ratio < r_thresh[0]:  # < 0.5
        label = "DISCARD"
    elif ratio < r_thresh[1]:  # < 0.9
        label = "CAPTION"
    elif ratio < r_thresh[2]:  # < 1.1
        label = "BODY_BOLD" if is_bold else "BODY"
    elif ratio < r_thresh[3]:  # < 1.35
        label = "CALLOUT_LABEL"
    elif ratio < r_thresh[4]:  # < 1.6
        label = "HEADING_3"
    elif ratio < r_thresh[5]:  # < 2.0
        label = "HEADING_2"
    elif ratio < r_thresh[6]:  # < 3.5
        label = "HEADING_1"
    else:  # >= 3.5
        text_stripped = block.text.strip()
        is_numeral = bool(re.fullmatch(r"\d{1,3}", text_stripped))
        is_high_ratio = ratio >= config.chapter_number_ratio_threshold
        if is_numeral or is_high_ratio:
            label = "CHAPTER_NUMBER"
        else:
            label = "CHAPTER_TITLE"

    return label, is_bold


def preprocess(doc: ExtractedDocument, config: PreprocessingConfig) -> PreprocessedDocument:
    """Preprocess ExtractedDocument according to Phase 3 v3 frozen architecture specification."""
    doc_body_size, page_baselines = _establish_baselines(doc, config)
    printable_width = config.page_width - config.left_margin - config.right_margin

    preprocessed_pages: List[PreprocessedPage] = []
    all_output_blocks: List[PreprocessedBlock] = []

    for page in doc.pages:
        body_size = page_baselines.get(page.page_number, doc_body_size)
        raw_blocks = page.blocks

        # 1. Classify blocks
        classified_blocks: List[dict] = []
        for p2_idx, b in enumerate(raw_blocks):
            label, is_bold = _classify_block(b, body_size, config)
            classified_blocks.append({
                "p2_idx": p2_idx,
                "block": b,
                "label": label,
                "is_bold": is_bold,
                "sidebar_unit_id": None,
                "sidebar_type": None,
            })

        # 2. Detect sidebars & full-width spans
        current_sidebar_id = None
        current_sidebar_type = None

        for item in classified_blocks:
            b = item["block"]
            b_width = b.x1 - b.x0
            is_full_width = b_width >= (config.full_width_threshold * printable_width)
            norm_text = " ".join(b.text.upper().split())
            is_known_label = norm_text in config.sidebar_labels

            if is_known_label:
                item["label"] = "SIDEBAR_HEADER"
                current_sidebar_id = f"sidebar_p{page.page_number}_b{item['p2_idx']}"
                current_sidebar_type = norm_text
                item["sidebar_unit_id"] = current_sidebar_id
                item["sidebar_type"] = current_sidebar_type
            elif current_sidebar_id is not None and is_full_width:
                item["label"] = "SIDEBAR_BODY"
                item["sidebar_unit_id"] = current_sidebar_id
                item["sidebar_type"] = current_sidebar_type
            else:
                current_sidebar_id = None
                current_sidebar_type = None
                if is_full_width:
                    if item["label"] in ["HEADING_1", "CHAPTER_TITLE", "CHAPTER_NUMBER"]:
                        item["label"] = "CHAPTER_INTRO_SPAN"
                    elif item["label"] not in config.sidebar_labels and item["label"] not in ["BODY", "BODY_BOLD", "HEADING_2", "HEADING_3", "CAPTION"]:
                        item["label"] = "UNKNOWN_SPAN"

        # 3. Group sidebar units if present
        grouped_items: List[dict] = []
        i = 0
        while i < len(classified_blocks):
            item = classified_blocks[i]
            if item["sidebar_unit_id"] is not None:
                sb_id = item["sidebar_unit_id"]
                sb_type = item["sidebar_type"]
                sb_group = []
                while i < len(classified_blocks) and classified_blocks[i]["sidebar_unit_id"] == sb_id:
                    sb_group.append(classified_blocks[i])
                    i += 1
                
                # Combine constituent blocks into a single SidebarUnit item
                first_item = sb_group[0]
                combined_text = "\n".join(g["block"].text for g in sb_group)
                min_x0 = min(g["block"].x0 for g in sb_group)
                min_y0 = min(g["block"].y0 for g in sb_group)
                max_x1 = max(g["block"].x1 for g in sb_group)
                max_y1 = max(g["block"].y1 for g in sb_group)
                source_blocks = [(page.page_number, g["p2_idx"]) for g in sb_group]

                grouped_items.append({
                    "p2_idx": first_item["p2_idx"],
                    "block": TextBlock(
                        text=combined_text,
                        x0=min_x0,
                        y0=min_y0,
                        x1=max_x1,
                        y1=max_y1,
                        font_size=first_item["block"].font_size,
                        font_name=first_item["block"].font_name,
                    ),
                    "label": "SIDEBAR_BODY",
                    "is_bold": first_item["is_bold"],
                    "sidebar_unit_id": sb_id,
                    "sidebar_type": sb_type,
                    "source_blocks": source_blocks,
                })
            else:
                item["source_blocks"] = [(page.page_number, item["p2_idx"])]
                grouped_items.append(item)
                i += 1

        # 4. Remove headers, footers, and discards (before reading order)
        filtered_items: List[dict] = []
        total_grouped = len(grouped_items)

        for item in grouped_items:
            label = item["label"]
            b = item["block"]

            if label == "DISCARD":
                continue

            # Safety guard check
            ratio = (b.font_size / body_size) if b.font_size is not None else 1.0
            is_large_heading = ratio >= config.ratio_thresholds[5]  # >= 1.6 (HEADING_2 or larger)
            is_only_block = (total_grouped == 1)

            # Header check
            if b.y0 < config.header_zone_height:
                if not (is_only_block or is_large_heading):
                    continue  # Remove HEADER

            # Footer / Page Number check
            if b.y1 > (config.page_height - config.footer_zone_height):
                if not (is_only_block or is_large_heading):
                    continue  # Remove FOOTER / PAGE_NUMBER

            filtered_items.append(item)

        if not filtered_items:
            preprocessed_pages.append(PreprocessedPage(
                page_number=page.page_number,
                page_type="SINGLE_COLUMN",
                blocks=[],
            ))
            continue

        # 5. Determine zone per remaining block
        for item in filtered_items:
            b = item["block"]
            b_width = b.x1 - b.x0
            midpoint_x = (b.x0 + b.x1) / 2.0
            is_full = (b_width >= config.full_width_threshold * printable_width) or (item["sidebar_unit_id"] is not None) or (item["label"] == "CHAPTER_INTRO_SPAN")

            if is_full:
                item["zone"] = "FULL_WIDTH"
            elif midpoint_x < config.column_split_x:
                item["zone"] = "LEFT"
            else:
                item["zone"] = "RIGHT"

        # 6. Reconstruct reading order
        left_count = sum(1 for item in filtered_items if item["zone"] == "LEFT")
        right_count = sum(1 for item in filtered_items if item["zone"] == "RIGHT")

        if left_count < config.single_col_block_threshold or right_count < config.single_col_block_threshold:
            page_type = "SINGLE_COLUMN"
            ordered_items = sorted(filtered_items, key=lambda x: (x["block"].y0, x["block"].x0))
        else:
            page_type = "TWO_COLUMN"
            full_width_items = sorted([it for it in filtered_items if it["zone"] == "FULL_WIDTH"], key=lambda x: (x["block"].y0, x["block"].x0))
            left_items = sorted([it for it in filtered_items if it["zone"] == "LEFT"], key=lambda x: (x["block"].y0, x["block"].x0))
            right_items = sorted([it for it in filtered_items if it["zone"] == "RIGHT"], key=lambda x: (x["block"].y0, x["block"].x0))

            ordered_items = []
            li, ri = 0, 0
            for fw in full_width_items:
                while li < len(left_items) and left_items[li]["block"].y0 < fw["block"].y0:
                    ordered_items.append(left_items[li])
                    li += 1
                while ri < len(right_items) and right_items[ri]["block"].y0 < fw["block"].y0:
                    ordered_items.append(right_items[ri])
                    ri += 1
                ordered_items.append(fw)
            ordered_items.extend(left_items[li:])
            ordered_items.extend(right_items[ri:])

        # 7. Compute gaps between consecutive blocks in the same column
        page_pblocks: List[PreprocessedBlock] = []
        for idx, item in enumerate(ordered_items):
            b = item["block"]
            gap_before_pt = None
            is_para_break = None
            is_sec_break = None

            if idx > 0:
                prev_item = ordered_items[idx - 1]
                prev_b = prev_item["block"]
                if item["zone"] == prev_item["zone"]:
                    gap = b.y0 - prev_b.y1
                    gap_before_pt = gap
                    if gap < config.gap_line_pt * config.line_continuation_factor:
                        is_para_break = False
                        is_sec_break = False
                    elif gap < config.gap_para_pt * config.para_break_factor:
                        is_para_break = True
                        is_sec_break = False
                    else:
                        is_para_break = True
                        is_sec_break = True

            pblock = PreprocessedBlock(
                source_page=page.page_number,
                p2_block_index=item["p2_idx"],
                reading_order_index=idx,
                text=b.text,
                x0=b.x0,
                y0=b.y0,
                x1=b.x1,
                y1=b.y1,
                label=item["label"],
                font_size=b.font_size,
                font_name=b.font_name,
                is_bold=item["is_bold"],
                zone=item["zone"],
                page_type=page_type,
                gap_before_pt=gap_before_pt,
                is_paragraph_break=is_para_break,
                is_section_break=is_sec_break,
                sidebar_unit_id=item["sidebar_unit_id"],
                sidebar_type=item["sidebar_type"],
                source_blocks=item["source_blocks"],
            )
            page_pblocks.append(pblock)
            all_output_blocks.append(pblock)

        preprocessed_pages.append(PreprocessedPage(
            page_number=page.page_number,
            page_type=page_type,
            blocks=page_pblocks,
        ))

    # 8. Text normalization & De-hyphenation across all blocks
    dehyphenated_boundary_indices: Set[int] = set()

    for i in range(len(all_output_blocks) - 1):
        b_i = all_output_blocks[i]
        b_next = all_output_blocks[i + 1]

        # De-hyphenation operates within the same page only
        if b_i.source_page != b_next.source_page:
            continue

        # Trigger check:
        # (a) b_i.text ends with letter followed by "-"
        # (b) b_next.text starts with lowercase letter
        # (c) candidate joined word contains no additional hyphens
        m_curr = re.search(r'([a-zA-Z]+)-$', b_i.text)
        m_next = re.match(r'^([a-z]+)', b_next.text)

        if m_curr and m_next:
            w1 = m_curr.group(1)
            w2 = m_next.group(1)
            candidate_word = w1 + w2
            if "-" not in candidate_word:
                # Remove trailing "-" from b_i.text ONLY
                b_i.text = b_i.text[:-1]
                # b_next.text is LEFT UNCHANGED
                dehyphenated_boundary_indices.add(i)

    # Whitespace collapse within each block (preserving newlines for SidebarUnits)
    for b in all_output_blocks:
        lines = [re.sub(r"[ \t\r]+", " ", line).strip() for line in b.text.split("\n")]
        b.text = "\n".join(line for line in lines if line)

    # 9. Build provenance and full_text (Section 9.5)
    DEFAULT_SEP = "\n\n"
    separators = [DEFAULT_SEP] * len(all_output_blocks)
    if all_output_blocks:
        separators[-1] = ""

    for i in dehyphenated_boundary_indices:
        separators[i] = ""

    full_text = "".join(b.text + sep for b, sep in zip(all_output_blocks, separators))

    provenance: List[ProvenanceSpan] = []
    char_pos = 0
    for b, sep in zip(all_output_blocks, separators):
        span_start = char_pos
        span_end = char_pos + len(b.text) + len(sep)
        provenance.append(ProvenanceSpan(
            char_start=span_start,
            char_end=span_end,
            source_blocks=b.source_blocks or [(b.source_page, b.p2_block_index)],
            label=b.label,
        ))
        char_pos = span_end

    # Postcondition verification for Invariants A-D
    if provenance:
        assert char_pos == len(full_text), f"Invariant C failed: char_pos {char_pos} != len(full_text) {len(full_text)}"
        assert provenance[0].char_start == 0, f"Invariant B failed: start {provenance[0].char_start} != 0"
        assert provenance[-1].char_end == len(full_text), f"Invariant C failed: end {provenance[-1].char_end} != len(full_text)"
        for idx in range(len(provenance) - 1):
            assert provenance[idx].char_end == provenance[idx + 1].char_start, f"Invariant A failed at index {idx}"
        reconstructed = "".join(full_text[s.char_start:s.char_end] for s in provenance)
        assert reconstructed == full_text, "Invariant D failed: reconstructed full_text does not match"

    return PreprocessedDocument(
        doc_id=doc.doc_id,
        pages=preprocessed_pages,
        full_text=full_text,
        provenance=provenance,
        config_used=config,
    )
