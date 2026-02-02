#!/usr/bin/env python3
"""
Generate LaTeX table from bar_plot_v11.py data processing logic.
Table structure:
- Training Strategy: Low/High NCB with different percentages (5%, 20%, 35%)
- Columns: Number, Vanilla ACC, Quantity-Stressing (ACC1, ACC2), Source-Stressing (ACC1, ACC2)
- Grouped by model: Qwen3-A3B-30B-Instruct-2507, Qwen3-A3B-30B-Thinking-2507
"""

import json
import argparse
import numpy as np
import pandas as pd
import os
import re
from collections import Counter
from typing import List, Any, Dict, Optional, Tuple
from tqdm import tqdm

# Import functions from bar_plot_v11.py
def parse_mode_generic(mode_str):
    """
    Parse mode string into parts.
    Examples:
    - asch_misleading_std -> part1=asch_misleading, part2=default, part3=std
    - asch_conflict_cfg0_std -> part1=asch_conflict, part2=cfg0, part3=std
    - source_misleading_low_std -> part1=source_misleading, part2=low, part3=std
    """
    segments = mode_str.split('_')
    part3 = segments[-1]  # Always the last part (std/cot)
    
    # Known part2 patterns (configurations)
    known_part2_patterns = {
        'cfg0', 'cfg1', 'cfg2', 'cfg3', 'cfg4', 'cfg5', 'cfg6',
        'low', 'medium', 'high',
        'default'
    }
    
    # Check if second-to-last segment looks like a part2 (configuration)
    if len(segments) >= 3:
        potential_part2 = segments[-2]
        # Check if it matches known patterns or looks like a config
        if (potential_part2 in known_part2_patterns or 
            potential_part2.startswith('cfg') or
            potential_part2 in ['low', 'medium', 'high']):
            part2 = potential_part2
            part1_segments = segments[:-2]
        else:
            # No part2, everything before part3 is part1
            part2 = "default"
            part1_segments = segments[:-1]
    else:
        part2 = "default"
        part1_segments = segments[:-1]
    
    part1 = '_'.join(part1_segments)
    
    return {
        'original': mode_str,
        'part1': part1,
        'part2': part2,
        'part3': part3
    }

class EntityNormalizer:
    def __init__(self):
        self.reject_words = {
            "none", "n/a", "unknown", "null", "nil", "not available", "no answer", "i don't know", "not_attempted"
        }

    def _universal_clean(self, text):
        if not text: return None
        if isinstance(text, dict):
            text = text.get('final_answer', text.get('answer', text.get('text', str(text))))
        text = str(text).lower()
        if "\n" in text or "\r" in text:
            text = text.replace("\n", " ").replace("\r", " ")
        if text.strip() in self.reject_words: return None
        pattern = r'\s*\([^()]*\)'
        while re.search(pattern, text):
            text = re.sub(pattern, '', text)
        text = re.sub(r'[^\w\s\.\-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize(self, raw_text):
        if str(raw_text) == "NOT_ATTEMPTED": return "not_attempted"
        text = self._universal_clean(raw_text)
        return text

NORMALIZER = EntityNormalizer()
def normalize_entity(text): return NORMALIZER.normalize(text)

# Global switch: Control the Accuracy matching mode
ACCURACY_MATCH_MODE = "sentence"

def extract_raw_text_list(raw_data: List[Any], stage: str) -> List[str]:
    text_list = []
    if not raw_data:
        return []
    for item in raw_data:
        content = ""
        if isinstance(item, dict):
            if "followup_response_raw" in item:
                content = item["followup_response_raw"]
            elif "followup_parsed" in item:
                parsed = item["followup_parsed"]
                if isinstance(parsed, dict):
                    content = parsed.get('text', str(parsed))
                else:
                    content = str(parsed)
            elif "text" in item:
                content = item["text"]
            elif "final_answer" in item:
                content = item["final_answer"]
            elif "answer" in item:
                content = item["answer"]
            else:
                content = str(item)
        else:
            content = str(item)
        text_list.append(content.lower())
    return text_list

def calculate_confidence(entities):
    valid = [normalize_entity(e) for e in entities if normalize_entity(e)]
    if not valid: return 0.0
    _, count = Counter(valid).most_common(1)[0]
    return count / len(valid)

def calculate_coverage(entities):
    if not entities: return 0.0
    rejected = sum(1 for e in entities if not normalize_entity(e) or normalize_entity(e) == "not_attempted")
    return 1.0 - (rejected / len(entities))

def calculate_loose_accuracy(entities, golden, raw_sentences):
    """
    Determine matching mode based on ACCURACY_MATCH_MODE:
    - "entity":   Only match between entity e and golden
    - "sentence": Only match in the corresponding raw sentence text
    - "both":     First match in entity, then match in sentence if not found
    """
    global ACCURACY_MATCH_MODE

    valid_indices = [i for i, e in enumerate(entities) if normalize_entity(e)]
    if not valid_indices:
        return 0.0
    g_norm = normalize_entity(golden)
    if not g_norm:
        return 0.0

    correct = 0
    for idx in valid_indices:
        e = normalize_entity(entities[idx])
        if not e:
            continue

        match_found = False

        # 1) Entity-level matching
        if ACCURACY_MATCH_MODE in ("entity", "both"):
            if g_norm in e or e in g_norm:
                match_found = True

        # 2) Sentence-level matching (raw_sentences is cleaned & lower by extract_raw_text_list)
        if (not match_found
            and ACCURACY_MATCH_MODE in ("sentence", "both")
            and raw_sentences
            and idx < len(raw_sentences)):
            raw_text = raw_sentences[idx]
            if isinstance(raw_text, dict):
                raw_text = str(raw_text)
            raw_text = str(raw_text).lower()
            if g_norm in raw_text or e in g_norm:
                match_found = True

        if match_found:
            correct += 1

    return correct / len(valid_indices)

def get_metrics_for_stage(item, mode_full, stage, level=None) -> Optional[Dict[str, float]]:
    if stage == "Initial":
        entity_key = f"extracted_entities_{mode_full}"
        response_key = f"resp_{mode_full}"
    else:
        suffix = f"_lvl{level}"
        # Try multiple possible key formats for followup data
        possible_entity_keys = [
            f"extracted_followup_entities_{mode_full}{suffix}",
            f"extracted_followup_entities_followup_{mode_full}{suffix}",
        ]
        possible_response_keys = [
            f"followup_{mode_full}{suffix}",
            f"followup_followup_{mode_full}{suffix}",
        ]
        
        # Find the first existing key
        entity_key = None
        for key in possible_entity_keys:
            if key in item:
                entity_key = key
                break
        if not entity_key:
            entity_key = possible_entity_keys[0]  # Default to first format
        
        response_key = None
        for key in possible_response_keys:
            if key in item:
                response_key = key
                break
        if not response_key:
            response_key = possible_response_keys[0]  # Default to first format

    raw_responses = item.get(response_key, [])
    extracted_entities = item.get(entity_key, [])
    
    # Check if entities exist and are not all None/empty
    if not extracted_entities:
        return None
    
    # Check if there are any valid (non-None, non-empty) entities
    valid_entities = [e for e in extracted_entities if normalize_entity(e)]
    if not valid_entities:
        return None
    
    golden = item.get("original_answer")
    if not golden:
        return None
    raw_text_list = extract_raw_text_list(raw_responses, stage)
    
    return {
        "Accuracy": calculate_loose_accuracy(extracted_entities, golden, raw_text_list),
        "Coverage": calculate_coverage(extracted_entities),
    }

def get_baseline_metrics(item):
    try:
        golden = item.get("original_answer")
        if not golden: return None
        baseline_entities = item.get("oq_entities", [])
        baseline_responses = item.get("oq_responses", [])
        if not baseline_entities:
            return {
                "Accuracy": 0.0, 
                "Coverage": 0.0,
                "Confidence": 0.0
            }
        
        baseline_text_list = extract_raw_text_list(baseline_responses, "Baseline")
        return {
            "Accuracy": calculate_loose_accuracy(baseline_entities, golden, baseline_text_list),
            "Coverage": calculate_coverage(baseline_entities),
            "Confidence": calculate_confidence(baseline_entities)
        }
    except:
        return None

def get_grouping_logic(df, quantile):
    """Group data by quantile (Bottom/Top)"""
    scores = df['Score']
    lower_threshold = scores.quantile(quantile)
    upper_threshold = scores.quantile(1.0 - quantile)
    
    def get_group(score):
        if score <= lower_threshold:
            return f"Bottom {int(quantile*100)}%"
        elif score >= upper_threshold:
            return f"Top {int(quantile*100)}%"
        else:
            return "Middle"
    
    return df['Score'].apply(get_group)

def extract_model_name(input_file: str) -> str:
    """
    Extract model name from input file path or filename.
    Examples:
    - Qwen3-30B-A3B-Instruct-2507_fact_belief_2000_nq -> Qwen3-A3B-30B-Instruct-2507
    - Qwen3-30B-A3B-Thinking-2507_fact_belief_2000_nq -> Qwen3-A3B-30B-Thinking-2507
    """
    filename = os.path.basename(input_file)
    full_path = input_file.lower()
    
    # Check for Instruct model
    if "instruct" in filename.lower() or "inst" in filename.lower():
        return "Qwen3-A3B-30B-Instruct-2507"
    # Check for Thinking model
    elif "thinking" in filename.lower() or "think" in filename.lower():
        return "Qwen3-A3B-30B-Thinking-2507"
    else:
        # Fallback: try to extract from path
        path_parts = full_path.split('/')
        for part in path_parts:
            if ("inst" in part or "instruct" in part) and "qwen" in part:
                return "Qwen3-A3B-30B-Instruct-2507"
            elif ("think" in part or "thinking" in part) and "qwen" in part:
                return "Qwen3-A3B-30B-Thinking-2507"
        
        # Last resort: check if user provided model name via argument
        # (This would require modifying main() to accept model_name argument)
        print(f"  WARNING: Could not determine model name from {input_file}")
        print(f"  Please specify model name manually or check file path")
        return "Unknown-Model"

def calculate_overall_statistics(df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    """
    Calculate the mean of all metrics for all samples, regardless of quantile.
    """
    # Only keep Accuracy
    df_acc = df[df['Metric'] == 'Accuracy'].copy()
    if df_acc.empty:
        print(f"WARNING: No Accuracy data found for {model_name}")
        return None

    # Initial and Followup
    df_initial = df_acc[df_acc['Stage'] == 'Initial']
    df_followup = None
    followup_stage_name = None
    for lvl in [2, 1, 3]:
        stage_name = f'Followup_Lvl{lvl}'
        df_check = df_acc[df_acc['Stage'] == stage_name]
        if not df_check.empty:
            df_followup = df_check
            followup_stage_name = stage_name
            break
    if df_followup is None:
        df_followup = pd.DataFrame()

    # Configuration selection is the same as the original logic
    asch_experiments = ['asch_conflict', 'asch_misleading']
    source_experiments = ['source_conflict', 'source_misleading']

    asch_part2_list = sorted(df_initial[df_initial['part1'].isin(asch_experiments)]['part2'].unique())
    source_part2_list = sorted(df_initial[df_initial['part1'].isin(source_experiments)]['part2'].unique())

    asch_cfg = None
    if asch_part2_list:
        cfg_list = [p for p in asch_part2_list if p.startswith('cfg')]
        if cfg_list:
            cfg_list_sorted = sorted(cfg_list, key=lambda x: int(x.replace('cfg', '')) if x.replace('cfg', '').isdigit() else 999)
            asch_cfg = cfg_list_sorted[-1]
    source_cfg = None
    if source_part2_list:
        if 'high' in source_part2_list:
            source_cfg = 'high'
        elif source_part2_list:
            source_cfg = source_part2_list[-1]

    # Number of samples
    count = len(df_initial['SampleID'].unique())

    # Vanilla ACC
    vanilla_acc = df_initial['Baseline'].mean()

    # Quantity-Stressing ACC
    quantity_df_list = []
    if asch_cfg:
        asch_conflict_df = df_initial[
            (df_initial['part1'] == 'asch_conflict') &
            (df_initial['part2'] == asch_cfg)
        ]
        quantity_df_list.append(asch_conflict_df)
    asch_misleading_df = df_initial[df_initial['part1'] == 'asch_misleading']
    quantity_df_list.append(asch_misleading_df)
    if quantity_df_list:
        quantity_df = pd.concat(quantity_df_list, ignore_index=True)
    else:
        quantity_df = df_initial[df_initial['part1'].isin(asch_experiments)]
    quantity_std_df = quantity_df[quantity_df['part3'] == 'std']
    quantity_cot_df = quantity_df[quantity_df['part3'] == 'cot']
    quantity_std = quantity_std_df['Value'].mean() if not quantity_std_df.empty else None
    quantity_cot = quantity_cot_df['Value'].mean() if not quantity_cot_df.empty else None

    # Source-Stressing ACC
    source_df = df_initial[df_initial['part1'].isin(source_experiments)]
    if source_cfg:
        source_df = source_df[source_df['part2'] == source_cfg]
    source_std_df = source_df[source_df['part3'] == 'std']
    source_cot_df = source_df[source_df['part3'] == 'cot']
    source_std = source_std_df['Value'].mean() if not source_std_df.empty else None
    source_cot = source_cot_df['Value'].mean() if not source_cot_df.empty else None

    # Followup ACC
    quantity_followup_std = None
    source_followup_std = None
    if not df_followup.empty:
        quantity_followup_df_list = []
        if asch_cfg:
            asch_conflict_followup_df = df_followup[
                (df_followup['part1'] == 'asch_conflict') &
                (df_followup['part2'] == asch_cfg)
            ]
            quantity_followup_df_list.append(asch_conflict_followup_df)
        asch_misleading_followup_df = df_followup[df_followup['part1'] == 'asch_misleading']
        quantity_followup_df_list.append(asch_misleading_followup_df)
        if quantity_followup_df_list:
            quantity_followup_df = pd.concat(quantity_followup_df_list, ignore_index=True)
        else:
            quantity_followup_df = df_followup[df_followup['part1'].isin(asch_experiments)]
        quantity_followup_std_df = quantity_followup_df[quantity_followup_df['part3'] == 'std']
        quantity_followup_std = quantity_followup_std_df['Value'].mean() if not quantity_followup_std_df.empty else None

        source_followup_df = df_followup[df_followup['part1'].isin(source_experiments)]
        if source_cfg:
            source_followup_df = source_followup_df[source_followup_df['part2'] == source_cfg]
        source_followup_std_df = source_followup_df[source_followup_df['part3'] == 'std']
        source_followup_std = source_followup_std_df['Value'].mean() if not source_followup_std_df.empty else None

    results = {
        'number': count,
        'vanilla_acc': vanilla_acc if vanilla_acc is not None else None,
        'quantity_acc': quantity_std if quantity_std is not None else None,
        'quantity_cot_acc': quantity_cot if quantity_cot is not None else None,
        'quantity_followup_acc': quantity_followup_std if quantity_followup_std is not None else None,
        'source_acc': source_std if source_std is not None else None,
        'source_cot_acc': source_cot if source_cot is not None else None,
        'source_followup_acc': source_followup_std if source_followup_std is not None else None,
    }
    return results
def process_data_for_table(input_file: str, min_py: float = 0.8, acc_match_mode: str = "entity"):
    """
    Process data similar to bar_plot_v11.py and return DataFrame.
    """
    global ACCURACY_MATCH_MODE
    ACCURACY_MATCH_MODE = acc_match_mode
    
    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Total samples in file: {len(data)}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    if not data:
        print("Error: Empty data file.")
        return None

    # Identify Modes
    found_modes = set()
    sample_size = min(len(data), 50)
    for item in data[:sample_size]:
        for key in item.keys():
            if key.startswith("extracted_entities_"):
                mode_name = key.replace("extracted_entities_", "")
                found_modes.add(mode_name)
    
    print(f"Detected Extraction Modes: {found_modes}")
    
    rows = []
    
    # Process Data
    processed_count = 0
    skipped_baseline = 0
    skipped_metrics = 0
    
    for mode_full in found_modes:
        parsed = parse_mode_generic(mode_full)
        part1 = parsed['part1']
        part2 = parsed['part2']
        part3 = parsed['part3']
        
        # Only process target experiments
        if part1 not in ['asch_conflict', 'asch_misleading', 'source_conflict', 'source_misleading']:
            continue
        
        stages_to_process = [("Initial", None)]
        # Check followup levels: prefer lvl2, fallback to lvl1, then lvl3
        followup_levels_to_check = [2, 1, 3]  # Priority order: lvl2 > lvl1 > lvl3
        found_followup_level = None
        for lvl in followup_levels_to_check:
            key_check = f"extracted_followup_entities_{mode_full}_lvl{lvl}"
            has_lvl = False
            # Check more samples to ensure we find followup data if it exists
            check_samples = min(len(data), 500)
            for s_item in data[:check_samples]:
                if key_check in s_item:
                    # Also check if the entities list is not empty
                    entities = s_item.get(key_check, [])
                    if entities and any(e for e in entities if normalize_entity(e)):
                        has_lvl = True
                        break
            if has_lvl:
                found_followup_level = lvl
                stages_to_process.append((f"Followup_Lvl{lvl}", lvl))
                print(f"    Found Followup_Lvl{lvl} data for {mode_full}")
                break  # Use the first found level (highest priority)
        
        for idx, item in enumerate(data):
            belief_result = item.get("belief_result", {})
            p_y = belief_result.get("p_y")
            belief_score = belief_result.get("score")

            baseline_metrics = get_baseline_metrics(item)
            if not baseline_metrics:
                skipped_baseline += 1
                continue

            for stage_name, level in stages_to_process:
                metrics = get_metrics_for_stage(item, mode_full, stage_name, level)
                if not metrics:
                    skipped_metrics += 1
                    continue
                
                for metric_name, val in metrics.items():
                    rows.append({
                        "SampleID":idx,
                        "Score": belief_score,
                        "Mode": mode_full,
                        "part1": part1,
                        "part2": part2,
                        "part3": part3,
                        "Stage": stage_name,
                        "Metric": metric_name,
                        "Value": val,
                        "Baseline": baseline_metrics.get(metric_name, 0.0)
                    })
                    processed_count += 1
    
    print(f"\nProcessing summary:")
    print(f"  Processed rows: {processed_count}")
    print(f"  Skipped (no baseline): {skipped_baseline}")
    print(f"  Skipped (no metrics): {skipped_metrics}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid data found matching criteria (p_y filter).")
        return None
    df_initial = df[(df['Stage'] == 'Initial') & (df['Metric'] == 'Accuracy')]
    print(f"Samples with Initial/Accuracy: {len(df_initial['SampleID'].unique())}")
    
    print(f"\nCollected {len(df)} data points.")
    return df

def calculate_table_statistics(df: pd.DataFrame, quantile: float, model_name: str) -> Dict[str, Any]:
    """
    Calculate statistics for one quantile setting.
    Returns a dict with keys: strategy_name, number, vanilla_acc, quantity_acc1, quantity_acc2, source_acc1, source_acc2
    """
    # Filter to Accuracy only
    df_acc = df[df['Metric'] == 'Accuracy'].copy()
    
    if df_acc.empty:
        print(f"    WARNING: No Accuracy data found for quantile={quantile}")
        return None
    
    # Apply grouping
    df_acc['Group'] = get_grouping_logic(df_acc, quantile)
    
    # Debug: Print grouping info
    group_counts = df_acc['Group'].value_counts()
    print(f"    Group counts (quantile={quantile}): {dict(group_counts)}")
    
    # Filter out Middle group
    df_acc = df_acc[df_acc['Group'] != "Middle"].copy()
    
    if df_acc.empty:
        print(f"    WARNING: No data after filtering Middle group for quantile={quantile}")
        return None
    
    # Filter to Initial and Followup stages (for table)
    # Prefer lvl2, fallback to lvl1
    df_initial = df_acc[df_acc['Stage'] == 'Initial'].copy()
    df_followup = None
    followup_stage_name = None
    for lvl in [2, 1, 3]:  # Priority: lvl2 > lvl1 > lvl3
        stage_name = f'Followup_Lvl{lvl}'
        df_check = df_acc[df_acc['Stage'] == stage_name].copy()
        if not df_check.empty:
            df_followup = df_check
            followup_stage_name = stage_name
            print(f"    Using {stage_name} for followup data: {len(df_followup)} rows")
            break
    
    if df_followup is None:
        df_followup = pd.DataFrame()  # Empty DataFrame
    
    if df_initial.empty:
        print(f"    WARNING: No Initial stage data for quantile={quantile}")
        print(f"    Available stages: {df_acc['Stage'].unique()}")
        return None
    
    has_followup = not df_followup.empty
    if not has_followup:
        print(f"    WARNING: No Followup data found for quantile={quantile}")
        print(f"    Available stages in df_acc: {sorted(df_acc['Stage'].unique())}")
        print(f"    Total rows in df_acc: {len(df_acc)}")
    else:
        print(f"    Found {followup_stage_name} data: {len(df_followup)} rows")
        print(f"    {followup_stage_name} experiments: {sorted(df_followup['part1'].unique())}")
        print(f"    {followup_stage_name} part2: {sorted(df_followup['part2'].unique())}")
        print(f"    {followup_stage_name} part3: {sorted(df_followup['part3'].unique())}")
    
    # Select part2 configurations:
    # - Asch experiments: prefer cfg2 (as specified in requirements), fallback to cfg6, then last cfg
    # - Source experiments: prefer high, fallback to last config
    asch_experiments = ['asch_conflict', 'asch_misleading']
    source_experiments = ['source_conflict', 'source_misleading']
    
    # Get available part2 for each experiment type
    asch_part2_list = sorted(df_initial[df_initial['part1'].isin(asch_experiments)]['part2'].unique())
    source_part2_list = sorted(df_initial[df_initial['part1'].isin(source_experiments)]['part2'].unique())
    
    print(f"    Available Asch part2: {asch_part2_list}")
    print(f"    Available Source part2: {source_part2_list}")
    
    # Select target part2
    asch_cfg = None
    if asch_part2_list:
        cfg_list = [p for p in asch_part2_list if p.startswith('cfg')]
        # Prefer cfg6 (last cfg), fallback to last available cfg
        if cfg_list:
            # Sort cfg list to get the last one (cfg6 if exists)
            cfg_list_sorted = sorted(cfg_list, key=lambda x: int(x.replace('cfg', '')) if x.replace('cfg', '').isdigit() else 999)
            asch_cfg = cfg_list_sorted[-1]  # Last cfg (prefer cfg6)
        print(f"    Selected Asch cfg: {asch_cfg}")
    
    source_cfg = None
    if source_part2_list:
        if 'high' in source_part2_list:
            source_cfg = 'high'
        elif source_part2_list:
            source_cfg = source_part2_list[-1]
        print(f"    Selected Source cfg: {source_cfg}")
    
    # Count samples per group using the same method as bar_plot_v11.py
    # Use a specific part2 (like cfg2 for Asch, high for Source) and count rows
    # This gives consistent counts across different configurations
    group_sample_counts = {}
    
    # Use the selected part2 configurations for counting
    # For each group, count rows with the selected part2
    for group_name in df_acc['Group'].unique():
        if group_name == "Middle":
            continue
        
        # Count using a representative part2 configuration
        # For Asch experiments: use asch_cfg (e.g., cfg2)
        # For Source experiments: use source_cfg (e.g., high)
        # Count rows where part2 matches AND part3 is std (or cot, doesn't matter, same sample)
        df_group_initial = df_initial[df_initial['Group'] == group_name]
        
        # Count unique samples: use a combination that represents one row per sample
        # Each sample has one row per (part1, part2, part3) combination
        # For counting, we can use any part2/part3 combination, but let's use std with selected part2
        count_rows = []
        
        # Count using the same method as bar_plot_v11.py
        # bar_plot_v11.py counts rows for a specific (part2, Stage_part3) combination
        # For Initial_std with a specific part2, each sample contributes:
        # - 1 row for asch_conflict (if using asch_conflict only)
        # - 1 row for asch_misleading (if using asch_misleading only)
        # - 2 rows total if counting both Asch experiments
        # - Similarly for Source experiments
        
        # To match bar_plot_v11.py, we should count rows for ONE specific experiment
        # Let's use asch_conflict with selected cfg, or source_conflict if Asch not available
        count = 0
        
        if asch_cfg:
            # Count rows for asch_conflict only (one experiment per sample)
            asch_conflict_rows = df_group_initial[
                (df_group_initial['part1'] == 'asch_conflict') &
                (df_group_initial['part2'] == asch_cfg) &
                (df_group_initial['part3'] == 'std')
            ]
            count = len(asch_conflict_rows)
        elif source_cfg:
            # Fallback to source_conflict
            source_conflict_rows = df_group_initial[
                (df_group_initial['part1'] == 'source_conflict') &
                (df_group_initial['part2'] == source_cfg) &
                (df_group_initial['part3'] == 'std')
            ]
            count = len(source_conflict_rows)
        
        if count > 0:
            group_sample_counts[group_name] = count
            print(f"    Group {group_name}: {count} samples (using single experiment counting)")
        else:
            # Fallback: estimate from total rows
            # Each sample has ~20 modes * 2 metrics = 40 rows in Initial stage
            group_sample_counts[group_name] = len(df_group_initial) // 40
            print(f"    Group {group_name}: estimated {group_sample_counts[group_name]} samples (fallback)")
    
    # Process Low (Bottom) and High (Top) groups
    results = {}
    
    # Get actual group names from data
    bottom_group_name = f"Bottom {int(quantile*100)}%"
    top_group_name = f"Top {int(quantile*100)}%"
    
    print(f"    Looking for groups: {bottom_group_name}, {top_group_name}")
    print(f"    Available groups: {df_initial['Group'].unique()}")
    
    for group_name, group_label in [(bottom_group_name, "Low"), (top_group_name, "High")]:
        df_group = df_initial[df_initial['Group'] == group_name].copy()
        
        if df_group.empty:
            print(f"    WARNING: No data for group '{group_name}'")
            continue
        
        print(f"    Processing {group_name}: {len(df_group)} rows")
        
        # Count: use the pre-computed sample count
        count = group_sample_counts.get(group_name, len(df_group['Score'].unique()))
        print(f"      Count: {count} unique samples (from {len(df_group)} rows)")
        
        # Vanilla ACC: Baseline accuracy (use first available part2 for consistency)
        # Since Baseline is the same for all rows of the same sample, we can just take mean
        vanilla_acc = df_group['Baseline'].mean()
        
        # Quantity-Stressing: asch_conflict + asch_misleading
        # ACC1: Initial_std, ACC2: Initial_cot
        # Note: asch_conflict has cfg configs (cfg0, cfg1, cfg2, etc.)
        #       asch_misleading has no cfg, part2 is "default" or "misleading"
        # So we need to handle them separately:
        # - asch_conflict: use selected cfg (e.g., cfg2)
        # - asch_misleading: use its own part2 (usually "default")
        quantity_df_list = []
        
        # Add asch_conflict with selected cfg
        if asch_cfg:
            asch_conflict_df = df_group[
                (df_group['part1'] == 'asch_conflict') &
                (df_group['part2'] == asch_cfg)
            ]
            quantity_df_list.append(asch_conflict_df)
        
        # Add asch_misleading (it doesn't have cfg, so include all)
        asch_misleading_df = df_group[df_group['part1'] == 'asch_misleading']
        quantity_df_list.append(asch_misleading_df)
        
        # Combine
        if quantity_df_list:
            quantity_df = pd.concat(quantity_df_list, ignore_index=True)
        else:
            quantity_df = df_group[df_group['part1'].isin(asch_experiments)]
        
        quantity_std_df = quantity_df[quantity_df['part3'] == 'std']
        quantity_cot_df = quantity_df[quantity_df['part3'] == 'cot']
        quantity_std = quantity_std_df['Value'].mean() if not quantity_std_df.empty else None
        quantity_cot = quantity_cot_df['Value'].mean() if not quantity_cot_df.empty else None
        
        # Debug: show which configs are used
        if not quantity_std_df.empty:
            used_configs = sorted(quantity_std_df[['part1', 'part2', 'part3']].drop_duplicates().apply(
                lambda x: f"{x['part1']}_{x['part2']}_{x['part3']}", axis=1
            ).tolist())
            print(f"      Quantity-Stressing ACC1 (std): {used_configs}")
            print(f"        -> Average of: {', '.join(used_configs)}")
        if not quantity_cot_df.empty:
            used_configs = sorted(quantity_cot_df[['part1', 'part2', 'part3']].drop_duplicates().apply(
                lambda x: f"{x['part1']}_{x['part2']}_{x['part3']}", axis=1
            ).tolist())
            print(f"      Quantity-Stressing ACC2 (cot): {used_configs}")
            print(f"        -> Average of: {', '.join(used_configs)}")
        print(f"      Quantity-Stressing: std={len(quantity_std_df)} rows, cot={len(quantity_cot_df)} rows")
        
        # Source-Stressing: source_conflict + source_misleading
        # ACC1: Initial_std, ACC2: Initial_cot
        # Both use the same config: selected source_cfg (e.g., high)
        source_df = df_group[df_group['part1'].isin(source_experiments)]
        if source_cfg:
            source_df = source_df[source_df['part2'] == source_cfg]
        
        source_std_df = source_df[source_df['part3'] == 'std']
        source_cot_df = source_df[source_df['part3'] == 'cot']
        source_std = source_std_df['Value'].mean() if not source_std_df.empty else None
        source_cot = source_cot_df['Value'].mean() if not source_cot_df.empty else None
        
        # Debug: show which configs are used
        if not source_std_df.empty:
            used_configs = sorted(source_std_df[['part1', 'part2', 'part3']].drop_duplicates().apply(
                lambda x: f"{x['part1']}_{x['part2']}_{x['part3']}", axis=1
            ).tolist())
            print(f"      Source-Stressing ACC1 (std): {used_configs}")
            print(f"        -> Average of: {', '.join(used_configs)}")
        if not source_cot_df.empty:
            used_configs = sorted(source_cot_df[['part1', 'part2', 'part3']].drop_duplicates().apply(
                lambda x: f"{x['part1']}_{x['part2']}_{x['part3']}", axis=1
            ).tolist())
            print(f"      Source-Stressing ACC2 (cot): {used_configs}")
            print(f"        -> Average of: {', '.join(used_configs)}")
        print(f"      Source-Stressing: std={len(source_std_df)} rows, cot={len(source_cot_df)} rows")
        
        # Followup_Lvl1: same logic as Initial
        quantity_followup_std = None
        quantity_followup_cot = None
        source_followup_std = None
        source_followup_cot = None
        
        if has_followup:
            df_group_followup = df_followup[df_followup['Group'] == group_name].copy()
            print(f"      Followup data for {group_name}: {len(df_group_followup)} rows")
            
            if not df_group_followup.empty:
                print(f"        Followup part1: {sorted(df_group_followup['part1'].unique())}")
                print(f"        Followup part2: {sorted(df_group_followup['part2'].unique())}")
                print(f"        Followup part3: {sorted(df_group_followup['part3'].unique())}")
                
                # Quantity-Stressing Followup
                quantity_followup_df_list = []
                if asch_cfg:
                    asch_conflict_followup_df = df_group_followup[
                        (df_group_followup['part1'] == 'asch_conflict') &
                        (df_group_followup['part2'] == asch_cfg)
                    ]
                    print(f"        asch_conflict with {asch_cfg}: {len(asch_conflict_followup_df)} rows")
                    quantity_followup_df_list.append(asch_conflict_followup_df)
                asch_misleading_followup_df = df_group_followup[df_group_followup['part1'] == 'asch_misleading']
                print(f"        asch_misleading: {len(asch_misleading_followup_df)} rows")
                quantity_followup_df_list.append(asch_misleading_followup_df)
                
                if quantity_followup_df_list:
                    quantity_followup_df = pd.concat(quantity_followup_df_list, ignore_index=True)
                else:
                    quantity_followup_df = df_group_followup[df_group_followup['part1'].isin(asch_experiments)]
                
                quantity_followup_std_df = quantity_followup_df[quantity_followup_df['part3'] == 'std']
                quantity_followup_cot_df = quantity_followup_df[quantity_followup_df['part3'] == 'cot']
                quantity_followup_std = quantity_followup_std_df['Value'].mean() if not quantity_followup_std_df.empty else None
                quantity_followup_cot = quantity_followup_cot_df['Value'].mean() if not quantity_followup_cot_df.empty else None
                
                print(f"        Followup Quantity-Stressing: std={len(quantity_followup_std_df)} rows (acc={quantity_followup_std}), cot={len(quantity_followup_cot_df)} rows")
                
                # Source-Stressing Followup
                source_followup_df = df_group_followup[df_group_followup['part1'].isin(source_experiments)]
                print(f"        Source followup before filtering: {len(source_followup_df)} rows")
                if source_cfg:
                    source_followup_df = source_followup_df[source_followup_df['part2'] == source_cfg]
                    print(f"        Source followup after filtering by {source_cfg}: {len(source_followup_df)} rows")
                
                source_followup_std_df = source_followup_df[source_followup_df['part3'] == 'std']
                source_followup_cot_df = source_followup_df[source_followup_df['part3'] == 'cot']
                source_followup_std = source_followup_std_df['Value'].mean() if not source_followup_std_df.empty else None
                source_followup_cot = source_followup_cot_df['Value'].mean() if not source_followup_cot_df.empty else None
                
                print(f"        Followup Source-Stressing: std={len(source_followup_std_df)} rows (acc={source_followup_std}), cot={len(source_followup_cot_df)} rows")
            else:
                print(f"        WARNING: No followup data for group {group_name} after filtering!")
        else:
            print(f"      WARNING: No followup data available for {group_name}")
               
        strategy_name = f"{group_label} NCB-{int(quantile*100)}%"
        results[strategy_name] = {
            'number': count,
            'vanilla_acc': vanilla_acc if not np.isnan(vanilla_acc) else None,
            'quantity_acc': quantity_std if quantity_std is not None and not np.isnan(quantity_std) else None,
            'quantity_cot_acc': quantity_cot if quantity_cot is not None and not np.isnan(quantity_cot) else None,
            'quantity_followup_acc': quantity_followup_std if quantity_followup_std is not None and not np.isnan(quantity_followup_std) else None,
            'quantity_followup_cot_acc': quantity_followup_cot if quantity_followup_cot is not None and not np.isnan(quantity_followup_cot) else None,
            'source_acc': source_std if source_std is not None and not np.isnan(source_std) else None,
            'source_cot_acc': source_cot if source_cot is not None and not np.isnan(source_cot) else None,
            'source_followup_acc': source_followup_std if source_followup_std is not None and not np.isnan(source_followup_std) else None,
            'source_followup_cot_acc': source_followup_cot if source_followup_cot is not None and not np.isnan(source_followup_cot) else None,
        }
        
        print(f"      Results: vanilla={vanilla_acc:.3f}, q_acc1={quantity_std}, q_acc2={quantity_cot}, s_acc1={source_std}, s_acc2={source_cot}")
    
    return results

def format_value(val, default="--"):
    """Format a value for table display"""
    if val is None or np.isnan(val):
        return default
    # Format to 3 decimal places, remove trailing zeros
    formatted = f"{val:.3f}".rstrip('0').rstrip('.')
    return formatted if formatted else "0"

def generate_latex_table(results_dict: Dict[str, Dict[str, Any]], output_file: str, quantiles: List[float]):
    """
    Generate LaTeX table from results (optional, for saving to file).
    results_dict: {model_name: {strategy_name: {number, vanilla_acc, ...}}}
    quantiles: List of quantile values used
    """
    # Define model order
    model_order = ["Qwen3-A3B-30B-Instruct-2507", "Qwen3-A3B-30B-Thinking-2507"]
    
    # Define strategy order based on quantiles
    strategy_order = []
    for q in quantiles:
        strategy_order.append(f"Low NCB-{int(q*100)}%")
    for q in quantiles:
        strategy_order.append(f"High NCB-{int(q*100)}%")
    
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccccc}")
    lines.append("\\toprule")
    lines.append("\\multirow{2}{*}{\\textbf{Training Strategy}}")
    lines.append("& \\multirow{2}{*}{\\textbf{Number}}")
    lines.append("& \\multirow{2}{*}{\\textbf{Vanilla ACC}}")
    lines.append("& \\multicolumn{3}{c}{\\textbf{Quantity-Stressing}}")
    lines.append("& \\multicolumn{3}{c}{\\textbf{Source-Stressing}} \\\\")
    lines.append("")
    lines.append("\\cmidrule(lr){4-6}")
    lines.append("\\cmidrule(lr){7-9}")
    lines.append("& & & \\textbf{ACC} & \\textbf{COT\\_ACC} & \\textbf{Followup\\_ACC} & \\textbf{ACC} & \\textbf{COT\\_ACC} & \\textbf{Followup\\_ACC} \\\\")
    lines.append("")
    lines.append("\\midrule")
    
    # Process each model
    for model_name in model_order:
        if model_name not in results_dict:
            continue
        
        model_results = results_dict[model_name]
        
        # Model header
        lines.append(f"\\multicolumn{{9}}{{c}}{{\\cellcolor{{gray!20}}\\textbf{{{model_name}}}}} \\\\")
        lines.append("\\midrule")
        
        # Process strategies
        for strategy in strategy_order:
            if strategy not in model_results:
                # Empty row
                lines.append(f"{strategy} & & & & & & & & \\\\")
            else:
                r = model_results[strategy]
                lines.append(
                    f"{strategy} & "
                    f"{r.get('number', '--')} & "
                    f"{format_value(r.get('vanilla_acc'))} & "
                    f"{format_value(r.get('quantity_acc'))} & "
                    f"{format_value(r.get('quantity_cot_acc'))} & "
                    f"{format_value(r.get('quantity_followup_acc'))} & "
                    f"{format_value(r.get('source_acc'))} & "
                    f"{format_value(r.get('source_cot_acc'))} & "
                    f"{format_value(r.get('source_followup_acc'))} \\\\"
                )
        
        lines.append("\\midrule")
        lines.append("")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\caption{Main results under different training strategies. Generic-task performance is visually distinguished from stress-testing diagnostics via background shading. Followup columns show results after intervention.}")
    lines.append("\\label{tab:training_results}")
    lines.append("\\end{table*}")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\nLaTeX table saved to: {output_file}")

def print_table(results_dict: Dict[str, Dict[str, Any]], quantiles: List[float]):
    """
    Print formatted table to console.
    results_dict: {model_name: {strategy_name: {number, vanilla_acc, ...}}}
    quantiles: List of quantile values used
    """
    # Check if results_dict is empty
    if not results_dict:
        print("\nWARNING: No results to display!")
        return
    
    # Check if any model has results
    has_results = any(len(model_results) > 0 for model_results in results_dict.values())
    if not has_results:
        print("\nWARNING: All models have empty results!")
        print("Results dict structure:")
        for model_name, model_results in results_dict.items():
            print(f"  {model_name}: {len(model_results)} strategies")
        return
    
    # Use actual model names from results_dict, but prefer standard order if available
    standard_order = ["Qwen3-A3B-30B-Instruct-2507", "Qwen3-A3B-30B-Thinking-2507"]
    model_order = []
    # Add standard models first if they exist
    for model in standard_order:
        if model in results_dict:
            model_order.append(model)
    # Add any other models
    for model in results_dict.keys():
        if model not in model_order:
            model_order.append(model)
    
    # Define strategy order based on quantiles
    strategy_order = []
    for q in quantiles:
        strategy_order.append(f"Low NCB-{int(q*100)}%")
    for q in quantiles:
        strategy_order.append(f"High NCB-{int(q*100)}%")
    
    # Column widths
    col_widths = {
        'strategy': 20,
        'number': 8,
        'vanilla': 12,
        'q_acc': 10,
        'q_cot_acc': 10,
        'q_f_acc': 10,
        's_acc': 10,
        's_cot_acc': 10,
        's_f_acc': 10,
    }
    
    # Print header
    print("\n" + "="*110)
    print("Main results under different training strategies".center(110))
    print("="*110)
    print()
    
    # Print column headers
    header1 = (
        f"{'Training Strategy':<{col_widths['strategy']}} "
        f"{'Number':>{col_widths['number']}} "
        f"{'Vanilla ACC':>{col_widths['vanilla']}} "
        f"{'Quantity-Stressing':>{col_widths['q_acc'] + col_widths['q_cot_acc'] + col_widths['q_f_acc'] + 2}} "
        f"{'Source-Stressing':>{col_widths['s_acc'] + col_widths['s_cot_acc'] + col_widths['s_f_acc'] + 2}}"
    )
    print(header1)
    
    header2 = (
        f"{'':<{col_widths['strategy']}} "
        f"{'':>{col_widths['number']}} "
        f"{'':>{col_widths['vanilla']}} "
        f"{'ACC':>{col_widths['q_acc']}} "
        f"{'COT_ACC':>{col_widths['q_cot_acc']}} "
        f"{'Followup_ACC':>{col_widths['q_f_acc']}} "
        f"{'ACC':>{col_widths['s_acc']}} "
        f"{'COT_ACC':>{col_widths['s_cot_acc']}} "
        f"{'Followup_ACC':>{col_widths['s_f_acc']}}"
    )
    print(header2)
    print("-" * 110)
    
    # Process each model
    for model_name in model_order:
        if model_name not in results_dict:
            continue
        
        model_results = results_dict[model_name]
        
        # Model header
        print()
        print(f"  {model_name}".center(110))
        print("-" * 110)
        
        # Process strategies
        for strategy in strategy_order:
            if strategy not in model_results:
                # Empty row
                row = (
                    f"{strategy:<{col_widths['strategy']}} "
                    f"{'--':>{col_widths['number']}} "
                    f"{'--':>{col_widths['vanilla']}} "
                    f"{'--':>{col_widths['q_acc']}} "
                    f"{'--':>{col_widths['q_cot_acc']}} "
                    f"{'--':>{col_widths['q_f_acc']}} "
                    f"{'--':>{col_widths['s_acc']}} "
                    f"{'--':>{col_widths['s_cot_acc']}} "
                    f"{'--':>{col_widths['s_f_acc']}}"
                )
            else:
                r = model_results[strategy]
                row = (
                    f"{strategy:<{col_widths['strategy']}} "
                    f"{r.get('number', '--'):>{col_widths['number']}} "
                    f"{format_value(r.get('vanilla_acc')):>{col_widths['vanilla']}} "
                    f"{format_value(r.get('quantity_acc')):>{col_widths['q_acc']}} "
                    f"{format_value(r.get('quantity_cot_acc')):>{col_widths['q_cot_acc']}} "
                    f"{format_value(r.get('quantity_followup_acc')):>{col_widths['q_f_acc']}} "
                    f"{format_value(r.get('source_acc')):>{col_widths['s_acc']}} "
                    f"{format_value(r.get('source_cot_acc')):>{col_widths['s_cot_acc']}} "
                    f"{format_value(r.get('source_followup_acc')):>{col_widths['s_f_acc']}}"
                )
            print(row)
        
        print("-" * 110)
    
    print()
    print("="*110)
    print()

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from bar_plot_v11.py data")
    parser.add_argument("--input_files", nargs="+", required=True, 
                       help="Path(s) to JSON file(s). If multiple, will process each separately.")
    parser.add_argument("--model_names", nargs="+", default=None,
                       help="Model names corresponding to input_files (optional, auto-detected if not provided)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Optional: Save LaTeX table to file (if not provided, only prints to console)")
    parser.add_argument("--min_py", type=float, default=0.8, help="Minimum p_y")
    parser.add_argument(
        "--acc_match_mode",
        type=str,
        default="sentence",
        choices=["entity", "sentence", "both"],
        help="Accuracy matching mode: entity / sentence / both",
    )
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.05, 0.20, 0.35],
                       help="Quantile values to use (default: 0.05 0.20 0.35)")
    args = parser.parse_args()

    # Validate model_names if provided
    if args.model_names and len(args.model_names) != len(args.input_files):
        print("ERROR: Number of model_names must match number of input_files")
        return

    # Process each input file
    all_results = {}
    
    for idx, input_file in enumerate(args.input_files):
        if args.model_names:
            model_name = args.model_names[idx]
        else:
            model_name = extract_model_name(input_file)
        
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"Input file: {input_file}")
        print(f"{'='*60}")
        
        df = process_data_for_table(input_file, args.min_py, args.acc_match_mode)
        if df is None or df.empty:
            print(f"  WARNING: No data for {model_name}, skipping...")
            continue

        stats = calculate_overall_statistics(df, model_name)
        if stats:
            all_results[model_name] = stats
        else:
            print(f"    WARNING: No statistics returned for {model_name}")
    
    # Check if we have any results
    if not all_results:
        print("\nERROR: No results generated! Check the debug output above.")
        print("Possible issues:")
        print("  - No data matching the criteria")
        print("  - No target experiments found")
        print("  - All data filtered out")
        return
    
    # Print summary of results
    print(f"\n{'='*60}")
    print("Results Summary:")
    print(f"{'='*60}")
    for model_name, stats in all_results.items():
        print(f"\n{model_name}:")
        print(f"  number={stats.get('number', 'N/A')}, "
            f"vanilla={format_value(stats.get('vanilla_acc'))}, "
            f"q_acc={format_value(stats.get('quantity_acc'))}, "
            f"q_cot_acc={format_value(stats.get('quantity_cot_acc'))}, "
            f"s_acc={format_value(stats.get('source_acc'))}, "
            f"s_cot_acc={format_value(stats.get('source_cot_acc'))}, "
            f"q_f_acc={format_value(stats.get('quantity_followup_acc'))}, "
            f"s_f_acc={format_value(stats.get('source_followup_acc'))}")
    
    # Print table to console
    # print_table(all_results, args.quantiles)
    
    # Optionally save LaTeX version
    # if args.output_file:
    #     generate_latex_table(all_results, args.output_file, args.quantiles)
    #     print(f"LaTeX version also saved to: {args.output_file}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

