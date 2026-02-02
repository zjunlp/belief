#!/usr/bin/env python3
"""
Bar Plot V11: Combine all experiments in one 2x2 grid
- Each subplot is one experiment (part1)
- Within each subplot, all configurations (part2) are shown together
- Layout: 2 rows × 2 columns (4 experiments)
- Columns are wider (2:1 ratio)
- Only Accuracy metric (Coverage can be added later)
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
from collections import Counter
from typing import List, Any, Dict, Optional, Tuple
from tqdm import tqdm

# Import functions from v10
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

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']

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

# Global switch: Control Accuracy matching mode
# - "entity": Only do containment match on extracted entities (default)
# - "sentence": Only do containment match on original sentences
# - "both": First try entity match, then sentence match if failed
ACCURACY_MATCH_MODE = "entity"

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
    Decide matching mode based on global ACCURACY_MATCH_MODE:
    - "entity":   Only do containment match between entity e and golden
    - "sentence": Only find golden in corresponding raw sentence text
    - "both":     First try entity match, then sentence match if failed
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

        # 2) Sentence-level matching (raw_sentences cleaned & lowered by extract_raw_text_list)
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
        entity_key = f"extracted_followup_entities_{mode_full}{suffix}"
        response_key = f"followup_{mode_full}{suffix}"

    raw_responses = item.get(response_key, [])
    extracted_entities = item.get(entity_key, [])
    if not extracted_entities:
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

def get_grouping_logic(df, split_mode, config):
    scores = df['Score']
    if split_mode == "quantile":
        q = config.get('quantile', 0.25)
        lower_threshold = scores.quantile(q)
        upper_threshold = scores.quantile(1.0 - q)
        def get_group(score):
            if score <= lower_threshold:
                return f"Bottom {int(q*100)}%"
            elif score >= upper_threshold:
                return f"Top {int(q*100)}%"
            else:
                return "Middle"
        return df['Score'].apply(get_group)
        
    elif split_mode == "threshold":
        th = config.get('threshold', 0.8)
        def get_group(score):
            if score >= th:
                return f"High (≥ {th})"
            else:
                return f"Low (< {th})"
        return df['Score'].apply(get_group)
        
    elif split_mode == "certainty":
        th = 0.999
        def get_group(score):
            if score > th:
                return "Certain"
            else:
                return "Uncertain"
        return df['Score'].apply(get_group)
    return None

def plot_v11_combined(df, output_dir, split_mode, split_config, score_mode="geo_mean", min_py=0.8):
    """
    Plot all experiments in a 2x2 grid.
    Each subplot is one experiment, showing all its configurations.
    """
    # Filter to only the 4 main experiments
    target_experiments = ['asch_conflict', 'asch_misleading', 'source_misleading', 'source_conflict']
    df_filtered = df[df['part1'].isin(target_experiments)].copy()
    
    if df_filtered.empty:
        print("No data for target experiments")
        return
    
    # Apply Grouping
    df_filtered['Group'] = get_grouping_logic(df_filtered, split_mode, split_config)
    
    # Filter Middle if quantile
    if split_mode == "quantile":
        df_filtered = df_filtered[df_filtered['Group'] != "Middle"].copy()
    
    if df_filtered.empty:
        print("No data after grouping")
        return
    
    # Define Group Order (X-axis)
    unique_groups = sorted(df_filtered['Group'].unique())
    if split_mode == "quantile":
        unique_groups = sorted(unique_groups, key=lambda x: 0 if "Bottom" in x else 1)
    elif split_mode == "threshold":
        unique_groups = sorted(unique_groups, key=lambda x: 0 if "Low" in x else 1)
    elif split_mode == "certainty":
        unique_groups = sorted(unique_groups, key=lambda x: 0 if "Uncertain" in x else 1)
    
    # Create combination: Stage_part3 (like v10)
    df_filtered['Stage_part3'] = df_filtered['Stage'] + '_' + df_filtered['part3']
    
    # Filter to Accuracy only
    df_accuracy = df_filtered[df_filtered['Metric'] == 'Accuracy'].copy()
    
    # Create 2x2 grid with wider columns
    # Width ratio: columns are 2x wider than rows
    fig_width = 14.0  # Double-column width
    fig_height = 6.0   # Height for 2 rows
    # Share x-axis for same column, only show x-axis labels at bottom of each column
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), sharex='col')
    axes = axes.flatten()
    
    # Experiment order: asch_conflict, asch_misleading, source_misleading, source_conflict
    exp_order = ['asch_conflict', 'asch_misleading', 'source_misleading', 'source_conflict']
    
    title_suffix = ""
    if split_mode == "quantile":
        title_suffix = f"Top/Bottom {int(split_config.get('quantile',0.25)*100)}%"
    elif split_mode == "threshold":
        title_suffix = f"Threshold {split_config.get('threshold',0.8)}"
    elif split_mode == "certainty":
        title_suffix = "Certainty"
    
    # No overall title, leave top space for legend
    # fig.suptitle(
    #     f"Accuracy Comparison Across Experiments ({title_suffix}, {score_mode})",
    #     fontsize=14, fontweight='bold', y=0.98
    # )
    
    legend_handles = None
    legend_labels = None

    for idx, exp in enumerate(exp_order):
        ax = axes[idx]
        df_exp = df_accuracy[df_accuracy['part1'] == exp].copy()

        # First check what part2 values exist
        available_part2 = sorted(df_exp['part2'].unique()) if not df_exp.empty else []

        # Only look at specified configurations:
        # - Asch experiment: prefer cfg6; if no cfg6, use last cfg setting among all cfg*
        # - Source experiment: prefer high, if not available fallback to last config
        if exp.startswith("asch_"):
            cfg_part2 = [p for p in available_part2 if p.startswith("cfg")]
            if "cfg6" in cfg_part2:
                target_part2 = "cfg6"
            elif cfg_part2:
                target_part2 = cfg_part2[-1]
                print(
                    f"  WARNING: 'cfg6' not found for {exp}; "
                    f"using last cfg setting '{target_part2}' instead. "
                    f"Available cfg settings: {cfg_part2}"
                )
            else:
                target_part2 = None
        elif exp.startswith("source_"):
            preferred_part2 = "high"
            if preferred_part2 in available_part2:
                target_part2 = preferred_part2
            elif available_part2:
                target_part2 = available_part2[-1]
                print(
                    f"  WARNING: preferred part2 '{preferred_part2}' not found for {exp}; "
                    f"using '{target_part2}' instead. All available: {available_part2}"
                )
            else:
                target_part2 = None
        else:
            target_part2 = None

        if target_part2 is not None:
            df_exp = df_exp[df_exp['part2'] == target_part2].copy()
        
        # Debug: Print what data we have for this experiment
        print(f"\n  Processing {exp}:")
        print(f"    Total rows: {len(df_exp)}")
        if not df_exp.empty:
            print(f"    Unique part2 (after filter): {sorted(df_exp['part2'].unique())}")
            print(f"    Unique Stage_part3: {sorted(df_exp['Stage_part3'].unique())}")
            print(f"    Unique Groups: {sorted(df_exp['Group'].unique())}")
        
        if df_exp.empty:
            ax.text(0.5, 0.5, f'No data for {exp}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(exp.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            continue
        
        # Get unique part2 (configurations) for this experiment
        unique_part2 = sorted(df_exp['part2'].unique())
        
        # Get unique stages and part3 for ordering
        unique_stages = sorted(df_exp['Stage'].unique())
        unique_part3 = sorted(df_exp['part3'].unique())
        
        # Define Stage order: Initial first, then Followup_Lvl1, Lvl2, Lvl3
        stage_order = []
        if "Initial" in unique_stages:
            stage_order.append("Initial")
        for lvl in [1, 2, 3]:
            stage_name = f"Followup_Lvl{lvl}"
            if stage_name in unique_stages:
                stage_order.append(stage_name)
        for st in unique_stages:
            if st not in stage_order:
                stage_order.append(st)
        
        # For each part2, aggregate by (Group, Stage_part3)
        # Then combine all part2 results, using part2_Stage_part3 as hue
        # OR: aggregate across all part2 for each (Group, Stage_part3)
        # Let's use the second approach: aggregate across part2, keep Stage_part3 separate
        
        # Aggregate: Group by Group (X) and Stage_part3 (Hue), averaging across part2
        # This is similar to v9 but across all part2
        agg_data = df_exp.groupby(['Group', 'Stage_part3'])['Value'].mean().reset_index()
        
        # Debug: Print aggregation info
        print(f"    Aggregated data shape: {agg_data.shape}")
        print(f"    Sample aggregated values:")
        for _, row in agg_data.head(5).iterrows():
            print(f"      Group={row['Group']}, Stage_part3={row['Stage_part3']}, Value={row['Value']:.3f}")
        
        # Build hue order: Baseline first, then Stage_part3 combinations
        # Order: Initial_std, Initial_cot, Followup_Lvl1_std, Followup_Lvl1_cot, ...
        hue_order = ['Baseline']
        actual_combos = set(df_exp['Stage_part3'].unique())
        for stage in stage_order:
            for p3 in unique_part3:
                combo = f"{stage}_{p3}"
                if combo in actual_combos:
                    hue_order.append(combo)
        
        # Ensure all actual combinations are included
        for combo in sorted(actual_combos):
            if combo not in hue_order:
                hue_order.append(combo)
        
        print(f"    Hue order: {hue_order}")
        
        # Baseline: use first part2 and first Stage_part3 to avoid duplication
        first_part2 = unique_part2[0] if unique_part2 else None
        first_combo = hue_order[1] if len(hue_order) > 1 else None
        if first_part2 and first_combo:
            base_rows = df_exp[(df_exp['part2'] == first_part2) & (df_exp['Stage_part3'] == first_combo)].copy()
            base_rows['Stage_part3'] = 'Baseline'
            base_rows['Value'] = base_rows['Baseline']
            base_rows = base_rows[['Group', 'Stage_part3', 'Value']]
            plot_data = pd.concat([agg_data, base_rows], ignore_index=True)
        else:
            plot_data = agg_data
        
        # Color Palette: Baseline grey, then distinct colors for each combo
        palette = {'Baseline': (0.4, 0.4, 0.4)}  # Grey
        n_combos = len(hue_order) - 1  # Exclude Baseline
        if n_combos > 0:
            colors = sns.color_palette("Set2", n_colors=max(n_combos, 8))
            for i, combo in enumerate(hue_order[1:], 0):  # Skip Baseline
                palette[combo] = colors[i % len(colors)]
        
        # Plot
        sns.barplot(
            data=plot_data,
            x="Group",
            y="Value",
            hue="Stage_part3",
            order=unique_groups,
            hue_order=hue_order,
            palette=palette,
            edgecolor="white",
            linewidth=1,
            ax=ax
        )
        
        # Add counts: 
        # For consistency with baseline, we should count using the same approach
        # Baseline uses first_part2, so for other Stage_part3, we should also use first_part2 for counting
        # OR: count all part2 (since we averaged across part2)
        # Let's use first_part2 for consistency with baseline
        if first_part2:
            # Count using first_part2 for each Stage_part3 (consistent with baseline approach)
            counts_list = []
            for combo in hue_order[1:]:  # Skip Baseline
                combo_df = df_exp[(df_exp['part2'] == first_part2) & (df_exp['Stage_part3'] == combo)]
                combo_counts = combo_df.groupby('Group').size().reset_index(name='count')
                combo_counts['Stage_part3'] = combo
                counts_list.append(combo_counts)
            
            if counts_list:
                counts = pd.concat(counts_list, ignore_index=True)
            else:
                counts = pd.DataFrame(columns=['Group', 'Stage_part3', 'count'])
            
            # Baseline: use the same filtering as when creating baseline rows
            if first_combo:
                baseline_df = df_exp[(df_exp['part2'] == first_part2) & (df_exp['Stage_part3'] == first_combo)]
                baseline_counts = baseline_df.groupby('Group').size().reset_index(name='count')
                baseline_counts['Stage_part3'] = 'Baseline'
                all_counts = pd.concat([counts, baseline_counts[['Group', 'Stage_part3', 'count']]], ignore_index=True)
            else:
                all_counts = counts
        else:
            # Fallback: count all part2
            counts = df_exp.groupby(['Group', 'Stage_part3']).size().reset_index(name='count')
            all_counts = counts
        
        for i, container in enumerate(ax.containers):
            if i >= len(hue_order): break
            combo_name = hue_order[i]
            
            for j, bar in enumerate(container):
                if j >= len(unique_groups): break
                group_name = unique_groups[j]
                
                c_row = all_counts[(all_counts['Group'] == group_name) & (all_counts['Stage_part3'] == combo_name)]
                count = c_row['count'].values[0] if not c_row.empty else 0
                
                height = bar.get_height()
                if not np.isnan(height) and height > 0:
                    # Adjust text position for new y-axis range (0.5-1.0)
                    # Add small offset relative to the range
                    text_y = height + 0.005  # Small offset (0.5% of range)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        text_y,
                        f"n={count}",
                        ha='center', va='bottom', fontsize=8, color='black'
                    )
        
        # Save legend handles, only needed once
        if legend_handles is None and legend_labels is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Keep title for each experiment subplot
        exp_title = exp.replace('_', ' ').title()
        ax.set_title(exp_title, fontsize=11, fontweight='bold', pad=5)

        # Y-axis starts from 0.2
        ax.set_ylim(0.2, 1.0)
        ax.set_yticks(np.arange(0.2, 1.01, 0.05))

        # Only show x-axis labels and tick text at bottom subplot of each column
        row_idx = idx // 2
        if row_idx == 0:
            ax.set_xlabel("")
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel("Belief Group", fontsize=10)

        if idx in [0, 2]:  # Left column
            ax.set_ylabel("Accuracy", fontsize=10)
        else:
            ax.set_ylabel("")

        # X-axis labels written horizontally (no tilt)
        ax.tick_params(axis='x', rotation=0, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Individual subplots don't keep legend, unified at top of figure
        if ax.get_legend():
            ax.get_legend().remove()

    # Put a shared legend at top of overall figure
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Stage & Type",
            loc='upper center',
            bbox_to_anchor=(0.5, 1.03),
            ncol=min(len(legend_labels), 5),
            fontsize=8,
            title_fontsize=9,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    clean_split = split_mode
    final_filename = f"combined_all_experiments_{score_mode}_{clean_split}_v11.png"
    plt.savefig(os.path.join(output_dir, final_filename), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  -> Saved: {final_filename}")


def plot_v11_low_medium_high(df, output_dir, split_mode, split_config, score_mode="geo_mean", min_py=0.8):
    """
    Draw a separate figure comparing low / medium / high settings in Source experiment (only Accuracy, Initial_std).
    Layout: 1 row 2 columns, left: source_misleading, right: source_conflict.
    X-axis: Belief Group (Same as v11), Hue: part2 in {low, medium, high}.
    """
    target_experiments = ['source_misleading', 'source_conflict']
    df_filtered = df[df['part1'].isin(target_experiments)].copy()

    if df_filtered.empty:
        print("No data for source experiments (low/medium/high).")
        return

    # Groups (Bottom/Top, High/Low, etc.)
    df_filtered['Group'] = get_grouping_logic(df_filtered, split_mode, split_config)

    if split_mode == "quantile":
        df_filtered = df_filtered[df_filtered['Group'] != "Middle"].copy()

    if df_filtered.empty:
        print("No data after grouping for low/medium/high plot.")
        return

    # Only look at Accuracy, Initial stage, std
    df_acc = df_filtered[
        (df_filtered['Metric'] == 'Accuracy')
        & (df_filtered['Stage'] == 'Initial')
        & (df_filtered['part3'] == 'std')
    ].copy()

    if df_acc.empty:
        print("No Accuracy Initial_std data for low/medium/high plot.")
        return

    # Only keep low / medium / high settings
    cfg_levels = ['low', 'medium', 'high']
    df_acc = df_acc[df_acc['part2'].isin(cfg_levels)].copy()
    if df_acc.empty:
        print("No low/medium/high configs found for source experiments.")
        return

    # X-axis Group order
    unique_groups = sorted(df_acc['Group'].unique())
    if split_mode == "quantile":
        unique_groups = sorted(unique_groups, key=lambda x: 0 if "Bottom" in x else 1)
    elif split_mode == "threshold":
        unique_groups = sorted(unique_groups, key=lambda x: 0 if "Low" in x else 1)
    elif split_mode == "certainty":
        unique_groups = sorted(unique_groups, key=lambda x: 0 if "Uncertain" in x else 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    exp_order = target_experiments

    legend_handles = None
    legend_labels = None

    for idx, exp in enumerate(exp_order):
        ax = axes[idx]
        df_exp = df_acc[df_acc['part1'] == exp].copy()

        print(f"\n[low/med/high] Processing {exp}: rows={len(df_exp)}")
        if df_exp.empty:
            ax.text(0.5, 0.5, f'No data for {exp}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(exp.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            continue

        # Aggregate: average Accuracy for same Group, same part2
        agg = df_exp.groupby(['Group', 'part2'])['Value'].mean().reset_index()

        sns.barplot(
            data=agg,
            x="Group",
            y="Value",
            hue="part2",
            order=unique_groups,
            hue_order=cfg_levels,
            palette=sns.color_palette("Set2", n_colors=len(cfg_levels)),
            edgecolor="white",
            linewidth=1,
            ax=ax,
        )

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.get_legend():
            ax.get_legend().remove()

        ax.set_title(exp.replace('_', ' ').title(), fontsize=11, fontweight='bold', pad=5)
        ax.set_ylim(0.2, 1.0)
        ax.set_yticks(np.arange(0.2, 1.01, 0.1))
        ax.set_xlabel("Belief Group", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=0, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Config",
            loc='upper center',
            bbox_to_anchor=(0.5, 1.03),
            ncol=len(cfg_levels),
            fontsize=8,
            title_fontsize=9,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    clean_split = split_mode
    final_filename = f"source_low_medium_high_{score_mode}_{clean_split}_v11.png"
    plt.savefig(os.path.join(output_dir, final_filename), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  -> Saved: {final_filename}")

def main():
    parser = argparse.ArgumentParser(description="Combine all experiments in 2x2 grid (V11)")
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSON file")
    parser.add_argument("--output_dir", type=str, default="plots_bar_v11", help="Output directory")
    parser.add_argument("--quantile", type=float, default=0.25, help="Quantile fraction (default 0.25)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Threshold for belief score split")
    parser.add_argument("--min_py", type=float, default=0.8, help="Minimum p_y")
    parser.add_argument(
        "--acc_match_mode",
        type=str,
        default="entity",
        choices=["entity", "sentence", "both"],
        help="Accuracy matching mode: entity / sentence / both",
    )
    parser.add_argument("--score_mode", type=str, default="geo_mean", choices=["geo_mean", "arith_mean"])
    parser.add_argument("--modes", nargs="+", default=["quantile"], choices=["quantile", "threshold", "certainty"])
    args = parser.parse_args()

    # Set global matching mode
    global ACCURACY_MATCH_MODE
    ACCURACY_MATCH_MODE = args.acc_match_mode

    set_style()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input_file}...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if not data:
        print("Error: Empty data file.")
        return

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
    MIN_PY = args.min_py
    
    # Process Data (same as v10)
    for mode_full in found_modes:
        parsed = parse_mode_generic(mode_full)
        part1 = parsed['part1']
        part2 = parsed['part2']
        part3 = parsed['part3']
        
        stages_to_process = [("Initial", None)]
        for lvl in [1, 2, 3]:
            key_check = f"extracted_followup_entities_{mode_full}_lvl{lvl}"
            has_lvl = False
            for s_item in data[:100]:
                if key_check in s_item:
                    has_lvl = True
                    break
            if has_lvl:
                stages_to_process.append((f"Followup_Lvl{lvl}", lvl))
        
        for item in tqdm(data, desc=f"Analyzing {mode_full}", leave=False):
            belief_result = item.get("belief_result", {})
            p_y = belief_result.get("p_y")
            belief_score = belief_result.get("score")
            
            if p_y is None or float(p_y) <= MIN_PY:
                continue
            
            baseline_metrics = get_baseline_metrics(item)
            if not baseline_metrics:
                continue

            for stage_name, level in stages_to_process:
                metrics = get_metrics_for_stage(item, mode_full, stage_name, level)
                if not metrics:
                    continue
                
                for metric_name, val in metrics.items():
                    rows.append({
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

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid data found matching criteria (p_y filter).")
        return
        
    print(f"\nCollected {len(df)} data points.")
    
    # Debug: Check what experiments we have
    unique_part1 = df['part1'].unique()
    print(f"Detected experiments (part1): {sorted(unique_part1)}")
    
    # Check for target experiments
    target_experiments = ['asch_conflict', 'asch_misleading', 'source_misleading', 'source_conflict']
    for exp in target_experiments:
        df_exp = df[df['part1'] == exp]
        if df_exp.empty:
            print(f"  WARNING: No data found for {exp}")
        else:
            unique_part2 = sorted(df_exp['part2'].unique())
            print(f"  {exp}: {len(df_exp)} rows, part2={unique_part2}")
    
    split_config = {
        "quantile": args.quantile,
        "threshold": args.threshold
    }
    
    # Plot combined figure
    for split_mode in args.modes:
        plot_v11_combined(
            df, args.output_dir,
            split_mode=split_mode,
            split_config=split_config,
            score_mode=args.score_mode,
            min_py=args.min_py
        )
        # Draw an additional figure comparing low/medium/high for Source experiment
        plot_v11_low_medium_high(
            df, args.output_dir,
            split_mode=split_mode,
            split_config=split_config,
            score_mode=args.score_mode,
            min_py=args.min_py
        )
            
    print(f"\nAll Done (V11). Outputs in {args.output_dir}")

if __name__ == "__main__":
    main()
