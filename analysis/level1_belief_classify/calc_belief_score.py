from collections import Counter
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

import re

class EntityNormalizer:
    def __init__(self, prefix_rules=None, suffix_rules=None):
        """
        Initialize cleaner
        :param prefix_rules: Prefix regex list (loaded from config)
        :param suffix_rules: Suffix regex list (loaded from config)
        """
        self.prefix_rules = prefix_rules if prefix_rules else []
        self.suffix_rules = suffix_rules if suffix_rules else []
        self.QUALITY_LIMIT = 100
        
        # Define refusal word set (hardcoded common noise)
        self.reject_words = {
            "none", "n/a", "unknown", "null", "nil", "not available", "no answer"
        }

    def _universal_clean(self, text):
        """
        [General method] Revised: use whitelist mode to thoroughly clean brackets and quotes
        """
        if not text:
            return None
            
        # 0. [New feature] Newline check
        # If entity contains newlines, multi-line text was extracted, usually an error
        if "\n" in text or "\r" in text:
            return None
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Refusal word check
        if text.strip() in self.reject_words:
            return None
            
        # 3. [Fixed logic] Whitelist cleaning
        # Explanation:
        # [^\w\s\.\-] means: match everything that is "not" (word char, space, dot, hyphen)
        # i.e., replace all brackets(), quotes"", asterisks* etc. with empty
        text = re.sub(r'[^\w\s\.\-]', '', text)
        
        # 4. Normalize whitespace (prevent extra spaces after removing punctuation)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _specific_clean(self, text):
        """
        [Specialized method] Prefix/suffix handling depending on external config
        """
        if not text:
            return None

        # 1. Handle prefixes
        for pattern in self.prefix_rules:
            # Execute replacement
            new_text = re.sub(pattern, "", text).strip()
            # Only effective when remaining length > 1 (prevent deleting single char like "A")
            if len(new_text) > 1:
                text = new_text

        # 2. Handle suffixes
        for pattern in self.suffix_rules:
            # Execute replacement
            new_text = re.sub(pattern, "", text).strip()
            # [Safety check]: Only accept when remaining length > 2 after suffix removal
            # Prevent: "Gap Inc" -> "Gap" (OK), "IT Inc" -> "IT" (risky), "A Inc" -> "A" (reject)
            if len(new_text) > 2:
                text = new_text
            
        return text

    def normalize(self, raw_text):
        """
        Main entry function
        """
        if raw_text == "NOT_ATTEMPTED":
            return "not_attempted"
        # Phase 1: General cleaning
        text = self._universal_clean(raw_text)
        # If empty after general cleaning (e.g., None), return directly
        if not text:
            return None
        if len(text) > self.QUALITY_LIMIT:
            return None
            
        # Phase 2: Specialized cleaning (load config)
        text = self._specific_clean(text)
        
        return text


def analyze_and_split_data(data, output_dir="01_belief_diagnosis", quantile=0.25, balance_by_py=False, num_bins=5):
    """
    Input: data (complete data list with computed belief_result)
    Parameters: 
        output_dir: Directory to save results
        quantile: Split ratio (default 0.25, i.e., Top 25% and Bottom 25%)
        balance_by_py: Whether to balance groups by original question score (p_y)
        num_bins: Number of bins when balance_by_py=True
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"=== Starting data statistics and splitting (Quantile={quantile}, Balance_by_py={balance_by_py}) ===")

    # ==========================================
    # 1. Global Statistics
    # ==========================================
    total_count = len(data)
    valid_items = [d for d in data if d.get("belief_result", {}).get("valid", False)]
    valid_count = len(valid_items)
    
    valid_ratio = (valid_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"\n[1] Basic Statistics:")
    print(f"    - Total raw data: {total_count}")
    print(f"    - Valid data (y=y*): {valid_count}")
    print(f"    - Valid ratio: {valid_ratio:.2f}%")
    print(f"    - Filtered data (y!=y*): {total_count - valid_count}")

    if valid_count == 0:
        print("Warning: No Valid data, cannot proceed with analysis.")
        return

    # ==========================================
    # 2. Score Distribution Analysis
    # ==========================================
    # Extract scores from all Valid data
    scores = [d["belief_result"]["score"] for d in valid_items]
    scores_np = np.array(scores)
    
    # Extract original question scores
    p_ys = [d["belief_result"].get("p_y", 0.0) for d in valid_items]
    p_ys_np = np.array(p_ys)
    
    print(f"\n[2] Score Distribution Statistics (Valid data only):")
    print(f"    - Belief Score Mean: {np.mean(scores_np):.4f}")
    print(f"    - Belief Score Median: {np.median(scores_np):.4f}")
    print(f"    - Belief Score Std: {np.std(scores_np):.4f}")
    print(f"    - Belief Score Max: {np.max(scores_np):.4f}")
    print(f"    - Belief Score Min: {np.min(scores_np):.4f}")
    print(f"    - Original question score (p_y) Mean: {np.mean(p_ys_np):.4f}")
    print(f"    - Original question score (p_y) Median: {np.median(p_ys_np):.4f}")

    # --- Core chart: Plot Belief Score histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(scores_np, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Belief Scores (Valid Samples)', fontsize=15)
    plt.xlabel('Belief Score (Geometric Mean)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    
    # Save figure
    plot_path = os.path.join(output_dir, 'belief_score_distribution.png')
    plt.savefig(plot_path)
    print(f"    - Distribution histogram saved to: {plot_path}")
    plt.close()
    
    # --- Original question score histogram ---
    plt.figure(figsize=(10, 6))
    plt.hist(p_ys_np, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Original Question Scores (p_y)', fontsize=15)
    plt.xlabel('Original Question Score (p_y)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    
    plot_path_py = os.path.join(output_dir, 'p_y_distribution.png')
    plt.savefig(plot_path_py)
    print(f"    - Original question score histogram saved to: {plot_path_py}")
    plt.close()

    # ==========================================
    # 3. Data Splitting (High vs Low)
    # ==========================================
    n = len(valid_items)
    cut_off_count = int(n * quantile)
    
    if cut_off_count == 0:
        print("Warning: Valid data too few, cannot split by ratio.")
        return
    
    if balance_by_py:
        print(f"\n[3] Balanced grouping based on original question score (Top/Bottom {quantile*100}%)")
        print(f"    - Using {num_bins} bins for stratified sampling")
        
        # Sort by original question score (p_y)
        valid_items_sorted_by_py = sorted(valid_items, key=lambda x: x["belief_result"].get("p_y", 0.0))
        
        # Calculate size of each bin
        bin_size = len(valid_items_sorted_by_py) // num_bins
        remainder = len(valid_items_sorted_by_py) % num_bins
        
        # Bin assignment
        bins = []
        start_idx = 0
        for i in range(num_bins):
            # Allocate remaining items
            current_bin_size = bin_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_bin_size
            bins.append(valid_items_sorted_by_py[start_idx:end_idx])
            start_idx = end_idx
        
        # Select high and low belief score samples from each bin
        group_high = []
        group_low = []
        
        # Calculate samples to select from each bin
        samples_per_bin_high = cut_off_count // num_bins
        samples_per_bin_low = cut_off_count // num_bins
        remainder_high = cut_off_count % num_bins
        remainder_low = cut_off_count % num_bins
        
        for i, bin_items in enumerate(bins):
            # Sort by belief score within each bin
            bin_items_sorted = sorted(bin_items, key=lambda x: x["belief_result"]["score"], reverse=True)
            
            # Calculate high score samples to select from current bin
            current_high_count = samples_per_bin_high + (1 if i < remainder_high else 0)
            # Select high score samples from current bin
            if current_high_count > 0 and len(bin_items_sorted) > 0:
                group_high.extend(bin_items_sorted[:current_high_count])
            
            # Calculate low score samples to select from current bin
            current_low_count = samples_per_bin_low + (1 if i < remainder_low else 0)
            # Select low score samples from current bin
            if current_low_count > 0 and len(bin_items_sorted) > 0:
                group_low.extend(bin_items_sorted[-current_low_count:])
        
        # Ensure both groups have correct count
        if len(group_high) > cut_off_count:
            group_high = group_high[:cut_off_count]
        if len(group_low) > cut_off_count:
            group_low = group_low[:cut_off_count]
        
        # If samples insufficient, try to supplement
        if len(group_high) < cut_off_count:
            # Sort all valid_items by belief score, select unselected high score samples
            all_high_scores = sorted([item for item in valid_items if item not in group_high and item not in group_low], 
                                    key=lambda x: x["belief_result"]["score"], reverse=True)
            group_high.extend(all_high_scores[:cut_off_count - len(group_high)])
        
        if len(group_low) < cut_off_count:
            # Sort all valid_items by belief score, select unselected low score samples
            all_low_scores = sorted([item for item in valid_items if item not in group_high and item not in group_low], 
                                   key=lambda x: x["belief_result"]["score"])
            group_low.extend(all_low_scores[:cut_off_count - len(group_low)])
    else:
        # Original grouping: sort by belief score only
        print(f"\n[3] Original grouping method (Top/Bottom {quantile*100}%):")
        valid_items.sort(key=lambda x: x["belief_result"]["score"], reverse=True)
        
        # Split
        group_high = valid_items[:cut_off_count]       # Top 25%
        group_low = valid_items[-cut_off_count:]       # Bottom 25%
    
    # Calculate middle part (if needed)
    group_middle = [item for item in valid_items if item not in group_high and item not in group_low]

    print(f"    - Group A (Robust/High): {len(group_high)} items")
    if group_high:
        print(f"      -> Belief score range: [{min(item['belief_result']['score'] for item in group_high):.4f}, {max(item['belief_result']['score'] for item in group_high):.4f}]")
        print(f"      -> Original question score (p_y) mean: {np.mean([item['belief_result'].get('p_y', 0.0) for item in group_high]):.4f}")
    
    print(f"    - Group B (Memorized/Low): {len(group_low)} items")
    if group_low:
        print(f"      -> Belief score range: [{min(item['belief_result']['score'] for item in group_low):.4f}, {max(item['belief_result']['score'] for item in group_low):.4f}]")
        print(f"      -> Original question score (p_y) mean: {np.mean([item['belief_result'].get('p_y', 0.0) for item in group_low]):.4f}")
    
    print(f"    - Discarded (Middle): {len(group_middle)} items (excluded from subsequent experiments)")
    
    # Verify grouping balance effect
    if group_high and group_low:
        # Get original question scores for both groups
        high_p_ys = [item['belief_result'].get('p_y', 0.0) for item in group_high]
        low_p_ys = [item['belief_result'].get('p_y', 0.0) for item in group_low]
        
        # Calculate basic statistics
        high_py_mean = np.mean(high_p_ys)
        low_py_mean = np.mean(low_p_ys)
        high_py_median = np.median(high_p_ys)
        low_py_median = np.median(low_p_ys)
        high_py_std = np.std(high_p_ys)
        low_py_std = np.std(low_p_ys)
        
        # Calculate difference
        py_diff_mean = high_py_mean - low_py_mean
        py_diff_median = high_py_median - low_py_median
        
        print(f"\n[4] Original question score balance analysis:")
        print(f"    - High group original question score statistics:")
        print(f"      -> Mean: {high_py_mean:.4f}, Median: {high_py_median:.4f}, Std: {high_py_std:.4f}")
        print(f"    - Low group original question score statistics:")
        print(f"      -> Mean: {low_py_mean:.4f}, Median: {low_py_median:.4f}, Std: {low_py_std:.4f}")
        print(f"    - Difference analysis:")
        print(f"      -> Mean difference: High group - Low group = {py_diff_mean:.4f}")
        print(f"      -> Median difference: High group - Low group = {py_diff_median:.4f}")
        
        # Evaluate balance effect
        if balance_by_py:
            # Define reasonable difference threshold (adjust as needed)
            threshold = 0.1
            if abs(py_diff_mean) <= threshold:
                print(f"    - ✓ Good balance: original question score mean difference within threshold ({threshold})")
            else:
                print(f"    - ⚠️  Fair balance: suggest adjusting bin count or trying other grouping strategies")
        else:
            if py_diff_mean > 0.2:
                print(f"    - ⚠️  Warning: Under original grouping, high group original question score is significantly higher than low group")
                print(f"    - 💡 Suggestion: Use --balance_by_py=True to enable balanced grouping")
        
        # Plot original question score distribution comparison for both groups
        plt.figure(figsize=(12, 6))
        plt.hist(high_p_ys, bins=30, alpha=0.6, color='blue', label='High Belief Group')
        plt.hist(low_p_ys, bins=30, alpha=0.6, color='red', label='Low Belief Group')
        plt.title('Comparison of Original Question Scores (p_y) Between Groups', fontsize=15)
        plt.xlabel('Original Question Score (p_y)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.5)
        
        # Add mean lines
        plt.axvline(x=high_py_mean, color='blue', linestyle='--', label=f'High Group Mean: {high_py_mean:.4f}')
        plt.axvline(x=low_py_mean, color='red', linestyle='--', label=f'Low Group Mean: {low_py_mean:.4f}')
        
        # Save figure
        comparison_plot_path = os.path.join(output_dir, 'p_y_comparison.png')
        plt.savefig(comparison_plot_path)
        print(f"    - Original question score distribution comparison saved to: {comparison_plot_path}")
        plt.close()

    # ==========================================
    # 4. Save Files
    # ==========================================
    path_high = os.path.join(output_dir, 'group_a_robust.json')
    path_low = os.path.join(output_dir, 'group_b_memorized.json')
    path_stats = os.path.join(output_dir, 'stats_summary.txt') # Save statistics log
    
    with open(path_high, 'w', encoding='utf-8') as f:
        json.dump(group_high, f, indent=2, ensure_ascii=False)
        
    with open(path_low, 'w', encoding='utf-8') as f:
        json.dump(group_low, f, indent=2, ensure_ascii=False)
        
    print(f"\n[4] Files saved:")
    print(f"    - Robust group:   {path_high}")
    print(f"    - Memorized group: {path_low}")


NORMALIZER = EntityNormalizer()
def normalize_entity(text):
    """
    Entity normalization function.
    Currently only strips leading/trailing spaces; can add lower() or more complex regex rules if needed.
    Note: Better to maintain consistency with entity cleaning logic in generation phase (e.g., EntityNormalizer).
    """
    if not isinstance(text, str):
        return str(text)
    return NORMALIZER.normalize(text)

def calculate_belief_metrics(data, match_type="strict", neighbor_agg="geo_mean"):
    """
    Input: data (list of dict)
    Parameters: 
        match_type: "strict" (exact match) or "loose" (containment)
        neighbor_agg: "geo_mean" (geometric mean, default) or "arith_mean" (arithmetic mean)
    Output: data with computed results (new belief_score fields added)
    """
    
    # Small value to prevent log(0)
    EPSILON = 1e-10
    
    results = []

    for item_idx, item in enumerate(data):
        # ==========================================
        # Step 1: Process Original Question
        # ==========================================
        conf_data = item.get("confidence", {})["consistency_confidence"]
        
        # Get ground truth answer
        golden_answer = normalize_entity(conf_data.get("golden_answer", ""))
        
        # Get sampled responses and Normalize
        raw_entities = conf_data.get("all_entities", [])
        normalized_entities = [normalize_entity(e) for e in raw_entities]
        
        if not normalized_entities:
            # Exception handling: no sampled data
            item["belief_result"] = {"valid": False, "reason": "no_samples"}
            continue

        # Calculate dominant answer (Majority Vote)
        counts = Counter(normalized_entities)
        dominant_answer, dominant_count = counts.most_common(1)[0]
        
        # Calculate P(y): probability of dominant answer
        p_y = dominant_count / len(normalized_entities)
        
        # Core check: whether y equals y*
        if match_type == "loose":
            # Loose mode: a in b or b in a
            # Note: Handle empty string case to avoid misjudgment
            if not dominant_answer or not golden_answer:
                is_correct = (dominant_answer == golden_answer)
            else:
                is_correct = (dominant_answer in golden_answer) or (golden_answer in dominant_answer)
        else:
            # Strict mode: exact match
            is_correct = (dominant_answer == golden_answer)
        
        if not is_correct:
            # If dominant answer is wrong, mark directly without further calculation
            item["belief_result"] = {
                "valid": False,
                "reason": "wrong_answer",
                "dominant_answer": dominant_answer,
                "golden_answer": golden_answer,
                "p_y": p_y,
                "score": 0.0  # Can also be set to None
            }
            continue

        # ==========================================
        # Step 2: Process Neighbor Questions
        # ==========================================
        neighbors = item.get("neighbor_questions", [])
        neighbor_probs = [] # Store all p_i
        
        for nq in neighbors:
            n_correct_answer = nq.get("correct_answer", "").strip()
            n_responses = nq.get("responses", [])
            
            if not n_responses:
                # If a neighbor has no samples, treat as all wrong (p=0)
                neighbor_probs.append(0.0)
                continue
            
            # Calculate p_i: probability of neighbor answering correctly
            # Logic: Exact Match (lowercase)
            correct_count = sum(1 for r in n_responses if r.strip().lower() == n_correct_answer.lower())
            p_i = correct_count / len(n_responses)
            neighbor_probs.append(p_i)
            
        # ==========================================
        # Step 3: Apply Belief Score formula
        # Formula 1 (Geo Mean): ln(Score) = ln(P(y)) + Mean(ln(p_i))  => Score = P(y) * GeometricMean(p_i)
        # Formula 2 (Arith Mean): ln(Score) = ln(P(y)) + ln(Mean(p_i)) => Score = P(y) * Mean(p_i)
        # ==========================================
        
        # 3.1 Prepare data, use clip to prevent log(0)
        p_y_safe = max(p_y, EPSILON)
        
        if not neighbor_probs:
            # If no neighbor data, cannot calculate consistency
            final_score = 0.0 # Or equal to p_y, depending on definition
            log_score = -999.0
        else:
            neighbor_probs_safe = np.clip(neighbor_probs, EPSILON, 1.0)
            
            if neighbor_agg == "arith_mean":
                # Arithmetic mean mode: average p_i first, then log, finally add to ln(P(y))
                # Equivalent to Score = P(y) * ArithmeticMean(p_i)
                mean_neighbors = np.mean(neighbor_probs) # Use original probabilities for average, safe version not needed since p_i >= 0
                mean_neighbors_safe = max(mean_neighbors, EPSILON)
                
                log_score = np.log(p_y_safe) + np.log(mean_neighbors_safe)
            elif neighbor_agg == "robust_geo_mean":
                # Robust geometric mean mode: remove lowest score, then compute geometric mean
                # If only one neighbor, don't remove
                if len(neighbor_probs) > 1:
                    min_idx = np.argmin(neighbor_probs)
                    neighbor_probs_filtered = [p for i, p in enumerate(neighbor_probs) if i != min_idx]
                else:
                    neighbor_probs_filtered = neighbor_probs
                
                neighbor_probs_filtered_safe = np.clip(neighbor_probs_filtered, EPSILON, 1.0)
                mean_log_neighbors = np.mean(np.log(neighbor_probs_filtered_safe))
                log_score = np.log(p_y_safe) + mean_log_neighbors
            else:
                # Geometric mean mode (default): log p_i first, then average, finally add to ln(P(y))
                # Equivalent to Score = P(y) * GeometricMean(p_i)
                # mean_log_neighbors = (1/k) * sum(ln(p_i))
                mean_log_neighbors = np.mean(np.log(neighbor_probs_safe))
                log_score = np.log(p_y_safe) + mean_log_neighbors
            
            # 3.4 Convert back to 0-1 space
            final_score = np.exp(log_score)

        # ==========================================
        # Step 4: Write results
        # ==========================================
        item["belief_result"] = {
            "valid": True,
            "score": float(final_score),  # Final score
            "p_y": float(p_y),            # Original question confidence
            "neighbor_probs": [float(p) for p in neighbor_probs], # Each neighbor's accuracy
            "log_score": float(log_score) if neighbor_probs else -999.0
        }

    return data


def main():
    """
    Command line entry:
    - Read data from JSON results after gen_nq/gen_oq
    - Calculate belief score and write back
    - Optional: do distribution analysis and split data by quantile
    - Support normal grouping and grouping strategy balanced by original question score
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSON file containing confidence info and neighbor_questions (e.g., file generated by gen_nq)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON file path with belief_result field",
    )
    parser.add_argument(
        "--output_valid",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--output_invalid",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--analysis_output_dir",
        type=str,
        default="01_belief_diagnosis",
        help="Directory to save statistics and grouping results",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.25,
        help="Quantile for Top/Bottom grouping (default 0.25 means 25%%)",
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Only calculate belief score, skip distribution analysis and data splitting",
    )
    parser.add_argument(
        "--balance_by_py",
        action="store_true",
        help="Whether to balance groups by original question score (p_y), ensuring small score difference between high and low groups",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=5,
        help="Number of bins when using balanced grouping (default 5)",
    )
    parser.add_argument(
        "--match_type",
        type=str,
        default="loose",
        choices=["strict", "loose"],
        help="Match mode: 'strict' (exact match) or 'loose' (containment)",
    )
    parser.add_argument(
        "--neighbor_agg",
        type=str,
        default="geo_mean",
        choices=["geo_mean", "arith_mean", "robust_geo_mean"],
        help="Neighbor aggregation: 'geo_mean' (geometric mean, stricter), 'arith_mean' (arithmetic mean), 'robust_geo_mean' (geometric mean after removing lowest)",
    )
    args = parser.parse_args()

    # 1. Load data
    print(f"[Belief] Loading data from {args.input_file} ...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Calculate belief metrics
    print(f"[Belief] Calculating belief scores (Match Type: {args.match_type}, Neighbor Agg: {args.neighbor_agg}) ...")
    processed_data = calculate_belief_metrics(data, match_type=args.match_type, neighbor_agg=args.neighbor_agg)

    # 3. Write JSON with belief_result
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"[Belief] Saved belief-annotated data to: {args.output_file}")

    # 3.1 Write valid and invalid belief_result data separately
    valid_data = [item for item in processed_data if item["belief_result"]["valid"]]
    invalid_data = [item for item in processed_data if not item["belief_result"]["valid"]]
    with open(args.output_valid, "w", encoding="utf-8") as f:
        json.dump(valid_data, f, ensure_ascii=False, indent=2)
    with open(args.output_invalid, "w", encoding="utf-8") as f:
        json.dump(invalid_data, f, ensure_ascii=False, indent=2)

    # 4. Optional: Statistics and grouping
    if not args.skip_analysis:
        analyze_and_split_data(
            processed_data,
            output_dir=args.analysis_output_dir,
            quantile=args.quantile,
            balance_by_py=args.balance_by_py,
            num_bins=args.num_bins,
        )


if __name__ == "__main__":
    """
    Usage examples:
    
    # Original grouping (sort by belief score only)
    # python calc_belief_score.py --input_file input.json --output_file output.json
    
    # Balanced grouping (ensure small original question score difference between high and low groups)
    # python calc_belief_score.py --input_file input.json --output_file output.json --balance_by_py
    
    # Adjust bin count (default 5 bins)
    # python calc_belief_score.py --input_file input.json --output_file output.json --balance_by_py --num_bins 10
    
    Advantages of balanced grouping:
    1. Ensure similar original question score (p_y) distribution between high and low belief score groups
    2. Avoid confusing experiment results: high group may have performed better due to higher original question scores
    3. Provide more accurate validation of belief score effectiveness
    
    Output files:
    - belief_result field added to original data
    - 01_belief_diagnosis/group_a_robust.json - High group data
    - 01_belief_diagnosis/group_b_memorized.json - Low group data
    - 01_belief_diagnosis/p_y_comparison.png - Original question score distribution comparison between two groups
    """
    main()