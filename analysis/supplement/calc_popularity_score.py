#!/usr/bin/env python3
"""
Calculate popularity and difficulty scores for data
Use LLM API to score OQ+OA pairs on two dimensions (1-10 scale each):
- Popularity: How common and well-known the question-answer pair is
- Difficulty: How difficult the question is to answer correctly
"""
import json
import argparse
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI
try:
    from zai import ZhipuAiClient
    _ZAI_AVAILABLE = True
except ImportError:
    _ZAI_AVAILABLE = False
    ZhipuAiClient = None

def create_popularity_prompt(question, answer):
    """
    Create a prompt for evaluating popularity and difficulty
    """
    prompt = f"""Please evaluate the following question-answer pair on two dimensions, each on a scale of 1-10:

Question: {question}
Answer: {answer}

Dimension A - Popularity (1-10):
Evaluate how popular, common, and well-known this question-answer pair is. Consider:
- Is this question frequently asked or commonly encountered?
- Does this knowledge belong to common sense or basic knowledge?
- How likely is this question-answer pair to appear frequently in training data?
- Is this information widely accessible and familiar to the general public?
Higher scores (8-10) indicate very popular/common content. Lower scores (1-3) indicate obscure/specialized content.

Dimension B - Difficulty (1-10):
Evaluate how difficult this question is to answer correctly. Consider:
- How much specialized knowledge or reasoning is required?
- Is this a straightforward factual question or does it require complex understanding?
- Would a typical person be able to answer this correctly?
- Does answering require domain expertise or advanced knowledge?
Higher scores (8-10) indicate very difficult questions. Lower scores (1-3) indicate easy/trivial questions.

Please output your evaluation in the following format exactly:
Popularity: [integer 1-10]
Difficulty: [integer 1-10]

Output only these two lines with no other text."""
    return prompt

def extract_scores_from_response(response_text):
    """
    Extract popularity and difficulty scores from model response
    Returns (popularity_score, difficulty_score) or (None, None) if extraction fails
    """
    import re
    response = response_text.strip()
    
    # Try to extract both scores using the expected format
    # Pattern: "Popularity: X" and "Difficulty: Y"
    popularity_match = re.search(r'popularity[：:]\s*(\d+)', response, re.IGNORECASE)
    difficulty_match = re.search(r'difficulty[：:]\s*(\d+)', response, re.IGNORECASE)
    
    popularity_score = None
    difficulty_score = None
    
    if popularity_match:
        try:
            score = int(popularity_match.group(1))
            if 1 <= score <= 10:
                popularity_score = score
        except:
            pass
    
    if difficulty_match:
        try:
            score = int(difficulty_match.group(1))
            if 1 <= score <= 10:
                difficulty_score = score
        except:
            pass
    
    # If we got both scores, return them
    if popularity_score is not None and difficulty_score is not None:
        return popularity_score, difficulty_score
    
    # Fallback: try to find two numbers in sequence
    all_numbers = re.findall(r'\b([1-9]|10)\b', response)
    if len(all_numbers) >= 2:
        try:
            score1 = int(all_numbers[0])
            score2 = int(all_numbers[1])
            if 1 <= score1 <= 10 and 1 <= score2 <= 10:
                # Assume first is popularity, second is difficulty
                return score1, score2
        except:
            pass
    
    # If still not found, return None for both
    return None, None

def process_data_with_api(data, output_file, provider='deepseek', max_workers=8, batch_size=None):
    """
    Process data in batch using API, score each OQ+OA pair
    Supports resume from checkpoint if output_file already exists
    """
    # Check if output file exists and load existing results
    existing_data = {}
    if os.path.exists(output_file):
        print(f"Found existing output file: {output_file}")
        print("Loading existing results to resume processing...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data_list = json.load(f)
                # Create a lookup dict by (question, answer) tuple
                for item in existing_data_list:
                    key = (item.get('original_question', ''), item.get('original_answer', ''))
                    existing_data[key] = item
            print(f"  Loaded {len(existing_data)} existing items")
        except Exception as e:
            print(f"  Warning: Failed to load existing file: {e}")
            print("  Will process all items from scratch")
    
    # Identify items that need processing
    items_to_process = []
    items_already_processed = []
    for item in data:
        key = (item.get('original_question', ''), item.get('original_answer', ''))
        if key in existing_data:
            existing_item = existing_data[key]
            # Check if both scores are available
            if (existing_item.get('popularity_score') is not None and 
                existing_item.get('difficulty_score') is not None):
                # Copy scores from existing data
                item['popularity_score'] = existing_item.get('popularity_score')
                item['difficulty_score'] = existing_item.get('difficulty_score')
                item['popularity_response'] = existing_item.get('popularity_response', '')
                items_already_processed.append(item)
                continue
        items_to_process.append((len(items_to_process), item))
    
    print(f"\nProcessing status:")
    print(f"  Total items: {len(data)}")
    print(f"  Already processed: {len(items_already_processed)}")
    print(f"  Need processing: {len(items_to_process)}")
    
    if len(items_to_process) == 0:
        print("\nAll items already processed. Skipping API calls.")
        popularity_scores = [item.get('popularity_score') for item in data 
                            if item.get('popularity_score') is not None]
        difficulty_scores = [item.get('difficulty_score') for item in data 
                            if item.get('difficulty_score') is not None]
        return data, popularity_scores, difficulty_scores
    
    # Initialize API client
    api_key = os.getenv("DEEPSEEK_API_KEY") if provider == "deepseek" else os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY environment variable not set.")
    
    base_url = "https://www.dmxapi.cn/v1" if provider == "deepseek" else None
    model_name = os.getenv("DEEPSEEK_MODEL_NAME", "DeepSeek-V3.2") if provider == "deepseek" else os.getenv("ZHIPU_MODEL_NAME", "glm-4-plus")
    
    # provider=provider,api_key=api_key,base_url=base_url,model_name=model_name,
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Build prompts only for items that need processing
    prompts = []
    for idx, item in items_to_process:
        question = item.get('original_question', '')
        answer = item.get('original_answer', '')
        prompt = create_popularity_prompt(question, answer)
        prompts.append(prompt)
    
    # Batch API calls with concurrent processing
    print(f"\nCalling API ({provider}) with {max_workers} workers for {len(prompts)} items...")
    
    def _call_api_single(prompt, retries=3):
        """Single API call with retry logic"""
        for attempt in range(retries):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 200,
                }
                response = client.chat.completions.create(**kwargs)
                # Extract response text
                if not response or not getattr(response, "choices", None):
                    return ""
                message = response.choices[0].message
                if message is None:
                    return ""
                if isinstance(message, dict):
                    content = message.get("content", "")
                else:
                    content = getattr(message, "content", "")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            parts.append(item.get("text", "") or item.get("content", ""))
                        else:
                            parts.append(str(item))
                    content = "".join(parts)
                elif not isinstance(content, str):
                    content = str(content)
                return content.strip()
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"API Error: {e}")
                    return ""
        return ""
    
    # Execute concurrent API calls
    responses = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_call_api_single, prompt): i 
            for i, prompt in enumerate(prompts)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(prompts), desc=f"Processing [{provider}]"):
            idx = future_to_idx[future]
            responses[idx] = future.result()
    
    # Extract scores and update data
    popularity_scores = []
    difficulty_scores = []
    failed_count = 0
    
    # First, collect scores from already processed items
    for item in items_already_processed:
        if item.get('popularity_score') is not None and item.get('difficulty_score') is not None:
            popularity_scores.append(item.get('popularity_score'))
            difficulty_scores.append(item.get('difficulty_score'))
    
    # Then, process new responses
    for (batch_idx, item), response in zip(items_to_process, responses):
        popularity_score, difficulty_score = extract_scores_from_response(response)
        if popularity_score is None or difficulty_score is None:
            failed_count += 1
            question_preview = item.get('original_question', '')[:50]
            print(f"Warning: Failed to extract scores (batch idx {batch_idx}): Q: {question_preview}... Response: {response[:100]}")
        
        item['popularity_score'] = popularity_score
        item['difficulty_score'] = difficulty_score
        item['popularity_response'] = response
        if popularity_score is not None and difficulty_score is not None:
            popularity_scores.append(popularity_score)
            difficulty_scores.append(difficulty_score)
    
    print(f"\nProcessing complete:")
    print(f"  Total items: {len(data)}")
    print(f"  Successfully extracted scores: {len(popularity_scores)}")
    print(f"  Failed extractions: {failed_count}")
    if popularity_scores:
        print(f"  Popularity score statistics:")
        print(f"    Mean: {sum(popularity_scores)/len(popularity_scores):.2f}")
        print(f"    Min: {min(popularity_scores)}")
        print(f"    Max: {max(popularity_scores)}")
        print(f"  Difficulty score statistics:")
        print(f"    Mean: {sum(difficulty_scores)/len(difficulty_scores):.2f}")
        print(f"    Min: {min(difficulty_scores)}")
        print(f"    Max: {max(difficulty_scores)}")
    
    # Save results (this will save all items, including previously processed ones)
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data)} items")
    
    return data, popularity_scores, difficulty_scores

def compare_groups(group_a_file, group_b_file, output_dir):
    """
    Compare popularity scores between two groups
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load data
    print("Loading data...")
    with open(group_a_file, 'r', encoding='utf-8') as f:
        group_a = json.load(f)
    with open(group_b_file, 'r', encoding='utf-8') as f:
        group_b = json.load(f)
    
    # Extract scores
    pop_scores_a = [item.get('popularity_score') for item in group_a if item.get('popularity_score') is not None]
    pop_scores_b = [item.get('popularity_score') for item in group_b if item.get('popularity_score') is not None]
    diff_scores_a = [item.get('difficulty_score') for item in group_a if item.get('difficulty_score') is not None]
    diff_scores_b = [item.get('difficulty_score') for item in group_b if item.get('difficulty_score') is not None]
    
    # Popularity comparison
    print(f"\n=== Popularity Score Comparison ===")
    print(f"Group A (Robust):")
    print(f"  Count: {len(pop_scores_a)}")
    if pop_scores_a:
        print(f"  Mean: {np.mean(pop_scores_a):.2f}")
        print(f"  Std: {np.std(pop_scores_a):.2f}")
        print(f"  Median: {np.median(pop_scores_a):.2f}")
        print(f"  Min: {np.min(pop_scores_a)}")
        print(f"  Max: {np.max(pop_scores_a)}")
    
    print(f"\nGroup B (Memorized):")
    print(f"  Count: {len(pop_scores_b)}")
    if pop_scores_b:
        print(f"  Mean: {np.mean(pop_scores_b):.2f}")
        print(f"  Std: {np.std(pop_scores_b):.2f}")
        print(f"  Median: {np.median(pop_scores_b):.2f}")
        print(f"  Min: {np.min(pop_scores_b)}")
        print(f"  Max: {np.max(pop_scores_b)}")
    
    if pop_scores_a and pop_scores_b:
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(pop_scores_a, pop_scores_b)
            print(f"\nStatistical Test (t-test):")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  Result: Significant difference (p < 0.05)")
            else:
                print(f"  Result: No significant difference (p >= 0.05)")
        except ImportError:
            print("\nNote: scipy not available, skipping statistical test")
    
    # Difficulty comparison
    print(f"\n=== Difficulty Score Comparison ===")
    print(f"Group A (Robust):")
    print(f"  Count: {len(diff_scores_a)}")
    if diff_scores_a:
        print(f"  Mean: {np.mean(diff_scores_a):.2f}")
        print(f"  Std: {np.std(diff_scores_a):.2f}")
        print(f"  Median: {np.median(diff_scores_a):.2f}")
        print(f"  Min: {np.min(diff_scores_a)}")
        print(f"  Max: {np.max(diff_scores_a)}")
    
    print(f"\nGroup B (Memorized):")
    print(f"  Count: {len(diff_scores_b)}")
    if diff_scores_b:
        print(f"  Mean: {np.mean(diff_scores_b):.2f}")
        print(f"  Std: {np.std(diff_scores_b):.2f}")
        print(f"  Median: {np.median(diff_scores_b):.2f}")
        print(f"  Min: {np.min(diff_scores_b)}")
        print(f"  Max: {np.max(diff_scores_b)}")
    
    if diff_scores_a and diff_scores_b:
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(diff_scores_a, diff_scores_b)
            print(f"\nStatistical Test (t-test):")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  Result: Significant difference (p < 0.05)")
            else:
                print(f"  Result: No significant difference (p >= 0.05)")
        except ImportError:
            pass
    
    # Plot distributions
    if pop_scores_a and pop_scores_b and diff_scores_a and diff_scores_b:
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
        })
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        boxplot_style = {
            'boxprops': {'linewidth': 2.0},
            'whiskerprops': {'linewidth': 2.0},
            'capprops': {'linewidth': 2.0},
            'medianprops': {'linewidth': 2.2},
            'flierprops': {'markersize': 4, 'markeredgewidth': 1.0},
        }
        
        # Popularity histogram (top-left)
        axes[0, 0].hist(pop_scores_a, bins=10, alpha=0.7, label='High NCB', color='blue', edgecolor='black')
        axes[0, 0].hist(pop_scores_b, bins=10, alpha=0.7, label='Low NCB', color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Popularity Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Popularity Distribution')
        axes[0, 0].legend(loc="upper center")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Difficulty histogram (top-right)
        axes[0, 1].hist(diff_scores_a, bins=10, alpha=0.7, label='High NCB', color='blue', edgecolor='black')
        axes[0, 1].hist(diff_scores_b, bins=10, alpha=0.7, label='Low NCB', color='red', edgecolor='black')
        axes[0, 1].set_xlabel('Difficulty Score')
        axes[0, 1].set_ylabel('')
        axes[0, 1].set_title('Difficulty Distribution')
        axes[0, 1].legend(loc="upper center")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Popularity box plot (bottom-left)
        axes[1, 0].boxplot([pop_scores_a, pop_scores_b], tick_labels=['High NCB', 'Low NCB'], **boxplot_style)
        axes[1, 0].set_ylabel('Popularity Score')
        axes[1, 0].set_title('Popularity Box Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Difficulty box plot (bottom-right)
        axes[1, 1].boxplot([diff_scores_a, diff_scores_b], tick_labels=['High NCB', 'Low NCB'], **boxplot_style)
        axes[1, 1].set_ylabel('')
        axes[1, 1].set_title('Difficulty Box Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'popularity_difficulty_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        # save as PDF for better quality
        pdf_file = os.path.join(output_dir, 'popularity_difficulty_comparison.pdf')
        plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plot to {plot_file} and {pdf_file}")

def main():
    parser = argparse.ArgumentParser(description='Calculate popularity and difficulty scores for OQ+OA pairs')
    parser.add_argument('--group_a', type=str, required=True, help='Path to group_a_robust.json')
    parser.add_argument('--group_b', type=str, required=True, help='Path to group_b_memorized.json')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--provider', type=str, default='deepseek', choices=['deepseek', 'zhipu'], help='API provider')
    parser.add_argument('--max_workers', type=int, default=8, help='Number of concurrent API calls')
    parser.add_argument('--skip_api', action='store_true', help='Skip API calls, only compare existing results')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.group_a)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process Group A
    output_a = os.path.join(args.output_dir, 'group_a_robust_with_popularity.json')
    if not args.skip_api:
        print("=" * 60)
        print("Processing Group A (Robust)...")
        print("=" * 60)
        with open(args.group_a, 'r', encoding='utf-8') as f:
            data_a = json.load(f)
        process_data_with_api(data_a, output_a, provider=args.provider, max_workers=args.max_workers)
    else:
        output_a = args.group_a
    
    # Process Group B
    output_b = os.path.join(args.output_dir, 'group_b_memorized_with_popularity.json')
    if not args.skip_api:
        print("\n" + "=" * 60)
        print("Processing Group B (Memorized)...")
        print("=" * 60)
        with open(args.group_b, 'r', encoding='utf-8') as f:
            data_b = json.load(f)
        process_data_with_api(data_b, output_b, provider=args.provider, max_workers=args.max_workers)
    else:
        output_b = args.group_b
    
    # Compare the two groups
    print("\n" + "=" * 60)
    print("Comparing Groups...")
    print("=" * 60)
    compare_groups(output_a, output_b, args.output_dir)
    
    print("\nDone!")

if __name__ == '__main__':
    main()

