#!/usr/bin/env python3

import argparse
import copy
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Ensure project root on path for `utils` imports (project root: confidence/analysis)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import flip_answer, load_json, save_json

from level2_belief_intervention.preprocess.llm_utils import default_sampling_params, init_llm
from level2_belief_intervention.preprocess.prompts import CONVERSION_PROMPT, REPLACEMENT_PROMPT


# =========
# 4-type generation (from hallu_nq_4type.py)
# =========
def build_hallu_variants(entry: Dict[str, Any], commonsense_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Prefer renamed key; fallback to legacy
    if "misleading_neighbor_questions" in entry:
        entry["right_hallu"] = entry.pop("misleading_neighbor_questions")
    elif "neighbor_questions" in entry:
        entry["right_hallu"] = entry.pop("neighbor_questions")
    else:
        entry["right_hallu"] = entry.get("right_hallu", [])

    wrong_hallu = []
    for item in entry["right_hallu"]:
        new_item = copy.deepcopy(item)
        new_item["correct_answer"] = flip_answer(new_item["correct_answer"])
        wrong_hallu.append(new_item)
    entry["wrong_hallu"] = wrong_hallu

    weak_hallu = []
    for item in entry["right_hallu"]:
        new_item = copy.deepcopy(item)
        new_item["correct_answer"] = "I don't know"
        new_item["expected_answer_type"] = "Uncertainty"
        weak_hallu.append(new_item)
    entry["weak_hallu"] = weak_hallu

    num_items = len(entry["right_hallu"])
    if num_items > 0:
        selected_cs = random.sample(commonsense_data, k=min(num_items, len(commonsense_data)))
        cs_hallu = []
        template = entry["right_hallu"][0] if entry["right_hallu"] else {}

        for cs_item in selected_cs:
            new_item = copy.deepcopy(template)
            new_item["question"] = cs_item["question"]
            new_item["correct_answer"] = cs_item["answer"]
            new_item["expected_answer_type"] = "Boolean"
            new_item["category"] = "commonsense"
            cs_hallu.append(new_item)
        entry["cs_hallu"] = cs_hallu
    else:
        entry["cs_hallu"] = []

    return entry


def run_gen_4type(args: argparse.Namespace) -> None:
    data = load_json(args.input_file)
    commonsense_data = load_json(args.commonsense_file)

    processed = [build_hallu_variants(copy.deepcopy(entry), commonsense_data) for entry in data]
    save_json(processed, args.output_file)
    print(f"Processed data saved to {args.output_file}")


# =========
# Conversion (from convert_questions.py)
# =========
def convert_question_to_statement(question_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        prompt = CONVERSION_PROMPT.format(
            question=question_data["question"],
            correct_answer=question_data["correct_answer"],
        )
        return {"prompt": prompt, "question_data": question_data}
    except Exception as e:
        print(f"Error in convert_question_to_statement: {str(e)}", file=sys.stderr)
        return None


def process_all_hallucinations(
    data: List[Dict[Any, Any]],
    llm,
    sampling_params,
    hallu_types: Tuple[str, ...] = ("right_hallu", "wrong_hallu", "weak_hallu", "cs_hallu"),
) -> List[Dict[str, Any]]:
    all_questions = []
    question_mapping = []

    print("Collecting questions from all hallucination types...")
    for item_idx, item in enumerate(data):
        for h_type in hallu_types:
            questions = item.get(h_type, [])
            for q_idx, question_data in enumerate(questions):
                question_info = convert_question_to_statement(question_data)
                if question_info:
                    all_questions.append(question_info)
                    question_mapping.append((item_idx, h_type, q_idx))

    if not all_questions:
        print("No questions found to convert.")
        return data

    print(f"Generating statements for {len(all_questions)} questions...")
    prompts = [q["prompt"] for q in all_questions]
    chat_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

    outputs = llm.chat(messages=chat_messages, sampling_params=sampling_params, use_tqdm=True)
    statements = [output.outputs[0].text.strip() for output in outputs]

    for i in range(len(data)):
        original_item = data[i]
        data[i] = {"metadata": original_item.copy()}
    for item in data:
        for h_type in hallu_types:
            item[f"converted_{h_type}"] = []

    print("Organizing results...")
    for i, (item_idx, h_type, q_idx) in enumerate(question_mapping):
        question_data = all_questions[i]["question_data"]
        statement = statements[i]

        converted_item = {
            "original_question": question_data["question"],
            "expected_answer_type": question_data["expected_answer_type"],
            "correct_answer": question_data["correct_answer"],
            "converted_statement": statement,
        }
        data[item_idx][f"converted_{h_type}"].append(converted_item)

    return data


# =========
# NQ conversion + hallucination replacement
# =========
def collect_base_neighbors(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect base original neighbor questions from the entry or metadata.
    """
    neighbors = entry.get("original_neighbor_questions")
    if neighbors is None:
        neighbors = entry.get("metadata", {}).get("original_neighbor_questions")
    if neighbors is None:
        neighbors = entry.get("neighbor_questions") or entry.get("metadata", {}).get("neighbor_questions")
    return neighbors or []


def convert_nq_to_statements(
    data: List[Dict[Any, Any]],
    llm,
    sampling_params,
) -> List[Optional[List[Dict[str, Any]]]]:
    """
    Convert base NQ neighbor QA pairs into statements.
    Returns a list aligned with data; each entry length matches original_neighbor_questions.
    """
    conversion_tasks = []
    mapping = []
    for idx, item in enumerate(data):
        neighbors = collect_base_neighbors(item)
        for n_i, qa in enumerate(neighbors):
            q = qa.get("question")
            a = qa.get("correct_answer")
            if not q or not a:
                continue
            prompt = CONVERSION_PROMPT.format(question=q, correct_answer=a)
            conversion_tasks.append({"prompt": prompt, "qa": qa, "local_idx": n_i})
            mapping.append((idx, n_i))

    if not conversion_tasks:
        return [None] * len(data)

    chat_messages = [[{"role": "user", "content": t["prompt"]}] for t in conversion_tasks]
    outputs = llm.chat(messages=chat_messages, sampling_params=sampling_params, use_tqdm=True)
    statements = [o.outputs[0].text.strip() for o in outputs]

    results: List[Optional[List[Dict[str, Any]]]] = [None] * len(data)
    for i, (idx, n_i) in enumerate(mapping):
        qa = conversion_tasks[i]["qa"]
        converted_item = {
            "original_question": qa.get("question"),
            "expected_answer_type": qa.get("expected_answer_type"),
            "correct_answer": qa.get("correct_answer"),
            "converted_statement": statements[i],
        }
        if results[idx] is None:
            results[idx] = []
        while len(results[idx]) <= n_i:
            results[idx].append({})
        results[idx][n_i] = converted_item
    return results


def convert_misleading_neighbors(
    data: List[Dict[Any, Any]],
    llm,
    sampling_params,
) -> List[Optional[List[Dict[str, Any]]]]:
    """
    Convert all misleading neighbor questions to statements.
    Returns a list aligned with data; each entry is a list of converted items or None.
    """
    tasks = []
    mapping = []  # (item_idx, neighbor_idx)
    for item_idx, item in enumerate(data):
        neighbors = item.get("misleading_neighbor_questions") or item.get("neighbor_questions") or []
        for n_idx, neighbor_q in enumerate(neighbors):
            q = neighbor_q.get("question", "")
            a = neighbor_q.get("correct_answer", "")
            if not q or not a:
                continue
            prompt = CONVERSION_PROMPT.format(question=q, correct_answer=a)
            tasks.append({"prompt": prompt, "neighbor": neighbor_q})
            mapping.append((item_idx, n_idx))

    if not tasks:
        return [None] * len(data)

    msgs = [[{"role": "user", "content": t["prompt"]}] for t in tasks]
    outputs = llm.chat(messages=msgs, sampling_params=sampling_params, use_tqdm=True)
    stmts = [o.outputs[0].text.strip() for o in outputs]

    results: List[Optional[List[Dict[str, Any]]]] = [None] * len(data)
    for i, (item_idx, n_idx) in enumerate(mapping):
        neighbor_q = tasks[i]["neighbor"]
        converted = {
            "original_question": neighbor_q.get("question", ""),
            "expected_answer_type": neighbor_q.get("expected_answer_type", ""),
            "correct_answer": neighbor_q.get("correct_answer", ""),
            "converted_statement": stmts[i],
        }
        if results[item_idx] is None:
            results[item_idx] = []
        # ensure order
        while len(results[item_idx]) <= n_idx:
            results[item_idx].append({})
        results[item_idx][n_idx] = converted

    return results


def replace_nq_subject_with_hallucination(
    data: List[Dict[Any, Any]],
    nq_statements: List[Optional[List[Dict[str, Any]]]],
    llm,
    sampling_params,
) -> List[Optional[List[Dict[str, Any]]]]:
    """
    Replace main subject in NQ statements with hallucination entity.
    Returns list aligned with data; each entry is a list (mirrors nq_statements per item).
    """
    replacement_tasks = []
    mapping = []  # (item_idx, local_idx)
    for idx, item in enumerate(data):
        nq_list = nq_statements[idx] if idx < len(nq_statements) else None
        if not nq_list:
            continue
        target_entity = (
            item.get("misleading_entity")
            or item.get("metadata", {}).get("misleading_entity")
        )
        for local_i, nq_item in enumerate(nq_list):
            if not nq_item:
                continue
            # FIXME: use original_answer instead of correct_answer
            original_entity = item.get("original_answer")
            if not target_entity or not original_entity:
                continue
            prompt = REPLACEMENT_PROMPT.format(
                original_entity=original_entity,
                target_entity=target_entity,
                statement=nq_item["converted_statement"],
            )
            replacement_tasks.append(
                {
                    "prompt": prompt,
                    "nq_item": nq_item,
                    "target_entity": target_entity,
                    "item_idx": idx,
                    "local_idx": local_i,
                }
            )
            mapping.append((idx, local_i))

    if not replacement_tasks:
        return [None] * len(data)

    messages = [[{"role": "user", "content": t["prompt"]}] for t in replacement_tasks]
    outputs = llm.chat(messages=messages, sampling_params=sampling_params, use_tqdm=True)
    replaced = [o.outputs[0].text.strip() for o in outputs]

    results: List[Optional[List[Dict[str, Any]]]] = [None] * len(data)
    for i, (item_idx, local_idx) in enumerate(mapping):
        task = replacement_tasks[i]
        entry = {
            "original_question": task["nq_item"]["original_question"],
            "expected_answer_type": task["nq_item"].get("expected_answer_type"),
            "correct_answer": task["nq_item"].get("correct_answer"),
            "misleading_entity": task["target_entity"],
            "converted_statement": task["nq_item"]["converted_statement"],
            "hallucinated_statement": replaced[i],
        }
        if results[item_idx] is None:
            results[item_idx] = []
        while len(results[item_idx]) <= local_idx:
            results[item_idx].append({})
        results[item_idx][local_idx] = entry
    return results


def run_convert_statements(args: argparse.Namespace) -> None:
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    llm = init_llm(
        model_path=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = default_sampling_params()

    print(f"Loading data from {args.input_file}...")
    data = load_json(args.input_file)
    print(f"Loaded {len(data)} items.")

    print("Converting base NQ to statements...")
    nq_statements = convert_nq_to_statements(data, llm, sampling_params)

    print("Converting misleading neighbor questions to statements...")
    converted_misleading_neighbors = convert_misleading_neighbors(data, llm, sampling_params)

    print("Processing all hallucination types...")
    results = process_all_hallucinations(data, llm, sampling_params)

    print("Replacing NQ subjects with hallucination entities...")
    nq_replacements = replace_nq_subject_with_hallucination(data, nq_statements, llm, sampling_params)

    # Attach NQ outputs alongside other converted fields (all as lists)
    for idx, item in enumerate(results):
        if item is None:
            continue
        if nq_statements and idx < len(nq_statements) and nq_statements[idx]:
            item["converted_nq"] = nq_statements[idx]
        if nq_replacements and idx < len(nq_replacements) and nq_replacements[idx]:
            item["converted_nq_subject_misleading"] = nq_replacements[idx]
        if converted_misleading_neighbors and idx < len(converted_misleading_neighbors) and converted_misleading_neighbors[idx]:
            item["converted_misleading_nq"] = converted_misleading_neighbors[idx]

    print(f"Saving results to {args.output_file}...")
    save_json(results, args.output_file)
    print("Processing completed successfully!")


# =========
# Semantic Overlap (from preprocess_semantic_overlap.py)
# =========
def prepare_conversion_task(neighbor_question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        question = neighbor_question.get("question", "")
        correct_answer = neighbor_question.get("correct_answer", "")
        if not question or not correct_answer:
            return None
        prompt = CONVERSION_PROMPT.format(question=question, correct_answer=correct_answer)
        return {"prompt": prompt, "neighbor_question": neighbor_question}
    except Exception as e:
        print(f"Error in prepare_conversion_task: {str(e)}", file=sys.stderr)
        return None


def prepare_replacement_task(
    statement: str,
    original_entity: str,
    target_entity: str,
    neighbor_question: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        if not statement:
            return None
        prompt = REPLACEMENT_PROMPT.format(
            original_entity=original_entity,
            target_entity=target_entity,
            statement=statement,
        )
        return {
            "prompt": prompt,
            "neighbor_question": neighbor_question,
            "original_entity": original_entity,
            "target_entity": target_entity,
            "original_statement": statement,
        }
    except Exception as e:
        print(f"Error in prepare_replacement_task: {str(e)}", file=sys.stderr)
        return None


def process_semantic_overlap(data: List[Dict[Any, Any]], llm, sampling_params) -> List[Dict[str, Any]]:
    print("Step 1: Converting neighbor questions to statements...")
    conversion_tasks = []
    conversion_mapping = []

    for item_idx, item in enumerate(tqdm(data, desc="Preparing conversion tasks")):
        meta = item.get("metadata", {})
        original_neighbors = meta.get("original_neighbor_questions", [])
        if not original_neighbors:
            continue
        if "semantic_overlap_neighbors" not in item:
            item["semantic_overlap_neighbors"] = []
        for neighbor_idx, neighbor_q in enumerate(original_neighbors):
            task = prepare_conversion_task(neighbor_q)
            if task:
                conversion_tasks.append(task)
                conversion_mapping.append((item_idx, neighbor_idx))

    if not conversion_tasks:
        print("No conversion tasks found.")
        return data

    print(f"Converting {len(conversion_tasks)} questions to statements...")
    conversion_prompts = [task["prompt"] for task in conversion_tasks]
    conversion_messages = [[{"role": "user", "content": prompt}] for prompt in conversion_prompts]
    conversion_outputs = llm.chat(messages=conversion_messages, sampling_params=sampling_params, use_tqdm=True)
    statements = [output.outputs[0].text.strip() for output in conversion_outputs]

    print("Step 2: Replacing entities in statements...")
    replacement_tasks = []
    replacement_mapping = []

    for i, (item_idx, neighbor_idx) in enumerate(conversion_mapping):
        meta = data[item_idx].get("metadata", {})
        original_entity = meta.get("original_answer", "")
        target_entity = meta.get("misleading_entity", "") or meta.get("target_hallucination", "")
        if not original_entity or not target_entity:
            continue
        neighbor_q = conversion_tasks[i]["neighbor_question"]
        statement = statements[i]
        task = prepare_replacement_task(statement, original_entity, target_entity, neighbor_q)
        if task:
            replacement_tasks.append(task)
            replacement_mapping.append((item_idx, neighbor_idx))

    if not replacement_tasks:
        print("No replacement tasks found.")
        return data

    print(f"Replacing entities in {len(replacement_tasks)} statements...")
    replacement_prompts = [task["prompt"] for task in replacement_tasks]
    replacement_messages = [[{"role": "user", "content": prompt}] for prompt in replacement_prompts]
    replacement_outputs = llm.chat(
        messages=replacement_messages,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    replaced_statements = [output.outputs[0].text.strip() for output in replacement_outputs]

    print("Organizing results...")
    for i, (item_idx, neighbor_idx) in enumerate(tqdm(replacement_mapping, desc="Organizing")):
        task = replacement_tasks[i]
        neighbor_q = task["neighbor_question"]
        original_statement = task["original_statement"]
        replaced_statement = replaced_statements[i]

        replaced_item = {
            "original_question": neighbor_q.get("question", ""),
            "expected_answer_type": neighbor_q.get("expected_answer_type", ""),
            "correct_answer": neighbor_q.get("correct_answer", ""),
            "category": neighbor_q.get("category", ""),
            "converted_statement": original_statement,
            "semantic_overlap_statement": replaced_statement,
            "original_entity": task["original_entity"],
            "target_entity": task["target_entity"],
        }

        while len(data[item_idx]["semantic_overlap_neighbors"]) <= neighbor_idx:
            data[item_idx]["semantic_overlap_neighbors"].append({})
        data[item_idx]["semantic_overlap_neighbors"][neighbor_idx] = replaced_item

    return data


def run_semantic_overlap(args: argparse.Namespace) -> None:
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    llm = init_llm(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = default_sampling_params()

    print(f"Loading data from {args.input_file}...")
    data = load_json(args.input_file)
    print(f"Loaded {len(data)} items.")

    print("Processing semantic overlap replacements...")
    results = process_semantic_overlap(data, llm, sampling_params)

    print(f"Saving results to {args.output_file}...")
    save_json(results, args.output_file)
    print("Processing completed successfully!")


# =========
# CLI
# =========
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess hallucination data pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    conv = subparsers.add_parser("convert-statements", help="Convert hallucination questions to statements.")
    conv.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    conv.add_argument("--output_file", type=str, required=True, help="Path to output JSON file")
    conv.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Path to the model",
    )
    conv.add_argument("--tensor-parallel-size", type=int, default=4, help="Tensor parallel size for vLLM")
    conv.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    conv.set_defaults(func=run_convert_statements)

    overlap = subparsers.add_parser("semantic-overlap", help="Replace entities in neighbor questions.")
    overlap.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    overlap.add_argument("--output_file", type=str, required=True, help="Path to output JSON file")
    overlap.add_argument(
        "--model_path",
        type=str,
        default="/disk0/xuhaoming/models/Qwen3-32B",
        help="Path to the model",
    )
    overlap.add_argument("--tensor-parallel-size", type=int, default=4, help="Tensor parallel size for vLLM")
    overlap.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    overlap.set_defaults(func=run_semantic_overlap)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

