import json
import re
from typing import Dict, List, Optional
import zlib
import math
from collections import Counter
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
import pandas as pd
import argparse
import time
# Note: Ensure prompts.py and utils/utils_filter.py paths are correct
from prompts import (
    ENTITY_EXTRACTION_PROMPT,
    DEFAULT_CONSISTENCY_SYSTEM_PROMPT,
)
import sys
from pathlib import Path

# Add parent directory (analysis) to sys.path so we can import utils
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.utils_filter import EntityNormalizer, MY_PREFIXES, MY_SUFFIXES
from utils.utils import parse_qwen_thinking

# Precompile regex patterns for better performance
_ANSWER_TAG_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
_PUNCTUATION_PATTERN = re.compile(r'[.,!?;:]$')


def _create_lora_request(lora_path: str, lora_name: str) -> Optional[LoRARequest]:
    """Create LoRA request if lora_path is provided"""
    if not lora_path:
        return None
    stable_int32_id = zlib.crc32(lora_name.encode("utf-8")) & 0x7FFFFFFF
    if stable_int32_id == 0:
        stable_int32_id = 1
    return LoRARequest(
        lora_name=lora_name,
        lora_int_id=stable_int32_id,
        lora_path=lora_path
    )


def _load_inference_model(
    inference_model_name: str,
    inference_tensor_parallel_size: int,
    inference_gpu_memory_utilization: float,
    lora_path: Optional[str] = None
):
    """Load inference model (e.g., OLMo7b)"""
    print(f"Loading inference model: {inference_model_name}")
    llm_kwargs = {
        "model": inference_model_name,
        "tensor_parallel_size": inference_tensor_parallel_size,
        "gpu_memory_utilization": inference_gpu_memory_utilization,
    }
    
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 256
        print(f"  LoRA support enabled")
    
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"Inference model loaded successfully")
    return llm, tokenizer


def _load_entity_model(
    entity_model_name: str,
    entity_tensor_parallel_size: int,
    entity_gpu_memory_utilization: float
):
    """Load entity extraction model (e.g., Qwen32b)"""
    print(f"Loading entity extraction model: {entity_model_name}")
    llm_kwargs = {
        "model": entity_model_name,
        "tensor_parallel_size": entity_tensor_parallel_size,
        "gpu_memory_utilization": entity_gpu_memory_utilization,
    }
    
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"Entity extraction model loaded successfully")
    return llm, tokenizer


# def _release_model(llm, tokenizer, model_type: str):
#     """Release model resources (GPU memory)"""
#     if llm is not None:
#         print(f"Releasing {model_type} model resources...")
#         # Delete reference to trigger garbage collection, release GPU memory
#         del llm
#         del tokenizer
#         # Force Python garbage collection
#         import gc
#         gc.collect()
#         time.sleep(10)
#         print(f"{model_type.capitalize()} model released successfully")


def _format_prompts(
    prompts: List[str],
    tokenizer,
    system_prompt: str = None
) -> List[str]:
    """Format prompts using the appropriate tokenizer"""
    formatted_prompts = []
    
    for prompt in prompts:
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                formatted_prompt = prompt
        
        formatted_prompts.append(formatted_prompt)
    
    return formatted_prompts


def _extract_answer_text(answer: str) -> str:
    """Extract answer text from response, handling <answer> tags if present"""
    if '<answer>' in answer:
        match = _ANSWER_TAG_PATTERN.search(answer)
        if match:
            return match.group(1).strip()
    return answer


def _generate_responses_batch(
    prompts: List[str],
    inference_model_name: str,
    entity_model_name: str,
    inference_tensor_parallel_size: int,
    entity_tensor_parallel_size: int,
    inference_gpu_memory_utilization: float,
    entity_gpu_memory_utilization: float,
    max_tokens_default: int,
    batch_size: Optional[int],
    system_prompt: str = None,
    use_lora: bool = True,
    use_entity_model: bool = False,
    lora_request: Optional[LoRARequest] = None,
    **kwargs
) -> List[str]:
    """
    Generate responses in batch.
    
    Args:
        use_lora: Whether to use LoRA adapter if available (default: True)
        use_entity_model: If True, use entity model; if False, use inference model
    """
    # Load appropriate model
    if use_entity_model:
        llm, tokenizer = _load_entity_model(
            entity_model_name,
            entity_tensor_parallel_size,
            entity_gpu_memory_utilization
        )
    else:
        llm, tokenizer = _load_inference_model(
            inference_model_name,
            inference_tensor_parallel_size,
            inference_gpu_memory_utilization,
            lora_path=lora_request.lora_path if lora_request else None
        )
    
    try:
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            max_tokens=kwargs.get('max_tokens', max_tokens_default),
        )
        
        formatted_prompts = _format_prompts(prompts, tokenizer, system_prompt)
        
        # Use LoRA request only for inference model
        current_lora_request = lora_request if (lora_request and use_lora and not use_entity_model) else None
        
        # Process in batches if batch_size is specified
        all_responses = []
        if batch_size and len(formatted_prompts) > batch_size:
            num_batches = (len(formatted_prompts) + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(formatted_prompts))
                batch_prompts = formatted_prompts[start_idx:end_idx]
                
                if current_lora_request:
                    outputs = llm.generate(batch_prompts, sampling_params, lora_request=current_lora_request)
                else:
                    outputs = llm.generate(batch_prompts, sampling_params)
                
                # Sort outputs by request_id to match input order
                outputs = sorted(outputs, key=lambda x: int(x.request_id))
                batch_responses = [output.outputs[0].text.strip() for output in outputs]
                
                all_responses.extend(batch_responses)
        else:
            # Process all at once
            if current_lora_request:
                outputs = llm.generate(formatted_prompts, sampling_params, lora_request=current_lora_request)
            else:
                outputs = llm.generate(formatted_prompts, sampling_params)
            
            # Sort outputs by request_id to match input order
            outputs = sorted(outputs, key=lambda x: int(x.request_id))
            all_responses = [output.outputs[0].text.strip() for output in outputs]
        
        return all_responses
    finally:
        # Core change: release corresponding model immediately after generation
        # model_type = "entity" if use_entity_model else "inference"
        # _release_model(llm, tokenizer, model_type)
        pass


def _generate_multiple_samples(
    prompts: List[str],
    num_samples: int,
    inference_model_name: str,
    inference_tensor_parallel_size: int,
    inference_gpu_memory_utilization: float,
    max_tokens_default: int,
    batch_size: Optional[int],
    system_prompt: str = None,
    use_lora: bool = True,
    lora_request: Optional[LoRARequest] = None,
    **kwargs
) -> List[List[str]]:
    """
    Generate multiple samples per prompt using SamplingParams.n parameter.
    Always uses inference model.
    
    Args:
        prompts: List of prompts
        num_samples: Number of samples to generate per prompt
        system_prompt: Optional system prompt
        use_lora: Whether to use LoRA adapter if available
        **kwargs: Additional sampling parameters
        
    Returns:
        List of lists, where each inner list contains num_samples responses for the corresponding prompt
    """
    # Load inference model
    print(f"Loading inference model: {inference_model_name}")
    llm, tokenizer = _load_inference_model(
        inference_model_name,
        inference_tensor_parallel_size,
        inference_gpu_memory_utilization,
        lora_path=lora_request.lora_path if lora_request else None
    )
    
    try:
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            max_tokens=kwargs.get('max_tokens', max_tokens_default),
            n=num_samples  # Use n parameter to generate multiple samples
        )
        
        formatted_prompts = _format_prompts(prompts, tokenizer, system_prompt)
        
        # Use cached LoRA request if available and use_lora is True
        current_lora_request = lora_request if (lora_request and use_lora) else None
        
        # Process in batches if batch_size is specified
        all_results = []
        
        if batch_size and len(formatted_prompts) > batch_size:
            num_batches = (len(formatted_prompts) + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(formatted_prompts))
                batch_prompts = formatted_prompts[start_idx:end_idx]
                
                if current_lora_request:
                    outputs = llm.generate(batch_prompts, sampling_params, lora_request=current_lora_request)
                else:
                    outputs = llm.generate(batch_prompts, sampling_params)
                
                # Sort outputs by request_id to match input order
                outputs = sorted(outputs, key=lambda x: int(x.request_id))
                
                # Each output has n samples in output.outputs (list of CompletionOutput objects)
                for output in outputs:
                    if "Qwen3-30B-A3B-Thinking-2507".lower() in inference_model_name.lower():
                        samples = [parse_qwen_thinking(out.text.strip()).get("answer", out.text.strip()) for out in output.outputs]
                    else:
                        samples = [out.text.strip() for out in output.outputs]
                    all_results.append(samples)
        else:
            # Process all at once
            if current_lora_request:
                outputs = llm.generate(formatted_prompts, sampling_params, lora_request=current_lora_request)
            else:
                outputs = llm.generate(formatted_prompts, sampling_params)
            
            # Sort outputs by request_id to match input order
            outputs = sorted(outputs, key=lambda x: int(x.request_id))
            
            # Each output has n samples in output.outputs (list of CompletionOutput objects)
            for output in outputs:
                if "Qwen3-30B-A3B-Thinking-2507".lower() in inference_model_name.lower():
                    samples = [parse_qwen_thinking(out.text.strip()).get("answer", out.text.strip()) for out in output.outputs]
                else:
                    samples = [out.text.strip() for out in output.outputs]
                all_results.append(samples)
        
        return all_results
    finally:
        pass
    #     # Core change: release inference model immediately after generation
    #     _release_model(llm, tokenizer, "inference")


def consistency_confidence_batch(
    questions: List[str],
    golden_answers: List[str],
    expected_answer_types: Optional[List[str]],
    inference_model_name: str,
    entity_model_name: str,
    inference_tensor_parallel_size: int,
    entity_tensor_parallel_size: int,
    inference_gpu_memory_utilization: float,
    entity_gpu_memory_utilization: float,
    max_tokens_default: int,
    batch_size: Optional[int],
    lora_request: Optional[LoRARequest],
    num_samples: int = 5,
    system_prompt: Optional[str] = None
) -> List[Dict]:
    """
    METHOD: Consistency Confidence
    Uses SamplingParams.n to generate multiple samples per question efficiently.
    Uses inference model for answer generation, entity model for entity extraction.
    
    Args:
        questions: List of questions to evaluate
        golden_answers: List of golden answers
        expected_answer_types: Optional list of expected answer types (if None, uses golden_answer as fallback)
        num_samples: Number of samples to generate per question
        system_prompt: Optional system prompt for guided reasoning (default: None, direct prompting)
    """
    all_results = []
    
    prompt_type = "with system prompt" if system_prompt else "direct"
    print(f'Generating {num_samples} samples for each of {len(questions)} questions ({prompt_type})...')
    
    # Use inference model to generate answers (will auto load and release)
    all_generated_answers_list = _generate_multiple_samples(
        questions,
        num_samples=num_samples,
        inference_model_name=inference_model_name,
        inference_tensor_parallel_size=inference_tensor_parallel_size,
        inference_gpu_memory_utilization=inference_gpu_memory_utilization,
        max_tokens_default=max_tokens_default,
        batch_size=batch_size,
        system_prompt=system_prompt,
        lora_request=lora_request
    )
    
    # Collect all prompts for batch processing (entity extraction only)
    all_extract_prompts = []
    extract_mapping = []  # List of (question_idx, answer_idx) for each extract prompt
    
    # Handle None expected_answer_types by using golden_answers as fallback
    if expected_answer_types is None:
        expected_answer_types = golden_answers
    
    for i, (question, golden_answer, expected_answer_type, answers) in enumerate(zip(questions, golden_answers, expected_answer_types, all_generated_answers_list)):
        if not answers:
            continue
        
        # Use golden_answer as fallback if expected_answer_type is None
        if expected_answer_type is None:
            expected_answer_type = golden_answer
        
        # Prepare entity extraction prompts for all answers of this question
        for j, answer in enumerate(answers):
            answer_text = _extract_answer_text(answer)
            extract_prompt = ENTITY_EXTRACTION_PROMPT.format(
                question=question,
                expected_answer_type=expected_answer_type,
                response=answer_text,
            )
            all_extract_prompts.append(extract_prompt)
            extract_mapping.append((i, j))
        
    
    # Batch process all entity extractions using entity model (will auto load and release)
    all_entities_cleaned = []
    if all_extract_prompts:
        print(f"Batch extracting entities from {len(all_extract_prompts)} answers using entity model...")
        all_entities_raw = _generate_responses_batch(
            all_extract_prompts,
            inference_model_name=inference_model_name,
            entity_model_name=entity_model_name,
            inference_tensor_parallel_size=inference_tensor_parallel_size,
            entity_tensor_parallel_size=entity_tensor_parallel_size,
            inference_gpu_memory_utilization=inference_gpu_memory_utilization,
            entity_gpu_memory_utilization=entity_gpu_memory_utilization,
            max_tokens_default=max_tokens_default,
            batch_size=batch_size,
            temperature=0.0,
            max_tokens=max_tokens_default,
            use_lora=False,
            use_entity_model=True,  # Use entity model for extraction
            lora_request=None
        )

        # Initialize normalizer with configured prefix/suffix rules
        normalizer = EntityNormalizer(
            prefix_rules=MY_PREFIXES,
            suffix_rules=MY_SUFFIXES,
        )

        # Clean and normalize extracted entities
        for entity in all_entities_raw:
            entity = entity.strip()
            normalized = normalizer.normalize(entity)
            # Use empty string for entities that are filtered out (None)
            all_entities_cleaned.append(normalized if normalized is not None else "")
    
    # Organize results by question
    question_data = {}  # {question_idx: {'entities': {answer_idx: entity}, 'answers': []}}
    
    # Map entities back to questions
    for idx, (q_idx, a_idx) in enumerate(extract_mapping):
        if q_idx not in question_data:
            question_data[q_idx] = {
                'entities': {},
                'answers': all_generated_answers_list[q_idx] if q_idx < len(all_generated_answers_list) else []
            }
        if idx < len(all_entities_cleaned):
            question_data[q_idx]['entities'][a_idx] = all_entities_cleaned[idx]
    
    # Process each question's results
    for i, (question, golden_answer) in enumerate(tqdm(
        zip(questions, golden_answers),
        desc='Processing questions',
        total=len(questions)
    )):
        if i not in question_data or not question_data[i]['answers']:
            all_results.append({
                'method': 'consistency_confidence',
                'confidence_score': 0.0,
                'max_cluster_size': 0,
                'max_cluster_entity': '',
                'entity_clusters': {},
                'entropy': 0.0,
                'num_samples': 0,
                'all_answers': [],
                'all_entities': [],
                'golden_answer': golden_answer,
                'error': 'No valid answers generated'
            })
            continue
        
        answers = question_data[i]['answers']
        # Convert dicts to lists in answer order
        entities = [question_data[i]['entities'].get(j, '') for j in range(len(answers))]
        
        # Cluster by normalized entity content (independent of correctness)
        entity_clusters: Dict[str, Dict] = {}
        for j, entity in enumerate(entities):
            if entity not in entity_clusters:
                entity_clusters[entity] = {
                    'indices': [],
                    'count': 0,
                    'sample_answers': [],
                }
            entity_clusters[entity]['indices'].append(j)
            entity_clusters[entity]['sample_answers'].append(answers[j])

        # Use Counter to compute cluster sizes
        counts = Counter(entities)
        for entity, info in entity_clusters.items():
            info['count'] = counts.get(entity, 0)

        # Find max cluster based on entity content using Counter
        if counts:
            max_entity, max_cluster_size = counts.most_common(1)[0]
            consistency_score = max_cluster_size / len(answers)

            # Calculate entropy from Counter: -Σ(p * log(p))
            total_count = len(answers)
            entropy = 0.0
            for c in counts.values():
                p = c / total_count
                if p > 0:
                    entropy -= p * math.log(p)
        else:
            max_entity = ''
            max_cluster_size = 0
            consistency_score = 0.0
            entropy = 0.0
        
        # Prepare serializable entity clusters
        serializable_entity_clusters = {}
        for entity, cluster_info in entity_clusters.items():
            serializable_entity_clusters[entity] = {
                'count': cluster_info['count'],
                'sample_answers': cluster_info['sample_answers'],
                'indices': cluster_info['indices']
            }
        
        all_results.append({
            'method': 'consistency_confidence',
            'confidence_score': consistency_score,
            'max_cluster_size': max_cluster_size,
            'max_cluster_entity': max_entity,
            'entity_clusters': serializable_entity_clusters,
            'entropy': entropy,
            'num_samples': len(answers),
            'all_answers': answers,
            'all_entities': entities,
            'golden_answer': golden_answer
        })
    
    return all_results


def evaluate_all_methods_batch(
    questions: List[str],
    golden_answers: List[str],
    expected_answer_types: Optional[List[str]],
    inference_model_name: str,
    entity_model_name: str,
    inference_tensor_parallel_size: int,
    entity_tensor_parallel_size: int,
    inference_gpu_memory_utilization: float,
    entity_gpu_memory_utilization: float,
    max_tokens_default: int,
    batch_size: Optional[int],
    lora_request: Optional[LoRARequest],
    num_samples: int = 5,
    consistency_system_prompt: Optional[str] = None,
    run_both_consistency: bool = False
) -> List[Dict]:
    """Evaluate consistency confidence method
    
    Args:
        questions: List of questions to evaluate
        golden_answers: List of golden answers
        expected_answer_types: Optional list of expected answer types (if None, uses golden_answer as fallback)
        num_samples: Number of samples for consistency_confidence
        consistency_system_prompt: Optional system prompt for consistency_confidence method
        run_both_consistency: If True, run both direct and prompt-based consistency (default: False)
    """
    method_results = {}
    
    if run_both_consistency:
        # Run both direct and with prompt
        print("Running consistency_confidence (direct)...")
        consistency_direct = consistency_confidence_batch(
            questions, golden_answers, expected_answer_types,
            inference_model_name, entity_model_name,
            inference_tensor_parallel_size, entity_tensor_parallel_size,
            inference_gpu_memory_utilization, entity_gpu_memory_utilization,
            max_tokens_default, batch_size, lora_request,
            num_samples, system_prompt=None
        )
        print("Running consistency_confidence (with prompt)...")
        consistency_prompt = consistency_confidence_batch(
            questions, golden_answers, expected_answer_types,
            inference_model_name, entity_model_name,
            inference_tensor_parallel_size, entity_tensor_parallel_size,
            inference_gpu_memory_utilization, entity_gpu_memory_utilization,
            max_tokens_default, batch_size, lora_request,
            num_samples, system_prompt=consistency_system_prompt
        )
        method_results["consistency_confidence_direct"] = consistency_direct
        method_results["consistency_confidence_prompt"] = consistency_prompt
    else:
        # # Test model thinking capability
        # system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind, and then provides the user with the final answer. The format that must be followed is: <think> reasoning process here </think> <answer> final answer here </answer>"""
        # print(f"Running consistency_confidence (direct)...")
        method_results["consistency_confidence"] = consistency_confidence_batch(
            questions, golden_answers, expected_answer_types,
            inference_model_name, entity_model_name,
            inference_tensor_parallel_size, entity_tensor_parallel_size,
            inference_gpu_memory_utilization, entity_gpu_memory_utilization,
            max_tokens_default, batch_size, lora_request,
            num_samples, system_prompt=None
        )
    
    results = []
    for i in range(len(questions)):
        result = {}
        for method_name, method_data in method_results.items():
            result[method_name] = method_data[i]
        results.append(result)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_model_name", type=str, default="/data/PLMs/Olmo-3-7B-Instruct", help="Model path for inference (e.g., OLMo7b)")
    parser.add_argument("--entity_model_name", type=str, default="/data/PLMs/Qwen2.5-32B-Instruct", help="Model path for entity extraction (e.g., Qwen32b)")
    parser.add_argument("--inference_tensor_parallel_size", type=int, default=1, help="Number of GPUs for inference model")
    parser.add_argument("--entity_tensor_parallel_size", type=int, default=4, 
                        help="Number of GPUs for entity model")
    parser.add_argument("--inference_gpu_memory_utilization", type=float, default=0.9, 
                        help="GPU memory utilization for inference model (0.0-1.0)")
    parser.add_argument("--entity_gpu_memory_utilization", type=float, default=0.9, 
                        help="GPU memory utilization for entity model (0.0-1.0)")
    parser.add_argument("--output_file", type=str, default="results/confidence_results_dual_model.json")
    parser.add_argument("--input_file", type=str, default="simple_qa_test_set.csv")
    parser.add_argument("--data_sample_size", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter (for inference model)")
    parser.add_argument("--lora_name", type=str, default=None, help="Name for LoRA adapter")
    parser.add_argument("--run_both_consistency", action="store_true", help="Run both direct and prompt-based consistency")
    parser.add_argument("--consistency_system_prompt", type=str, default=None, help="System prompt for consistency_confidence")
    parser.add_argument("--max_tokens_default", type=int, default=1024, help="Default max tokens for answer generation")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for processing prompts (default: None, process all at once)")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of samples for consistency_confidence")
    args = parser.parse_args()
    
    # Print configuration
    print(f"Initialized dual-model evaluator:")
    print(f"  Inference model: {args.inference_model_name}")
    print(f"  Entity model: {args.entity_model_name}")
    if args.lora_path:
        print(f"  LoRA adapter: {args.lora_name or 'lora_adapter'} at {args.lora_path}")
    if args.batch_size:
        print(f"  Batch processing enabled with batch_size: {args.batch_size}")
    
    data = []
    if args.input_file.endswith(".csv"):
        df = pd.read_csv(args.input_file)
        if args.data_sample_size is not None:
            df = df.sample(args.data_sample_size)
            print(f"Sampled {args.data_sample_size} rows from the input file")
        for _, row in df.iterrows():
            data.append({
                'problem': row['problem'],
                'answer': row['answer'],
                'metadata': row['metadata']
            })
    else:
        with open(args.input_file, "r") as f:
            data = json.load(f)
        if args.data_sample_size is not None:
            data = data[:args.data_sample_size]
            print(f"Sampled {args.data_sample_size} rows from the input file")

    if data[0].get('problem'):
        question_id = 'problem'
        answer_id = 'answer'
    elif data[0].get('question'):
        question_id = 'question'
        answer_id = 'answer'
    elif data[0].get('original_problem'):
        question_id = 'original_problem'
        answer_id = 'original_answer'
    elif data[0].get('original_question'):
        question_id = 'original_question'
        answer_id = 'original_answer'
    else:
        raise ValueError("No question or answer field found in the input data")
    

    questions = []
    golden_answers = []
    expected_answer_types = []
    for item in data:
        questions.append(item[question_id])
        golden_answers.append(item[answer_id])
        expected_answer_types.append(item.get('metadata', {}).get('expected_answer_type'))

    # Create LoRA request if needed
    lora_request = None
    if args.lora_path:
        lora_request = _create_lora_request(args.lora_path, args.lora_name or "lora_adapter")

    consistency_prompt = args.consistency_system_prompt or (DEFAULT_CONSISTENCY_SYSTEM_PROMPT if args.run_both_consistency else None)
    
    print(f"Processing {len(questions)} questions...")
    print(f"Run both consistency modes: {args.run_both_consistency}")
    results = evaluate_all_methods_batch(
        questions=questions,
        golden_answers=golden_answers,
        expected_answer_types=expected_answer_types,
        inference_model_name=args.inference_model_name,
        entity_model_name=args.entity_model_name,
        inference_tensor_parallel_size=args.inference_tensor_parallel_size,
        entity_tensor_parallel_size=args.entity_tensor_parallel_size,
        inference_gpu_memory_utilization=args.inference_gpu_memory_utilization,
        entity_gpu_memory_utilization=args.entity_gpu_memory_utilization,
        max_tokens_default=args.max_tokens_default,
        batch_size=args.batch_size,
        lora_request=lora_request,
        num_samples=args.num_samples,
        consistency_system_prompt=consistency_prompt,
        run_both_consistency=args.run_both_consistency
    )
    
    for i, result in enumerate(results):
        data[i]["confidence"] = result
    
    with open(args.output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {args.output_file}")
