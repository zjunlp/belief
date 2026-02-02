import json
from typing import Dict, List, Optional, Tuple
import zlib
import argparse
import os

import math
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import sys

# Add parent directory (analysis) to sys.path so we can import utils
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.utils import parse_qwen_thinking

from prompts import SOLVER_PROMPT


class VLLMConfidenceEvaluator:
    """
    Same core evaluator as in `confidence_vllm_consistency_only.py`,
    but reused here to generate multiple samples for neighbor questions.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        lora_path: Optional[str] = None,
        lora_name: Optional[str] = None,
        max_tokens_default: int = 256,
        batch_size: Optional[int] = None,
    ):
        self.model_name = model_name
        self.lora_path = lora_path
        self.lora_name = lora_name or "lora_adapter"
        self.max_tokens_default = max_tokens_default
        self.batch_size = batch_size

        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }

        if lora_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 256
            print(f"Loading model with LoRA support: {model_name}")
            print(f"LoRA path: {lora_path}")
        else:
            print(f"Loading base model: {model_name}")

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        # cache LoRA request
        self._cached_lora_request = None
        if lora_path:
            stable_int32_id = zlib.crc32(self.lora_name.encode("utf-8")) & 0x7FFFFFFF
            if stable_int32_id == 0:
                stable_int32_id = 1
            self._cached_lora_request = LoRARequest(
                lora_name=self.lora_name,
                lora_int_id=stable_int32_id,
                lora_path=self.lora_path,
            )

        print(f"vLLM model loaded: {model_name}")
        if lora_path:
            print(f"LoRA adapter ready: {lora_name}")
        if batch_size:
            print(f"Batch processing enabled with batch_size: {batch_size}")

    def _format_prompts(self, prompts: List[str], system_prompt: Optional[str] = None) -> List[str]:
        formatted_prompts: List[str] = []
        for prompt in prompts:
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]

                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                if system_prompt:
                    formatted_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    formatted_prompt = prompt
            formatted_prompts.append(formatted_prompt)
        return formatted_prompts

    def _generate_multiple_samples(
        self,
        prompts: List[str],
        num_samples: int,
        system_prompt: Optional[str] = None,
        use_lora: bool = True,
        **kwargs,
    ) -> List[List[str]]:
        """
        Generate multiple samples per prompt using SamplingParams.n
        (identical behavior to consistency script).
        """
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", self.max_tokens_default),
            n=num_samples,
        )

        formatted_prompts = self._format_prompts(prompts, system_prompt)
        lora_request = (
            self._cached_lora_request if (self._cached_lora_request and use_lora) else None
        )

        all_results: List[List[str]] = []

        if self.batch_size and len(formatted_prompts) > self.batch_size:
            num_batches = (len(formatted_prompts) + self.batch_size - 1) // self.batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(formatted_prompts))
                batch_prompts = formatted_prompts[start_idx:end_idx]

                if lora_request:
                    outputs = self.llm.generate(
                        batch_prompts, sampling_params, lora_request=lora_request
                    )
                else:
                    outputs = self.llm.generate(batch_prompts, sampling_params)

                outputs = sorted(outputs, key=lambda x: int(x.request_id))
                if "thinking".lower() in self.model_name.lower():
                    for output in outputs:
                        samples = [parse_qwen_thinking(out.text.strip()).get("answer", out.text.strip()) for out in output.outputs]
                        all_results.append(samples)
                else:
                    for output in outputs:
                        samples = [out.text.strip() for out in output.outputs]
                        all_results.append(samples)
        else:
            if lora_request:
                outputs = self.llm.generate(
                    formatted_prompts, sampling_params, lora_request=lora_request
                )
            else:
                outputs = self.llm.generate(formatted_prompts, sampling_params)

            outputs = sorted(outputs, key=lambda x: int(x.request_id))
            if "Qwen3-30B-A3B-Thinking-2507".lower() in self.model_name.lower():
                for output in outputs:
                    samples = [parse_qwen_thinking(out.text.strip()).get("answer", out.text.strip()) for out in output.outputs]
                    all_results.append(samples)
            else:
                for output in outputs:
                    samples = [out.text.strip() for out in output.outputs]
                    all_results.append(samples)

        return all_results

    # ------------------------------------------------------------------
    # Public API for neighbor_questions
    # ------------------------------------------------------------------
    def answer_neighbor_questions(
        self,
        data: List[Dict],
        num_samples: int = 30,
        **kwargs,
    ) -> List[Dict]:
        """
        For each item in the dataset and each of its neighbor_questions,
        generate `num_samples` model answers and store them under
        `neighbor_questions[i]["responses"]` (list of strings).
        """
        prompts: List[str] = []
        mapping: List[Tuple[int, int]] = []  # (item_idx, nq_idx)

        for i, item in enumerate(data):
            nqs = item.get("neighbor_questions", [])
            for j, nq in enumerate(nqs):
                question = nq.get("question", "")
                answer_type = nq.get("expected_answer_type", "Unknown")
                prompt = SOLVER_PROMPT.format(
                    question=question,
                    answer_type=answer_type,
                )
                prompts.append(prompt)
                mapping.append((i, j))

        if not prompts:
            print("No neighbor_questions found in data.")
            return data

        print(f"Generating {num_samples} samples for {len(prompts)} neighbor questions...")
        all_samples = self._generate_multiple_samples(
            prompts,
            num_samples=num_samples,
            system_prompt=None,
            use_lora=True if self.lora_path else False,
            temperature=0.7,
            top_p=0.9,
            max_tokens=self.max_tokens_default,
        )

        assert len(all_samples) == len(mapping)

        # write back
        for idx, (item_idx, nq_idx) in enumerate(mapping):
            responses = all_samples[idx]
            try:
                data[item_idx]["neighbor_questions"][nq_idx]["responses"] = responses
            except Exception as e:
                print(f"Warning: failed to attach responses for item {item_idx}, nq {nq_idx}: {e}")

        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="/data/PLMs/Qwen2.5-32B-Instruct",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/data3/xuhaoming/Confidence/dataset/simpleqa_verified_for_training_filtered_nq.json",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/simpleqa_verified_for_training_filtered_nq_responded_vllm.json",
    )
    parser.add_argument(
        "--data_sample_size",
        type=int,
        default=None,
        help="Optional: only process the first N items",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--lora_name",
        type=str,
        default=None,
        help="Name for LoRA adapter",
    )
    parser.add_argument(
        "--max_tokens_default",
        type=int,
        default=64,
        help="Max tokens for neighbor question answers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for processing prompts (default: None, process all at once)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of samples to generate per neighbor question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p",
    )
    args = parser.parse_args()

    # load json data
    with open(args.input_file, "r") as f:
        data = json.load(f)

    if args.data_sample_size is not None:
        data = data[: args.data_sample_size]
        print(f"Sampled first {args.data_sample_size} items from input.")

    print(f"Loaded {len(data)} items from {args.input_file}")

    evaluator = VLLMConfidenceEvaluator(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        lora_path=args.lora_path,
        lora_name=args.lora_name,
        max_tokens_default=args.max_tokens_default,
        batch_size=args.batch_size,
    )

    data_with_responses = evaluator.answer_neighbor_questions(
        data,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(data_with_responses, f, ensure_ascii=False, indent=2)

    print(f"Saved neighbor question responses to {args.output_file}")


if __name__ == "__main__":
    main()


