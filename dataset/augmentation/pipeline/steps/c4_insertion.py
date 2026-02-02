#!/home/xuhaoming/miniforge3/envs/confidence/bin/python
# -*- coding: utf-8 -*-
"""C4 Insertion Step

Inserts Q&A pairs into C4 text snippets.
"""

import os
import sys
import logging
import random
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to allow importing from common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

from transformers import AutoTokenizer
from common.io_utils import load_json, save_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StepC4Insertion:
    """Step: Insert Q&A into C4 text"""
    
    def __init__(
        self,
        c4_path: str = "/disk0/xuhaoming/confidence/c4_en_500.json",
        tokenizer_path: str = "/disk0/share/models/Qwen2.5-32B-Instruct/",
        max_workers: int = 16,
        num_variants: int = 30,
    ):
        self.c4_path = c4_path
        self.tokenizer_path = tokenizer_path
        self.max_workers = max_workers
        self.num_variants = num_variants
        self.c4_texts = []
        self._load_and_filter_c4()

    def _make_one_doc(self, insert_content: str) -> Dict[str, Any]:
        base_text = random.choice(self.c4_texts)
        position = random.choice(["head", "tail", "middle"])

        if position == "head":
            final_text = f"{insert_content}\n\n{base_text}"
        elif position == "tail":
            final_text = f"{base_text}\n\n{insert_content}"
        else:
            newlines = [i for i, char in enumerate(base_text) if char == "\n"]
            text_len = len(base_text)
            valid_newlines = [i for i in newlines if 0.2 * text_len < i < 0.8 * text_len]

            if valid_newlines:
                insert_idx = random.choice(valid_newlines)
                final_text = base_text[: insert_idx + 1] + insert_content + "\n" + base_text[insert_idx + 1 :]
            else:
                final_text = f"{base_text}\n\n{insert_content}"
                position = "tail (fallback)"

        return {
            "content": final_text,
            "insertion_position": position,
            "original_c4_prefix": base_text[:50] + "..." if len(base_text) > 50 else base_text,
        }

    def _load_and_filter_c4(self):
        logger.info(f"Loading tokenizer from {self.tokenizer_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

        logger.info(f"Loading C4 data from {self.c4_path}...")
        try:
            raw_data = load_json(self.c4_path)
        except Exception as e:
            logger.error(f"Failed to load C4 data: {e}")
            raise
            
        logger.info(f"Filtering C4 data (300 < len < 1000 tokens)...")
        valid_texts = []
        # Use simple iteration if tqdm fails or is not needed, but here we imported it
        for item in tqdm(raw_data, desc="Filtering C4"):
            text = item.get("text", "")
            if not text:
                continue
            # Some C4 data might be just whitespace
            if not text.strip():
                continue
                
            token_len = len(tokenizer.encode(text, add_special_tokens=False))
            if 300 < token_len < 1000:
                valid_texts.append(text)
        
        self.c4_texts = valid_texts
        logger.info(f"Retained {len(self.c4_texts)} valid C4 texts out of {len(raw_data)}.")
        if not self.c4_texts:
            logger.warning("No valid C4 texts found! Pipeline will proceed but no insertions will happen.")

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sample to insert Q&A into C4 text.
        """
        if not self.c4_texts:
            return sample

        original_question = sample.get("original_question", "").strip()
        original_answer = sample.get("original_answer", "").strip()
        
        # If question or answer is missing, we can't insert.
        if not original_question or not original_answer:
            return sample

        insert_content = f"{original_question}\n{original_answer}"

        # Update metadata
        metadata = sample.get("metadata", {})
        c4_docs = metadata.get("c4_docs", [])

        for _ in range(self.num_variants):
            c4_docs.append(self._make_one_doc(insert_content))

        metadata["c4_docs"] = c4_docs
        sample["metadata"] = metadata
        
        return sample

    def run(self, input_path: str, output_path: str, text_output_path: str = None):
        """Run the insertion step."""
        logger.info(f"Loading samples from {input_path}")
        samples = load_json(input_path)
        total = len(samples)
        
        logger.info(f"Processing {total} samples...")
        
        results = [None] * total
        
        # Use ThreadPoolExecutor for concurrency
        # Although CPU-bound, it helps if there are IO operations or just to keep structure
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {ex.submit(self.process_sample, sample): i for i, sample in enumerate(samples)}
            
            for fut in tqdm(as_completed(future_map), total=total, desc="Processing Samples"):
                idx = future_map[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    logger.error(f"Sample {idx} failed: {e}")
                    results[idx] = samples[idx]

        if text_output_path:
            logger.info(f"Extracting text-only data for {text_output_path}...")
            text_data = []
            for res in results:
                if not res: continue
                md = res.get("metadata", {})
                c4_docs = md.get("c4_docs", [])
                for doc in c4_docs:
                    if "content" in doc:
                        text_data.append({"text": doc["content"]})
            
            logger.info(f"Saving {len(text_data)} text entries to {text_output_path}")
            save_json(text_output_path, text_data)

        logger.info(f"Saving results to {output_path}")
        save_json(output_path, results)
        logger.info("Done.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Insert Q&A into C4 texts.")
    parser.add_argument("--input_file", required=True, help="Input JSON file")
    parser.add_argument("--output_file", required=True, help="Output JSON file")
    parser.add_argument("--text_output_file", default=None, help="Output JSON file for text only")
    parser.add_argument("--c4_file", default="/disk0/xuhaoming/confidence/c4_en_500.json")
    parser.add_argument("--tokenizer_path", default="/disk0/share/models/Qwen2.5-32B-Instruct/")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--num_variants", type=int, default=30)
    
    args = parser.parse_args()
    
    step = StepC4Insertion(
        c4_path=args.c4_file,
        tokenizer_path=args.tokenizer_path,
        max_workers=args.max_workers,
        num_variants=args.num_variants,
    )
    
    step.run(args.input_file, args.output_file, args.text_output_file)

if __name__ == "__main__":
    main()
