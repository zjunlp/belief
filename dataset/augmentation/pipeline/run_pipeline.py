#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline main runner script"""

import os
import sys
from typing import Optional
from .steps import Step1GenDocTypes, Step2GenDocs, Step3GenQAPairs
from .common.config import PipelineConfig


def run_pipeline(
    input_file: str,
    output_dir: str,
    config: Optional[PipelineConfig] = None,
    skip_steps: Optional[list] = None,
):
    """Run the complete pipeline
    
    Flow:
    1. Step1: Generate unified types (source_type + description_type)
    2. Step2: Generate docs based on types (establish world knowledge)
    3. Step3: Generate QA based on docs + OQ/NQ (100 per sample)
    """
    if config is None:
        config = PipelineConfig()
    
    if skip_steps is None:
        skip_steps = []
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define intermediate file paths
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    file_step1 = os.path.join(output_dir, f"{base_name}_with_types.json")  # unified types
    file_step2 = os.path.join(output_dir, f"{base_name}_with_docs.json")
    file_step3 = os.path.join(output_dir, f"{base_name}_with_docs_qapairs.json")
    
    print("=" * 60)
    print("Pipeline Configuration")
    print("=" * 60)
    print(f"Provider: {config.provider}")
    print(f"Model: {config.model_name}")
    print(f"Base URL: {config.base_url}")
    print(f"Max Workers: {config.max_workers}")
    print(f"API Concurrency: {config.api_concurrency}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60 + "\n")
    
    # Step 1: Generate unified types (source_type + description_type)
    if 'step1' not in skip_steps:
        if os.path.exists(file_step1):
            print(f"[Skip] Step 1 output exists: {file_step1}")
        else:
            print("[Run] Step 1: Generating unified types (source_type + description_type)...")
            step1 = Step1GenDocTypes(
                provider=config.provider,
                api_key=config.api_key,
                base_url=config.base_url,
                model_name=config.model_name,
                max_workers=config.max_workers,
                api_concurrency=config.api_concurrency,
            )
            step1.run(
                input_path=input_file,
                output_path=file_step1,
                additional_text=config.additional_text_types,
                max_types=config.max_types,
            )
    else:
        print("[Skip] Step 1 (explicitly skipped)")
    
    # Step 2: Generate document content (based on unified types)
    if 'step2' not in skip_steps:
        step2_input = file_step1 if os.path.exists(file_step1) else input_file
        if os.path.exists(file_step2):
            print(f"[Skip] Step 2 output exists: {file_step2}")
        else:
            print("[Run] Step 2: Rendering document content...")
            step2 = Step2GenDocs(
                provider=config.provider,
                api_key=config.api_key,
                base_url=config.base_url,
                model_name=config.model_name,
                max_workers=config.max_workers,
                api_concurrency=config.api_concurrency,
            )
            step2.run(
                input_path=step2_input,
                output_path=file_step2,
                additional_text=config.additional_text_docs,
            )
    else:
        print("[Skip] Step 2 (explicitly skipped)")
    
    # Step 3: Generate QA pairs
    if 'step3' not in skip_steps:
        step3_input = file_step2 if os.path.exists(file_step2) else input_file
        if os.path.exists(file_step3):
            print(f"[Skip] Step 3 output exists: {file_step3}")
        else:
            print("[Run] Step 3: Generating QA pairs (100 per sample)...")
            step3 = Step3GenQAPairs(
                provider=config.provider,
                api_key=config.api_key,
                base_url=config.base_url,
                model_name=config.model_name,
                max_workers=config.max_workers,
                api_concurrency=config.api_concurrency,
            )
            step3.run(
                input_path=step3_input,
                output_path=file_step3,
                qa_pairs_per_doc=config.qa_pairs_per_doc,
                additional_text=config.additional_text_qapairs,
                oq_learning_qa_pairs=config.oq_learning_qa_pairs,
                nq_learning_qa_pairs=config.nq_learning_qa_pairs,
                learning_qa_additional_text=config.learning_qa_additional_text,
                max_nqs_for_learning=config.max_nqs_for_learning,
                oq_nq_combined_qa_pairs=config.oq_nq_combined_qa_pairs,
            )
    else:
        print("[Skip] Step 3 (explicitly skipped)")
    
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print(f"Final output: {file_step3 if os.path.exists(file_step3) else file_step2 if os.path.exists(file_step2) else 'N/A'}")
    print("=" * 60)


def main():
    """Command line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the complete data augmentation pipeline.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--provider", type=str, default="deepseek", choices=["deepseek", "zhipu"])
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="https://www.dmxapi.cn/v1")
    parser.add_argument("--model_name", type=str, default="DeepSeek-V3.2")
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument("--api_concurrency", type=int, default=64)
    
    # Step 1 config (unified types: source_type + description_type)
    parser.add_argument("--max_types", type=int, default=2)
    parser.add_argument("--additional_text_types", type=str, default="")
    
    # Step 2 config
    parser.add_argument("--additional_text_docs", type=str, default="")
    
    # Step 3 config
    parser.add_argument("--qa_pairs_per_doc", type=int, default=10)
    parser.add_argument("--additional_text_qapairs", type=str, default="")
    parser.add_argument("--oq_learning_qa_pairs", type=int, default=8)
    parser.add_argument("--nq_learning_qa_pairs", type=int, default=5)
    parser.add_argument("--learning_qa_additional_text", type=str, default="")
    parser.add_argument("--max_nqs_for_learning", type=int, default=5)
    parser.add_argument("--oq_nq_combined_qa_pairs", type=int, default=10)
    
    # Pipeline control
    parser.add_argument("--skip_steps", type=str, nargs="+", default=[], help="Steps to skip (e.g., --skip_steps step1 step2)")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        max_workers=args.max_workers,
        api_concurrency=args.api_concurrency,
        max_types=args.max_types,
        additional_text_types=args.additional_text_types,
        additional_text_docs=args.additional_text_docs,
        qa_pairs_per_doc=args.qa_pairs_per_doc,
        additional_text_qapairs=args.additional_text_qapairs,
        oq_learning_qa_pairs=args.oq_learning_qa_pairs,
        nq_learning_qa_pairs=args.nq_learning_qa_pairs,
        learning_qa_additional_text=args.learning_qa_additional_text,
        max_nqs_for_learning=args.max_nqs_for_learning,
        oq_nq_combined_qa_pairs=args.oq_nq_combined_qa_pairs,
    )
    
    run_pipeline(
        input_file=args.input_file,
        output_dir=args.output_dir,
        config=config,
        skip_steps=args.skip_steps,
    )


if __name__ == "__main__":
    main()
