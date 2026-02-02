#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline configuration management"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Pipeline configuration class"""
    # Basic configuration
    provider: str = "deepseek"
    base_url: str = "https://www.dmxapi.cn/v1"
    model_name: str = "DeepSeek-V3.2"
    max_workers: int = 64
    api_concurrency: int = 64
    api_key: Optional[str] = None
    
    # Step 1: unified types config (source_type + description_type)
    max_types: int = 2
    additional_text_types: str = ""
    
    # Step 3: docs config
    additional_text_docs: str = ""
    
    # Step 4: qa_pairs config
    # Quota allocation (total 100 per sample):
    # - Document QA: 40
    # - OQ learning QA: 20
    # - NQ learning QA: 20 (about 5 per NQ, processing about 4 NQs)
    # - OQ-NQ combined QA: 20 (about 5 per NQ, processing about 4 NQs)
    qa_pairs_per_doc: int = 10  # QA pairs per document (to reach the target of 40 document QAs total)
    additional_text_qapairs: str = ""
    
    # Step 4 additional: Learning QA config
    oq_learning_qa_pairs: int = 20  # OQ learning QA count (target 20)
    nq_learning_qa_pairs: int = 5  # Learning QA count per NQ
    learning_qa_additional_text: str = ""  # Additional instructions for learning QA
    max_nqs_for_learning: int = 4  # Max NQs for learning QA and combined QA (4×5=20)
    oq_nq_combined_qa_pairs: int = 5  # OQ-NQ combined QA count per NQ (4×5=20)
    
    @classmethod
    def from_args(cls, args) -> "PipelineConfig":
        """Create configuration from argparse namespace"""
        return cls(
            provider=getattr(args, 'provider', 'deepseek'),
            base_url=getattr(args, 'base_url', 'https://www.dmxapi.cn/v1'),
            model_name=getattr(args, 'model_name', 'DeepSeek-V3.2'),
            max_workers=getattr(args, 'max_workers', 64),
            api_concurrency=getattr(args, 'api_concurrency', 64),
            api_key=getattr(args, 'api_key', None),
            max_types=getattr(args, 'max_types', 2),
            additional_text_types=getattr(args, 'additional_text_types', ''),
            additional_text_docs=getattr(args, 'additional_text_docs', ''),
            qa_pairs_per_doc=getattr(args, 'qa_pairs_per_doc', 10),
            additional_text_qapairs=getattr(args, 'additional_text_qapairs', ''),
            oq_learning_qa_pairs=getattr(args, 'oq_learning_qa_pairs', 8),
            nq_learning_qa_pairs=getattr(args, 'nq_learning_qa_pairs', 5),
            learning_qa_additional_text=getattr(args, 'learning_qa_additional_text', ''),
            max_nqs_for_learning=getattr(args, 'max_nqs_for_learning', 5),
            oq_nq_combined_qa_pairs=getattr(args, 'oq_nq_combined_qa_pairs', 10),
        )
