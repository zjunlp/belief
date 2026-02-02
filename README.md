<div align="center">

**Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency**

[![arXiv](https://img.shields.io/badge/arXiv-2601.05905-b31b1b.svg)](https://arxiv.org/abs/2601.05905)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

*Haoming Xu, Ningyuan Zhao, Yunzhi Yao, Weihong Xu, Hongru Wang, Xinle Deng, Shumin Deng, Jeff Z. Pan, Huajun Chen, Ningyu Zhang*

</div>

---

## 📖 Overview

Traditional point-wise confidence measures (e.g., self-consistency) can create an **illusion of knowing**: even when models answer correctly with perfect self-consistency, their answers can collapse under mild contextual interference.

This repository implements novel approaches to diagnose and improve LLM truthfulness:

- **🎯 Neighbor-Consistency Belief (NCB)**: A structural measure of belief robustness computed over conceptual neighborhoods
- **🔬 Cognitive Stress Tests**: Contextual interference simulating social pressure and authority bias
- **🛠️ Structure-Aware Training (SAT)**: Enforces context-invariant belief structure (~30% less degradation under stress)

---

## 🗂️ Repository Structure

### 📊 Level 1: Belief Diagnosis & Scoring

Compute NCB-style belief scores to assess model confidence robustness:

```
analysis/level1_belief_classify/
├── gen_oq_dual_model.py      # Generate multiple samples + entity extraction
├── gen_nq.py                  # Answer neighbor questions
├── calc_belief_score.py       # Compute belief scores & split groups
└── run_all.sh                 # End-to-end pipeline
```

### 🧪 Level 2: Contextual Interference Stress Tests

Test belief robustness under cognitive pressure:

```
analysis/level2_belief_intervention/
├── misleading_steering.py     # Asch-style peer pressure + source credibility
└── run.sh                     # Full pipeline: retrieval → stress test → analysis
```

### 🎓 Training Pipeline

Structure-Aware Training with TRL, DeepSpeed, and LoRA:

```
training/
├── finetune/
│   ├── train.py              # Unified training entry point
│   └── config/               # Hydra configurations
└── scripts/
    └── finetune_trl.sh       # Convenience launcher
```

---

## 🚀 Getting Started

### Installation

```bash
# Create and activate conda environment
conda create -n confidence python=3.10 -y
conda activate confidence

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

#### 1️⃣ Compute Belief Scores

Edit paths in `analysis/level1_belief_classify/run_all.sh` and run:

```bash
bash analysis/level1_belief_classify/run_all.sh
```

**Example data**: `dataset/fact_belief_2000_annotated_nq_refined_verified.json`

#### 2️⃣ Run Stress Tests

Configure paths and execute:

```bash
TAG=experiment \
ORIGIN_DATA_DIR=/path/to/level1_output \
WORK_DATA_DIR=./output \
HALLUCINATION_FILE=dataset/misleading_nq.json \
TEST_MODEL_PATH=/path/to/your/model \
JUDGE_MODEL_PATH=/path/to/judge/model \
bash analysis/level2_belief_intervention/run.sh
```

#### 3️⃣ Train Models

Launch LoRA fine-tuning:

```bash
bash training/scripts/finetune_trl.sh
```

---

## 📁 Dataset

Sample datasets are provided in the `dataset/` directory:

- `fact_belief_2000_annotated_nq_refined_verified.json` - Annotated facts with neighbor questions
- `misleading_nq.json` - Misleading neighbor facts for stress testing

---

## 🎯 Key Concepts

### Neighbor-Consistency Belief (NCB)

NCB measures how robust a model's beliefs are by testing consistency across semantically related questions (neighbors), rather than just the same question multiple times.

### Cognitive Stress Tests

Two types of contextual interference:

- **👥 Peer Pressure**: Asch-style social consensus (misleading entities)
- **📚 Authority Bias**: High-credibility source influence (misleading neighbor facts)

### Structure-Aware Training (SAT)

Training approach that enforces context-invariant belief structures, making models more resistant to contextual interference.

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@misc{xu2026illusionsconfidencediagnosingllm,
  title={Illusions of Confidence? Diagnosing LLM Truthfulness via Neighborhood Consistency}, 
  author={Haoming Xu and Ningyuan Zhao and Yunzhi Yao and Weihong Xu and Hongru Wang and Xinle Deng and Shumin Deng and Jeff Z. Pan and Huajun Chen and Ningyu Zhang},
  year={2026},
  eprint={2601.05905},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.05905}
}
```

---

## 🙏 Acknowledgments

We thank the authors and maintainers of:

- **Datasets**: [SimpleQA](https://github.com/openai/simple-evals), [HotpotQA](https://hotpotqa.github.io/), [SciQ](https://allenai.org/data/sciq)
- **Libraries**: [Hugging Face Transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [PEFT](https://github.com/huggingface/peft), [vLLM](https://github.com/vllm-project/vllm)