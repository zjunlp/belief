#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   finetune.py
@Time    :   2025/10/03 19:54:03
@Author  :   haoming
@Version :   1.0
@Desc    :   Unified training script supporting SFT, DPO, GRPO, and CONTEXT-INJECTION
'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, GRPOTrainer, GRPOConfig
from peft import LoraConfig
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_model_identifiers_from_yaml, get_checkpoint
from reward import get_reward_function, init_consistency_reward, init_acc_reward
from dataset import (
    SimpleQASFTDataset,
    BeliefAugmentedSFTDataset,
    ContextInjectionDataset,
    ContextInjectionDataCollator,
    ContextDistillationDataset,
    ContextDistillationDataCollator,
    ContextDistillationWithKLDataset,
    ContextDistillationWithKLDataCollator,
    ContextDistillationWithKLSFTDataset,
    ContextDistillationWithKLSFTDataCollator,
    DPODataset,
    GRPODataset,
    KLDivergenceDataset,
    MixedSFTDataset,
    MixedSFTDatasetEXP
)
from context_injection_trainer import ContextInjectionTrainer
from context_distillation_trainer import ContextDistillationTrainer
from kl_trainer import KLDivergenceTrainer, TokenizedQASFTDataset, TwoDatasetWrapper, TwoDatasetDataCollator
from context_distill_kl_trainer import ContextDistillKLTrainer
from context_distill_kl_sft_trainer import ContextDistillKLKLTrainer, ContextDistillKLSFTTrainer
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# os.environ["SWANLAB_MODE"] = "disabled"

@hydra.main(version_base=None, config_path="./config", config_name="finetune_lora")
def finetune(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    set_seed(cfg.seed)

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    data_file = cfg.data_path

    # --nproc_per_node gives the number of GPUs per = num_devices. take it from torchrun/os.environ
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # save the cfg file if master process
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("The pad token is not set, set it to the eos token")
    
    # Determine training method: sft (default), dpo, grpo, or context_injection
    training_method = getattr(cfg, 'training_method', 'sft')
    use_context_injection = getattr(cfg, 'context_injection', False)
    
    # Override training_method if context_injection is True (for backward compatibility)
    if use_context_injection:
        training_method = 'context_injection'
    
    print(f"Training method: {training_method}")
    
    # Load appropriate dataset based on training method
    data_collator = None
    belief_setting = getattr(cfg, 'belief_setting', 'none')
    if training_method == 'dpo':
        print(f"Loading DPO dataset from: {data_file}")
        dataset = DPODataset(
            data_file,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
        )
    elif training_method == 'grpo':
        print(f"Loading GRPO dataset from: {data_file}")
        use_system_prompt = getattr(cfg, 'use_system_prompt', True)
        use_answer_tags = getattr(cfg, 'use_answer_tags', True)
        print(f"  - System prompt: {use_system_prompt}")
        print(f"  - Answer tags: {use_answer_tags}")
        dataset = GRPODataset(
            data_file,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
            use_system_prompt=use_system_prompt,
            use_answer_tags=use_answer_tags,
        )
    elif training_method == 'context_injection':
        print(f"Loading CONTEXT-INJECTION dataset from: {data_file}")
        max_samples = getattr(cfg, 'max_samples', None)
        if max_samples is not None:
            print(f"  - Limiting dataset to max_samples: {max_samples}")
        dataset = ContextInjectionDataset(
            data_file,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
            use_chat_template=True,
            only_use_answers=getattr(cfg.context_injection, 'only_use_answers', True),
            max_samples=max_samples,
        )
        print(f"only_use_answers: {getattr(cfg.context_injection, 'only_use_answers', True)}")
        data_collator = ContextInjectionDataCollator(tokenizer)
    elif training_method == 'context_distillation':
        print(f"Loading CONTEXT-DISTILLATION dataset from: {data_file}")
        # FIXME: NOT IMPLEMENTED max_samples for context distillation
        max_samples = getattr(cfg, 'max_samples', None)
        if max_samples is not None:
            print(f"  - Limiting dataset to max_samples: {max_samples}")
        dataset = ContextDistillationDataset(
            data_file,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
        )
        data_collator = ContextDistillationDataCollator(tokenizer)
    elif training_method == 'context_distill_kl':
        print(f"Loading CONTEXT-DISTILL-KL dataset from: {data_file}")
        cd_dataset = ContextDistillationDataset(
            data_file,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
        )
        kl_data_path = getattr(cfg.context_distill_kl, 'kl_data_path', None)
        if kl_data_path is None or str(kl_data_path).strip() == "":
            raise ValueError("context_distill_kl.kl_data_path is required")

        print(f"Loading CONTEXT-DISTILL-KL KL dataset from: {kl_data_path}")
        kl_dataset = TokenizedQASFTDataset(
            kl_data_path,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
        )

        kl_sampling = getattr(cfg.context_distill_kl, 'kl_sampling', 'random')
        sft_data_path = getattr(cfg.context_distill_kl, 'sft_data_path', None)
        if sft_data_path is not None and str(sft_data_path).strip() != "":
            print(f"Loading CONTEXT-DISTILL-KL SFT dataset from: {sft_data_path}")
            sft_dataset = TokenizedQASFTDataset(
                sft_data_path,
                tokenizer=tokenizer,
                max_seq_length=cfg.max_length,
            )
            sft_sampling = getattr(cfg.context_distill_kl, 'sft_sampling', 'random')
            dataset = ContextDistillationWithKLSFTDataset(
                cd_dataset,
                kl_dataset,
                sft_dataset,
                kl_sampling=kl_sampling,
                sft_sampling=sft_sampling,
            )
            data_collator = ContextDistillationWithKLSFTDataCollator(tokenizer)
        else:
            dataset = ContextDistillationWithKLDataset(cd_dataset, kl_dataset, kl_sampling=kl_sampling)
            data_collator = ContextDistillationWithKLDataCollator(tokenizer)
    elif training_method == 'kl_sft':
        print(f"Loading KL-SFT main dataset from: {data_file}")
        dataset = TokenizedQASFTDataset(
            data_file,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
        )
    elif training_method == "mixsft":
        print(f"Loading Mix SFT dataset from: {data_file}")
        dataset = MixedSFTDatasetEXP(
            data_file,
            cfg.doc_data_path,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
        )
    elif training_method == 'mixed_context_kl':
        try:
            from dataset import MixedContextKLDataset, MixedContextKLDataCollator
            from mixed_context_kl_trainer import MixedContextKLTrainer
        except ImportError:
            raise ValueError(
                "training_method 'mixed_context_kl' is currently unavailable in this workspace. "
                "The required dataset/collator/trainer modules are missing."
            )
        print(f"Loading Mixed Context-KL dataset")
        print(f"  - Context distillation dataset from: {data_file}")
        
        # Load context distillation dataset
        cd_dataset = ContextDistillationDataset(
            data_file,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
        )
        
        # Load KL dataset
        kl_data_path = getattr(cfg.mixed_context_kl, 'kl_data_path', None)
        if kl_data_path is None:
            raise ValueError("kl_data_path is required for mixed_context_kl training method")
        
        print(f"  - KL dataset from: {kl_data_path}")
        kl_dataset = TokenizedQASFTDataset(
            kl_data_path,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
        )
        
        # Create mixed dataset
        kl_sampling = getattr(cfg.mixed_context_kl, 'kl_sampling', 'random')
        dataset = MixedContextKLDataset(
            cd_dataset,
            kl_dataset,
            sampling=kl_sampling
        )
        data_collator = MixedContextKLDataCollator(tokenizer)
    else:
        if belief_setting and belief_setting.lower() != 'none':
            print(f"Loading Belief-Augmented SFT dataset ({belief_setting}) from: {data_file}")
            dataset = BeliefAugmentedSFTDataset(
                data_file,
                tokenizer=tokenizer,
                max_seq_length=cfg.max_length,
                belief_setting=belief_setting,
                max_belief_rounds=getattr(cfg, 'belief_max_rounds', 3),
                system_prompt=getattr(cfg, 'belief_system_prompt', None),
            )
        else:
            print(f"Loading SFT dataset from: {data_file}")
            dataset = SimpleQASFTDataset(
                data_file,
                tokenizer=tokenizer,
                max_seq_length=cfg.max_length,
            )

    max_steps = int(cfg.num_epochs*len(dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    
    # Create experiment name with more details
    belief_suffix = ""
    if training_method == 'sft' and belief_setting and belief_setting.lower() != 'none':
        belief_suffix = f"{belief_setting}-r{getattr(cfg, 'belief_max_rounds', 3)}"

    experiment_name = f"{training_method}_{cfg.model_family}{belief_suffix}_lr{cfg.lr}_ep{cfg.num_epochs}_bs{batch_size}x{gradient_accumulation_steps}"
    
    # Only initialize SwanLab on main process
    is_main_process = os.environ.get('LOCAL_RANK') is None or int(os.environ.get('LOCAL_RANK', '0')) == 0
    
    if is_main_process:
        print(f"[Main Process] Initializing SwanLab with experiment name: {experiment_name}")
        
        # Manually initialize swanlab run to ensure experiment name is set
        swanlab.init(
            project="SFT_belief",
            workspace="haomingx",
            experiment_name=experiment_name,
            description=f"{training_method.upper()} training on {cfg.model_family}",
            config={
                "framework": "TRL",
                "training_method": training_method,
                "model": cfg.model_family,
                "learning_rate": cfg.lr,
                "num_epochs": cfg.num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_generations": getattr(cfg, 'num_generations', None),
                "dataset_size": len(dataset),
                "max_steps": max_steps,
            },
            resume=cfg.resume_from_checkpoint,
            id=cfg.swanlab_id if cfg.swanlab_id and cfg.swanlab_id != "" else None
        )
    
    # Create callback (will use the already initialized run)
    swanlab_callback = SwanLabCallback()

    common_args = {
        "output_dir": cfg.save_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": max(1, max_steps//cfg.num_epochs),
        "max_steps": max_steps,
        "learning_rate": cfg.lr,
        "bf16": True,
        "bf16_full_eval": True,
        "logging_strategy": "steps",
        "logging_dir": f'{cfg.save_dir}/logs',
        "logging_steps": cfg.log_nums,
        "optim": "paged_adamw_32bit",
        "save_steps": max(1, max_steps//cfg.save_nums),
        "save_only_model": False,
        "ddp_find_unused_parameters": False,
        "eval_strategy": "no",
        "deepspeed": cfg.ds_config,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
        # "report_to": "swanlab",
    }
    
    # Create config based on training method
    if training_method == 'dpo':
        training_config = DPOConfig(
            **common_args,
            beta=getattr(cfg.dpo, 'beta', 0.1),
            loss_type=getattr(cfg.dpo, 'loss_type', 'sigmoid'),
            max_length=cfg.max_length,
            max_prompt_length=getattr(cfg, 'max_prompt_length', 512),
        )
    elif training_method == 'grpo':
        training_config = GRPOConfig(**common_args,
            # use_vllm=True,
            # vllm_mode="colocate",
            # vllm_gpu_memory_utilization=0.1,
            # vllm_tensor_parallel_size=1,
            # vllm_enable_sleep_mode=False,
            loss_type=cfg.loss_type if cfg.loss_type else "grpo", # grpo, dapo, bnpo, dr_grpo
            num_generations=cfg.num_generations if cfg.num_generations else 8,
            generation_batch_size=cfg.generation_batch_size if cfg.generation_batch_size else 8,
            temperature=getattr(cfg, 'temperature', 0.7),
            top_p=getattr(cfg, 'top_p', 0.9),
            beta=getattr(cfg, 'beta', 0.001),
            # max_prompt_length=cfg.max_prompt_length,
            # max_completion_length=cfg.max_completion_length,
        )
    elif training_method == "mixsft":
        training_config = TrainingArguments(
            **common_args,
        )
    elif training_method == 'mixed_context_kl':
        training_config = TrainingArguments(
            **common_args,
        )
    elif training_method == 'context_distill_kl':
        training_config = TrainingArguments(
            **common_args,
        )
    else:
        training_config = SFTConfig(
            **common_args,
            max_length=cfg.max_length,
            packing=False,
        )
    
    print(f"Loading model from: {model_id}")
    # if "llama" in model_id.lower():
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_id, 
    #         dtype=torch.bfloat16, 
    #         trust_remote_code=True
    #     )
    # elif "qwen" in model_id.lower():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        trust_remote_code=True
    )
        
    if model_cfg["gradient_checkpointing"] == "true":
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # Configure LoRA
    if cfg.LoRA.r != 0:
        peft_config = LoraConfig(
            r=cfg.LoRA.r, 
            lora_alpha=cfg.LoRA.alpha, 
            target_modules="all-linear",
            lora_dropout=cfg.LoRA.dropout,
        )
    else:
        peft_config = None
    print(training_config)
    print(peft_config)
    
    # Create trainer based on training method
    print(f"Creating {training_method.upper()} trainer...")
    
    if training_method == 'dpo':
        trainer = DPOTrainer(
            model=model,
            # For training PEFT adapters with DPO there is no need to pass a reference model.
            ref_model=None,
            args=training_config,
            train_dataset=dataset.dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[swanlab_callback],
        )
    elif training_method == 'grpo':
        # Get reward type from config (default: 'acc')
        reward_type = getattr(cfg, 'reward_type', 'acc')
        
        print(f"Using reward type: {reward_type}")
        
        # Initialize LLM-based verification if judge model is configured
        judge_model = getattr(cfg, 'judge_model', None)
        use_api = getattr(cfg, 'use_api', False)
        judge_device_map = getattr(cfg, 'judge_device_map', "4,5,6,7")
        if judge_model or use_api:
            if use_api:
                api_key = getattr(cfg, 'api_key', None)
                base_url = getattr(cfg, 'base_url', 'https://api.deepseek.com')
                judge_model = "deepseek-chat"
            else:
                api_key = None
                base_url = None
            if reward_type == 'consistency':
                # Use OpenAI-compatible API for entity extraction
                print(f"Initializing API-based consistency reward with judge model: {judge_model}")
                init_consistency_reward(judge_model, api_key, base_url)
            elif reward_type == 'acc' or reward_type == 'accuracy':
                # Use LLM-based answer verification
                print(f"Initializing LLM-based ACC reward with judge model: {judge_model}")
                init_acc_reward(judge_model, api_key, base_url,use_api,judge_device_map)
        else:
            if reward_type == 'consistency':
                print("Using rule-based entity extraction for consistency reward")
            elif reward_type == 'acc' or reward_type == 'accuracy':
                print("Using rule-based matching for ACC reward")
        
        # Get reward function from registry
        reward_func = get_reward_function(reward_type)
        
        trainer = GRPOTrainer(
            model=model,
            args=training_config,
            train_dataset=dataset.dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            reward_funcs=reward_func,
            callbacks=[swanlab_callback],
        )
    elif training_method == 'context_injection':
        trainer = ContextInjectionTrainer(
            distill_temperature=getattr(cfg.context_injection, 'distill_temperature', 1.0),
            use_distillation=getattr(cfg.context_injection, 'use_distillation', True),
            model=model,
            args=training_config,
            processing_class=tokenizer,
            train_dataset=dataset.dataset,
            data_collator=data_collator,
            peft_config=peft_config,
            callbacks=[swanlab_callback],
        )

    elif training_method == 'context_distillation':
        trainer = ContextDistillationTrainer(
            distill_temperature=getattr(cfg.context_distillation, 'distill_temperature', 1.0),
            use_distillation=getattr(cfg.context_distillation, 'use_distillation', True),
            # collect_steps=getattr(cfg.context_distillation, 'collect_steps', 1),
            # teacher_lora_path=getattr(cfg, 'teacher_lora_path', None),
            # teacher_adapter_name=getattr(cfg, 'teacher_adapter_name', 'teacher'),
            # student_adapter_name=getattr(cfg, 'student_adapter_name', None),
            model=model,
            args=training_config,
            processing_class=tokenizer,
            train_dataset=dataset.dataset,
            data_collator=data_collator,
            peft_config=peft_config,
            callbacks=[swanlab_callback],
        )
    elif training_method == 'context_distill_kl':
        distill_temperature = getattr(cfg.context_distill_kl, 'distill_temperature', 1.0)
        kl_weight = getattr(cfg.context_distill_kl, 'kl_weight', 0.1)
        kl_temperature = getattr(cfg.context_distill_kl, 'kl_temperature', 1.0)
        sft_weight = getattr(cfg.context_distill_kl, 'sft_weight', 1.0)
        teacher_model_id = getattr(cfg.context_distill_kl, 'teacher_model_id', None)
        ref_model_id = getattr(cfg.context_distill_kl, 'ref_model_id', None)

        if teacher_model_id is None or str(teacher_model_id).strip() == "":
            raise ValueError("context_distill_kl.teacher_model_id is required")

        print(f"Loading teacher model from: {teacher_model_id}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_id,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        ref_model = None
        if ref_model_id is not None and str(ref_model_id).strip() != "":
            print(f"Loading KL reference model from: {ref_model_id}")
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model_id,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )

        if peft_config is not None:
            from peft import get_peft_model
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        print(
            f"Initializing CONTEXT-DISTILL-KL Trainer with distill_temperature={distill_temperature}, "
            f"kl_weight={kl_weight}, kl_temperature={kl_temperature}, "
            f"teacher_model=provided, ref_model={'provided' if ref_model is not None else 'none'}"
        )

        sft_data_path = getattr(cfg.context_distill_kl, 'sft_data_path', None)
        if sft_data_path is not None and str(sft_data_path).strip() != "":
            print(f"  - SFT enabled: sft_weight={sft_weight}")
            trainer = ContextDistillKLKLTrainer(
                model=model,
                teacher_model=teacher_model,
                ref_model=ref_model,
                distill_temperature=distill_temperature,
                kl_weight=kl_weight,
                kl_temperature=kl_temperature,
                sft_weight=sft_weight,
                args=training_config,
                processing_class=tokenizer,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[swanlab_callback],
            )
        else:
            trainer = ContextDistillKLTrainer(
                model=model,
                teacher_model=teacher_model,
                ref_model=ref_model,
                distill_temperature=distill_temperature,
                kl_weight=kl_weight,
                kl_temperature=kl_temperature,
                args=training_config,
                processing_class=tokenizer,
                train_dataset=dataset,
                data_collator=data_collator,
                callbacks=[swanlab_callback],
            )
    elif training_method == 'kl_sft':
        kl_weight = getattr(cfg.kl_sft, 'kl_weight', 0.1)
        kl_temperature = getattr(cfg.kl_sft, 'kl_temperature', 1.0)
        kl_data_path = getattr(cfg.kl_sft, 'kl_data_path', None)
        ref_model_id = getattr(cfg.kl_sft, 'ref_model_id', None)
        kl_sampling = getattr(cfg.kl_sft, 'kl_sampling', 'random')
        
        if kl_data_path is None:
            raise ValueError("kl_data_path is required for kl_sft training method")
        
        print(f"Loading KL constraint dataset from: {kl_data_path}")
        kl_dataset = TokenizedQASFTDataset(
            kl_data_path,
            tokenizer=tokenizer,
            max_seq_length=cfg.max_length,
        )
        
        two_dataset = TwoDatasetWrapper(dataset, kl_dataset, kl_sampling=kl_sampling)
        data_collator = TwoDatasetDataCollator(tokenizer)
        
        ref_model = None
        if ref_model_id is not None and str(ref_model_id).strip() != "":
            print(f"Loading KL reference model from: {ref_model_id}")
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model_id,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        if peft_config is not None:
            from peft import get_peft_model
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        print(
            f"Initializing KL-SFT Trainer with kl_weight={kl_weight}, kl_temperature={kl_temperature}, "
            f"ref_model={'provided' if ref_model is not None else 'model.disable_adapter()'}"
        )
        
        trainer = KLDivergenceTrainer(
            model=model,
            ref_model=ref_model,
            kl_weight=kl_weight,
            kl_temperature=kl_temperature,
            args=training_config,
            processing_class=tokenizer,
            train_dataset=two_dataset,
            data_collator=data_collator,
            callbacks=[swanlab_callback],
        )
    elif training_method == "mixsft":
        # TO enable packing
        # use DataCollatorWithFlattening
        from transformers import DataCollatorWithFlattening
        data_collator = DataCollatorWithFlattening()

        if peft_config is not None:
            from peft import get_peft_model
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
        trainer = Trainer(
            model=model,
            args=training_config,
            train_dataset=dataset.dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[swanlab_callback],
        )
        # # detokenize the input_ids, first and last sentence
        # input_ids = trainer.train_dataset[0]["input_ids"]
        # print(tokenizer.decode(input_ids))
        # print([(a,b) for a, b in zip(input_ids, trainer.train_dataset[0]["labels"])])
        # # print the last sentence
        # input_ids = trainer.train_dataset[-1]["input_ids"]
        # print(tokenizer.decode(input_ids))
        # print([(a,b) for a, b in zip(input_ids, trainer.train_dataset[-1]["labels"])])
        # import sys
        # sys.exit()

    elif training_method == 'mixed_context_kl':
        try:
            from mixed_context_kl_trainer import MixedContextKLTrainer
        except ImportError:
            raise ValueError(
                "training_method 'mixed_context_kl' is currently unavailable in this workspace. "
                "The required trainer module is missing."
            )
        # Get configuration parameters
        distill_temperature = getattr(cfg.mixed_context_kl, 'distill_temperature', 1.0)
        kl_weight = getattr(cfg.mixed_context_kl, 'kl_weight', 0.1)
        kl_temperature = getattr(cfg.mixed_context_kl, 'kl_temperature', 1.0)
        ref_model_id = getattr(cfg.mixed_context_kl, 'ref_model_id', None)
        
        ref_model = None
        if ref_model_id is not None and str(ref_model_id).strip() != "":
            print(f"Loading KL reference model from: {ref_model_id}")
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model_id,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        
        if peft_config is not None:
            from peft import get_peft_model
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        print(
            f"Initializing Mixed Context-KL Trainer with "
            f"distill_temperature={distill_temperature}, kl_weight={kl_weight}, "
            f"kl_temperature={kl_temperature}, "
            f"ref_model={'provided' if ref_model is not None else 'model.disable_adapter()'}"
        )
        
        trainer = MixedContextKLTrainer(
            model=model,
            ref_model=ref_model,
            distill_temperature=distill_temperature,
            kl_weight=kl_weight,
            kl_temperature=kl_temperature,
            args=training_config,
            processing_class=tokenizer,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[swanlab_callback],
        )

    else:
        trainer = SFTTrainer(
            model=model,
            args=training_config,
            processing_class=tokenizer,
            train_dataset=dataset.dataset,
            eval_dataset=dataset.dataset,
            peft_config=peft_config,
            callbacks=[swanlab_callback],
        )
    
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    print(f"\n{'='*60}")
    print(f"Starting {training_method.upper()} training")
    print(f"{'='*60}")
    print(f"Model: {cfg.model_family}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Max steps: {max_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Save directory: {cfg.save_dir}")
    print(f"Resume from checkpoint: {cfg.resume_from_checkpoint}")
    print(f"{'='*60}\n")
    
    trainer.train(resume_from_checkpoint=get_checkpoint(training_config) if cfg.resume_from_checkpoint else None)

    # Save the final model
    trainer.model.config.use_cache = True
    trainer.save_model(cfg.save_dir)
    # tokenizer.save_pretrained(cfg.save_dir)
    
    print(f"\n{'='*60}")
    print(f"{training_method.upper()} training completed!")
    print(f"Model saved to: {cfg.save_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    finetune()
