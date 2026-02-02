#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2025/10/03 19:56:13
@Author  :   haoming
@Version :   1.0
'''

import os
import torch
import torch.nn as nn
import yaml
from transformers.trainer_utils import get_last_checkpoint

def get_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def get_model_identifiers_from_yaml(model_family):
    '''
    Qwen2.5-7B-Instruct:
        hf_key: "Qwen2.5-7B-Instruct"
        gradient_checkpointing: "true"
    '''
    model_configs  = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config", "model_config.yaml")
    with open(config_path, "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )