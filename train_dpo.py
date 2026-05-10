"""
DPO (Direct Preference Optimization) Baseline Training Script

Implements Direct Preference Optimization as a baseline method for comparison
with contrastive perplexity.

DPO directly optimizes the policy to prefer non-toxic over toxic completions
without requiring a separate reward model.

Usage:
    python train_dpo.py --model_name mistralai/Mistral-7B-v0.1

Authors: Tassail Klein, Moin Nabi
License: Apache 2.0
"""

import transformers
from dataclasses import dataclass, field
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import TrainerCallback, GenerationConfig
from transformers.utils import PaddingStrategy
from datasets import load_dataset
import wandb
from random import choice
import random
import os
import torch
from trl import DPOTrainer
import sys
import argparse
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==================== 命令行参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
args_cmd = parser.parse_args()

# ==================== 加载数据集 ====================
train_dataset = load_dataset("json", data_files="dpo_data.json", split="train")
original_columns = train_dataset.column_names

def return_prompt_and_responses(samples):
    return {
        "prompt": [
            f"### Input: ```{input}```\n ### Output: "
            for input in samples["input"]
        ],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"],
    }

train_dataset = train_dataset.map(
    return_prompt_and_responses,
    batched=True,
    remove_columns=original_columns
)

# ==================== 训练参数 ====================
class Args:
    pass

args = Args()
args.description = "detox-DPO"
args.tags = None
args.use_4bit_quantization = True
args.use_8bit_quantization = False
args.output_dir = "results"
args.model_name = args_cmd.model_name
args.bnb_4bit_quant_type = "nf4"
args.bnb_4bit_compute_dtype = "float16"
args.use_nested_quant = True
args.lora_target_modules = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
args.per_device_train_batch_size = 2
args.gradient_accumulation_steps = 3
args.num_train_epochs = 1
args.bf16 = True
args.lora_r = 64
args.weight_decay = 0.0
args.lr_scheduler_type = "linear"
args.eval_steps = None
args.save_steps = 10
args.warmup_steps = 0
args.warmup_ratio = 0.0
args.use_gradient_checkpointing = True
args.optim = "adamw_torch"
args.num_workers = 4
args.beta = 0.1
args.push_to_hub = False
args.lora_alpha = 16
args.lora_dropout = 0.1
args.seed = 42
args.learning_rate = 2e-4
args.fp16 = False
args.logging_steps = 10

# ==================== Wandb ====================
wandb_project = args.description
if args.tags:
    args.tags = [item.strip() for item in args.tags.split(',')]
    wandb.init(project=wandb_project, tags=args.tags)
else:
    wandb.init(project=wandb_project)

output_name = wandb.run.name if wandb.run.name else 'dummy-run'
args.output_dir = os.path.join(args.output_dir, output_name)
os.makedirs(args.output_dir, exist_ok=True)

use_wandb = True

# ==================== 4bit 量化配置 ====================
compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=args.use_nested_quant,
    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype
)

device_map = "auto"

# ==================== 加载模型 & Tokenizer ====================
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=compute_dtype,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# ==================== LoRA 配置 ====================
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=[item.strip() for item in args.lora_target_modules.split(',')],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

# ==================== DPO Trainer ====================
dpo_trainer = DPOTrainer(
    model=model,
    peft_config=lora_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    beta=args.beta,
    max_length=512,
    max_prompt_length=128,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.use_gradient_checkpointing,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        evaluation_strategy="no",
        save_strategy="no",
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        optim=args.optim,
        push_to_hub=args.push_to_hub,
        report_to="wandb",
        run_name=wandb.run.name if use_wandb else None,
        dataloader_num_workers=args.num_workers,
        seed=args.seed
    ),
)

# ==================== 开始训练 ====================
dpo_trainer.train()
dpo_trainer.save_model(args.output_dir)