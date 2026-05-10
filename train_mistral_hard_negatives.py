"""
Contrastive Perplexity Training with Hard Negatives (Mistral)

This variant uses hard negative mining to generate more challenging toxic examples
that are semantically similar to non-toxic text, improving the robustness of detoxification.

Hard negatives are toxic paraphrases that are:
- Semantically similar to non-toxic counterparts
- Matched in length and model probability
- Generated through adversarial paraphrasing

Usage:
    python train_mistral_hard_negatives.py \
        --model_name mistralai/Mistral-7B-v0.1 \
        --num_pos 5 --num_neg 5 --alpha 100.0

Authors: Tassilo Klein, Moin Nabi
License: Apache 2.0
"""
from models.patch_utils import patch_causal_lm_for_loss_reduction
import argparse
import glob
import einops
import json
import sys
import os
import random
from collections import defaultdict
from tqdm import tqdm, trange
from datasets import Dataset, load_dataset
import torch
from torch import nn
from transformers import HfArgumentParser, TrainingArguments, Trainer
from torch.nn import CrossEntropyLoss
from typing import Any, Dict, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import wandb
from random import choice

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,  # <-- changed
    set_peft_model_state_dict,
)

from transformers import (
    MistralConfig,
    MistralForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Define and parse arguments.
@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    alpha: Optional[float] = field(default=100.0)
    beta: Optional[float] = field(default=1.0)
    detox: Optional[float] = field(default=1.0)
    toxify: Optional[float] = field(default=1.0)
    tau: Optional[float] = field(default=1.0)
    smoothing: Optional[float] = field(default=0.0)
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    num_pos: Optional[int] = field(default=6)
    num_neg: Optional[int] = field(default=6)
    num_rnd: Optional[int] = field(default=6)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[float] = field(default=0.001)
    max_prompts_per_group: Optional[int] = field(default=500)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    max_seq_length: Optional[int] = field(default=512)
    model_name: Optional[str] = field(
        default="mistralai/Mistral-7B-Instruct-v0.1",
        metadata={"help": "The model that you want to train from the Hugging Face hub."},
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    use_instruction: Optional[bool] = field(default=False)
    use_llama_chat_format: Optional[bool] = field(default=False)
    use_chat_format: Optional[bool] = field(default=False)

    # quantization flags (your code uses 4bit by default)
    use_8bit: Optional[bool] = field(default=False)
    use_4bit: Optional[bool] = field(default=True)
    use_nested_quant: Optional[bool] = field(default=False)
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16")
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")

    num_train_epochs: Optional[int] = field(default=4)
    fp16: Optional[bool] = field(default=False)
    bf16: Optional[bool] = field(default=False)
    packing: Optional[bool] = field(default=False)
    gradient_checkpointing: Optional[bool] = field(default=True)

    optim: Optional[str] = field(default="adamw_torch")
    lr_scheduler_type: str = field(default="constant")
    max_steps: int = field(default=5000)
    warmup_steps: int = field(default=0)
    warmup_ratio: float = field(default=0.0)
    save_steps: int = field(default=1000)
    eval_steps: int = field(default=None)
    logging_steps: int = field(default=10)
    output_dir: str = field(default="root/autodl-tmp/results")
    data_dir: str = field(default="data/paraphrase/safeNLP_processed")
    use_flash_attn: Optional[bool] = field(default=False)

    use_peft_lora: Optional[bool] = field(default=False)

    use_8bit_quantization: Optional[bool] = field(default=False)
    use_4bit_quantization: Optional[bool] = field(default=True)
    use_gradient_checkpointing: Optional[bool] = field(default=False)

    dataset_text_field: str = field(default="text")
    push_to_hub: Optional[bool] = field(default=False)
    num_workers: int = field(default=4)
    debug: Optional[bool] = field(default=False)


def main(args: ScriptArguments):
    # ----------------------------
    # indices used later in loss
    # ----------------------------
    idx_vec = []
    for _ in range(int(args.per_device_train_batch_size)):
        idx_vec.append(
            torch.cat(
                (
                    2 * torch.ones(int(args.num_pos), 1),
                    3 * torch.ones(int(args.num_neg), 1),
                    4 * torch.ones(int(args.num_rnd), 1),
                ),
                dim=0,
            )
        )
    idx_vec = torch.cat(idx_vec)
    seq_pos_idx = torch.where(idx_vec == 2)[0]
    seq_neg_idx = torch.where(idx_vec == 3)[0]
    seq_rnd_idx = torch.where(idx_vec == 4)[0]

    wandb_project = "Seq2SeqDetox_non_instruct_hard_negatives"
    wandb_run_name = ""
    wandb.init(project=wandb_project)

    output_name = wandb.run.name if wandb.run and wandb.run.name else "dummy-run"
    args.output_dir = os.path.join(args.output_dir, output_name)

    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    resume_from_checkpoint = None

    # ----------------------------
    # quant config
    # ----------------------------
    bnb_config = None
    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

        if compute_dtype == torch.float16:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)

    device_map = "auto"

    config = MistralConfig.from_pretrained(args.model_name)
    model = MistralForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=bnb_config,
        use_flash_attention_2=True,  # you can later replace with attn_implementation
    )
    model = patch_causal_lm_for_loss_reduction(model)
    # IMPORTANT: for 4bit/8bit training preparation
    model = prepare_model_for_kbit_training(model)

    # ----------------------------
    # LoRA
    # ----------------------------
    if len(args.lora_target_modules) > 0:
        lora_config = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            target_modules=[item for item in args.lora_target_modules.split(",")],
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        lora_config = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    cutoff_len = 256

    def tokenize_text(text: str, add_eos_token: bool = True) -> Dict[str, Any]:
        result = tokenizer(
            text,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if add_eos_token and len(result["input_ids"]) < cutoff_len:
            if len(result["input_ids"]) == 0 or result["input_ids"][-1] != tokenizer.eos_token_id:
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)
        return result

    random.seed(42)
    np.random.seed(42)

    llama2_prompting_template = """[INST] {prompt} [/INST]"""
    tulu_prompting_template = """<|user|>\n{prompt}\n<|assistant|>\n"""

    # ----------------------------
    # load dataset jsons
    # ----------------------------
    pos_prompts: Dict[str, list] = {}
    neg_prompts: Dict[str, list] = {}
    rnd_neg_prompts: Dict[str, list] = {}

    groups = [
        "asian",
        "black",
        "chinese",
        "jewish",
        "latino",
        "lgbtq",
        "mental_disability",
        "mexican",
        "middle_east",
        "muslim",
        "native_american",
        "physical_disability",
        "women",
    ]
    group_files = [f"neutral_{g}.json" for g in groups]

    # ============================
    # 方案1改动：使用绝对路径列表传入 data_files，避免 fs.protocol tuple 拼接 bug
    # ============================
    train_files = [os.path.join(args.data_dir, f) for f in group_files]
    dataset = load_dataset("json", data_files={"train": train_files})
    # ============================

    remove_id = []
    group_set = set([x["id"] for x in dataset["train"]])
    for i in group_set:
        rnd_neg_prompts[i] = []

    for idx, data in enumerate(tqdm(dataset["train"])):
        without_current_group = [x for x in group_set if not (x.startswith(data["id"][: data["id"].find("-")]))]

        # build pos/neg dicts
        for num, curr_dict, tag in zip(
            [int(args.num_pos), int(args.num_neg)],
            [pos_prompts, neg_prompts],
            ["paraphrases", "paraphrases_toxic"],
        ):
            prompt_list = []
            for prompt in data[tag]:
                prompt = prompt.replace("\\\\", "\\")
                prompt = prompt.replace("\\n", "\n")

                if args.use_llama_chat_format:
                    prompt = llama2_prompting_template.format(prompt="Complete the following: " + prompt)
                elif args.use_chat_format:
                    prompt = "<|user|>\nComplete the following: " + prompt + "\n<|assistant|>\nA:"
                prompt_list.append(prompt)

                # rnd negatives pool uses toxic paraphrases
                if tag == "paraphrases_toxic":
                    NUM_RAND_SAMPLES = 100
                    for _ in range(NUM_RAND_SAMPLES):
                        rnd_neg = choice(without_current_group)
                        rnd_neg_prompts[rnd_neg].append(prompt)

            curr_dict[data["id"]] = prompt_list

            # if pos or neg pool too small, mark for removal
            if len(curr_dict[data["id"]]) < num:
                remove_id.append(idx)

    dataset["train"] = dataset["train"].select((i for i in range(len(dataset["train"])) if i not in set(remove_id)))
    print(f"Number of removed items: {len(np.unique(remove_id))}")

    # ----------------------------
    # map into simple (text,label) dataset
    # ----------------------------
    def build_train_rows(dp, **kwargs):
        if args.use_llama_chat_format:
            ref_text = llama2_prompting_template.format(prompt="Complete the following: " + dp["instruction"])
        elif args.use_chat_format:
            ref_text = "<|user|>\nComplete the following: " + dp["instruction"] + "\n<|assistant|>\nA:"
        else:
            ref_text = dp["input"]

        return {"input_text": ref_text, "label": dp["id"]}

    train_ds = dataset["train"].shuffle(seed=42).map(build_train_rows).remove_columns(
        ["input", "id", "paraphrases", "paraphrases_toxic", "group"]
    )

    # ----------------------------
    # Data collator (FIXED)
    # ----------------------------
    @dataclass
    class MyDataCollatorForSeq2Seq:
        tokenizer: PreTrainedTokenizerBase
        pos_dict: dict
        neg_dict: dict
        rnd_dict: dict
        num_pos: int
        num_hard_negs: int
        num_rnd_negs: int
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        return_tensors: str = "pt"

        def __call__(self, pre_features, return_tensors=None):
            if return_tensors is None:
                return_tensors = self.return_tensors

            features = []

            for feature in pre_features:
                label = feature["label"]
                ref_text = feature["input_text"]

                # Pools
                pos_pool = self.pos_dict.get(label, [])
                neg_pool = self.neg_dict.get(label, [])
                rnd_pool = self.rnd_dict.get(label, [])

                if len(pos_pool) < self.num_pos or len(neg_pool) < self.num_hard_negs or len(rnd_pool) < self.num_rnd_negs:
                    continue

                pos_idx = np.random.choice(len(pos_pool), size=self.num_pos, replace=(len(pos_pool) < self.num_pos))
                neg_idx = np.random.choice(len(neg_pool), size=self.num_hard_negs, replace=(len(neg_pool) < self.num_hard_negs))
                rnd_idx = np.random.choice(len(rnd_pool), size=self.num_rnd_negs, replace=(len(rnd_pool) < self.num_rnd_negs))

                for i in pos_idx:
                    features.append(tokenize_text(pos_pool[int(i)]))

                for i in neg_idx:
                    features.append(tokenize_text(neg_pool[int(i)]))

                for i in rnd_idx:
                    features.append(tokenize_text(rnd_pool[int(i)]))

            if len(features) == 0:
                raise ValueError(
                    "DataCollator produced an empty batch. "
                    "Most likely too many samples are being skipped due to insufficient pos/neg/rnd pools. "
                    "Try reducing --num_pos/--num_neg/--num_rnd or inspect dataset."
                )

            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            return batch

    # ----------------------------
    # Trainer
    # ----------------------------
    class CustomTrainer(transformers.Trainer):
        def __init__(self, num_pos, num_neg, num_rnd, batch_size, alpha, beta, detox, toxify, tau, smoothing, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_pos = num_pos
            self.num_neg = num_neg
            self.num_rnd = num_rnd
            self.batch_size = batch_size
            self.alpha = alpha
            self.beta = beta
            self.detox = detox
            self.toxify = toxify
            self.tau = tau
            self.smoothing = smoothing

        def compute_loss(self, model, inputs, return_outputs=False):
            proxy_inputs = inputs["input_ids"]
            proxy_masks = inputs["attention_mask"]
            proxy_labels = inputs["input_ids"]

            output = model(
                input_ids=proxy_inputs,
                attention_mask=proxy_masks,
                labels=proxy_labels,
                return_dict=True,
                loss_reduction="none",
            )

            loss = einops.rearrange(output.loss, "(B S) -> B S", B=inputs["input_ids"].shape[0])
            loss = torch.nan_to_num(loss, nan=5.0, posinf=5.0)

            loss_fct = CrossEntropyLoss(label_smoothing=float(args.smoothing))

            NUM_POS = int(args.num_pos)
            NUM_NEG = int(args.num_neg)
            NUM_RAND_NEG = int(args.num_rnd)
            NUM_ELEMENTS = NUM_POS + NUM_NEG + NUM_RAND_NEG
            micro_batch_size = int(self.batch_size)

            true_micro_batch_size = int(inputs["input_ids"].shape[0] / NUM_ELEMENTS)
            index_tensor = np.arange(true_micro_batch_size * NUM_ELEMENTS)

            firstLevel_pos_idx = index_tensor[seq_pos_idx].flatten()
            firstLevel_neg_idx = index_tensor[seq_neg_idx].flatten()
            firstLevel_rnd_idx = index_tensor[seq_rnd_idx].flatten()

            ref_pos_avg = torch.mean(torch.mean(loss[firstLevel_pos_idx], dim=1))

            pos_scores_ = -torch.abs(torch.mean(loss[firstLevel_pos_idx], dim=1) - ref_pos_avg) / 1.0
            neg_scores_ = -torch.abs(torch.mean(loss[firstLevel_neg_idx], dim=1) - ref_pos_avg) / 1.0
            rnd_scores_ = -torch.abs(torch.mean(loss[firstLevel_rnd_idx], dim=1) - ref_pos_avg) / 1.0

            neg_scores_ = einops.rearrange(neg_scores_, "(a b)-> a b", a=micro_batch_size)
            rnd_scores_ = einops.rearrange(rnd_scores_, "(a b)-> a b", a=micro_batch_size)
            pos_scores_ = einops.rearrange(pos_scores_, "(a b)-> a b", a=micro_batch_size)

            final_scores = torch.cat([torch.exp(pos_scores_), torch.exp(neg_scores_), torch.exp(rnd_scores_)], 1) / float(self.tau)
            label_probs_ = torch.cat(
                [torch.ones_like(pos_scores_), torch.zeros_like(neg_scores_), torch.zeros_like(rnd_scores_)], 1
            ).to(final_scores.device)

            weights = torch.cat(
                [torch.zeros_like(pos_scores_), float(self.alpha) * torch.ones_like(neg_scores_), torch.zeros_like(rnd_scores_)],
                1,
            ).to(final_scores.device)

            final_scores = final_scores + weights
            constrastive_loss_perperplexity = loss_fct(final_scores, label_probs_) / (NUM_POS + 1 + NUM_RAND_NEG)

            wandb.log({"train/perplexity_loss": constrastive_loss_perperplexity.item()})
            return constrastive_loss_perperplexity

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_ds,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=int(args.per_device_train_batch_size),
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            gradient_checkpointing=bool(args.use_gradient_checkpointing),
            warmup_steps=int(args.warmup_steps),
            warmup_ratio=float(args.warmup_ratio),
            num_train_epochs=int(args.num_train_epochs),
            learning_rate=float(args.learning_rate),
            fp16=bool(args.fp16),
            bf16=bool(args.bf16),
            remove_unused_columns=False,
            logging_steps=int(args.logging_steps),
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=int(args.save_steps),
            save_total_limit=6,
            dataloader_num_workers=int(args.num_workers),
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=False,
            run_name=wandb_run_name if use_wandb else None,
            dataloader_drop_last=True,
            output_dir=args.output_dir,
            optim=args.optim,
            push_to_hub=bool(args.push_to_hub),
            report_to="wandb",
        ),
        data_collator=MyDataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pos_dict=pos_prompts,
            neg_dict=neg_prompts,
            rnd_dict=rnd_neg_prompts,
            num_pos=int(args.num_pos),
            num_hard_negs=int(args.num_neg),
            num_rnd_negs=int(args.num_rnd),
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        num_pos=int(args.num_pos),
        num_neg=int(args.num_neg),
        num_rnd=int(args.num_rnd),
        batch_size=int(args.per_device_train_batch_size),
        alpha=float(args.alpha),
        beta=float(args.beta),
        detox=float(args.detox),
        toxify=float(args.toxify),
        tau=float(args.tau),
        smoothing=float(args.smoothing),
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer._signature_columns = ["input_ids", "attention_mask"]

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)