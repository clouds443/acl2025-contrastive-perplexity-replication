#
# SPDX-FileCopyrightText: 2025 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import itertools
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Compute perplexity of model on data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or huggingface model name")
    parser.add_argument("--data_dir", type=str, default="data/paraphrase/safeNLP_processed", help="Path to data directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (mostly for non-quantized runs)")
    parser.add_argument("--num_toxic_samples", type=int, default=4, help="Number of toxic samples to include")
    parser.add_argument("--num_neutral_samples", type=int, default=0, help="Number of neutral samples to include")
    parser.add_argument("--stride", type=int, default=512, help="Stride for perplexity calculation")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # 关键：强制限制窗口长度，避免 attention O(seq^2) 爆显存
    parser.add_argument("--max_length", type=int, default=2048, help="Max window length for sliding perplexity")
    return parser.parse_args()


def get_model_device(model):
    # 对 device_map="auto" 的模型，model.device 有时不可用，用第一个参数的 device 更稳
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def compute_ppl_for_text(model, tokenizer, text, max_length, stride):
    """
    对单条文本计算 token-level 加权的 perplexity（标准做法）。
    返回：(total_nll, total_tokens)
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids_all = enc.input_ids  # shape (1, seq)
    seq_len = input_ids_all.size(1)
    if seq_len < 2:
        return 0.0, 0

    device = get_model_device(model)

    total_nll = 0.0
    total_tokens = 0
    prev_end_loc = 0

    # sliding window over this sample only
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0:
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
            continue

        input_ids = input_ids_all[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # 只算最后 trg_len 个 token

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss  # mean over valid tokens in this window

        total_nll += float(loss.item()) * trg_len
        total_tokens += int(trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return total_nll, total_tokens


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model from {args.model_path}...")

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # 建议：统一用 device_map="auto"，更不容易 OOM（必要时会 CPU offload）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("Loading dataset...")
    groups = [
        "asian",
        "black",
        "chinese",
        "jewish",
        "latino",
        "lgbtq",
        "mental_disability",
        "mexican",
        "middle-east",
        "muslim",
        "native-american",
        "physical_disability",
        "women",
    ]
    data_files = [f"neutral_{group}.json" for group in groups]
    valid_files = [f for f in data_files if os.path.exists(os.path.join(args.data_dir, f))]

    if not valid_files:
        print(f"Error: No data files found in {args.data_dir}")
        return

    dataset = load_dataset("json", data_dir=args.data_dir, data_files={"train": valid_files})

    # build evaluation list
    list1d = []
    if args.num_toxic_samples > 0:
        print(f"Sampling Mixture - Toxic: {args.num_toxic_samples}, Neutral: {args.num_neutral_samples}")
        list2d = []
        for x in dataset["train"]:
            toxic_samples = []
            neutral_samples = []

            if "paraphrases_toxic" in x and len(x["paraphrases_toxic"]) > 0:
                toxic_samples = random.sample(
                    x["paraphrases_toxic"], min(args.num_toxic_samples, len(x["paraphrases_toxic"]))
                )

            if "paraphrases" in x and len(x["paraphrases"]) > 0 and args.num_neutral_samples > 0:
                neutral_samples = random.sample(
                    x["paraphrases"], min(args.num_neutral_samples, len(x["paraphrases"]))
                )

            list2d.append(toxic_samples + neutral_samples)

        list1d = list(itertools.chain(*list2d))
    else:
        print("Sampling Positive (Neutral) only")
        num_samples = max(4, args.num_neutral_samples)
        list2d = []
        for x in dataset["train"]:
            if "paraphrases" in x and len(x["paraphrases"]) > 0:
                list2d.append(random.sample(x["paraphrases"], min(num_samples, len(x["paraphrases"]))))

        list1d = list(itertools.chain(*list2d))

    if not list1d:
        print("No samples found to evaluate.")
        return

    print(f"Evaluating perplexity on {len(list1d)} samples... (per-sample, max_length={args.max_length}, stride={args.stride})")

    # token-weighted aggregation across samples
    total_nll = 0.0
    total_tokens = 0

    for text in tqdm(list1d, desc="Perplexity"):
        nll, ntok = compute_ppl_for_text(model, tokenizer, text, max_length=args.max_length, stride=args.stride)
        total_nll += nll
        total_tokens += ntok

    if total_tokens == 0:
        print("Could not compute perplexity (no valid tokens).")
        return

    avg_nll = total_nll / total_tokens
    ppl = float(np.exp(avg_nll))

    print(f"Model: {args.model_path}")
    print(f"Avg NLL: {avg_nll:.6f} over {total_tokens} tokens")
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)