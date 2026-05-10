#
# SPDX-FileCopyrightText: 2025 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#

import warnings
warnings.filterwarnings("ignore")

import argparse
import glob
import json
import os
import numpy as np
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

try:
    import torchvision
    torchvision.disable_beta_transforms_warning()
except:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Generate paraphrases using LLM")
    parser.add_argument("--input_dir", type=str, default="data/eval/toxigen", help="Input directory containing text files")
    parser.add_argument("--output_dir", type=str, default="data/paraphrase/toxigen", help="Output directory for JSON files")
    parser.add_argument("--model_name", type=str, default="TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ", help="Model name or path")
    parser.add_argument("--model_basename", type=str, default="model", help="Model basename for GPTQ")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--num_prompts", type=int, default=500, help="Number of prompts to process per group")
    parser.add_argument("--num_paraphrases", type=int, default=10, help="Number of paraphrases to generate per prompt")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    return parser.parse_args()

def jdump(obj, f_path, indent=4, default=str):
    dir_name = os.path.dirname(f_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)
    with open(f_path, "w") as f:
        json.dump(obj, f, indent=indent, default=default)

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        legacy=False
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoGPTQForCausalLM.from_quantized(
        args.model_name,
        model_basename=args.model_basename,
        use_safetensors=True,
        trust_remote_code=False,
        device=args.device,
        use_triton=False,
        quantize_config=None,
        disable_exllamav2=True,
    )
    model.eval()
    return model, tokenizer

def generate_paraphrases(args, model, tokenizer):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    prompt_files = glob.glob(os.path.join(args.input_dir, "neutral*.txt"))

    # 🔥 开启 GPU 加速
    torch.backends.cudnn.benchmark = True

    for task_file in tqdm(prompt_files, desc="Processing files"):
        with open(task_file, "r") as f:
            group_prompts = [line.strip() for line in f][:args.num_prompts]

        all_data = []
        group_name = os.path.basename(task_file).split(".")[0]

        for prompt in tqdm(group_prompts, desc="Prompts", leave=True):
            item = {"input": prompt}

            # 生成正常 + 有毒
            for j in range(2):
                paraphrases = []
                sents = [prompt.strip()]
                postfix = "" if j == 0 else "_toxic"

                # 🔥 批量生成，大幅提速
                for i in range(0, args.num_paraphrases, args.batch_size):
                    current_batch_size = min(args.batch_size, args.num_paraphrases - i)
                    prompt_templates = []

                    for _ in range(current_batch_size):
                        if j == 0:
                            prompt_template = f'''### HUMAN: Paraphrase the following sentence:
"{prompt}"

### RESPONSE:
'''
                        else:
                            prompt_template = f'''### HUMAN: Paraphrase the following sentence in a very toxic way:
"{prompt}"

### RESPONSE:
'''
                        prompt_templates.append(prompt_template)

                    inputs = tokenizer(
                        prompt_templates,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256
                    ).to(args.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=64,        # 🔥 缩短生成长度
                            do_sample=True,
                            top_k=20,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    for res in responses:
                        if "### RESPONSE:" in res:
                            filtered = res.split("### RESPONSE:")[-1].strip().lower()
                        else:
                            filtered = res.strip().lower()
                        paraphrases.append(filtered)

                # ROUGE 排序
                all_para_tokens = [scorer._tokenizer.tokenize(p) for p in paraphrases]
                ref_tokens = scorer._tokenizer.tokenize(prompt)
                scores = [
                    rouge_scorer._score_lcs(ref_tokens, pt).fmeasure
                    for pt in all_para_tokens
                ]

                top_paraphrases = {
                    paraphrases[i]: scores[i]
                    for i in np.argsort(scores)[-10:][::-1]
                }
                item["paraphrases" + postfix] = top_paraphrases

            all_data.append(item)

        out_file = os.path.join(args.output_dir, f"{group_name}.json")
        jdump(all_data, out_file)

if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = load_model(args)
    generate_paraphrases(args, model, tokenizer)