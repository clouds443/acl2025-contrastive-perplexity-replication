#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
from typing import Any, Dict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser("Generate samples from RealToxicityPrompts (RTP).")
    p.add_argument("--model_path", required=True, type=str)
    p.add_argument("--out_jsonl", required=True, type=str)

    # dataset
    p.add_argument("--dataset_name", type=str, default="allenai/real-toxicity-prompts")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")

    # generation
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--do_sample", action="store_true", help="Sampling; if false, greedy")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50, help="0 disables top-k")
    p.add_argument("--typical_p", type=float, default=None, help="e.g. 0.95; None disables")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--min_new_tokens", type=int, default=1)
    p.add_argument("--max_chars", type=int, default=200)

    # anti-degeneration (important)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)

    # stop if model starts spamming punctuation
    p.add_argument("--stop_on_punct_run", action="store_true", help="Trim output when repeated punctuation is detected")
    p.add_argument("--punct_run_len", type=int, default=20, help="How many repeated punct chars triggers trimming")

    return p.parse_args()


def extract_prompt(row: Dict[str, Any]) -> str:
    if "prompt" in row:
        pr = row["prompt"]
        if isinstance(pr, dict) and "text" in pr:
            return str(pr["text"])
        return str(pr)
    if "text" in row:
        return str(row["text"])
    raise KeyError(f"Cannot find prompt field. Available keys: {list(row.keys())}")


_PUNCT_RUN_RE_TEMPLATE = r"([!?.])\1{{{n},}}"


def trim_on_punct_run(text: str, run_len: int) -> str:
    """
    If there is a run of repeated punctuation like !!!!!!!!!!, cut at the start of that run.
    """
    pat = re.compile(_PUNCT_RUN_RE_TEMPLATE.format(n=run_len))
    m = pat.search(text)
    if not m:
        return text
    return text[: m.start()].rstrip()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading dataset {args.dataset_name} split={args.split} ...")
    ds = load_dataset(args.dataset_name, split=args.split)

    idxs = list(range(len(ds)))
    if args.shuffle:
        random.shuffle(idxs)
    idxs = idxs[: args.num_samples]

    quant = None
    if args.use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

    print(f"Loading model from {args.model_path} ...")
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        quantization_config=quant,
        trust_remote_code=True,
    )
    model.eval()

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    device = next(model.parameters()).device

    print(f"Generating {len(idxs)} samples...")
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for n, i in enumerate(idxs):
            row = ds[i]
            prompt = extract_prompt(row).strip()

            inputs = tok(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs = dict(
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            if args.top_k and args.top_k > 0:
                gen_kwargs["top_k"] = args.top_k
            if args.typical_p is not None:
                gen_kwargs["typical_p"] = args.typical_p

            with torch.no_grad():
                gen_ids = model.generate(**inputs, **gen_kwargs)

            full_text = tok.decode(gen_ids[0], skip_special_tokens=True)

            continuation = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            if args.stop_on_punct_run:
                continuation = trim_on_punct_run(continuation, run_len=args.punct_run_len)

            continuation_200 = continuation[: args.max_chars]

            out = {
                "dataset": args.dataset_name,
                "split": args.split,
                "index": int(i),
                "prompt": prompt,
                "generation": continuation_200,
                "generation_full": continuation,
                "gen_params": {
                    "do_sample": bool(args.do_sample),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "top_k": int(args.top_k),
                    "typical_p": None if args.typical_p is None else float(args.typical_p),
                    "max_new_tokens": int(args.max_new_tokens),
                    "max_chars": int(args.max_chars),
                    "seed": int(args.seed),
                    "use_4bit": bool(args.use_4bit),
                    "repetition_penalty": float(args.repetition_penalty),
                    "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
                    "stop_on_punct_run": bool(args.stop_on_punct_run),
                    "punct_run_len": int(args.punct_run_len),
                },
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

            if (n + 1) % 10 == 0:
                print(f"  {n+1}/{len(idxs)} done")

    print(f"Saved generations to {args.out_jsonl}")


if __name__ == "__main__":
    main()