from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

# 原始基座模型
BASE_MODEL = "mistralai/Mistral-7B-v0.1"
# 你的 LoRA 路径（优先尝试 checkpoint 里的）
LORA_PATH = "./autodl-tmp/results/youthful-snow-16/checkpoint-1000"
# 合并后模型保存的文件夹
SAVE_PATH = "./autodl-tmp/mistral-detox-merged"

# 先检查路径是否存在
if not os.path.exists(LORA_PATH):
    print(f"路径不存在：{LORA_PATH}")
    # 再试外面的路径
    LORA_PATH = "./autodl-tmp/results/youthful-snow-16"
    print(f"尝试外层路径：{LORA_PATH}")
    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError(f"LoRA 路径不存在，请检查：{LORA_PATH}")

# 检查 adapter_config.json 是否存在
config_path = os.path.join(LORA_PATH, "adapter_config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"找不到 adapter_config.json，请检查路径：{config_path}")

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=False  # 关键修复，解决 Mistral tokenizer 报错
)

print(f"Loading LoRA from: {LORA_PATH}")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("Merging LoRA into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {SAVE_PATH}...")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("✅ Done! Merged model is ready.")