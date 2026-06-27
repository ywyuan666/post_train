"""
GRPO 评估 —— 使用 GSM8K 数学题（3.1）
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset
import re

def rule_reward(generated: str, ground_truth: str) -> float:
    gen_clean = generated.strip().replace(",", "")
    gt_clean = ground_truth.strip().replace(",", "")
    return 1.0 if gt_clean in gen_clean else 0.0

def grpo_advantage(rewards):
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    return (rewards - mean) / std

def generate_with_model(model, tokenizer, prompt, device, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("A:")[-1].strip()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "HuggingFaceTB/SmolLM-135M"
    lora_path = "./output_standard_full/lora_standard_final.pt"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    target = get_target_modules_from_preset("standard")
    model = apply_lora_to_model(model, r=8, lora_alpha=16, target_modules=target)
    lora_state = torch.load(lora_path, map_location=device)
    load_lora_state_dict(model, lora_state)
    model.to(device)
    model.eval()

    # 加载 GSM8K 前 5 题
    dataset = load_dataset("gsm8k", "main", split="train[:5]")
    for idx, sample in enumerate(dataset):
        question = sample["question"]
        answer_raw = sample["answer"]
        gt = answer_raw.split("####")[-1].strip()
        prompt = f"Q: {question}\nA:"

        print(f"\n📝 问题 {idx+1}: {question}")
        print(f"🎯 正确答案: {gt}")

        responses, rewards = [], []
        for i in range(4):
            resp = generate_with_model(model, tokenizer, prompt, device)
            r = rule_reward(resp, gt)
            responses.append(resp)
            rewards.append(r)
            print(f"  回复{i+1}: {resp[:80]}... → 奖励: {r}")

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        advantages = grpo_advantage(rewards_t)
        print("优势:", advantages.tolist())

if __name__ == "__main__":
    main()