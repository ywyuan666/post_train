"""
GRPO 评估 —— 可调采样温度（3.3）
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset

# ====== 可在这里修改温度、top_p ======
TEMPERATURE = 1.0
TOP_P = 0.9
# ====================================

def rule_reward(generated: str, ground_truth: str) -> float:
    return 1.0 if ground_truth in generated else 0.0

def grpo_advantage(rewards):
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    return (rewards - mean) / std

def generate_with_model(model, tokenizer, prompt, device, max_new_tokens=32):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("### Response:")[-1].strip()

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

    prompt = "### Instruction:\nWhat is 2 + 2?\n\n### Response:\n"
    ground_truth = "4"

    print(f"\n📝 Prompt: {prompt}")
    print(f"🎯 Ground Truth: {ground_truth}")
    print(f"🔧 采样参数: temperature={TEMPERATURE}, top_p={TOP_P}\n")

    responses, rewards = [], []
    for i in range(4):
        resp = generate_with_model(model, tokenizer, prompt, device)
        r = rule_reward(resp, ground_truth)
        responses.append(resp)
        rewards.append(r)
        print(f"  回复{i+1}: {resp[:80]}... → 奖励: {r}")

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    advantages = grpo_advantage(rewards_t)

    print("\n📊 GRPO 优势计算:")
    print(f"Rewards:    {rewards_t.tolist()}")
    print(f"Mean:       {rewards_t.mean().item():.4f}")
    print(f"Std:        {rewards_t.std().item():.4f}")
    print(f"Advantages: {advantages.tolist()}")

if __name__ == "__main__":
    main()