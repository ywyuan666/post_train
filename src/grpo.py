"""
GRPO (Group Relative Policy Optimization) 规则奖励与优势计算验证
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset


def rule_reward(generated_text: str, ground_truth: str) -> float:
    """
    简单规则奖励：生成文本包含正确答案即给 1.0，否则 0.0
    实际可扩展为更复杂的规则（如数学题解析）
    """
    # 移除空格和标点，简单子串匹配
    gen_clean = generated_text.lower().strip()
    gt_clean = ground_truth.lower().strip()
    return 1.0 if gt_clean in gen_clean else 0.0


def grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    GRPO 优势计算：组内标准化 (reward - mean) / std
    """
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    return (rewards - mean) / std


def generate_with_model(model, tokenizer, prompt, device, max_new_tokens=32):
    """生成回复文本"""
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
    # 提取 Response 部分
    response = full_text.split("### Response:")[-1].strip()
    return response


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "HuggingFaceTB/SmolLM-135M"
    
    print("📦 加载微调模型（作为策略模型）...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    target_modules = get_target_modules_from_preset("standard")
    model = apply_lora_to_model(model, r=8, lora_alpha=16, target_modules=target_modules)
    lora_state = torch.load("./output_standard/lora_standard_final.pt", map_location=device)
    load_lora_state_dict(model, lora_state)
    
    # ✅ 关键修复：将模型整体显式移动到目标设备（确保 LoRA 参数也在 GPU）
    model = model.to(device)
    model.eval()
    
    # 构造一个数学问题（便于规则奖励判断）
    prompt = "### Instruction:\nWhat is 2 + 2?\n\n### Response:\n"
    ground_truth = "4"
    
    print(f"\n📝 Prompt: {prompt}")
    print(f"🎯 Ground Truth: {ground_truth}\n")
    
    # 模拟 Group Sampling (G=4)
    group_size = 4
    responses = []
    rewards = []
    
    print("🔁 采样 4 个回复并计算奖励...")
    for i in range(group_size):
        resp = generate_with_model(model, tokenizer, prompt, device)
        r = rule_reward(resp, ground_truth)
        responses.append(resp)
        rewards.append(r)
        print(f"  [{i+1}] 回复: {resp[:60]}... → 奖励: {r}")
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    advantages = grpo_advantage(rewards_tensor)
    
    print("\n📊 GRPO 优势计算:")
    print(f"Rewards:    {rewards_tensor.tolist()}")
    print(f"Mean:       {rewards_tensor.mean().item():.4f}")
    print(f"Std:        {rewards_tensor.std().item():.4f}")
    print(f"Advantages: {advantages.tolist()}")
    
    # ✅ 修改后的验证逻辑（更友好的提示）
    print("\n✅ 验证：")
    if rewards_tensor.std().item() == 0:
        print("  组内所有回复奖励相同，优势全为 0，无法区分优劣。")
    else:
        for i, (r, adv) in enumerate(zip(rewards, advantages)):
            if r == 1.0 and adv > 0:
                print(f"  回复 {i+1} 正确且优势为正 ✓")
            elif r == 0.0 and adv < 0:
                print(f"  回复 {i+1} 错误且优势为负 ✓")
            else:
                print(f"  回复 {i+1} 状态：奖励={r:.0f}，优势={adv:.3f}")


if __name__ == "__main__":
    main()