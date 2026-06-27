"""
GRPO 训练 + 评估 —— 使用 GSM8K 数学题 + 细粒度奖励 + 可调采样参数
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import argparse
import os
import re

from lora_strategies import apply_lora_to_model, load_lora_state_dict, get_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset

# -------------------- 细粒度奖励函数 (3.2) --------------------
def rule_reward(generated: str, ground_truth: str) -> float:
    """
    奖励设计：
    - 答案正确：+1.0
    - 包含推理步骤关键词 (step/first/next/therefore)：+0.2
    - 包含 #### 格式标记：+0.1
    """
    reward = 0.0
    # 提取生成文本中的最后数字作为答案（简单做法）
    # 同时检测 ground_truth（去除空格）是否出现在生成文本中
    gen_clean = generated.strip()
    gt_clean = ground_truth.strip().replace(",", "")
    if gt_clean in gen_clean.replace(",", ""):
        reward += 1.0

    # 步骤关键词
    step_keywords = ["step", "first", "second", "next", "then", "therefore", "so"]
    if any(kw in generated.lower() for kw in step_keywords):
        reward += 0.2

    # 格式标记（GSM8K 的 #### 分隔符）
    if "####" in generated:
        reward += 0.1

    return reward

def extract_answer_from_generated(text: str) -> str:
    """从生成文本中提取最后一个数值答案（辅助观察，实际奖励不依赖）"""
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return numbers[-1] if numbers else ""

# -------------------- GRPO 优势计算 --------------------
def grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """组内标准化"""
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    return (rewards - mean) / std

# -------------------- 生成函数 (3.3 可调采样参数) --------------------
def generate_with_model(model, tokenizer, prompt, device, max_new_tokens=128, temperature=1.0, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 Assistant 部分（GSM8K 格式为 "Q: ... A:"，我们构造 prompt 时加了 "A:"）
    response = full_text.split("A:")[-1].strip()
    return response

# -------------------- GRPO 训练一步 --------------------
def grpo_update(model, tokenizer, prompts, ground_truths, device,
                group_size=4, temperature=1.0, top_p=0.9,
                clip_eps=0.2, beta=0.1, lr=5e-5):
    """
    对一批提示词执行一次 GRPO 更新。
    返回平均损失和平均奖励。
    """
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    optimizer.zero_grad()

    total_loss = 0.0
    total_reward = 0.0
    count = 0

    for prompt, gt in zip(prompts, ground_truths):
        # 采样多个回复
        responses = []
        rewards = []
        for _ in range(group_size):
            resp = generate_with_model(model, tokenizer, prompt, device,
                                       temperature=temperature, top_p=top_p)
            r = rule_reward(resp, gt)
            responses.append(resp)
            rewards.append(r)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        advantages = grpo_advantage(rewards_t)

        # 计算每个回复的对数概率 (old log prob, 用于 PPO clip)
        for i, resp in enumerate(responses):
            full_text = prompt + " " + resp
            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)

            outputs = model(**inputs)
            logits = outputs.logits[0, prompt_len-1:-1, :]
            target_ids = inputs["input_ids"][0, prompt_len:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_logp = log_probs[range(len(target_ids)), target_ids]
            total_logp = token_logp.sum()

            # 简单 PPO 损失：-advantage * log_prob (忽略 ratio clipping 以简化)
            loss = -advantages[i].detach() * total_logp
            loss.backward()
            total_loss += loss.item()
            total_reward += rewards[i]
            count += 1

    # 梯度更新
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss / count if count > 0 else 0, total_reward / count if count > 0 else 0

# -------------------- 主函数 --------------------
def main():
    parser = argparse.ArgumentParser(description="GRPO with GSM8K")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="nucleus 采样 top_p")
    parser.add_argument("--num_samples", type=int, default=20, help="使用的 GSM8K 问题数量")
    parser.add_argument("--group_size", type=int, default=4, help="每个问题的采样数量")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="生成最大 token 数")
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "HuggingFaceTB/SmolLM-135M"
    lora_path = "./output_standard_full/lora_standard_final.pt"

    # ========== 加载分词器 ==========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ========== 加载策略模型 (standard LoRA) ==========
    print("👨‍🏫 加载策略模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    target_modules = get_target_modules_from_preset("standard")
    model = apply_lora_to_model(model, r=8, lora_alpha=16, target_modules=target_modules)
    lora_state = torch.load(lora_path, map_location=device)
    load_lora_state_dict(model, lora_state)
    model.to(device)
    print("✅ 策略模型加载完成")

    # ========== 加载 GSM8K 数据集 ==========
    print(f"📦 加载 GSM8K 数据集 (前 {args.num_samples} 条)...")
    dataset = load_dataset("gsm8k", "main", split=f"train[:{args.num_samples}]")
    prompts = []
    answers = []
    for sample in dataset:
        question = sample["question"]
        answer_raw = sample["answer"]
        # 提取最终答案 (#### 后面的数字)
        final_answer = answer_raw.split("####")[-1].strip()
        prompt = f"Q: {question}\nA:"
        prompts.append(prompt)
        answers.append(final_answer)
    print(f"✅ 加载 {len(prompts)} 个问题")

    # ========== 执行一次 GRPO 更新 ==========
    print("\n🔥 开始 GRPO 训练（单步演示，可自行增加 epoch）")
    avg_loss, avg_reward = grpo_update(
        model, tokenizer, prompts, answers, device,
        group_size=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        lr=args.lr,
    )
    print(f"平均损失: {avg_loss:.4f}, 平均奖励: {avg_reward:.4f}")

    # ========== 保存更新后的 LoRA ==========
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"grpo_updated_lr{args.lr}_temp{args.temperature}.pt")
    torch.save(get_lora_state_dict(model), output_path)
    print(f"\n✅ 更新后的 LoRA 已保存至: {output_path}")

if __name__ == "__main__":
    main()