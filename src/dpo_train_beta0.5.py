"""
DPO 训练 —— 使用 Anthropic HH-RLHF 数据集
支持命令行参数: --beta, --lr
示例:
    python dpo_train.py --beta 0.1 --lr 5e-5
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

from lora_strategies import apply_lora_to_model, load_lora_state_dict, get_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset

# -------------------- 辅助函数 --------------------
def extract_prompt_response(full_text: str):
    """从 HH-RLHF 的对话文本中切分出 prompt 和最后的 assistant 回复"""
    parts = full_text.rsplit("Assistant:", 1)
    if len(parts) == 2:
        prompt = parts[0].strip() + "\nAssistant:"
        response = parts[1].strip()
    else:
        prompt = ""
        response = full_text.strip()
    return prompt, response

def compute_policy_log_prob(model, tokenizer, prompt: str, response: str, device):
    """
    计算策略模型的对数概率，保留梯度（用于参数更新）
    返回标量张量，requires_grad=True
    """
    full_text = prompt + " " + response
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_tokens)

    outputs = model(**inputs)                # 正常前向，保留梯度
    logits = outputs.logits[0, prompt_len-1:-1, :]   # (response_len, vocab)
    target_ids = inputs["input_ids"][0, prompt_len:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[range(len(target_ids)), target_ids]
    return token_log_probs.sum()             # 标量，带梯度

def compute_ref_log_prob(model, tokenizer, prompt: str, response: str, device):
    """
    计算参考模型的对数概率，不保留梯度（冻结模型）
    返回 detach 后的标量张量
    """
    with torch.no_grad():
        full_text = prompt + " " + response
        inputs = tokenizer(full_text, return_tensors="pt").to(device)

        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)

        outputs = model(**inputs)
        logits = outputs.logits[0, prompt_len-1:-1, :]
        target_ids = inputs["input_ids"][0, prompt_len:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[range(len(target_ids)), target_ids]
    return token_log_probs.sum().detach()

# -------------------- DPO 损失 --------------------
def dpo_loss(policy_chosen_logp, policy_rejected_logp,
             ref_chosen_logp, ref_rejected_logp, beta=0.1):
    """
    policy_* 带梯度，ref_* 已 detach 无梯度
    """
    policy_log_ratio = policy_chosen_logp - policy_rejected_logp
    ref_log_ratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_log_ratio - ref_log_ratio)
    loss = -F.logsigmoid(logits).mean()
    return loss

# -------------------- 主函数 --------------------
def main():
    parser = argparse.ArgumentParser(description="DPO Training on HH-RLHF")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of training samples")
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()

    # ========== 基本配置 ==========
    model_name = "HuggingFaceTB/SmolLM-135M"
    lora_path = "./output_standard_full/lora_standard_final.pt"   # 你的 SFT LoRA 权重
    beta = args.beta
    learning_rate = args.lr
    num_epochs = args.max_epochs
    batch_size = 1
    gradient_accumulation_steps = 4
    output_dir = "./dpo_output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 DPO 训练配置: beta={beta}, lr={learning_rate}, samples={args.num_samples}")

    # ========== 加载分词器 ==========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ========== 1. 策略模型 (policy = SFT LoRA) ==========
    print("👨‍🏫 加载策略模型...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    target_modules = get_target_modules_from_preset("standard")
    policy_model = apply_lora_to_model(policy_model, r=8, lora_alpha=16, target_modules=target_modules)
    lora_state = torch.load(lora_path, map_location=device)
    load_lora_state_dict(policy_model, lora_state)
    policy_model.to(device)
    policy_model.train()
    print("✅ 策略模型加载完成")

    # ========== 2. 参考模型 (基座，冻结) ==========
    print("📚 加载参考模型（基座）...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    ref_model.to(device)
    ref_model.eval()
    # 确保参考模型所有参数都不会被更新
    for param in ref_model.parameters():
        param.requires_grad = False
    print("✅ 参考模型加载完成")

    # ========== 3. 数据 ==========
    print(f"📦 加载 HH-RLHF 数据集 (前 {args.num_samples} 条)...")
    dataset = load_dataset("Anthropic/hh-rlhf", split=f"train[:{args.num_samples}]")
    data = []
    for sample in dataset:
        chosen_text = sample["chosen"]
        rejected_text = sample["rejected"]
        prompt_c, resp_c = extract_prompt_response(chosen_text)
        prompt_r, resp_r = extract_prompt_response(rejected_text)
        prompt = prompt_c if len(prompt_c) > len(prompt_r) else prompt_r
        if len(prompt) < 5:
            continue
        data.append({"prompt": prompt, "chosen": resp_c, "rejected": resp_r})
    print(f"✅ 解析到 {len(data)} 条有效偏好对")

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # ========== 4. 优化器（仅策略模型可训练参数） ==========
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ========== 5. 训练循环 ==========
    print("\n🔥 开始 DPO 训练...")
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(progress_bar):
            prompt = batch["prompt"][0]
            chosen = batch["chosen"][0]
            rejected = batch["rejected"][0]

            # 策略模型（保留梯度）
            policy_chosen_logp = compute_policy_log_prob(policy_model, tokenizer, prompt, chosen, device)
            policy_rejected_logp = compute_policy_log_prob(policy_model, tokenizer, prompt, rejected, device)

            # 参考模型（无梯度）
            ref_chosen_logp = compute_ref_log_prob(ref_model, tokenizer, prompt, chosen, device)
            ref_rejected_logp = compute_ref_log_prob(ref_model, tokenizer, prompt, rejected, device)

            loss = dpo_loss(policy_chosen_logp, policy_rejected_logp,
                            ref_chosen_logp, ref_rejected_logp, beta=beta)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均 DPO Loss: {avg_loss:.4f}")

        # 保存当前 epoch 的 LoRA 权重
        os.makedirs(output_dir, exist_ok=True)
        torch.save(get_lora_state_dict(policy_model),
                   os.path.join(output_dir, f"dpo_beta{beta}_lr{learning_rate}_epoch{epoch+1}.pt"))

    print(f"\n✅ DPO 训练完成！模型保存在 {output_dir}/")

if __name__ == "__main__":
    main()