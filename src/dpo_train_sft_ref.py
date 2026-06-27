"""
DPO 训练 —— 使用 SFT 模型（标准 LoRA）作为参考
数据集: Anthropic HH-RLHF
策略模型: 标准 LoRA（初始与参考相同）
参考模型: 标准 LoRA（冻结，不更新）
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
    """计算策略模型的对数概率（保留梯度）"""
    full_text = prompt + " " + response
    inputs = tokenizer(full_text, return_tensors="pt").to(device)

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_tokens)

    outputs = model(**inputs)
    logits = outputs.logits[0, prompt_len-1:-1, :]  # (response_len, vocab)
    target_ids = inputs["input_ids"][0, prompt_len:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[range(len(target_ids)), target_ids]
    return token_log_probs.sum()

def compute_ref_log_prob(model, tokenizer, prompt: str, response: str, device):
    """计算参考模型的对数概率（无梯度）"""
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
    policy_log_ratio = policy_chosen_logp - policy_rejected_logp
    ref_log_ratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_log_ratio - ref_log_ratio)
    loss = -F.logsigmoid(logits).mean()
    return loss

# -------------------- 训练主函数 --------------------
def main():
    model_name = "HuggingFaceTB/SmolLM-135M"
    # 标准 LoRA 权重路径（SFT 模型，同时用于策略和参考初始化）
    lora_path = "./output_standard_full/lora_standard_final.pt"
    beta = 0.1
    learning_rate = 5e-5
    num_epochs = 1
    batch_size = 1
    gradient_accumulation_steps = 4
    output_dir = "./dpo_sft_ref_output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 开始 DPO 训练准备（参考模型 = SFT LoRA）...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 加载基座模型，然后为策略模型注入 LoRA 并加载 SFT 权重
    print("👨‍🏫 加载策略模型（SFT LoRA）...")
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

    # 加载基座模型，再为参考模型注入 LoRA 并加载相同的 SFT 权重，然后冻结
    print("📚 加载参考模型（SFT LoRA，冻结）...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    ref_model = apply_lora_to_model(ref_model, r=8, lora_alpha=16, target_modules=target_modules)
    # 加载相同的 SFT 权重
    ref_lora_state = torch.load(lora_path, map_location=device)
    load_lora_state_dict(ref_model, ref_lora_state)
    ref_model.to(device)
    ref_model.eval()
    # 确保参考模型的所有参数都不参与梯度计算
    for param in ref_model.parameters():
        param.requires_grad = False
    print("✅ 参考模型加载完成")

    # 数据集
    print("📦 加载 HH-RLHF 数据集...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train[:200]")
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
    print(f"✅ 解析到 {len(data)} 条偏好对")

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # 优化器（只优化策略模型的 LoRA 参数）
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    print("\n🔥 开始 DPO 训练...")
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(progress_bar):
            prompt = batch["prompt"][0]
            chosen = batch["chosen"][0]
            rejected = batch["rejected"][0]

            # 策略模型对数概率（需要梯度）
            policy_chosen_logp = compute_policy_log_prob(policy_model, tokenizer, prompt, chosen, device)
            policy_rejected_logp = compute_policy_log_prob(policy_model, tokenizer, prompt, rejected, device)

            # 参考模型对数概率（无梯度）
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

        os.makedirs(output_dir, exist_ok=True)
        torch.save(get_lora_state_dict(policy_model), f"{output_dir}/dpo_epoch{epoch+1}.pt")

    print(f"\n✅ DPO 训练完成！模型保存在 {output_dir}/")

if __name__ == "__main__":
    main()