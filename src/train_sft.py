"""
监督微调（SFT）训练脚本
使用 Alpaca-GPT4 数据集，手动 LoRA 实现
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import argparse

# 导入我们的 LoRA 实现
from lora import apply_lora_to_model, get_lora_state_dict


def format_alpaca(example):
    """将 Alpaca 格式转换为训练文本"""
    if example.get("input", "").strip():
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
    return {"text": text}


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize 并构建 labels（仅对 Response 部分计算 loss）"""
    texts = examples["text"]
    
    # Tokenize 全部文本
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    labels = []
    for i, text in enumerate(texts):
        # 找到 "### Response:" 的位置
        response_start = text.find("### Response:")
        if response_start == -1:
            # 如果找不到，对整个序列计算 loss
            labels.append(tokenized["input_ids"][i].copy())
            continue
        
        # Tokenize prompt 部分（Response 之前的内容）
        prompt_text = text[:response_start + len("### Response:")]
        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        
        # 创建 labels：prompt 部分为 -100，response 部分为真实 token
        label = tokenized["input_ids"][i].copy()
        label[:prompt_len] = [-100] * prompt_len
        labels.append(label)
    
    tokenized["labels"] = labels
    return tokenized


def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=4):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        # 移动到设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})
    
    return total_loss / len(dataloader)


def generate_response(model, tokenizer, instruction, input_text="", max_new_tokens=128):
    """生成回复（用于验证）"""
    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取 Response 部分
    response_part = response.split("### Response:")[-1].strip()
    return response_part


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./lora_output")
    args = parser.parse_args()
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    
    # 加载模型和分词器
    print(f"📦 加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        device_map="auto" if device.type == "cuda" else None,
    )
    
    # 应用 LoRA
    print("🔧 应用 LoRA 适配器...")
    model = apply_lora_to_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # 课件推荐最小集
    )
    model.to(device)
    
    # 加载数据集
    print(f"📚 加载数据集: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    
    # 取子集（控制训练时间）
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"📊 数据集大小: {len(dataset)} 条")
    
    # 格式化数据
    dataset = dataset.map(format_alpaca)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # DataLoader
    def collate_fn(batch):
        # Padding
        max_len = max(len(item["input_ids"]) for item in batch)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # 优化器和调度器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * total_steps),
        num_training_steps=total_steps,
    )
    
    # 训练循环
    print("\n🎯 开始训练...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler, device,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        print(f"Epoch {epoch + 1} 平均 Loss: {avg_loss:.4f}")
        
        # 保存 checkpoint
        lora_state = get_lora_state_dict(model)
        torch.save(lora_state, os.path.join(args.output_dir, f"lora_epoch_{epoch+1}.pt"))
    
    # 最终保存
    lora_state = get_lora_state_dict(model)
    torch.save(lora_state, os.path.join(args.output_dir, "lora_final.pt"))
    print(f"\n✅ 训练完成！LoRA 权重已保存到 {args.output_dir}")
    
    # 测试生成
    print("\n🧪 测试生成...")
    test_instructions = [
        ("Write a short poem about AI.", ""),
        ("Explain what is machine learning in simple terms.", ""),
    ]
    
    for instruction, input_text in test_instructions:
        print(f"\n📝 Instruction: {instruction}")
        response = generate_response(model, tokenizer, instruction, input_text)
        print(f"🤖 Response: {response}")


if __name__ == "__main__":
    main()