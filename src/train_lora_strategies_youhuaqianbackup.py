"""
LoRA 目标模块策略对比实验
支持四种预设：minimal / standard / attention_mlp / all_linear
修复了全量数据训练的维度错误
"""
import os
import argparse
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

# 导入我们自己的 LoRA 实现
from lora_strategies import apply_lora_to_model, get_lora_state_dict


# -------------------- 预设策略映射 --------------------
PRESET_TARGETS = {
    "minimal": ["q_proj", "v_proj"],
    "standard": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attention_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "all_linear": ["all-linear"],   # 特殊标记
}


def get_target_modules_from_preset(preset: str):
    """根据预设名称返回目标模块列表"""
    if preset not in PRESET_TARGETS:
        raise ValueError(f"未知预设: {preset}，可选: {list(PRESET_TARGETS.keys())}")
    return PRESET_TARGETS[preset]


# -------------------- 数据处理 --------------------
def format_alpaca(example):
    """Alpaca 格式化为训练文本"""
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
    """
    Tokenize 并构建仅对 Response 部分计算 loss 的 labels。
    确保 input_ids 与 labels 长度完全一致，避免维度错误。
    """
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    all_labels = []
    for i, text in enumerate(texts):
        input_ids = tokenized["input_ids"][i]
        
        # 找到 "### Response:" 的位置
        response_start = text.find("### Response:")
        if response_start == -1:
            # 若找不到，对整个序列计算 loss
            all_labels.append(input_ids.copy())
            continue
        
        # 获取 prompt 部分的 token 数量（不添加特殊 token）
        prompt_text = text[:response_start + len("### Response:")]
        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        
        # 创建 labels：复制 input_ids，并将 prompt 部分设为 -100
        labels = input_ids.copy()
        # 截断保护：如果 prompt_len 超过 input_ids 长度，全部 mask
        if prompt_len >= len(labels):
            labels = [-100] * len(labels)
        else:
            labels[:prompt_len] = [-100] * prompt_len
        
        all_labels.append(labels)
    
    tokenized["labels"] = all_labels
    return tokenized


def collate_fn(batch, tokenizer):
    """动态 padding，自动适配 batch 内最长序列"""
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


# -------------------- 训练函数 --------------------
def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=4):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="LoRA 策略对比实验")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--max_samples", type=int, default=-1, help="使用的样本数，-1 表示全量")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./lora_strategy_output")
    parser.add_argument("--preset", type=str, required=True,
                        choices=["minimal", "standard", "attention_mlp", "all_linear"],
                        help="选择 LoRA 目标模块策略")
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 设备: {device}")

    # 加载模型
    print(f"📦 加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        device_map="auto" if device.type == "cuda" else None,
    )

    # 获取目标模块列表
    target_modules = get_target_modules_from_preset(args.preset)
    print(f"🎯 策略: {args.preset} -> 目标模块: {target_modules}")

    # 应用 LoRA
    model = apply_lora_to_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
    )
    model.to(device)

    # 加载数据
    print(f"📚 数据集: {args.dataset_name} (取前 {args.max_samples if args.max_samples > 0 else '全量'} 条)")
    dataset = load_dataset(args.dataset_name, split="train")
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    elif args.max_samples == -1:
        pass  # 使用全量数据

    dataset = dataset.map(format_alpaca, load_from_cache_file=False)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # 优化器
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * total_steps),
        num_training_steps=total_steps,
    )

    # 训练
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n🎯 开始训练...")
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device,
                               gradient_accumulation_steps=args.gradient_accumulation_steps)
        print(f"Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}")

        # 保存 checkpoint
        lora_state = get_lora_state_dict(model)
        ckpt_path = os.path.join(args.output_dir, f"lora_{args.preset}_epoch{epoch+1}.pt")
        torch.save(lora_state, ckpt_path)

    # 最终保存
    final_path = os.path.join(args.output_dir, f"lora_{args.preset}_final.pt")
    torch.save(get_lora_state_dict(model), final_path)
    print(f"\n✅ 训练完成！LoRA 权重已保存至: {final_path}")


if __name__ == "__main__":
    main()