"""
LoRA 目标模块策略对比实验 - 全量数据稳定版（新版 AMP + lora_dropout）
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

from lora_strategies import apply_lora_to_model, get_lora_state_dict

PRESET_TARGETS = {
    "minimal": ["q_proj", "v_proj"],
    "standard": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attention_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "all_linear": ["all-linear"],
}

def get_target_modules_from_preset(preset: str):
    if preset not in PRESET_TARGETS:
        raise ValueError(f"未知预设: {preset}")
    return PRESET_TARGETS[preset]

def format_alpaca(example):
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
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    labels = []
    for i, text in enumerate(texts):
        response_start = text.find("### Response:")
        if response_start == -1:
            labels.append(tokenized["input_ids"][i].copy())
            continue
        prompt_text = text[:response_start + len("### Response:")]
        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        label = tokenized["input_ids"][i].copy()
        label[:prompt_len] = [-100] * prompt_len
        labels.append(label)
    tokenized["labels"] = labels
    return tokenized

def collate_fn(batch, tokenizer):
    valid_batch = []
    for item in batch:
        if not all(l == -100 for l in item["labels"]):
            valid_batch.append(item)
    if len(valid_batch) == 0:
        return None

    max_len = max(len(item["input_ids"]) for item in valid_batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in valid_batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append([tokenizer.pad_token_id] * pad_len + item["input_ids"])
        attention_mask.append([0] * pad_len + item["attention_mask"])
        raw_label = item["labels"]
        if len(raw_label) != len(item["input_ids"]):
            if len(raw_label) < len(item["input_ids"]):
                raw_label = [-100] * (len(item["input_ids"]) - len(raw_label)) + raw_label
            else:
                raw_label = raw_label[:len(item["input_ids"])]
        label_padded = [-100] * pad_len + raw_label
        labels.append(label_padded)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, gradient_accumulation_steps=4):
    model.train()
    total_loss = 0
    valid_steps = 0
    progress_bar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        if batch is None:
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 新版 AMP 写法
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        loss_val = loss.item() * gradient_accumulation_steps
        if not torch.isnan(torch.tensor(loss_val)):
            total_loss += loss_val
            valid_steps += 1
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
        else:
            progress_bar.set_postfix({"loss": "NaN(skipped)"})

    avg_loss = total_loss / valid_steps if valid_steps > 0 else float('nan')
    return avg_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA input dropout rate")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./lora_strategy_output")
    parser.add_argument("--preset", type=str, required=True,
                        choices=["minimal", "standard", "attention_mlp", "all_linear"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    target_modules = get_target_modules_from_preset(args.preset)
    print(f"🎯 策略: {args.preset} -> 目标模块: {target_modules}")

    model = apply_lora_to_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,   # 新增 dropout 参数
        freeze_non_lora=True,
    )
    model.to(device)

    print(f"📚 数据集: {args.dataset_name} (取前 {args.max_samples} 条)")
    dataset = load_dataset(args.dataset_name, split="train")
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    dataset = dataset.map(format_alpaca)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.filter(lambda x: not all(l == -100 for l in x["labels"]))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler('cuda')  # 新版 API

    os.makedirs(args.output_dir, exist_ok=True)
    print("\n🎯 开始训练（全量数据，新版 AMP）...")
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, scaler, device,
                               gradient_accumulation_steps=args.gradient_accumulation_steps)
        print(f"Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}")

        lora_state = get_lora_state_dict(model)
        ckpt_path = os.path.join(args.output_dir, f"lora_{args.preset}_epoch{epoch+1}.pt")
        torch.save(lora_state, ckpt_path)

    final_path = os.path.join(args.output_dir, f"lora_{args.preset}_final.pt")
    torch.save(get_lora_state_dict(model), final_path)
    print(f"\n✅ 训练完成！LoRA 权重已保存至: {final_path}")

if __name__ == "__main__":
    main()