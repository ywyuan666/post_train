"""
LoRA 目标模块策略对比实验 - 全量数据稳定版 (新版 AMP + lora_dropout + 验证集与早停)
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

@torch.no_grad()
def evaluate_epoch(model, dataloader, device):
    """计算验证集上的平均损失"""
    model.eval()
    total_loss = 0
    valid_steps = 0
    for batch in tqdm(dataloader, desc="Validation"):
        if batch is None:
            continue
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        loss_val = loss.item()
        if not torch.isnan(torch.tensor(loss_val)):
            total_loss += loss_val
            valid_steps += 1

    return total_loss / valid_steps if valid_steps > 0 else float('nan')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=10, help="最大训练 epoch 数（早停可能提前停止）")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA input dropout rate")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./lora_strategy_output")
    parser.add_argument("--preset", type=str, required=True,
                        choices=["minimal", "standard", "attention_mlp", "all_linear"])
    # 早停相关参数
    parser.add_argument("--val_split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--patience", type=int, default=2, help="早停耐心值（连续多少个 epoch 不下降则停止）")
    parser.add_argument("--enable_early_stop", action="store_true", help="启用早停")
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
        lora_dropout=args.lora_dropout,
        freeze_non_lora=True,
    )
    model.to(device)

    # ---------- 数据集划分 ----------
    print(f"📚 加载数据集: {args.dataset_name} (取前 {args.max_samples} 条)")
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

    # 划分训练/验证集（如果启用早停或指定了 val_split）
    if args.enable_early_stop or args.val_split > 0:
        split_dataset = dataset.train_test_split(test_size=args.val_split, seed=42)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
        print(f"📊 训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条")
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer),
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler('cuda')

    os.makedirs(args.output_dir, exist_ok=True)
    print("\n🎯 开始训练（全量数据，新版 AMP）...")

    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        avg_train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device,
                                     gradient_accumulation_steps=args.gradient_accumulation_steps)
        print(f"Epoch {epoch+1} 训练平均 Loss: {avg_train_loss:.4f}")

        # ---------- 验证 ----------
        if val_loader is not None:
            val_loss = evaluate_epoch(model, val_loader, device)
            print(f"Epoch {epoch+1} 验证平均 Loss: {val_loss:.4f}")

            # 早停与最佳 checkpoint 保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                # 保存最佳模型
                best_path = os.path.join(args.output_dir, f"lora_{args.preset}_best.pt")
                torch.save(get_lora_state_dict(model), best_path)
                print(f"✅ 验证 loss 下降，保存最佳模型至: {best_path}")
            else:
                patience_counter += 1
                print(f"⚠️ 验证 loss 未下降 ({patience_counter}/{args.patience})")

            if args.enable_early_stop and patience_counter >= args.patience:
                print(f"🛑 早停触发！最佳 epoch: {best_epoch}, 验证 loss: {best_val_loss:.4f}")
                break

        # 仍然按 epoch 保存一份 checkpoint（可选项）
        lora_state = get_lora_state_dict(model)
        ckpt_path = os.path.join(args.output_dir, f"lora_{args.preset}_epoch{epoch+1}.pt")
        torch.save(lora_state, ckpt_path)

    # 最终保存与报告
    final_path = os.path.join(args.output_dir, f"lora_{args.preset}_final.pt")
    torch.save(get_lora_state_dict(model), final_path)
    print(f"\n✅ 训练完成！最终模型保存至: {final_path}")
    if best_epoch > 0:
        print(f"🏆 最佳验证 loss {best_val_loss:.4f} 发生在 epoch {best_epoch}，模型已保存至 {os.path.join(args.output_dir, f'lora_{args.preset}_best.pt')}")


if __name__ == "__main__":
    main()