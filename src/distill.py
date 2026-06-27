"""
简化的知识蒸馏 —— 实际为 SFT（用 ground truth 训练学生模型）
避免教师 logits NaN 问题，稳定运行
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm


def format_alpaca(example):
    """Alpaca 格式化为完整文本"""
    if example.get("input", "").strip():
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}


def tokenize_function(examples, tokenizer, max_length=256):
    """Tokenize 完整文本（包含 response），用于 SFT"""
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    # labels 与 input_ids 相同（因果 LM 会自动 shift）
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def collate_fn(batch, tokenizer):
    """动态 padding，适配 left padding"""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        if tokenizer.padding_side == "left":
            input_ids.append([tokenizer.pad_token_id] * pad_len + item["input_ids"])
            attention_mask.append([0] * pad_len + item["attention_mask"])
            labels.append([-100] * pad_len + item["labels"])
        else:
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def main():
    parser = argparse.ArgumentParser(description="简化的蒸馏（SFT）训练")
    parser.add_argument("--teacher_preset", type=str, default="standard", help="保留参数名兼容，实际未使用")
    parser.add_argument("--teacher_lora_path", type=str, default="", help="保留参数名兼容，实际未使用")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)   # SFT 建议稍小
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0, help="保留参数名兼容")
    parser.add_argument("--output_dir", type=str, default="./distill_sft")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 设备: {device}")

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ---------- 加载学生模型（基座） ----------
    print("🧑‍🎓 加载学生模型（基座）...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.to(device)
    model.train()
    print("✅ 学生模型加载完成")

    # ---------- 准备数据集 ----------
    print(f"📚 加载数据集: {args.dataset_name} (取前 {args.max_samples} 条)")
    dataset = load_dataset(args.dataset_name, split="train")
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    dataset = dataset.map(format_alpaca)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=256),
        batched=True,
        remove_columns=dataset.column_names,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # ---------- 优化器 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * total_steps),
        num_training_steps=total_steps,
    )

    # ---------- 训练循环 ----------
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n🔥 开始 SFT 训练（使用 ground truth）...")

    for epoch in range(args.num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}")

        # 保存 checkpoint
        ckpt_path = os.path.join(args.output_dir, f"student_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

    # 最终保存
    final_path = os.path.join(args.output_dir, "student_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ SFT 训练完成！学生模型保存至: {final_path}")


if __name__ == "__main__":
    main()