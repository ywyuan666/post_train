"""
知识蒸馏（Knowledge Distillation）- 最终稳定版（带 NaN 跳过）
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

from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset


def format_alpaca(example):
    if example.get("input", "").strip():
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}


def tokenize_for_distill(examples, tokenizer, max_length=256):
    texts = examples["text"]
    prompts = []
    for text in texts:
        resp_pos = text.find("### Response:")
        if resp_pos != -1:
            prompt = text[:resp_pos + len("### Response:")]
        else:
            prompt = text
        prompts.append(prompt)
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}


def collate_fn(batch, tokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        if tokenizer.padding_side == "left":
            input_ids.append([tokenizer.pad_token_id] * pad_len + item["input_ids"])
            attention_mask.append([0] * pad_len + item["attention_mask"])
        else:
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
    }


def safe_distillation_loss(student_logits, teacher_logits, temperature=1.0):
    """
    返回 loss 和 is_nan 标志
    """
    # 检查输入是否包含 NaN
    if torch.isnan(student_logits).any() or torch.isnan(teacher_logits).any():
        return torch.tensor(0.0, device=student_logits.device), True

    # 强制 float32
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()

    # 截断
    teacher_logits = torch.clamp(teacher_logits, min=-50.0, max=50.0)
    student_logits = torch.clamp(student_logits, min=-50.0, max=50.0)

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1).clamp(min=1e-7, max=1.0)

    loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    if torch.isnan(loss_kd):
        return torch.tensor(0.0, device=student_logits.device), True
    return loss_kd, False


def generate_teacher_logits(teacher_model, tokenizer, input_ids, attention_mask, max_new_tokens=64):
    teacher_model.eval()
    with torch.no_grad():
        outputs = teacher_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen_ids = outputs.sequences[:, input_ids.shape[1]:]
        logits = torch.stack(outputs.scores, dim=1)
        # 替换 NaN 为 0
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        logits = torch.clamp(logits, min=-50.0, max=50.0)
    return gen_ids, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_preset", type=str, default="standard")
    parser.add_argument("--teacher_lora_path", type=str, default="./output_standard/lora_standard_final.pt")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./distill_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 教师模型
    print("👨‍🏫 加载教师模型...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    target_modules = get_target_modules_from_preset(args.teacher_preset)
    teacher = apply_lora_to_model(teacher, r=8, lora_alpha=16, target_modules=target_modules)
    lora_state = torch.load(args.teacher_lora_path, map_location=device)
    load_lora_state_dict(teacher, lora_state)
    teacher.to(device)
    teacher.eval()
    print("✅ 教师模型加载完成")

    # 学生模型
    print("🧑‍🎓 加载学生模型（基座）...")
    student = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    student.to(device)
    student.train()
    print("✅ 学生模型加载完成")

    # 数据
    print(f"📚 加载数据集: {args.dataset_name} (取前 {args.max_samples} 条)")
    dataset = load_dataset(args.dataset_name, split="train")
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = dataset.map(format_alpaca)
    dataset = dataset.map(
        lambda x: tokenize_for_distill(x, tokenizer, max_length=256),
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * total_steps),
        num_training_steps=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print("\n🔥 开始知识蒸馏训练...")

    for epoch in range(args.num_epochs):
        total_loss = 0
        skipped_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                gen_ids, teacher_logits = generate_teacher_logits(
                    teacher, tokenizer, input_ids, attention_mask, max_new_tokens=64
                )
                teacher_logits = teacher_logits.to(device)
                gen_ids = gen_ids.to(device)

            student_input_ids = torch.cat([input_ids, gen_ids], dim=1)
            student_attention_mask = torch.cat([
                attention_mask,
                torch.ones_like(gen_ids, device=device)
            ], dim=1)

            student_outputs = student(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask,
            )
            student_logits = student_outputs.logits[:, input_ids.shape[1]:, :]

            loss_kd, is_nan = safe_distillation_loss(student_logits, teacher_logits, temperature=args.temperature)

            if is_nan:
                skipped_batches += 1
                progress_bar.set_postfix({"loss": "NaN(skipped)", "skipped": skipped_batches})
                # 清空梯度，避免污染下一步
                optimizer.zero_grad()
                continue

            loss = loss_kd / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})

        avg_loss = total_loss / (len(dataloader) - skipped_batches) if (len(dataloader) - skipped_batches) > 0 else 0
        print(f"Epoch {epoch+1} 平均蒸馏 Loss: {avg_loss:.4f}  (跳过 {skipped_batches} 个 NaN batch)")

        ckpt_path = os.path.join(args.output_dir, f"student_epoch{epoch+1}.pt")
        torch.save(student.state_dict(), ckpt_path)

    final_path = os.path.join(args.output_dir, "student_final.pt")
    torch.save(student.state_dict(), final_path)
    print(f"\n✅ 蒸馏完成！学生模型保存至: {final_path}")


if __name__ == "__main__":
    main()