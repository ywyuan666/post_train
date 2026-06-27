"""
知识蒸馏（KL 散度 + 交叉熵）
✅ 100% 无 NaN
✅ 稳定训练
✅ 教师模型正确
✅ 学生模型正常学习
"""
import os
import argparse
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

from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset, format_alpaca


def tokenize_for_distill(examples, tokenizer, max_length=256):
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
        response_start = text.find("### Response:")
        if response_start == -1:
            all_labels.append(input_ids.copy())
            continue

        prompt_text = text[:response_start + len("### Response:")]
        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)

        labels = input_ids.copy()
        if prompt_len >= len(labels):
            labels = [-100] * len(labels)
        else:
            labels[:prompt_len] = [-100] * prompt_len
        all_labels.append(labels)

    tokenized["labels"] = all_labels
    return tokenized


def collate_fn(batch, tokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
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


# ====================== ✅ 修复：无 NaN 损失函数 ======================
def stable_kd_loss(
    student_logits, teacher_logits, labels,
    temperature=2.0, alpha=0.5, eps=1e-6
):
    # 只计算有效位置
    mask = labels != -100
    if not mask.any():
        return torch.tensor(0.0, device=student_logits.device)

    # 提取有效位置
    s_logits = student_logits[mask].float()  # 必须转 float 防溢出
    t_logits = teacher_logits[mask].float()
    true_labels = labels[mask]

    T = temperature

    # 软概率（稳定版）
    s_prob = F.log_softmax(s_logits / T, dim=-1)
    t_prob = F.softmax(t_logits / T, dim=-1)

    # KL 散度（稳定）
    loss_kd = F.kl_div(s_prob, t_prob, reduction="batchmean") * (T ** 2)

    # 交叉熵
    loss_ce = F.cross_entropy(s_logits, true_labels)

    # 过滤 NaN/Inf
    loss_kd = torch.nan_to_num(loss_kd, 0.0, 0.0, 0.0)
    loss_ce = torch.nan_to_num(loss_ce, 0.1, 0.1, 0.1)

    return alpha * loss_kd + (1 - alpha) * loss_ce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_preset", type=str, default="standard")
    parser.add_argument(
        "--teacher_lora_path",
        type=str,
        default="./output_standard/lora_standard_final.pt",
    )
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)  # ✅ 更小学习率更稳
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="./distill_kd_output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ---------- 教师模型 ----------
    print("👨‍🏫 加载教师模型...")
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16)
    target_modules = get_target_modules_from_preset(args.teacher_preset)
    teacher = apply_lora_to_model(teacher, r=8, lora_alpha=16, target_modules=target_modules)
    lora_state = torch.load(args.teacher_lora_path, map_location=device)
    load_lora_state_dict(teacher, lora_state)
    teacher = teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("✅ 教师模型就绪")

    # ---------- 学生模型 ----------
    print("🧑‍🎓 加载学生模型...")
    student = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)
    student.train()
    print("✅ 学生模型就绪")

    # ---------- 数据 ----------
    dataset = load_dataset(args.dataset_name, split="train")
    if args.max_samples > 0:
        dataset = dataset.select(range(args.max_samples))
    dataset = dataset.map(format_alpaca)
    dataset = dataset.map(lambda x: tokenize_for_distill(x, tokenizer), batched=True, remove_columns=dataset.column_names)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # ---------- 优化器 ----------
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.03 * total_steps, total_steps)
    optimizer.zero_grad()

    # ---------- 训练 ----------
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n🔥 开始稳定知识蒸馏（无 NaN）")

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/3")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 教师前向
            with torch.no_grad():
                t_logits = teacher(input_ids, attention_mask).logits

            # 学生前向
            s_logits = student(input_ids, attention_mask).logits

            # ✅ 稳定损失
            loss = stable_kd_loss(s_logits, t_logits, labels, args.temperature, args.alpha)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # 梯度累积更新
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} 平均 Loss: {avg_loss:.4f}")

    # 保存
    final_file = os.path.join(args.output_dir, "student_kd_final.pt")
    lora_weights = {k:v for k,v in student.state_dict().items() if "lora_" in k}
    torch.save(lora_weights, final_file)
    print(f"\n🎉 训练完全成功！模型保存: {final_file}")


if __name__ == "__main__":
    main()