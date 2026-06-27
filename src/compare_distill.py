"""对比教师模型、蒸馏学生模型、基座模型"""
import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset

def generate(model, tokenizer, prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    gen_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = gen_tokens / elapsed
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, tokens_per_sec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_preset", type=str, default="standard")
    parser.add_argument("--teacher_lora", type=str, default="./output_standard/lora_standard_final.pt")
    parser.add_argument("--student_weights", type=str, default="./distill_standard/student_final.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 测试指令
    test_instruction = "Give three tips for healthy eating."
    prompt = f"### Instruction:\n{test_instruction}\n\n### Response:\n"

    print("=" * 60)
    print("🔵 1. 基座模型 (Base)")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    base_model.eval()
    base_resp, base_speed = generate(base_model, tokenizer, prompt)
    print(f"回复: {base_resp.split('### Response:')[-1].strip()[:200]}...")
    print(f"生成速度: {base_speed:.2f} tokens/sec\n")

    print("🟢 2. 教师模型 (Teacher = Base + LoRA)")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    target_modules = get_target_modules_from_preset(args.teacher_preset)
    teacher = apply_lora_to_model(teacher, r=8, lora_alpha=16, target_modules=target_modules)
    lora_state = torch.load(args.teacher_lora, map_location=device)
    load_lora_state_dict(teacher, lora_state)
    teacher.eval()
    teacher_resp, teacher_speed = generate(teacher, tokenizer, prompt)
    print(f"回复: {teacher_resp.split('### Response:')[-1].strip()[:200]}...")
    print(f"生成速度: {teacher_speed:.2f} tokens/sec\n")

    print("🟡 3. 蒸馏学生模型 (Student)")
    student = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    student.load_state_dict(torch.load(args.student_weights, map_location=device))
    student.eval()
    student_resp, student_speed = generate(student, tokenizer, prompt)
    print(f"回复: {student_resp.split('### Response:')[-1].strip()[:200]}...")
    print(f"生成速度: {student_speed:.2f} tokens/sec\n")

    print("=" * 60)

if __name__ == "__main__":
    main()