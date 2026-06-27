"""
评估 LoRA 模型在 GSM8K 数学题上的生成效果
用法：
    python eval_grpo.py --lora_path ./output_standard_full/lora_standard_final.pt --num_samples 5
"""
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA 权重路径")
    parser.add_argument("--num_samples", type=int, default=5, help="测试题目数量")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="最大生成长度")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "HuggingFaceTB/SmolLM-135M"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 加载基座模型，注入 LoRA 并加载权重
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    target = get_target_modules_from_preset("standard")
    model = apply_lora_to_model(model, r=8, lora_alpha=16, target_modules=target)
    state = torch.load(args.lora_path, map_location=device)
    load_lora_state_dict(model, state)
    model.to(device)
    model.eval()

    # 加载 GSM8K 数据集（前 num_samples 题）
    dataset = load_dataset("gsm8k", "main", split=f"train[:{args.num_samples}]")
    for idx, sample in enumerate(dataset):
        question = sample["question"]
        answer_raw = sample["answer"]
        gt = answer_raw.split("####")[-1].strip()
        prompt = f"Q: {question}\nA:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.split("A:")[-1].strip()

        print(f"\n📝 问题 {idx+1}: {question}")
        print(f"🎯 正确答案: {gt}")
        print(f"🤖 模型回答: {response[:200]}")   # 只显示前200字符
        print("-" * 60)

if __name__ == "__main__":
    main()