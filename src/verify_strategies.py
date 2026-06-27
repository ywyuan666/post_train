"""对比四种策略的生成效果"""
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset   # 复用预设映射

def generate(model, tokenizer, prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--preset", type=str, required=True,
                        choices=["minimal", "standard", "attention_mlp", "all_linear"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "HuggingFaceTB/SmolLM-135M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )

    target_modules = get_target_modules_from_preset(args.preset)
    model = apply_lora_to_model(model, r=8, lora_alpha=16, target_modules=target_modules)

    lora_state = torch.load(args.lora_path, map_location=device)
    load_lora_state_dict(model, lora_state)
    model.to(device)
    model.eval()

    test_instruction = "Explain the importance of recycling in three sentences."
    prompt = f"### Instruction:\n{test_instruction}\n\n### Response:\n"

    print(f"\n📌 策略: {args.preset}")
    print(f"📌 指令: {test_instruction}")
    response = generate(model, tokenizer, prompt)
    print(f"🤖 回复:\n{response.split('### Response:')[-1].strip()}\n")

if __name__ == "__main__":
    main()