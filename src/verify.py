"""
验证 LoRA 微调效果
对比基座模型 vs 微调后模型的输出
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora import apply_lora_to_model, load_lora_state_dict


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "HuggingFaceTB/SmolLM-135M"
    
    print("📦 加载基座模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        device_map="auto" if device.type == "cuda" else None,
    )
    base_model.to(device)
    base_model.eval()
    
    print("🔧 加载微调后模型...")
    tuned_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        device_map="auto" if device.type == "cuda" else None,
    )
    tuned_model = apply_lora_to_model(tuned_model, r=8, lora_alpha=16)
    lora_state = torch.load("./lora_output/lora_final.pt", map_location=device)
    load_lora_state_dict(tuned_model, lora_state)
    tuned_model.to(device)
    tuned_model.eval()
    
    # 测试样例
    test_cases = [
        ("Give three tips for healthy eating.", ""),
        ("What is the capital of France?", ""),
        ("Write a short email to schedule a meeting.", ""),
    ]
    
    print("\n" + "="*60)
    print("📊 对比结果")
    print("="*60)
    
    for i, (instruction, input_text) in enumerate(test_cases, 1):
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        print(f"\n--- 测试 {i} ---")
        print(f"Instruction: {instruction}")
        
        # 基座模型
        base_response = generate(base_model, tokenizer, prompt)
        print(f"\n🔵 基座模型:\n{base_response.split('### Response:')[-1].strip()[:200]}...")
        
        # 微调模型
        tuned_response = generate(tuned_model, tokenizer, prompt)
        print(f"\n🟢 微调后:\n{tuned_response.split('### Response:')[-1].strip()[:200]}...")
        
        print("-"*40)


if __name__ == "__main__":
    main()