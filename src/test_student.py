import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora_strategies import apply_lora_to_model
from train_lora_strategies import get_target_modules_from_preset

# 配置
model_name = "HuggingFaceTB/SmolLM-135M"
lora_path = "./distill_kd_output/student_kd_final.pt"
device = "cuda"

print("🧑‍🎓 加载蒸馏后的学生模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 加载基座
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 挂载 LoRA
target_modules = get_target_modules_from_preset("standard")
model = apply_lora_to_model(model, r=8, lora_alpha=16, target_modules=target_modules)

# 加载权重 ✅ 关键修复：直接加载到 GPU
lora_weights = torch.load(lora_path, map_location=device)
model.load_state_dict(lora_weights, strict=False)

# ✅ 最关键：整个模型搬到 GPU
model = model.to(device)
model.eval()
print("✅ 模型加载完成")

# 测试对话
def ask(question):
    prompt = f"### Instruction:\n{question}\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-"*50)

# 开始测试
ask("What is AI?")
ask("Explain machine learning in simple terms.")