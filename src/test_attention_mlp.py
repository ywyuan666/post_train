"""
测试 attention_mlp 策略微调后的模型生成效果
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset

model_name = 'HuggingFaceTB/SmolLM-135M'
preset = 'attention_mlp'
lora_path = './output_attention_mlp_full/lora_attention_mlp_final.pt'

print(f"🚀 加载模型: {model_name} (策略: {preset})")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)
target_modules = get_target_modules_from_preset(preset)
model = apply_lora_to_model(model, r=8, lora_alpha=16, target_modules=target_modules)
state = torch.load(lora_path, map_location='cuda')
load_lora_state_dict(model, state)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

print("✅ 模型加载完成\n")

prompt = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors='pt').to(device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=32, temperature=0.7, do_sample=True)

response = tokenizer.decode(out[0], skip_special_tokens=True)
print("📝 生成结果:")
print(response)