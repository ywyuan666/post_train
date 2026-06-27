"""
测试蒸馏学生模型的生成效果，并与教师模型对比
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from lora_strategies import apply_lora_to_model, load_lora_state_dict
from train_lora_strategies import get_target_modules_from_preset

model_name = "HuggingFaceTB/SmolLM-135M"
teacher_preset = "standard"
teacher_lora_path = "./output_standard/lora_standard_final.pt"   # 教师权重
student_model_path = "./distill_kd_output/student_kd_final.pt"    # 学生权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🚀 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ---------- 加载教师模型 ----------
print("👨‍🏫 加载教师模型（standard LoRA）...")
teacher = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
target_modules = get_target_modules_from_preset(teacher_preset)
teacher = apply_lora_to_model(teacher, r=8, lora_alpha=16, target_modules=target_modules)
state = torch.load(teacher_lora_path, map_location=device)
load_lora_state_dict(teacher, state)
teacher = teacher.to(device)
teacher.eval()

# ---------- 加载学生模型 ----------
print("🧑‍🎓 加载蒸馏学生模型...")
student = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
student.load_state_dict(torch.load(student_model_path, map_location=device))
student = student.to(device)
student.eval()

prompt = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\n📝 Prompt:", prompt)
print("\n🔮 生成教师回复...")
with torch.no_grad():
    teacher_out = teacher.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True)
teacher_response = tokenizer.decode(teacher_out[0], skip_special_tokens=True).split("### Response:")[-1].strip()
print("👨‍🏫 教师:", teacher_response[:200])

print("\n🔮 生成学生回复...")
with torch.no_grad():
    student_out = student.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True)
student_response = tokenizer.decode(student_out[0], skip_special_tokens=True).split("### Response:")[-1].strip()
print("🧑‍🎓 学生:", student_response[:200])