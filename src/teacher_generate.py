from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# 加载你训练好的最佳 LoRA 教师模型
model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# 加载你训练好的权重
model.load_state_dict(torch.load("output_standard/lora_standard_best.pt"), strict=False)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 数学问题（50 条）
questions = [
    "1+2+3+4=?", "5*6-7=?", "10-3+2=?", "8/2+4=?", "12+13-5=?",
    "7*3-9=?", "20/4+6=?", "15+15-10=?", "9*2-8=?", "30/5+3=",
    "2+2*3=?", "10-2*4=?", "6+6/2=?", "18-3*5=?", "4*4-10=",
    "25/5+7=?", "11+11-4=?", "6*5-12=?", "32/8+9=?", "14+16-20=",
    "7+3*4=?", "20-5*3=?", "9+12/3=?", "22-2*7=?", "5*5-15=",
    "40/8+2=?", "13+17-8=?", "8*4-10=?", "36/6+11=", "28+12-30=",
    "10+5*2=?", "30-4*6=?", "12+18/3=?", "25-3*6=?", "6*6-20=",
    "50/10+15=", "21+19-25=", "9*5-20=?", "48/4+2=?", "33+17-40=",
    "8+4*3=?", "15-3*2=?", "20+10/2=?", "35-5*4=?", "7*7-30=",
    "60/6+5=?", "24+26-35=", "10*3-15=?", "42/7+8=?", "19+21-32="
]

data = []
for q in questions:
    prompt = f"Question: {q}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.7)
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    data.append({"prompt": prompt, "teacher_answer": ans})

# 保存 50 条蒸馏数据
with open("distill_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 教师模型生成 50 条数据完成：distill_data.json")
