import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

# 超参数
model_name = "HuggingFaceTB/SmolLM-135M"
batch_size = 8
lr = 5e-5
epochs = 3

# 加载数据
with open("distill_data.json") as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 数据集
class DistillDataset(Dataset):
    def __len__(self): return len(data)
    def __getitem__(self, i):
        item = data[i]
        return tokenizer(
            item["prompt"] + " " + item["teacher_answer"],
            truncation=True, max_length=128, padding="max_length", return_tensors="pt"
        )

loader = DataLoader(DistillDataset(), batch_size=batch_size, shuffle=True)

# 学生模型 = 全新基座模型
student = AutoModelForCausalLM.from_pretrained(model_name)
student.to("cuda" if torch.cuda.is_available() else "cpu")
opt = torch.optim.AdamW(student.parameters(), lr=lr)

# 训练
student.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].squeeze().to(student.device)
        attention_mask = batch["attention_mask"].squeeze().to(student.device)
        loss = student(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, loss: {total_loss/len(loader):.4f}")

torch.save(student.state_dict(), "student_distilled.pt")
print("✅ 蒸馏训练完成，学生模型保存为 student_distilled.pt")
print("📊 蒸馏完成：教师 -> 学生，50 条推理轨迹训练")
