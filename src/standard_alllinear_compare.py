import re
import matplotlib.pyplot as plt

def extract_losses(log_file):
    """提取每个 epoch 的训练和验证平均 loss"""
    train_loss = []
    val_loss = []
    with open(log_file) as f:
        for line in f:
            # 匹配“Epoch X 训练平均 Loss: Y.YYYY”
            m_train = re.search(r"Epoch\s+\d+\s+训练平均 Loss:\s*([\d.]+)", line)
            if m_train:
                train_loss.append(float(m_train.group(1)))

            # 匹配“Epoch X 验证平均 Loss: Y.YYYY”
            m_val = re.search(r"Epoch\s+\d+\s+验证平均 Loss:\s*([\d.]+)", line)
            if m_val:
                val_loss.append(float(m_val.group(1)))
    return train_loss, val_loss

# 读取两种策略的损失
train_std, val_std = extract_losses("train_standard_full.log")
train_all, val_all = extract_losses("train_all_linear_full.log")

# 绘制对比图
plt.figure(figsize=(10, 6))
epochs_std = range(1, len(train_std) + 1)
epochs_all = range(1, len(train_all) + 1)

plt.plot(epochs_std, train_std, 'o-', label='Standard Train', color='#1f77b4')
if val_std:
    plt.plot(epochs_std, val_std, 's--', label='Standard Val', color='#1f77b4')

plt.plot(epochs_all, train_all, 'o-', label='All_linear Train', color='#d62728')
if val_all:
    plt.plot(epochs_all, val_all, 's--', label='All_linear Val', color='#d62728')

plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Standard vs All_linear (Full Alpaca, 52k samples)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("standard_alllinear_compare.png", dpi=150)
print("✅ 图片已保存为 standard_alllinear_compare.png")