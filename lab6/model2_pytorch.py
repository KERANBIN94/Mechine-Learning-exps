import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import os

# ==================== 参数设置 & 保存目录 ====================
save_dir = "lab6/output_model2_pytorch/results"

MAX_VOCAB_SIZE = 20000  # 词汇表大小
MAX_SEQUENCE_LENGTH = 200  # 最大序列长度
EMBEDDING_DIM = 128
LSTM_UNITS = 128
NUM_CLASSES = 3
BATCH_SIZE = 128
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==================== 数据读取 ====================
train_df = pd.read_csv("lab6/data/review/drugsComTrain_raw.csv", encoding="utf-8")
test_df = pd.read_csv("lab6/data/review/drugsComTest_raw.csv", encoding="utf-8")

print(f"训练样本数: {len(train_df)}")
print(f"测试样本数: {len(test_df)}")


# ==================== 情感标签生成 ====================
def rating_to_sentiment(rating):
    if rating <= 4:
        return 0  # 消极
    elif rating <= 6:
        return 1  # 中性
    else:
        return 2  # 积极


train_labels = train_df["rating"].apply(rating_to_sentiment).values
test_labels = test_df["rating"].apply(rating_to_sentiment).values

train_texts = train_df["review"].astype(str).tolist()
test_texts = test_df["review"].astype(str).tolist()


# ==================== 纯 Python 文本预处理 & 词汇表构建 ====================
def basic_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()


print("Building vocabulary from training data...")
word_counter = Counter()
for text in tqdm(train_texts):
    tokens = basic_tokenize(text)
    word_counter.update(tokens)

most_common = word_counter.most_common(MAX_VOCAB_SIZE - 4)

vocab = {"<pad>": 0, "<unk>": 1}
idx = 2
for word, _ in most_common:
    vocab[word] = idx
    idx += 1

VOCAB_SIZE = len(vocab)
print(f"词汇表大小: {VOCAB_SIZE}")


def text_to_sequence(text, vocab, max_len):
    tokens = basic_tokenize(text)[:max_len]
    seq = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    return seq


print("Converting texts to sequences...")
train_sequences = [
    text_to_sequence(text, vocab, MAX_SEQUENCE_LENGTH) for text in tqdm(train_texts)
]
test_sequences = [
    text_to_sequence(text, vocab, MAX_SEQUENCE_LENGTH) for text in tqdm(test_texts)
]


def pad_sequences(sequences, max_len, pad_idx=0):
    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [pad_idx] * (max_len - len(seq)))
    return np.array(padded, dtype=np.int64)


train_data = pad_sequences(train_sequences, MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, MAX_SEQUENCE_LENGTH)

print(f"第一条训练评论序列长度: {len(train_sequences[0])}")


# ==================== Dataset 和 DataLoader ====================
class ReviewDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


train_dataset = ReviewDataset(train_data, train_labels)
test_dataset = ReviewDataset(test_data, test_labels)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==================== 模型定义 ====================
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, lstm_units, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(lstm_units, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.dropout(x)
        return self.fc(x)


model = SentimentLSTM(VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数量: {total_params:,}")

# ==================== 训练循环 ====================
train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(1, EPOCHS + 1):
    # 训练
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_losses.append(epoch_loss / len(train_loader))
    train_accs.append(correct / total)

    # 测试
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in tqdm(test_loader, desc=f"Epoch {epoch}/{EPOCHS} [Test]"):
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_losses.append(test_loss / len(test_loader))
    test_accs.append(correct / total)

    print(
        f"Epoch {epoch} | Train Loss: {train_losses[-1]:.4f} Acc: {train_accs[-1]:.4f} | "
        f"Test Loss: {test_losses[-1]:.4f} Acc: {test_accs[-1]:.4f}"
    )

# ==================== 保存训练曲线（不再显示）===================
curve_path = os.path.join(save_dir, "training_curves.png")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(test_accs, label="Test Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(curve_path)
plt.close()  # 关闭图像，释放内存

print(f"\n训练曲线已保存至: {curve_path}")

# 保存最佳模型（可选）
model_save_path = "output_model2_pytorch/sentiment_lstm_best.pth"
os.makedirs("output_model2_pytorch", exist_ok=True)
best_test_acc = max(test_accs)
best_epoch = test_accs.index(best_test_acc) + 1
print(f"最佳测试准确率: {best_test_acc:.4f} (Epoch {best_epoch})")

# 如果需要保存模型权重，可取消下方注释
# torch.save(model.state_dict(), model_save_path)
# print(f"模型权重已保存至: {model_save_path}")

print(f"\n最终测试准确率: {test_accs[-1]:.4f}")
print(f"所有结果保存在目录: {save_dir}")
