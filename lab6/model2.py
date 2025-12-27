import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
import os

# ==================== 参数设置 & 保存目录 ====================
save_dir = "lab6/output_model2/results"


MAX_NUM_WORDS = 20000  # 词汇表大小
MAX_SEQUENCE_LENGTH = 200  # 最大序列长度
EMBEDDING_DIM = 128
LSTM_UNITS = 128
NUM_CLASSES = 3
BATCH_SIZE = 128
EPOCHS = 10

# ==================== 读取数据 ====================
train_df = pd.read_csv("lab6/data/review/drugsComTrain_raw.csv")
test_df = pd.read_csv("lab6/data/review/drugsComTest_raw.csv")

print(f"训练样本数: {len(train_df)}")
print(f"测试样本数: {len(test_df)}")


# ==================== 生成情感标签 ====================
def get_sentiment(rating):
    if rating <= 4:
        return 0  # 消极
    elif rating <= 6:
        return 1  # 中性
    else:
        return 2  # 积极


train_df["sentiment"] = train_df["rating"].apply(get_sentiment)
test_df["sentiment"] = test_df["rating"].apply(get_sentiment)

train_texts = train_df["review"].values
train_labels = to_categorical(train_df["sentiment"].values, num_classes=NUM_CLASSES)

test_texts = test_df["review"].values
test_labels = to_categorical(test_df["sentiment"].values, num_classes=NUM_CLASSES)

# ==================== 文本预处理 ====================
# 只在训练集上fit tokenizer
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

print(f"第一条评论序列化后长度: {len(train_sequences[0])}")

# ==================== 构建模型 ====================
model = Sequential()
model.add(
    Embedding(
        input_dim=MAX_NUM_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
    )
)
model.add(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(NUM_CLASSES, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ==================== 训练模型 ====================
history = model.fit(
    train_data,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    verbose=1,  # 保持训练进度打印
)

# ==================== 评估模型 ====================
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
print(f"测试准确率: {test_acc:.4f}")

# ==================== 保存训练曲线（不再显示）===================
curve_path = os.path.join(save_dir, "training_curves.png")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(curve_path)
plt.close()  # 关闭图像，释放内存

print(f"\n训练曲线已保存至: {curve_path}")

# ==================== 可选：保存模型权重 ====================
model_save_path = "lab6/output_model2/sentiment_lstm_keras.h5"
os.makedirs("lab6/output_model2", exist_ok=True)
print(f"\n所有结果保存在目录: {save_dir}")
