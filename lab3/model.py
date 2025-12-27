import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# ====================== 1. 数据加载与预处理 ======================
train_path = "lab3/data/train.csv"
test_path = "lab3/data/test.csv"

# 读取原始数据（带表头）
train_df = pd.read_csv(train_path, header=0)
test_df = pd.read_csv(test_path, header=0)

# 目标列名称（根据你提供的信息）
label_col = "income"  # 可能是 "make over 50K a year or not" 或 "income"

# 将标签转为 0/1
train_df[label_col] = (
    train_df[label_col]
    .map({" <=50K": 0, " >50K": 1, "<=50K": 0, ">50K": 1, " <=50K.": 0, " >50K.": 1})
    .astype(int)
)

# ------------------- 缺失值填众数 -------------------
categorical_cols = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
numerical_cols = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

# 对所有列（包括数值列）都用众数填补
for col in train_df.columns:
    if col == label_col:
        continue
    mode_val = train_df[col].mode()[0]
    train_df[col] = train_df[col].fillna(mode_val)
    test_df[col] = test_df[col].fillna(mode_val)

# ------------------- One-Hot 编码 -------------------
all_cols = categorical_cols + numerical_cols
train_processed = pd.get_dummies(
    train_df[all_cols], columns=categorical_cols, drop_first=False
)
test_processed = pd.get_dummies(
    test_df[all_cols], columns=categorical_cols, drop_first=False
)

# 对齐训练集和测试集的列（test 中可能缺少某些类别）
test_processed = test_processed.reindex(columns=train_processed.columns, fill_value=0)

# 转为 numpy
X = train_processed.values.astype(np.float64)
y = train_df[label_col].values.astype(np.float64)
X_test = test_processed.values.astype(np.float64)

print(f"处理后特征维度: {X.shape[1]}")  # 应该是 106 维

# ====================== 2. 正态标准化（Z-score） ======================
normalize_indices = [
    train_processed.columns.get_loc("age"),
    train_processed.columns.get_loc("fnlwgt"),
    train_processed.columns.get_loc("education_num"),
    train_processed.columns.get_loc("capital_gain"),
    train_processed.columns.get_loc("capital_loss"),
    train_processed.columns.get_loc("hours_per_week"),
]

# 计算训练集均值和标准差
X_mean = X[:, normalize_indices].mean(axis=0)
X_std = X[:, normalize_indices].std(axis=0) + 1e-8  # 防止除0

X[:, normalize_indices] = (X[:, normalize_indices] - X_mean) / X_std
X_test[:, normalize_indices] = (X_test[:, normalize_indices] - X_mean) / X_std

# ====================== 3. 分割验证集 ======================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ====================== 4. Logistic Regression 实现 ======================
def sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)


def compute_loss(y_pred, y_label, w, lamda):
    cross_entropy = -np.mean(
        y_label * np.log(y_pred) + (1 - y_label) * np.log(1 - y_pred)
    )
    l2_reg = lamda * np.sum(w**2)
    return cross_entropy + l2_reg


def compute_accuracy(y_pred, y_true):
    return np.mean((y_pred >= 0.5) == y_true)


class LogisticRegression:
    def __init__(self, lamda=0.0):
        self.lamda = lamda
        self.w = None
        self.b = None

    def fit(self, X, y, lr=0.01, epochs=1000, verbose=True):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(epochs):
            # 前向
            z = X @ self.w + self.b
            y_pred = sigmoid(z)

            # 误差
            error = y_pred - y

            # 梯度（带 L2）
            w_grad = np.mean(error[:, np.newaxis] * X, axis=0) + 2 * self.lamda * self.w
            b_grad = np.mean(error)

            # 更新
            self.w -= lr * w_grad
            self.b -= lr * b_grad

            # 记录
            train_loss = compute_loss(y_pred, y, self.w, self.lamda)
            train_acc = compute_accuracy(y_pred, y)

            # validation
            val_pred = self.predict_proba(X_val)
            val_loss = compute_loss(val_pred, y_val, self.w, self.lamda)
            val_acc = compute_accuracy(val_pred, y_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            if verbose and (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
                )

        return train_losses, val_losses, train_accs, val_accs

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ====================== 5. 训练（可自行调参） ======================
lamda = 0.001  # L2 正则化强度
lr = 0.05  # 学习率（对标准化后的数据可大一点）
epochs = 10000

model = LogisticRegression(lamda=lamda)
train_losses, val_losses, train_accs, val_accs = model.fit(
    X_train, y_train, lr=lr, epochs=epochs, verbose=True
)

# ====================== 6. 画图 ======================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()

# ====================== 7. 测试集预测 ======================
y_test_pred = model.predict(X_test)

# 生成提交文件
submission = pd.DataFrame(
    {
        "id": np.arange(1, len(y_test_pred) + 1),  # 1~16281
        "label": y_test_pred,
    }
)

submission.to_csv("lab3/output.csv", index=False)
print("预测完成！output.csv 已生成（共 {} 条）".format(len(submission)))

# ====================== 8.（可选）查看特征重要性 ======================
feature_importance = pd.Series(np.abs(model.w), index=train_processed.columns)
print("\nTop 15 最重要的特征：")
print(feature_importance.nlargest(15))
