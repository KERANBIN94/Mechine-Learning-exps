import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ====================== 1. 读取数据 ======================
print("正在读取数据...")
train_raw = pd.read_csv("lab2/data/train.csv", header=0)
test_raw = pd.read_csv("lab2/data/test.csv", header=0)

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

train_raw.columns = columns
test_raw.columns = columns[:-1]

# ====================== 2. 处理训练集标签 ======================
train_raw["income"] = train_raw["income"].str.strip().map({"<=50K": 0, ">50K": 1})

# ====================== 3. 使用众数填补缺失值 ======================
print("正在使用众数填补缺失值...")
for col in ["workclass", "occupation", "native_country"]:
    # 去除空格
    train_raw[col] = train_raw[col].str.strip()
    test_raw[col] = test_raw[col].str.strip()

    # 用训练集的众数填补
    mode_val = train_raw[col].mode()[0]

    train_raw[col] = train_raw[col].replace("?", mode_val)
    test_raw[col] = test_raw[col].replace("?", mode_val)

# ====================== 4. 划分特征 ======================
continuous_cols = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
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

# ====================== 5. One-Hot 编码 ======================
print("正在进行 One-Hot 编码...")
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(train_raw[categorical_cols])

train_cat = encoder.transform(train_raw[categorical_cols])
test_cat = encoder.transform(test_raw[categorical_cols])

train_cont = train_raw[continuous_cols].values
test_cont = test_raw[continuous_cols].values

X_train = np.hstack([train_cont, train_cat])  # (32561, 106)
X_test = np.hstack([test_cont, test_cat])  # (16281, 106)
y_train = train_raw["income"].values

print(f"训练集特征维度: {X_train.shape}")
print(f"测试集特征维度 : {X_test.shape}  → 成功得到 106 维")

# ====================== 6. 标准化 ======================
print("正在标准化...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====================== 7. 概率生成模型（共享协方差）======================
print("正在训练概率生成模型（LDA）...")
n_features = X_train.shape[1]

class_0 = X_train[y_train == 0]
class_1 = X_train[y_train == 1]

N0, N1 = len(class_0), len(class_1)
N = N0 + N1

prior_0 = N0 / N
prior_1 = N1 / N

mu_0 = class_0.mean(axis=0)
mu_1 = class_1.mean(axis=0)

# 共享协方差矩阵
centered_0 = class_0 - mu_0
centered_1 = class_1 - mu_1
Sigma = (centered_0.T @ centered_0 + centered_1.T @ centered_1) / (N - 2)
Sigma += np.eye(n_features) * 1e-6
Sigma_inv = np.linalg.inv(Sigma)

# 判别函数参数
w = Sigma_inv @ (mu_1 - mu_0)
b = (
    -0.5 * (mu_1 @ Sigma_inv @ mu_1)
    + 0.5 * (mu_0 @ Sigma_inv @ mu_0)
    + np.log(prior_1 / prior_0)
)

# ====================== 8. 预测测试集 ======================
print("正在预测测试集...")
logits = X_test @ w + b
prob_1 = 1 / (1 + np.exp(-logits))
y_pred = (prob_1 >= 0.5).astype(int)

# ====================== 9. 生成提交文件 ======================
# 如果 test.csv 第一列是 id，就用它；否则用 1,2,3,...
if "id" in test_raw.columns:
    ids = test_raw["id"].values
elif test_raw.shape[1] >= 1 and test_raw.iloc[0, 0] in [1, 2, 3, 4]:  # 第一列可能是 id
    ids = test_raw.iloc[:, 0].values
else:
    ids = np.arange(1, len(y_pred) + 1)

result = pd.DataFrame({"id": ids, "label": y_pred})

result.to_csv("predict.csv", index=False)

print("=== 预测完成！===")
print(f"predict.csv 已生成，共 {len(y_pred)} 条记录")
print(f"预测为 >50K 的人数: {y_pred.sum()} ({y_pred.mean() * 100:.2f}%)")
