import numpy as np
import pandas as pd
import csv


# ====================== 1. 数据读取与预处理 ======================
def load_training_data(path="train.csv"):
    # 使用 errors='ignore' 以防止编码错误，与测试集的加载方式保持一致
    raw = pd.read_csv(path, encoding="big5")
    raw = raw.iloc[:, 3:]
    # 将 'NR' (无降雨) 替换为 0，并将所有数据转换为数值类型
    raw[raw == "NR"] = "0"
    raw_data = raw.to_numpy().astype(float)

    # 训练数据为12个月。每个月有20天的数据，每天有18个特征（污染物）。
    # 数据是交错排列的。我们需要将其从原始的 (4320, 24) 形状重塑为 (12, 18, 480)。
    # 480小时 = 20天 * 24小时/天。

    all_months_data = []
    for month in range(12):
        # 提取当前月份的数据块
        # 每个月有 18个特征 * 20天 = 360 行
        month_block = raw_data[month * 360 : (month + 1) * 360, :]

        month_features = []
        for feature_idx in range(18):
            # 对于当前特征，从月份数据块中选出属于它的20行
            # 数据是交错的，所以我们从特征的索引开始，每隔18行取一次
            feature_rows = month_block[feature_idx::18, :]
            # 将 (20, 24) 的数组展平，得到连续480小时的读数
            month_features.append(feature_rows.flatten())

        all_months_data.append(np.array(month_features))

    # 将12个月的数据堆叠起来，得到最终的 (12, 18, 480) 数组
    return np.array(all_months_data)


def create_dataset(data, feature_hours=9):
    """
    每连续10小时为一个样本，前9小时所有18维特征 → 第10小时PM2.5
    """
    x_list, y_list = [], []
    for month in range(12):
        for start in range(480 - feature_hours):
            x = data[month, :, start : start + feature_hours]  # (18, 9)
            y = data[month, 9, start + feature_hours]  # PM2.5 在第10维（索引9）
            x_list.append(x.flatten())
            y_list.append(y)
    X = np.array(x_list, dtype=float)  # (样本数, 162)
    Y = np.array(y_list, dtype=float)
    return X, Y


# ====================== 2. 测试数据处理 ======================
def load_test_data(path="test.csv"):
    raw = pd.read_csv(path, header=None, encoding="big5")
    raw = raw.iloc[:, 2:]  # 去掉 id 和监测项目

    test_x = []
    for i in range(0, len(raw), 18):
        block = raw.iloc[i : i + 18, :9].values
        # 处理 'NR'
        block = np.where(block == "NR", "0", block).astype(float)
        test_x.append(block.flatten())
    test_x = np.array(test_x, dtype=float)  # (240, 162)
    return test_x

# ====================== 4. AdaGrad 线性回归 ======================
class LinearRegressionAdaGrad:
    def __init__(self, lr=0.01, epochs=10000, epsilon=1e-8):
        self.lr = lr
        self.epochs = epochs
        self.epsilon = epsilon

    def fit(self, X, y):
        # X: (n_samples, 162), 手动加上 bias 列
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)  # (n, 163)
        self.w = np.zeros(X.shape[1])
        ada = np.zeros(X.shape[1])  # 累积梯度平方

        for epoch in range(1, self.epochs + 1):
            y_pred = X.dot(self.w)
            loss = y_pred - y
            grad = 2 * X.T.dot(loss) / len(X)  # (163,)
            ada += grad**2
            self.w -= self.lr * grad / (np.sqrt(ada) + self.epsilon)

            if epoch % 1000 == 0 or epoch <= 10:
                cost = np.sqrt(np.mean(loss**2))
                print(f"Epoch {epoch:5d} | RMSE: {cost:.4f}")

    def predict(self, X):
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return X.dot(self.w)


def manual_adagrad_version():
    print("\n=== AdaGrad 线性回归 ===")
    train_data = load_training_data()
    X_train, y_train = create_dataset(train_data)

    # 简单归一化（对每一列做 z-score）
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std

    model = LinearRegressionAdaGrad(lr=0.8, epochs=30000)  # lr 调大一点收敛快
    model.fit(X_train, y_train)

    X_test = load_test_data()
    X_test = (X_test - mean) / std
    test_pred = model.predict(X_test)

    write_submission(test_pred, filename="predict.csv")
    print("AdaGrad 版本预测完成，已保存为 predict.csv")
    return test_pred


# ====================== 5. 结果写入 ======================
def write_submission(pred, filename="predict.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "value"])
        for i, v in enumerate(pred):
            writer.writerow([f"id_{i}", max(0, v)])  # PM2.5 不能为负数


# ====================== 主程序 ======================
if __name__ == "__main__":    
    # 只运行手动 AdaGrad 版本
    manual_pred = manual_adagrad_version()
