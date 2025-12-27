# used_car_regression_fixed_nan.py
# 最终修复版：解决 NaN 和 price 异常问题

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ==================== 配置路径 ====================
TRAIN_PATH = "homework1/used_car/used_car_train.csv"
TEST_PATH = "homework1/used_car/used_car_test.csv"
OUTPUT_DIR = "homework1/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def solve():
    print("Step 1: 正在读取数据...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    test_id = test["SaleID"]
    train_count = len(train)

    df = pd.concat([train, test], axis=0, ignore_index=True)

    print("Step 2: 特征工程与清洗（确保无 NaN）...")

    # 2.1 处理日期 → used_days
    def process_date(d):
        try:
            s = str(int(float(d)))
            if len(s) == 8 and s[4:6] == "00":
                s = s[:4] + "01" + s[6:]
            return s
        except:  # noqa: E722
            return np.nan

    df["regDate"] = df["regDate"].apply(process_date)
    df["creatDate"] = df["creatDate"].apply(process_date)

    df["regDate"] = pd.to_datetime(df["regDate"], format="%Y%m%d", errors="coerce")
    df["creatDate"] = pd.to_datetime(df["creatDate"], format="%Y%m%d", errors="coerce")

    df["used_days"] = (df["creatDate"] - df["regDate"]).dt.days

    # 填充 used_days（中位数更稳健）
    used_days_median = df["used_days"].median()
    if np.isnan(used_days_median):
        used_days_median = 1825  # 默认5年
    df["used_days"] = df["used_days"].fillna(used_days_median)

    # 2.2 power 清洗
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    power_median = df["power"].median()
    if np.isnan(power_median):
        power_median = 110  # 常见中位数
    df["power"] = df["power"].fillna(power_median)
    df["power"] = df["power"].clip(lower=0, upper=600)

    # 2.3 kilometer 清洗
    df["kilometer"] = pd.to_numeric(df["kilometer"], errors="coerce")
    km_median = df["kilometer"].median()
    df["kilometer"] = df["kilometer"].fillna(km_median)

    # 2.4 notRepairedDamage
    df["notRepairedDamage"] = df["notRepairedDamage"].replace("-", np.nan)
    df["notRepairedDamage"] = pd.to_numeric(df["notRepairedDamage"], errors="coerce")
    nr_mode = df["notRepairedDamage"].mode()
    nr_fill = nr_mode[0] if not nr_mode.empty else 0.0
    df["notRepairedDamage"] = df["notRepairedDamage"].fillna(nr_fill)

    # 2.5 类别特征填充（众数）
    cat_cols = ["model", "brand", "bodyType", "fuelType", "gearbox"]
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else -1
            df[col] = df[col].fillna(fill_val)

    # 2.6 删除无用列
    df = df.drop(["seller", "offerType"], axis=1, errors="ignore")

    # 2.7 One-Hot 编码
    one_hot_cols = ["bodyType", "fuelType", "gearbox"]
    one_hot_cols = [c for c in one_hot_cols if c in df.columns]
    df = pd.get_dummies(df, columns=one_hot_cols, dummy_na=False)

    # 2.8 构造特征列表（v_* 直接使用）
    v_cols = [f"v_{i}" for i in range(15)]
    one_hot_generated = [
        c for c in df.columns if c.startswith(("bodyType_", "fuelType_", "gearbox_"))
    ]

    feature_cols = ["power", "kilometer", "used_days"] + v_cols + one_hot_generated
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    train_X = X[:train_count]
    test_X = X[train_count:]

    # 2.9 关键：检查并确保 train_X 无 NaN
    if train_X.isnull().any().any():
        print("警告：train_X 中仍有 NaN，正在用中位数填充...")
        train_X = train_X.fillna(train_X.median())
    if test_X.isnull().any().any():
        test_X = test_X.fillna(train_X.median())  # 用训练集统计值填充测试集

    # 目标变量清洗（price 必须 > 0）
    train_price = train["price"].copy()
    print(
        f"原始 price 统计: min={train_price.min()}, max={train_price.max()}, 零值数量={(train_price <= 0).sum()}"
    )

    # 将 price <= 0 的样本替换为中位数（或删除，但这里替换更安全）
    train_price.median()
    train_price = train_price.clip(lower=1)  # 至少为1
    train_y = np.log1p(train_price)  # 现在安全

    print("Step 3: 标准化 + 模型训练...")
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    model = Ridge(alpha=20.0, random_state=42)
    model.fit(train_X_scaled, train_y)

    # 训练集评估
    train_pred = model.predict(train_X_scaled)
    train_mae = mean_absolute_error(np.expm1(train_y), np.expm1(train_pred))
    print(f"训练集 MAE: {train_mae:.2f}")

    print("Step 4: 生成提交文件...")
    test_pred = model.predict(test_X_scaled)
    test_price = np.expm1(test_pred)
    test_price = np.maximum(test_price, 50)  # 避免过低

    submission = pd.DataFrame({"SaleID": test_id, "price": test_price})
    submit_path = os.path.join(OUTPUT_DIR, "submission_regression.csv")
    submission.to_csv(submit_path, index=False)
    print(f"预测完成！文件已保存: {submit_path}")


if __name__ == "__main__":
    solve()
