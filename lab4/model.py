import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# ==========================================
# 1. 环境检查与设置
# ==========================================
print("TensorFlow version:", tf.__version__)

# ==========================================
# 2. 数据预处理模块
# ==========================================
print("\n--- Step 2: Data Preprocessing ---")
# 2.1 加载训练集和测试集
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.mnist.load_data()

# 2.2 重塑形状并转换为 float32 (展平为 784 维向量)
X_train = X_train_raw.reshape(60000, 784).astype('float32')
X_test = X_test_raw.reshape(10000, 784).astype('float32')

# 2.3 归一化 (将像素值缩放到 0-1 之间)
X_train /= 255.0
X_test /= 255.0

# 2.4 One-hot 编码 (将标签 0-9 转换为 10 维二进制向量)
Y_train = keras.utils.to_categorical(y_train_raw, num_classes=10)
Y_test = keras.utils.to_categorical(y_test_raw, num_classes=10)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# ==========================================
# 3. 基础感知机模型 (Baseline Model)
# ==========================================
print("\n--- Step 3: Training Baseline Model (Single Layer) ---")
base_model = Sequential([
    Dense(units=10, input_shape=(784,), activation='softmax', name='Output_Layer')
])

base_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 训练基础模型 200 epochs
base_history = base_model.fit(X_train, Y_train, batch_size=128, epochs=200, 
                              verbose=0, validation_split=0.2)
base_score = base_model.evaluate(X_test, Y_test, verbose=0)
print(f"Base Model Test Accuracy: {base_score[1]:.4f}")

# ==========================================
# 4. 优化神经网络 (Optimized MLP)
# ==========================================
print("\n--- Step 4: Training Optimized Model (Multi-Layer) ---")
# 架构：输入(784) -> 隐藏1(128, ReLU) -> 隐藏2(128, ReLU) -> 输出(10, Softmax)
opt_model = Sequential([
    Dense(128, input_shape=(784,), activation='relu', name='Hidden_1'),
    Dense(128, activation='relu', name='Hidden_2'),
    Dense(10, activation='softmax', name='Output_Layer')
])

opt_model.summary()

opt_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 训练优化模型 20 epochs
opt_history = opt_model.fit(X_train, Y_train, batch_size=128, epochs=20, 
                            verbose=1, validation_split=0.2)
opt_score = opt_model.evaluate(X_test, Y_test, verbose=0)
print(f"Optimized Model Test Accuracy: {opt_score[1]:.4f}")

# ==========================================
# 5. 结果可视化模块 (产出报告图片)
# ==========================================
print("\n--- Step 5: Generating Report Images ---")

def generate_report_plots(history, model, X_test_data, y_true_labels):
    # 适配 TF 1.14.0 的键名：'acc' 和 'val_acc'
    plt.figure(figsize=(12, 5))
    
    # 5.1 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'], label='Train Acc', color='blue')
    plt.plot(history.history['val_acc'], label='Val Acc', color='orange', linestyle='--')
    plt.title('Optimized Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 5.2 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red')
    plt.plot(history.history['val_loss'], label='Val Loss', color='darkred', linestyle='--')
    plt.title('Optimized Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lab4/output/training_curves.png')
    print("Saved: training_curves.png")
    
    # 5.3 绘制错误样本分析
    # 注意：TF 1.14 的 predict 返回的是概率分布
    predictions = model.predict(X_test_data)
    pred_labels = np.argmax(predictions, axis=1)
    error_indices = np.where(pred_labels != y_true_labels)[0]
    
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(error_indices[:6]):
        plt.subplot(1, 6, i+1)
        plt.imshow(X_test_data[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True:{y_true_labels[idx]}\nPred:{pred_labels[idx]}", color='red')
        plt.axis('off')
    
    plt.suptitle("Error Sample Analysis (Top 6 Miss-classified)")
    plt.tight_layout()
    plt.savefig('lab4/output/error_analysis.png')
    print("Saved: error_analysis.png")
    plt.show()

# 执行绘图
generate_report_plots(opt_history, opt_model, X_test, y_test_raw)