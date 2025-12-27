import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Reshape, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

print("TensorFlow版本:", tf.__version__)

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 预处理
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

y_test_cls = y_test  # 真实类别（整数）

# 参数配置
img_size = 28
img_size_flat = 784
img_shape = (28, 28)
img_shape_full = (28, 28, 1)
num_channels = 1
num_classes = 10

print(img_size, img_size_flat, img_shape, img_shape_full, num_classes, num_channels)


# 绘图函数（正常图像）
def plot_images(images, cls_true, cls_pred=None, num_images=9):
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i].reshape(img_shape), cmap="binary")
            if cls_pred is None:
                xlabel = f"True: {cls_true[i]}"
            else:
                xlabel = f"True: {cls_true[i]}, Pred: {cls_pred[i]}"
            ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


# 测试绘图
plot_images(x_test[0:9], y_test_cls[0:9])


# 错分图像函数
def plot_example_errors(cls_pred, correct):
    incorrect = not correct
    images = x_test[incorrect]
    cls_pred_err = cls_pred[incorrect]
    cls_true_err = y_test_cls[incorrect]
    plot_images(images[0:9], cls_true_err[0:9], cls_pred_err[0:9])


# ------------------ 序列模型 ------------------
model = Sequential()
model.add(Input(shape=(img_size_flat,)))
model.add(Reshape(img_shape_full))
model.add(Conv2D(16, 5, padding="same", activation="relu", name="layer_conv1"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(36, 5, padding="same", activation="relu", name="layer_conv2"))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.summary()

optimizer = Adam(learning_rate=1e-3)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# 训练
model.fit(x_train.reshape(-1, img_size_flat), y_train_cat, epochs=1, batch_size=128)

# 评估
result = model.evaluate(x_test.reshape(-1, img_size_flat), y_test_cat)
for name, value in zip(model.metrics_names, result):
    print(name, value)

# 预测前9张
pred = model.predict(x_test[0:9].reshape(-1, img_size_flat))
cls_pred = np.argmax(pred, axis=1)
plot_images(x_test[0:9], y_test_cls[0:9], cls_pred)

# 错分
pred_all = model.predict(x_test.reshape(-1, img_size_flat))
cls_pred_all = np.argmax(pred_all, axis=1)
correct = cls_pred_all == y_test_cls
plot_example_errors(cls_pred_all, correct)

# ------------------ 功能模型 + 保存加载 + 可视化 ------------------

# 示例：保存模型
path_model = "my_mnist_model.keras"
model.save(path_model)  
