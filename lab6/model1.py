import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from pytorch_msssim import SSIM  # 新增：用于结构相似性损失

# ==================== 创建保存目录 ====================
save_dir = "lab6/output_model1/results"
os.makedirs(save_dir, exist_ok=True)

# ==================== 数据预处理 ====================
IMG_SIZE = 384

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
    ]
)

# ==================== 超参数 ====================
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 66
AE_EPOCHS = 50
CNN_EPOCHS = 40
LR = 0.0005
NOISE_FACTOR = 0.1  # 降低噪声强度，避免过度平滑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "lab6/data/covid19"

train_dataset = datasets.ImageFolder(root=data_dir + "/train", transform=transform)
test_dataset = datasets.ImageFolder(root=data_dir + "/noisy_test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)


# ==================== 模型定义 ====================
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


autoencoder = Autoencoder().to(device)
cnn = CNN(num_classes=3).to(device)

# ==================== 损失函数与优化器（关键：混合损失）===================
mse_loss = nn.MSELoss()
ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)  # SSIM 值越大越好


def ae_loss_function(recon_x, x):
    # SSIM 需要 [0,1] 范围的图像
    x_ssim = (x + 1) / 2
    recon_x_ssim = (recon_x + 1) / 2

    mse = mse_loss(recon_x, x)
    ssim_val = ssim_module(recon_x_ssim, x_ssim)
    ssim_loss = 1 - ssim_val  # 转为 loss

    total_loss = mse + 0.5 * ssim_loss  # 可调整权重（如 0.5、1.0）
    return total_loss


optimizer_ae = optim.Adam(autoencoder.parameters(), lr=LR, weight_decay=1e-4)

criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-4)
scheduler_cnn = ReduceLROnPlateau(optimizer_cnn, mode="max", factor=0.5, patience=5)


# ==================== 工具函数 ====================
def add_noise(inputs, noise_factor=NOISE_FACTOR):
    noise = torch.randn_like(inputs) * noise_factor
    return torch.clamp(inputs + noise, -1.0, 1.0)


# 存储训练过程
ae_losses = []
train_losses_cnn = []
train_accs_cnn = []
test_losses_cnn = []
test_accs_cnn = []


# ==================== 训练自编码器 ====================
def train_ae(epoch):
    autoencoder.train()
    train_loss = 0
    for data, _ in tqdm(train_loader, desc=f"AE Epoch {epoch}"):
        data = data.to(device)
        noisy_data = add_noise(data)

        optimizer_ae.zero_grad()
        output = autoencoder(noisy_data)
        loss = ae_loss_function(output, data)  # 使用混合损失
        loss.backward()
        optimizer_ae.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    ae_losses.append(avg_loss)
    print(f"AE Epoch {epoch} - Avg Loss: {avg_loss:.6f}")


# ==================== 训练CNN ====================
def train_cnn(epoch):
    cnn.train()
    autoencoder.eval()
    train_loss = 0
    correct = 0
    total = 0

    for data, labels in tqdm(train_loader, desc=f"CNN Train Epoch {epoch}"):
        data, labels = data.to(device), labels.to(device)
        noisy_data = add_noise(data)

        optimizer_cnn.zero_grad()
        with torch.no_grad():
            denoised = autoencoder(noisy_data)
        outputs = cnn(denoised)

        loss = criterion_cnn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1.0)
        optimizer_cnn.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = train_loss / len(train_loader)
    acc = 100.0 * correct / total
    train_losses_cnn.append(avg_loss)
    train_accs_cnn.append(acc)
    print(f"CNN Epoch {epoch} - Train Loss: {avg_loss:.6f} | Train Acc: {acc:.2f}%")
    return acc


def test_cnn(epoch):
    cnn.eval()
    autoencoder.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            denoised = autoencoder(data)
            outputs = cnn(denoised)

            loss = criterion_cnn(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    acc = 100.0 * correct / total
    test_losses_cnn.append(avg_loss)
    test_accs_cnn.append(acc)
    print(f"CNN Epoch {epoch} - Test Loss: {avg_loss:.6f} | Test Acc: {acc:.2f}%")
    return acc


print("=== Training Autoencoder ===")
for epoch in range(1, AE_EPOCHS + 1):
    train_ae(epoch)

os.makedirs("lab6/output_model1", exist_ok=True)
torch.save(autoencoder.state_dict(), "lab6/output_model1/autoencoder.pth")

print("\n=== Training CNN ===")
best_test_acc = 0
for epoch in range(1, CNN_EPOCHS + 1):
    train_acc = train_cnn(epoch)
    test_acc = test_cnn(epoch)
    scheduler_cnn.step(test_acc)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(cnn.state_dict(), "lab6/output_model1/cnn.pth")

# ==================== 保存可视化结果 ====================
autoencoder.eval()
with torch.no_grad():

    def denorm(img):
        return (img * 0.5 + 0.5).clamp(0, 1)

    # 1. 训练集去噪演示
    train_iter = iter(train_loader)
    clean_data, _ = next(train_iter)
    clean_data = clean_data[:8].to(device)
    noisy_demo = add_noise(clean_data)
    denoised_demo = autoencoder(noisy_demo)

    fig, axes = plt.subplots(3, 8, figsize=(20, 7))
    for i in range(8):
        axes[0, i].imshow(denorm(clean_data[i]).cpu().squeeze(), cmap="gray")
        axes[0, i].set_title("Original")
        axes[1, i].imshow(denorm(noisy_demo[i]).cpu().squeeze(), cmap="gray")
        axes[1, i].set_title("Noisy")
        axes[2, i].imshow(denorm(denoised_demo[i]).cpu().squeeze(), cmap="gray")
        axes[2, i].set_title("Denoised")
        for ax in axes[:, i]:
            ax.axis("off")
    plt.suptitle("Autoencoder Denoising Effect (Demo on Clean Train Images)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "denoising_demo_train.png"))
    plt.close(fig)
    print(f"去噪演示图已保存: {save_dir}/denoising_demo_train.png")

    # 2. 测试集去噪效果
    test_iter = iter(test_loader)
    noisy_test, _ = next(test_iter)
    noisy_test = noisy_test[:8].to(device)
    denoised_test = autoencoder(noisy_test)

    fig2, axes2 = plt.subplots(2, 8, figsize=(20, 5))
    for i in range(8):
        axes2[0, i].imshow(denorm(noisy_test[i]).cpu().squeeze(), cmap="gray")
        axes2[0, i].set_title("Noisy Test")
        axes2[1, i].imshow(denorm(denoised_test[i]).cpu().squeeze(), cmap="gray")
        axes2[1, i].set_title("Denoised")
        for ax in axes2[:, i]:
            ax.axis("off")
    plt.suptitle("Denoising on Noisy Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "denoising_test.png"))
    plt.close(fig2)
    print(f"测试集去噪图已保存: {save_dir}/denoising_test.png")

    # 3. CNN 训练曲线
    fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(ae_losses)
    ax1.set_title("Autoencoder Training Loss (MSE+SSIM)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(train_losses_cnn, label="Train Loss")
    ax2.plot(test_losses_cnn, label="Test Loss")
    ax2.legend()
    ax2.set_title("CNN Loss")
    ax2.set_xlabel("Epoch")

    ax3.plot(train_accs_cnn, label="Train Acc")
    ax3.plot(test_accs_cnn, label="Test Acc")
    ax3.legend()
    ax3.set_title("CNN Accuracy")
    ax3.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cnn_training_curves.png"))
    plt.close(fig3)
    print(f"训练曲线已保存: {save_dir}/cnn_training_curves.png")

print(f"\nBest Test Accuracy: {best_test_acc:.2f}%")
print(f"所有结果图像已保存至: {save_dir}/")
