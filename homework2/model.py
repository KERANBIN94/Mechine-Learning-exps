import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm

# ==========================
# 1. 数据集定义（关键修复：强制 3 通道）
# ==========================
class VideoDataset(Dataset):
    def __init__(self, csv_file, video_dir, num_frames=16, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_path_col = self.labels.columns[0]
        filename = self.labels.iloc[idx][video_path_col]
        label = int(self.labels.iloc[idx, 1])
        video_path = os.path.join(self.video_dir, filename)
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if total_frames < self.num_frames:
            indices = np.arange(self.num_frames) % total_frames
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)

        # 1. 强制转为 3 通道 (T, H, W, 3)
        if frames.ndim == 3:
            frames = np.stack([frames] * 3, axis=-1)
        elif frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        elif frames.shape[-1] != 3:
            frames = frames[..., :3]

        # 2. 转为 Tensor 并调整为 (T, C, H, W)
        # 这样遍历 frames 时，每一帧就是 (C, H, W)，符合 transform 的要求
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

        # 3. 逐帧应用 transform
        if self.transform:
            # 此时 frame 的形状是 (C, H, W)，Normalize 不会再报错
            frames = torch.stack([self.transform(frame) for frame in frames])

        frames = frames.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)

        return frames, label


# ==========================
# 2. 模型定义
# ==========================
class VideoClassifier(nn.Module):
    def __init__(
        self, num_frames=16, hidden_dim=512, nhead=8, num_layers=2, num_classes=2
    ):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.feature_dim = hidden_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        features = self.feature_extractor(x)
        features = features.squeeze(-1).squeeze(-1)
        features = features.view(B, T, self.feature_dim)

        features = features + self.pos_embedding
        out = self.transformer(features)
        out = out.mean(dim=1)

        logits = self.classifier(out)
        return logits


# ==========================
# 3. 训练与评估函数
# ==========================
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    for frames, labels in tqdm(loader, desc="Training"):
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(frames)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Evaluating"):
            frames, labels = frames.to(device), labels.to(device)
            logits = model(frames)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
    return correct / len(loader.dataset)


# ==========================
# 4. 主函数
# ==========================
def main():
    root_dir = "homework2/data"
    train_video_dir = root_dir
    test_video_dir = root_dir

    train_csv = os.path.join(root_dir, "labels_train.csv")
    test_csv = os.path.join(root_dir, "labels_test.csv")

    num_frames = 8
    batch_size = 4
    num_epochs = 20
    learning_rate = 1e-4
    use_fp16 = True

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = VideoDataset(
        train_csv, train_video_dir, num_frames=num_frames, transform=transform
    )
    test_dataset = VideoDataset(
        test_csv, test_video_dir, num_frames=num_frames, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = VideoClassifier(num_frames=num_frames, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = (
        torch.amp.GradScaler("cuda") if use_fp16 and device.type == "cuda" else None
    )

    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        test_acc = eval_epoch(model, test_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_video_classifier.pth")

    print(f"训练完成！最佳测试准确率: {best_acc:.4f}")
    print("模型已保存为 best_video_classifier.pth")


if __name__ == "__main__":
    main()
