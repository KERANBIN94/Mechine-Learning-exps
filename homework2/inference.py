import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ==========================================
# 1. 引用你之前的模型定义 (确保与训练时一致)
# ==========================================
class VideoClassifier(nn.Module):
    def __init__(self, num_frames=8, hidden_dim=512, nhead=8, num_layers=2, num_classes=2):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None) # 推理时不需要再下载预训练权重
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = hidden_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        features = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        features = features.view(B, T, self.feature_dim)
        features = features + self.pos_embedding
        out = self.transformer(features).mean(dim=1)
        return self.classifier(out)

# ==========================================
# 2. 推理专用工具函数
# ==========================================
def process_video_for_inference(video_path, num_frames=8, transform=None):
    """读取视频并返回适合分类模型的 Tensor 和 适合 BLIP 的 PIL 图片"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()

    # 处理分类用的 Tensor
    proc_frames = torch.from_numpy(frames).float() / 255.0
    proc_frames = proc_frames.permute(3, 0, 1, 2) # (C, T, H, W)
    
    if transform:
        # 逐帧应用 transform (注意之前的维度修复逻辑)
        # 先转为 (T, C, H, W) 迭代，再转回 (C, T, H, W)
        t_frames = proc_frames.permute(1, 0, 2, 3)
        t_frames = torch.stack([transform(f) for f in t_frames])
        proc_frames = t_frames.permute(1, 0, 2, 3)

    # 取中间一帧作为 BLIP 的输入图片
    mid_frame = frames[num_frames // 2]
    pil_image = Image.fromarray(mid_frame)
    
    return proc_frames.unsqueeze(0), pil_image # 增加 Batch 维度

# ==========================================
# 3. 主推理流程
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 配置参数 ---
    root_dir = "homework2/data"
    test_csv = os.path.join(root_dir, "labels_test.csv")
    video_dir = root_dir # CSV 里已经包含子目录名
    model_path = "homework2/output/best_video_classifier.pth"
    num_frames = 8

    # --- 1. 加载分类模型 ---
    classifier = VideoClassifier(num_frames=num_frames).to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()

    # --- 2. 加载 BLIP 模型 ---
    print("Loading BLIP model (this may take a while)...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    # --- 3. 数据预处理 ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 4. 开始推理 ---
    test_df = pd.read_csv(test_csv)
    results = []

    print(f"Starting inference on {len(test_df)} videos...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        video_rel_path = row[test_df.columns[0]]
        video_path = os.path.join(video_dir, video_rel_path)
        
        try:
            # 读取并处理
            input_tensor, pil_img = process_video_for_inference(video_path, num_frames, transform)
            input_tensor = input_tensor.to(device)

            # A. 执行分类
            with torch.no_grad():
                logits = classifier(input_tensor)
                pred_label = torch.argmax(logits, dim=1).item()

            # B. 执行文字描述 (BLIP)
            inputs = blip_processor(pil_img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = blip_model.generate(**inputs, max_new_tokens=20)
                caption = blip_processor.decode(out[0], skip_special_tokens=True)

            results.append({
                "video_path": video_rel_path,
                "predicted_label": pred_label,
                "blip_caption": caption
            })

        except Exception as e:
            print(f"Error processing {video_rel_path}: {e}")

    # --- 5. 保存结果 ---
    output_df = pd.DataFrame(results)
    output_df.to_csv("homework2/output/final_results.csv", index=False)
    print("\nInference complete! Results saved to 'final_results.csv'.")
    print(output_df.head())

if __name__ == "__main__":
    main()