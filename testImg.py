import torch
from efficientNetModel import *
from config import *
from utils import *
from PIL import Image
import numpy as np

# Load args từ config
args = cArgs()

# Load model đã huấn luyện
model = EfficientNet('b0', num_classes=1).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Load trọng số đã lưu
load_checkpoint(args.save_path, model, optimizer, args.lr)


# Đưa model vào chế độ đánh giá
model.eval()


from PIL import Image
import torch
import torchvision.transforms as transforms

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh về kích thước phù hợp với mô hình
        transforms.ToTensor(),          # Chuyển ảnh thành Tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo ImageNet
    ])

    image = Image.open(image_path).convert("RGB")  # Đọc ảnh và chuyển thành RGB
    image = transform(image)  # Áp dụng transform
    image = image.unsqueeze(0)  # Thêm batch dimension (1, C, H, W) để phù hợp với mô hình
    
    return image


def predict(image_path):
    img = preprocess_image(image_path)

    # Dự đoán
    with torch.no_grad():
        output = model(img).squeeze(1)

    # Áp dụng sigmoid vì BCEWithLogitsLoss đã dùng khi train
    prob = torch.sigmoid(output).item()

    # Xác định lớp dự đoán
    pred_class = "Dog" if prob <= 0.5 else "Wolf"

    if(prob <= 0.5):
        print(f"Predicted Class: {pred_class} (Confidence: {(1 - prob) * 100:.2f}%)")
    else:
        print(f"Predicted Class: {pred_class} (Confidence: {prob * 100:.2f}%)")

import tkinter as tk
from tkinter import filedialog

# Mở hộp thoại chọn file
def select_image():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh để dự đoán",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )
    return file_path

# Chạy chọn ảnh và dự đoán
image_path = select_image()
if image_path:
    predict(image_path)
else:
    print("Không có ảnh nào được chọn.")
