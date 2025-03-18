import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms

class DogsVsWolvesDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        super(DogsVsWolvesDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.dog_files = os.listdir(os.path.join(root_dir, 'dogs'))
        self.wolf_files = os.listdir(os.path.join(root_dir, 'wolves'))

        self.dog_files = [(x, 'dog') for x in self.dog_files]
        self.wolf_files = [(x, 'wolf') for x in self.wolf_files]

        self.file_list = self.dog_files + self.wolf_files
        random.seed(123)
        random.shuffle(self.file_list)

        # Định nghĩa transform cho ảnh
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # Resize về 224x224
            transforms.ToTensor(),  # Chuyển thành Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo ImageNet
        ])

        self.label_to_class = {'dog': 0, 'wolf': 1}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_name, image_label = self.file_list[idx]
        image_path = os.path.join(self.root_dir, 'dogs', image_name) if image_label == 'dog' \
            else os.path.join(self.root_dir, 'wolves', image_name)
        
        img = Image.open(image_path).convert('RGB')  # Đảm bảo ảnh có 3 kênh (RGB)
        img = self.transform(img)  # Áp dụng transform

        return img, self.label_to_class[image_label]  # Trả về tensor ảnh và label
