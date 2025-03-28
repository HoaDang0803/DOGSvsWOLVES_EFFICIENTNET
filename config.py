import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2


class cArgs():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = 3e-4  # learning rate
        self.batch_size = 64
        self.train_dir = 'processed_data/train'
        self.val_dir = 'processed_data/val'
        self.save_model = True
        self.load_checkpoint = False
        self.save_path = 'checkpoint.pth.tar'
        self.load_path = 'checkpoint1.pth.tar'
        self.epochs = 10
        self.save_frequency = 5
        self.validation_frequency = 5


label_to_class = {'dog': 0, 'wolf': 1}


transform_dogs = A.Sequential(
    [
        A.Resize(224, 224),  # for efficientnet b0
        A.Normalize(mean=[-0.0969, -0.0358, -0.0497], std=[1.1164, 1.1151, 1.1473], max_pixel_value=255.0),
        ToTensorV2()
    ]
)

transform_wolves = transform_dogs