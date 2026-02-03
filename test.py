import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import cv2
import math

from math import sqrt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

set_seed(42)

print(f"Torch: {torch.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Numpy: {np.__version__}")
print(f"OpenCV: {cv2.__version__}")

### Model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1x1 = nn.InstanceNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv1x1(out)
        out = self.bn1x1(out)
        
        out += shortcut
        return self.relu(out)
    
class MetadataBlock(nn.Module):
    def __init__(self, in_dim=3, feat_dim=512):
        super().__init__()
        self.importance_mlp = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.Sigmoid() 
        )
        
    def forward(self, feat, meta):
        importance = self.importance_mlp(meta).unsqueeze(-1).unsqueeze(-1)
        return feat * importance 

class AWBModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(AWBModel, self).__init__()

        # --- Энкодер (сжимающий путь) ---
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Бутылочное горлышко (Bottleneck) ---
        self.bottleneck = ConvBlock(256, 512)
        
        # --- Декодер (расширяющий путь) ---
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.dec3 = ConvBlock(512, 256) # 256 (от upconv) + 256 (от skip) = 512
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.dec2 = ConvBlock(256, 128) # 128 (от upconv) + 128 (от skip) = 256
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upconv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.dec1 = ConvBlock(128, 64)  # 64 (от upconv) + 64 (от skip) = 128
        
        # --- Финальный слой для получения выходного изображения ---
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_activation = nn.Sigmoid() 
        self.meta_datablock = MetadataBlock() 

    def calculate_white_point(self, img: torch.Tensor, weight_map: torch.Tensor):
        return (img * weight_map).mean(dim=[2, 3])

    def forward(self, x, metadata):
        # metadata: (B, 3)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        b = self.bottleneck(p3)
        b = self.meta_datablock(b, metadata)

        d3 = self.upsample3(b)
        d3 = self.upconv3(d3)
        d3 = torch.cat((d3, e3), dim=1) 
        d3 = self.dec3(d3)
        
        d2 = self.upsample2(d3)
        d2 = self.upconv2(d2)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upsample1(d2)
        d1 = self.upconv1(d1)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        out = self.final_activation(out)
        return self.calculate_white_point(x, out)
    

### Dataset
class CustomDataset(Dataset):
    def __init__(self, path: str, part='train', fold: int = -1):
        self.data = pd.read_csv(path)
        self.part = part
        self.metadata = pd.read_csv('metadata.csv')

        self.transform_val = transforms.Compose([
            transforms.Resize(size=(224, 224), antialias=True),
        ])

        self.full_images = self.data['names'].values.tolist()

    def __len__(self):
        return len(self.full_images)

    def read_image(self, path2img: str):
        img = cv2.imread(str(path2img), cv2.IMREAD_UNCHANGED)
        img = img / np.max(img)

        if img.shape[-1] == 3:
            img = img[..., ::-1]

        return img.copy()

    def get_metadata(self, img_path: str):
        d = self.metadata[self.metadata['names'] == img_path].copy()
        d['ExposureTime'] = d['ExposureTime'].apply(lambda x: 1/float(x.split('/')[-1]))
        return torch.tensor(d[['ExposureTime', 'ISO', 'LightValue']].values[0], dtype=torch.float32)

    def __getitem__(self, idx):
        img_path = self.full_images[idx]
        img = self.read_image(img_path)
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img).float()

        metadata = self.get_metadata(img_path)
        img_tensor = self.transform_val(img_tensor)
        return img_tensor, img_path, metadata

test_dataset = CustomDataset("test.csv", part="test")
test_data = DataLoader(test_dataset, batch_size=16, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for fold in range(5):
    fold = fold + 1

    model = AWBModel()
    model = model.to(device)

    model.load_state_dict(torch.load(f'best_grey_world_{fold}.pth'))
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_imgs, batch_ids, batch_metadata in tqdm(test_data, desc="Генерация предсказаний"):
            batch_imgs = batch_imgs.to(device).float()
            batch_metadata = batch_metadata.to(device).float()
            batch_outputs = model(batch_imgs, batch_metadata)

            # Нормализация векторов white point
            batch_outputs = batch_outputs / batch_outputs.norm(dim=1, keepdim=True)

            batch_predictions = [
                {
                    'names': img_id,
                    'wp_r': float(output[0]),
                    'wp_g': float(output[1]),
                    'wp_b': float(output[2])
                }
                for img_id, output in zip(batch_ids, batch_outputs)
            ]

            predictions.extend(batch_predictions)

    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(f'submission{fold}.csv', index=False)
    print(f'Сохранён самбит номер: {fold}') 

sub1 = pd.read_csv('submission1.csv')
sub2 = pd.read_csv('submission2.csv')
sub3 = pd.read_csv('submission3.csv')
sub4 = pd.read_csv('submission4.csv')
sub5 = pd.read_csv('submission5.csv')

sub_avg = sub1.copy()
sub_avg['wp_r'] = (sub1['wp_r'] + sub2['wp_r'] + sub3['wp_r'] + sub4['wp_r'] + sub5['wp_r']) / 5
sub_avg['wp_g'] = (sub1['wp_g'] + sub2['wp_g'] + sub3['wp_g'] + sub4['wp_g'] + sub5['wp_g']) / 5
sub_avg['wp_b'] = (sub1['wp_b'] + sub2['wp_b'] + sub3['wp_b'] + sub4['wp_b'] + sub5['wp_b']) / 5

sub_avg.to_csv('submission.csv', index=False)
print('Ансамбыль САМБИТ СОХРАНЕН.')
print('Отправлять надо: submission.csv!!!')