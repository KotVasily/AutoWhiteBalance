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

### Metric and Common
HIST_BINS = [116, 100]
HIST_RANGE_FLAT = [-sqrt(3), sqrt(3), -1, 2] 
HIST_VERT_PADD, HIST_HORR_PADD = 6, 14

def read_hist(path2hist):
    hist = cv2.imread(str(path2hist), cv2.IMREAD_UNCHANGED)
    hist = hist[HIST_HORR_PADD:-HIST_HORR_PADD,
                HIST_VERT_PADD:-HIST_VERT_PADD]
    hist = hist.astype(np.float32) / 255
    return hist.astype(np.float32)
    
def get_coords(hist: np.ndarray):
    alphas = np.linspace(HIST_RANGE_FLAT[3], HIST_RANGE_FLAT[2], HIST_BINS[1])
    betas = np.linspace(HIST_RANGE_FLAT[0], HIST_RANGE_FLAT[1], HIST_BINS[0])
    alphas, betas = np.meshgrid(alphas, betas, indexing='ij')
    grid = np.stack((alphas, betas), axis=-1)
    return grid[hist.astype(bool)].T


def rgb2chrom(rgb: np.ndarray):
    rgb = rgb.astype(np.float32)
    r, g, b  = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Избегаем деления на ноль добавляя малое значение к знаменателю
    denominator = r + g + b + 1e-10
    
    alpha = (2 * b  - (r + g)) / denominator
    beta = sqrt(3) * (r - g) / denominator
    
    alpha = np.where(np.isinf(alpha), 0, alpha)
    beta = np.where(np.isinf(beta), 0, beta)
    return np.stack((alpha, beta), axis=0)


def dist2hist(hist: np.ndarray, white_points: np.ndarray) -> float:
    hist_coords = get_coords(hist)
    if hist_coords.shape[1] == 0:
        alpha_min, alpha_max = HIST_RANGE_FLAT[2], HIST_RANGE_FLAT[3]
        beta_min, beta_max = HIST_RANGE_FLAT[0], HIST_RANGE_FLAT[1]
        diag = sqrt((alpha_max - alpha_min)**2 + (beta_max - beta_min)**2)
        return float(diag)

    chrom = rgb2chrom(white_points)
    diff = hist_coords - chrom[:, np.newaxis]
    min_dist = np.linalg.norm(diff, ord=2, axis=0).min()
    return float(min_dist)


def angle(gt: np.ndarray, pred: np.ndarray):
    '''
    gt, pred arrays shape of Nx3
    '''
    gt_norm = np.linalg.norm(gt, ord=2, axis=-1) + 1e-10
    pred_norm = np.linalg.norm(pred, ord=2, axis=-1) + 1e-10
    dot_prod = (gt * pred).sum(axis=-1)
    cosine = dot_prod / (gt_norm * pred_norm)
    angle = np.arccos(np.clip(cosine, -1, 1))
    return np.rad2deg(angle).mean()


def final_metric(pred_wp: np.ndarray, gt_wp: np.ndarray, hist: np.ndarray,
                 weight1: float = 1, weight2: float = 1):
    angle_value = angle(gt_wp, pred_wp)
    dist_value = dist2hist(hist, pred_wp)
    return (weight1 * dist_value + weight2 * angle_value).item()


def score(pred_wps: np.ndarray, gt_wps: np.ndarray, hists: list[np.ndarray]) -> float:
    N = pred_wps.shape[0]
    total = 0.0
    for i in range(N):
        dist_hist = final_metric(pred_wps[i], gt_wps[i], hists[i], weight1=1, weight2=0.1)
        total += dist_hist

    avg = total / N
    return 1 / (1 + avg)

class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = F.normalize(prediction, p=2, dim=1, eps=1e-8)
        target_norm = F.normalize(target, p=2, dim=1, eps=1e-8)
        cos_theta = torch.sum(pred_norm * target_norm, dim=1).clamp(-1.0, 1.0)
        sin_theta = torch.norm(torch.cross(pred_norm, target_norm, dim=1), dim=1).clamp_min(1e-8)
        angle = torch.atan2(sin_theta, cos_theta)
        angle_deg = torch.rad2deg(angle)
        return angle_deg.mean()

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
    
model = AWBModel()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Общее количество параметров: {total_params / 1e6:.2f}M")
print(f"Обучаемых параметров: {trainable_params / 1e6:.2f}M")

### Dataset
def _temp_to_rgb(temp_kelvin: float):
    """Convert color temperature in Kelvin to normalized RGB multipliers."""
    t = temp_kelvin / 100.0

    # Red
    if t <= 66:
        red = 255
    else:
        red = t - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = max(0, min(255, red))

    # Green
    if t <= 66:
        green = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        green = t - 60
        green = 288.1221695283 * (green ** -0.0755148492)
    green = max(0, min(255, green))

    # Blue
    if t >= 66:
        blue = 255
    elif t <= 19:
        blue = 0
    else:
        blue = t - 10
        blue = 138.5177312231 * math.log(blue) - 305.0447927307
        blue = max(0, min(255, blue))

    rgb = torch.tensor([red, green, blue], dtype=torch.float32) / 255.0
    return rgb / rgb.mean()  # нормализация по яркости

class CustomDataset(Dataset):
    def __init__(self, path: str, part='train', fold: int = -1):
        self.data = pd.read_csv(path)
        self.part = part
        self.metadata = pd.read_csv('metadata.csv')

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize(size=(224, 224), antialias=True),
        ])

        self.full_images = self.data['names'].values.tolist()
        self.full_hists = self.data['names'].apply(lambda x: x.split('/')[-1]).values.tolist()

        if part == 'test' and fold == -1:
            self.image_files = self.full_images
            self.path_hist = self.full_hists
        else:
            val_images = self.full_images[fold-100:fold]
            val_hists = self.full_hists[fold-100:fold]

            if part == 'val':
                self.image_files = val_images
                self.path_hist = val_hists
            if part == 'train':
                self.image_files = [p for p in self.full_images if p not in val_images]
                self.path_hist = [p for p in self.full_hists if p not in val_hists]

    def __len__(self):
        return len(self.image_files)

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

    def augmentation(self, img_tensor: torch.Tensor, white_point_tensor: torch.Tensor):
        if torch.rand(1).item() > 0.1:
            return img_tensor, white_point_tensor

        temp = torch.empty(1).uniform_(2500, 7500).item()
        rgb_gain = _temp_to_rgb(temp)

        img_tensor_aug = torch.clamp(img_tensor * rgb_gain.view(3, 1, 1), 0, 1)
        white_point_tensor_aug = white_point_tensor * rgb_gain

        img_tensor_aug = img_tensor_aug / img_tensor_aug.max()
        return img_tensor_aug, white_point_tensor_aug

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = self.read_image(img_path)
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img).float()

        metadata = self.get_metadata(img_path)

        if self.part == 'train' or self.part == 'val':
            white_point = self.data.loc[self.data['names'] == img_path, ['wp_r', 'wp_g', 'wp_b']].values[0]
            white_point_tensor = torch.from_numpy(white_point).float()

            hist_path = self.path_hist[idx]
            hist = read_hist(f'train_histograms/{hist_path}')
            hist_tensor = torch.from_numpy(hist).float()

            if self.part == 'train':
                img_tensor = self.transform_train(img_tensor)
                img_tensor, white_point_tensor = self.augmentation(img_tensor, white_point_tensor)
                return img_tensor, white_point_tensor, hist_tensor, img_path, metadata
            else:
                img_tensor = self.transform_val(img_tensor)
                return img_tensor, white_point_tensor, hist_tensor, img_path, metadata
        else:
            img_tensor = self.transform_val(img_tensor)
            return img_tensor, img_path, metadata

def get_data_loader(batch_size: int, fold: int = -1):
    train_dataset = CustomDataset("train.csv", part="train", fold=fold)
    test_dataset = CustomDataset("test.csv", part="test")
    val_dataset = CustomDataset("train.csv", part="val", fold=fold)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader 

### Train 
best_kfolds = []

for i in [1, 2, 3, 4, 5]:
    set_seed(42)
    fold = i * 100

    print('='*60)
    print(f'FOLD: {i}')
    print('='*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AWBModel()
    model = model.to(device)
    criterion = AngularLoss()
    criterion.to(device)
    train_data, val_data, test_data = get_data_loader(batch_size=16, fold=fold)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0002)

    best_metric_score = 0.0
    num_epochs = 35
    early_stopping_rounds = 10
    no_improvement_count = 0
    accumulation_steps = 2

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, white_points, hist, _, metadata) in enumerate(tqdm(train_data)):
            white_points = white_points.to(device).float()
            images = images.to(device).float()
            hist = hist.to(device).float()
            metadata = metadata.to(device).float()

            outputs = model(images, metadata)
            loss = criterion(outputs, white_points)
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * images.size(0) * accumulation_steps

        train_loss = train_loss / len(train_data.dataset)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_gt = []
        all_hists = []

        with torch.no_grad():
            for images, white_points, hist, _, metadata in tqdm(val_data):
                images = images.to(device).float()
                white_points = white_points.to(device).float()
                hist = hist.to(device).float()
                metadata = metadata.to(device).float()

                outputs = model(images, metadata)
                loss = criterion(outputs, white_points)
                val_loss += loss.item() * images.size(0)

                all_preds.append(outputs.detach().cpu().numpy())
                all_gt.append(white_points.detach().cpu().numpy())
                all_hists.append(hist.detach().cpu().numpy())

        total_val_samples = len(val_data.dataset)
        val_loss = val_loss / total_val_samples

        all_preds = np.vstack(all_preds)
        all_gt = np.vstack(all_gt)
        all_hists = np.vstack(all_hists)

        metric_score = score(all_preds, all_gt, all_hists)

        print(f'Эпоха {epoch+1}/{num_epochs}, Потери при обучении: {train_loss}, Потери при валидации: {val_loss}, Метрика: {metric_score:.4f}')

        if metric_score > best_metric_score:
            best_metric_score = metric_score
            torch.save(model.state_dict(), f'best_grey_world_{i}.pth')
            print(f'НОВАЯ ЛУЧШАЯ МОДЕЛЬ СОХРАНЕНА! Метрика: {best_metric_score:.4f}')
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_rounds:
            print(f'Early stopping на эпохе {epoch+1}. Нет улучшений в течение {early_stopping_rounds} эпох.')
            break

    best_kfolds.append(best_metric_score) 

print(f"Средняя по folds: {np.mean(best_kfolds):.4f}")
print("Обучения модели завершено")