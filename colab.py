"""
Sim-to-Real: 柔性微丝单缝衍射测径（Colab 高斯光束修正版）
解决物理盲点：引入激光高斯包络，真实还原边缘条纹衰减
"""

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# --------------------------- 配置与超参 (回归你最精准的设定) ---------------------------
@dataclass
class DiffractionConfig:
    pixels: int = 640
    pixel_size: float = 0.000204
    wavelength: float = 650e-9
    distance: float = 1.5
    d_min: float = 40e-6     # 构建 40~125 µm 数据集搜索范围，极大降低误差
    d_max: float = 125e-6
    noise_std_range: Tuple[float, float] = (0.005, 0.05)
    background_range: Tuple[float, float] = (0.0, 0.1)

def simulate_curve(diameter: torch.Tensor, cfg: DiffractionConfig) -> torch.Tensor:
    if diameter.dim() == 0:
        diameter = diameter.unsqueeze(0)
    device = diameter.device
    pixels = cfg.pixels
    x = (torch.arange(pixels, device=device) - pixels // 2) * cfg.pixel_size

    # 1. 理想衍射物理方程
    alpha = (math.pi * diameter.unsqueeze(1) / (cfg.wavelength * cfg.distance)) * x + 1e-9
    diffraction = (torch.sin(alpha) / alpha) ** 2

    # 2. 【核心修改：激光笔高斯光束包络 (Gaussian Beam Profile)】
    # 真实的激光笔能量向两侧急剧衰减，乘上这个包络后，边缘波纹才会和你的真实照片一致！
    beam_width = torch.empty(diameter.size(0), 1, device=device).uniform_(0.015, 0.04)
    intensity = diffraction * torch.exp(- (x.unsqueeze(0) ** 2) / (beam_width ** 2))

    # 3. 适度光晕与底噪
    halo_width = torch.empty(diameter.size(0), 1, device=device).uniform_(0.01, 0.03)
    halo_amp = torch.empty(diameter.size(0), 1, device=device).uniform_(0.2, 0.8)
    halo = halo_amp * torch.exp(- (x.unsqueeze(0) ** 2) / (2 * (halo_width ** 2)))

    background = torch.empty(diameter.size(0), 1, device=device).uniform_(0.1, 0.3)
    noise_std = torch.empty(diameter.size(0), 1, device=device).uniform_(0.01, 0.04)

    intensity = intensity + halo + background + torch.randn_like(intensity) * noise_std
    
    # 4. 过曝截断与归一化
    intensity = torch.clamp(intensity, min=0.0, max=1.2)
    min_val = intensity.min(dim=1, keepdim=True).values
    intensity = intensity - min_val
    max_val = intensity.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    intensity = intensity / max_val

    return intensity

def generate_dataset(num_samples: int, cfg: DiffractionConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    diameters = torch.empty(num_samples, device=device).uniform_(cfg.d_min, cfg.d_max)
    curves = simulate_curve(diameters, cfg)
    return curves.float(), diameters.float()

# --------------------------- 扎实的 CNN 网络架构 ---------------------------
class CNNRegressor(nn.Module):
    def __init__(self, pixels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),  # 提取 16 个核心特征
            
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

# --------------------------- 训练与评估 ---------------------------
def train_model(
    num_samples: int,
    epochs: int,
    batch_size: int,
    cfg: DiffractionConfig,
    device: torch.device,
    save_best_path: str
):
    print(f"🖥️ 设备: {device} | 样本: {num_samples} | 架构: Gaussian-Corrected CNN")
    curves, labels_m = generate_dataset(num_samples, cfg, device)
    labels_um = labels_m * 1e6
    dataset = TensorDataset(curves.unsqueeze(1), labels_um)

    val_size = int(len(dataset) * 0.1)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    model = CNNRegressor(cfg.pixels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    # 极其顺滑的余弦退火学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
        
        scheduler.step()

        model.eval()
        with torch.no_grad():
            abs_err = []
            for x_val, y_val in val_loader:
                pred = model(x_val)
                abs_err.append((pred - y_val).abs())
            val_mae = torch.cat(abs_err).mean().item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Train MAE: {epoch_loss/len(train_ds):.2f} µm | Val MAE: {val_mae:.2f} µm")

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {"state_dict": model.state_dict(), "pixel_size": cfg.pixel_size}
            torch.save(best_state, save_best_path)

    if best_state:
        model.load_state_dict(best_state["state_dict"])
    return model, best_mae

def image_to_curve(image_path: str, cfg: DiffractionConfig, band: int = 30) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape
    center_x = w // 2
    signal = arr[:, max(0, center_x - band) : min(w, center_x + band)].mean(axis=1)
    peak_idx = int(np.argmax(signal))
    half_p = cfg.pixels // 2
    curve = np.zeros(cfg.pixels, dtype=np.float32)
    start_y, end_y = peak_idx - half_p, peak_idx + half_p
    valid_start, valid_end = max(0, start_y), min(h, end_y)
    c_start, c_end = max(0, -start_y), max(0, -start_y) + (valid_end - valid_start)
    if valid_end > valid_start: curve[c_start:c_end] = signal[valid_start:valid_end]
    curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-6)
    return curve.astype(np.float32)

if __name__ == "__main__":
    start = time.time()
    cfg = DiffractionConfig(pixel_size=0.000204)
    
    # 强制 Colab 绝对路径
    model_path = "/content/diffraction_cnn.pt"
    image_path = "/content/Diffraction.jpg"

    if not os.path.exists(image_path):
        print(f"⚠️ 请点击左侧 📁 图标上传照片，并重命名为 {image_path}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("🚀 启动 50 轮高精度 GPU 训练...")
        
        # 10万样本跑 50 轮
        model, mae_um = train_model(
            num_samples=100_000,
            epochs=50,
            batch_size=512,
            cfg=cfg,
            device=device,
            save_best_path=model_path
        )

        raw_curve = image_to_curve(image_path, cfg, band=7)
        
        model.eval()
        with torch.no_grad():
            curve_t = torch.from_numpy(raw_curve).float().unsqueeze(0).unsqueeze(0).to(device)
            pred_um = model(curve_t).item()

        print("\n" + "="*45)
        print(f"🎉 训练完成 | 预测直径: {pred_um:.2f} µm | 最优 MAE: {mae_um:.2f} µm")
        print(f"⏱️ 用时 {time.time() - start:.2f} 秒")
        print("="*45)

        # ---------- 最终精准出图 ----------
        print("🎨 正在生成含高斯修正的物理对比图...")
        x_axis_cm = (np.arange(cfg.pixels) - cfg.pixels // 2) * cfg.pixel_size * 100
        with torch.no_grad():
            d_tensor = torch.tensor([pred_um * 1e-6], device=device)
            x_plot = torch.tensor(x_axis_cm / 100).to(device)
            alpha = (math.pi * d_tensor / (cfg.wavelength * cfg.distance)) * x_plot + 1e-9
            diff_curve = (torch.sin(alpha) / alpha) ** 2
            
            # 画图时同样套上高斯包络，完美契合实拍
            ideal_curve = diff_curve * torch.exp(- (x_plot ** 2) / (0.025 ** 2))
            ideal_curve = ideal_curve.squeeze().cpu().numpy()
            
            baseline = np.median(raw_curve[:40]) 
            ideal_curve = ideal_curve * (1 - baseline) + baseline

        plt.figure(figsize=(10, 5), dpi=300)
        plt.scatter(x_axis_cm, raw_curve, s=6, c='gray', alpha=0.6, label='Raw Sensor Data')
        plt.plot(x_axis_cm, ideal_curve, 'r-', linewidth=2.5, label=f'Gaussian AI Predicted (d = {pred_um:.2f} $\\mu m$)')
        plt.title('Gaussian Beam Physics-Informed Alignment', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Position on Screen relative to center (cm)', fontsize=12)
        plt.ylabel('Normalized Intensity', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.xlim([-5, 5])
        plt.ylim([0, 1.1])
        plt.tight_layout()
        plt.savefig("/content/Paper_Figure.png", format='png', bbox_inches='tight')
        plt.show()