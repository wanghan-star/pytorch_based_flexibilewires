"""
Sim-to-Real: 单缝衍射直径端到端测量（数据生成 + 1D-CNN/MLP 训练 + 推理示例）

概述
1) 用物理公式在 numpy/torch 生成 1D 衍射光强曲线（640 像素，0.000204 m/px）。
2) 随机采样 d∈[20µm,150µm]，加入噪声、背景、过曝，构建合成数据。
3) 训练 1D-CNN 或 MLP 回归直径（单位：µm）。
4) 使用实拍曲线进行推理，验证 Sim-to-Real 泛化。

运行提示
- 建议优先使用 GPU (CUDA)。
- 默认快速示例仅 1 epoch，可按需增大 num_samples 与 epochs。
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
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, random_split


# --------------------------- 配置与超参 ---------------------------
@dataclass
class DiffractionConfig:
    pixels: int = 640                  # 灰度曲线长度 (像素)
    pixel_size: float = 0.000204      # m/px，来自 15cm/(1333-598)
    wavelength: float = 650e-9        # m，激光波长 650nm
    distance: float = 1.5             # m，光阑到屏幕距离
    d_min: float = 40e-6              # m，40 µm
    d_max: float = 125e-6             # m，125 µm
    noise_std_range: Tuple[float, float] = (0.005, 0.05)  # 噪声强度范围
    background_range: Tuple[float, float] = (0.0, 0.1)    # 背景偏置
    saturation_prob: float = 0.35     # 出现中心过曝的概率
    saturation_clip: float = 0.85     # 过曝时的阈值上限


def simulate_curve(diameter: torch.Tensor, cfg: DiffractionConfig) -> torch.Tensor:
    if diameter.dim() == 0:
        diameter = diameter.unsqueeze(0)
    device = diameter.device
    pixels = cfg.pixels
    x = (torch.arange(pixels, device=device) - pixels // 2) * cfg.pixel_size  # [P]

    # 1. 纯物理理想衍射
    alpha = (math.pi * diameter.unsqueeze(1) / (cfg.wavelength * cfg.distance)) * x + 1e-9
    intensity = (torch.sin(alpha) / alpha) ** 2

    # 2. 【核心绝杀】：模拟真实照片中巨大的“激光散射光晕 (Halo)”
    # 用高斯函数强行在中心加一坨光晕，逼迫 AI 学会忽略变胖的中心
    halo_width = torch.empty(diameter.size(0), 1, device=device).uniform_(0.01, 0.04)
    halo_amp = torch.empty(diameter.size(0), 1, device=device).uniform_(0.5, 2.0)
    # x shape [P], need to broadcast to [N, P]
    halo = halo_amp * torch.exp(- (x.unsqueeze(0) ** 2) / (2 * (halo_width ** 2)))

    # 3. 模拟你照片里极高的环境底噪 (0.2~0.5)
    background = torch.empty(diameter.size(0), 1, device=device).uniform_(0.2, 0.5)
    noise_std = torch.empty(diameter.size(0), 1, device=device).uniform_(0.01, 0.05)

    # 叠加所有干扰
    intensity = intensity + halo + background + torch.randn_like(intensity) * noise_std

    # 4. 模拟过曝截断
    intensity = torch.clamp(intensity, min=0.0, max=1.2)

    # 5. 归一化 (与你的图片预处理完全对齐)
    min_val = intensity.min(dim=1, keepdim=True).values
    intensity = intensity - min_val
    max_val = intensity.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    intensity = intensity / max_val

    return intensity


def generate_dataset(num_samples: int, cfg: DiffractionConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """批量生成数据集: curves shape [N, pixels], labels shape [N]."""
    diameters = torch.empty(num_samples, device=device).uniform_(cfg.d_min, cfg.d_max)
    curves = simulate_curve(diameters, cfg)
    return curves.float(), diameters.float()


# --------------------------- 模型定义 ---------------------------
class CNNRegressor(nn.Module):
    def __init__(self, pixels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, pixels]
        return self.net(x).squeeze(1)


class MLPRegressor(nn.Module):
    def __init__(self, pixels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pixels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, pixels]
        x = x.squeeze(1)
        return self.net(x).squeeze(1)


# --------------------------- 训练与评估 ---------------------------
def train_model(
    num_samples: int = 20_000,
    epochs: int = 5,
    batch_size: int = 256,
    use_cnn: bool = True,
    lr: float = 1e-3,
    val_ratio: float = 0.1,
    seed: int = 42,
    cfg: Optional[DiffractionConfig] = None,
    device: Optional[torch.device] = None,
    init_state_path: Optional[str] = None,
    save_best_path: Optional[str] = None,
):
    """训练模型并返回 (model, val_mae_um)。"""
    cfg = cfg or DiffractionConfig()
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"设备: {device}; CNN: {use_cnn}; 样本: {num_samples}; 像素: {cfg.pixels}")
    curves, labels_m = generate_dataset(num_samples, cfg, device)
    labels_um = labels_m * 1e6  # 将目标从米转为微米，避免极小数值导致训练不稳定
    dataset = TensorDataset(curves.unsqueeze(1), labels_um)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    model = (CNNRegressor(cfg.pixels) if use_cnn else MLPRegressor(cfg.pixels)).to(device)
    if init_state_path and os.path.exists(init_state_path):
        state = torch.load(init_state_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        print(f"已从 {init_state_path} 加载初始权重")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE 更直观

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
        epoch_loss /= train_size

        # 验证 MAE
        model.eval()
        with torch.no_grad():
            abs_err = []
            for x_val, y_val in val_loader:
                pred = model(x_val)
                abs_err.append((pred - y_val).abs())
            val_mae = torch.cat(abs_err).mean().item()
        print(f"Epoch {epoch:02d} | Train L1: {epoch_loss:.4e} | Val MAE: {val_mae:.2f} µm")

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {"state_dict": model.state_dict(), "pixel_size": cfg.pixel_size}
            if save_best_path:
                torch.save(best_state, save_best_path)
                print(f"保存最优权重 -> {save_best_path} (MAE {best_mae:.2f} µm)")

    if best_state and save_best_path:
        # 加载最优状态用于返回
        model.load_state_dict(best_state["state_dict"])
        val_mae = best_mae
    return model, val_mae  # 直接返回 µm 误差


# --------------------------- 推理/实战 ---------------------------
def predict_diameter(model: nn.Module, curve: Union[np.ndarray, torch.Tensor], cfg: DiffractionConfig) -> float:
    """对单条曲线推理直径，返回 µm。curve 为 1D ndarray 或 Tensor。"""
    model.eval()
    device = next(model.parameters()).device
    if isinstance(curve, np.ndarray):
        curve_t = torch.from_numpy(curve.astype(np.float32))
    else:
        curve_t = curve.float().cpu()
    curve_t = curve_t / (curve_t.max() + 1e-6)
    # 如果提供的是 [P]，扩为 [1,1,P]
    if curve_t.dim() == 1:
        curve_t = curve_t.unsqueeze(0)
    curve_t = curve_t.unsqueeze(1).to(device)
    with torch.no_grad():
        pred = model(curve_t).item()  # 模型已在 µm 空间训练
    return pred


def load_real_curve_from_csv(path: str) -> np.ndarray:
    """示例：从 CSV 读入一列光强数据 (无需表头)。"""
    arr = np.loadtxt(path, delimiter=",", ndmin=1)
    return arr.astype(np.float32)


# --------------------------- 实拍图像 -> 1D 曲线 ---------------------------
def image_to_curve(
    image_path: str,
    cfg: DiffractionConfig,
    band: int = 30,
    invert: bool = False,
) -> np.ndarray:
    """将衍射照片转为 1D 光强曲线（自动中心对齐、对称截取）。

    1) 取图像中心列左右 band 范围求均值，得到竖直方向信号；
    2) 可选 invert 处理；
    3) 找到竖直方向最亮像素作为真实物理中心；
    4) 以该峰值为中心，对称截取 cfg.pixels 段，越界处自动零填充；
    5) 归一化到 [0,1]。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    # 1. 取中间区域，沿竖直方向求平均
    center_x = w // 2
    signal = arr[:, max(0, center_x - band) : min(w, center_x + band)].mean(axis=1)

    if invert:
        signal = signal.max() - signal

    # 2. 找到真实光斑的绝对物理中心（最亮像素的位置）
    peak_idx = int(np.argmax(signal))

    # 3. 以真实峰值为中心，严格截取 cfg.pixels 个像素
    half_p = cfg.pixels // 2
    start_y = peak_idx - half_p
    end_y = peak_idx + half_p

    # 4. 提取并做越界保护
    curve = np.zeros(cfg.pixels, dtype=np.float32)
    valid_start = max(0, start_y)
    valid_end = min(h, end_y)

    c_start = max(0, -start_y)
    c_end = c_start + (valid_end - valid_start)

    if valid_end > valid_start:  # 确保截取到了数据
        curve[c_start:c_end] = signal[valid_start:valid_end]

    # 5. 归一化
    curve = curve - curve.min()
    curve = curve / (curve.max() + 1e-6)

    return curve.astype(np.float32)


def load_model(model_path: str, cfg: DiffractionConfig, use_cnn: bool = True, device: Optional[torch.device] = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (CNNRegressor(cfg.pixels) if use_cnn else MLPRegressor(cfg.pixels)).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def train_and_infer_on_image(
    image_path: str,
    num_samples: int = 20_000,
    epochs: int = 8,
    batch_size: int = 256,
    use_cnn: bool = True,
    band: int = 30,
    invert: bool = False,
    pixel_size: Optional[float] = None,
    save_path: Optional[str] = "physics_test/diffraction_cnn.pt",
    init_state_path: Optional[str] = None,
    save_best_path: Optional[str] = None,
    seed: int = 42,
):
    """一键训练 + 推理照片直径。

    - num_samples / epochs：可加大以提升泛化。
    - band：取中线上下 band 行做均值，抑制噪声。
    - invert：若条纹暗、背景亮，可置 True。
    - pixel_size：如需与真实几何标定对齐可传入，否则用默认。
    - save_path：保存权重路径，None 则不保存。
    返回 (model, val_mae_um, pred_um)。
    """

    default_px = DiffractionConfig().pixel_size
    cfg = DiffractionConfig(pixel_size=pixel_size or default_px)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"训练配置 | device={device} | samples={num_samples} | epochs={epochs} | "
        f"batch={batch_size} | band={band} | invert={invert} | pixel_size={cfg.pixel_size}"
    )

    model, val_mae_um = train_model(
        num_samples=num_samples,
        epochs=epochs,
        batch_size=batch_size,
        use_cnn=use_cnn,
        cfg=cfg,
        device=device,
        seed=seed,
        init_state_path=init_state_path,
        save_best_path=save_best_path or save_path,
    )

    if save_path:
        torch.save({"state_dict": model.state_dict(), "pixel_size": cfg.pixel_size}, save_path)
        print(f"权重已保存: {save_path}")

    curve = image_to_curve(image_path, cfg, band=band, invert=invert)
    pred_um = predict_diameter(model, curve, cfg)
    print(f"照片 {os.path.basename(image_path)} 预测直径: {pred_um:.2f} µm | 验证 MAE: {val_mae_um:.2f} µm")
    return model, val_mae_um, pred_um


if __name__ == "__main__":
    # 如果已有训练好的权重，直接纯推理；否则回退到训练+推理。
    cfg = DiffractionConfig(pixel_size=0.000204)
    model_path = os.path.join(os.path.dirname(__file__), "diffraction_cnn.pt")
    image_path = os.path.join(os.path.dirname(__file__), "Diffraction.jpg")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"默认图片不存在: {image_path}")

    if os.path.exists(model_path):
        model = load_model(model_path, cfg, use_cnn=True)
        curve = image_to_curve(image_path, cfg, band=7)
        pred_um = predict_diameter(model, curve, cfg)
        print(f"🎉 对齐后预测直径: {pred_um:.2f} µm")
    else:
        start = time.time()
        _, mae_um, pred_um = train_and_infer_on_image(
            image_path=image_path,
            num_samples=20_000,
            epochs=20,
            batch_size=256,
            band=7,
            invert=False,
            pixel_size=cfg.pixel_size,
            save_path=model_path,
            init_state_path=None,
            save_best_path=model_path,
        )

        print(
            f"训练完成 | 预测: {pred_um:.2f} µm | 最优验证 MAE: {mae_um:.2f} µm | 用时 {time.time() - start:.2f} s"
        )