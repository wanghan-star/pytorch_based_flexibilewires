# Sim-to-Real 单缝衍射直径估计

基于物理仿真生成 1D 光强曲线，用 1D-CNN/MLP 回归头发丝/狭缝直径（输出单位：µm）。

## 直接运行（默认 20,000 样本，20 epoch）
```powershell
cd /d d:\python_project
python -u physics_test\py.py
```
> 需有 `physics_test/Diffraction.jpg`。默认权重保存为 `physics_test/diffraction_cnn.pt`，可根据硬件适当调整 batch_size。

## 自定义训练调用
```powershell
python - <<"PY"
from physics_test import py
model, val_mae = py.train_model(
    num_samples=100_000,  # 仿真样本数
    epochs=20,
    batch_size=512,
    use_cnn=True,
)
# 保存
import torch
torch.save(model.state_dict(), "diffraction_cnn.pt")
PY
```

## 用真实曲线推理
```python
import torch
import numpy as np
from physics_test import py

# 载入模型
cfg = py.DiffractionConfig()
model = py.CNNRegressor(cfg.pixels)
model.load_state_dict(torch.load("diffraction_cnn.pt", map_location="cpu"))
model.eval()

# 载入真实光强曲线（例如 CSV 一列光强）
curve = py.load_real_curve_from_csv("your_curve.csv")
pred_um = py.predict_diameter(model, curve, cfg)
print(f"预测直径: {pred_um:.2f} µm")
```

## 数据建模要点
- 物理公式：$I(x) = (\sin \alpha / \alpha)^2$, 其中 $\alpha = \frac{\pi d}{\lambda L} x$。
- 像素尺度：640 px，`pixel_size=0.000204 m/px` (15 cm / (1333-598)).
- 随机化：直径 $d\in[20, 150]\,\mu m$，高斯噪声、背景偏置、中心过曝概率，提升泛化。
- 指标：训练/验证采用 MAE，输出单位为 µm（标签已在训练阶段从米换算为 µm）。

## 目录
- `py.py`：数据生成、模型、训练、推理全流程脚本。
- `README.md`：使用说明（本文件）。
