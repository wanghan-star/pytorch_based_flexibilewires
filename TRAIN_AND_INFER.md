# 扩大量级训练与推理指南

一键训练 + 照片推理示例，使用 `train_and_infer_on_image`。

## 示例：加大数据量与 epoch（可用于自定义调用）
```powershell
cd /d d:\python_project
python - <<"PY"
from physics_test import py

# 配置
image_path = "physics_test/Diffraction.jpg"
model_path = "physics_test/diffraction_cnn.pt"

model, mae_um, pred_um = py.train_and_infer_on_image(
    image_path=image_path,
    num_samples=20000,   # 数据量
    epochs=8,            # 训练轮数（可加大，例如 20）
    batch_size=256,
    band=7,              # 取中心±7行均值
    invert=False,        # 若条纹暗背景亮可改 True
    pixel_size=0.000204, # 如需重新标定可修改
    save_path=model_path,
)
print(f"最终预测: {pred_um:.2f} µm, 验证 MAE: {mae_um:.2f} µm")
PY
```

## 若训练时间受限
- 可先用 `num_samples=8000, epochs=4` 试跑。
- 有 GPU 时自动使用 CUDA；CPU 环境建议分段训练或减少 batch_size。

## 仅推理已训练权重
```powershell
python - <<"PY"
from physics_test import py
import torch
cfg = py.DiffractionConfig()
ckpt = torch.load("physics_test/diffraction_cnn.pt", map_location="cpu")
cfg = py.DiffractionConfig(pixel_size=ckpt.get("pixel_size", cfg.pixel_size))
model = py.CNNRegressor(cfg.pixels)
model.load_state_dict(ckpt.get("state_dict", ckpt))
model.eval()
curve = py.image_to_curve("physics_test/Diffraction.jpg", cfg, band=7, invert=False)
pred = py.predict_diameter(model, curve, cfg)
print(f"预测直径: {pred:.2f} µm")
PY
```
