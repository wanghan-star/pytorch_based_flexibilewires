import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ============ 核心参数 ============
IMAGE_PATH = 'D:\python_project\physics_test\Diffraction.jpg' 
L = 1.5     # 头发到墙壁的距离 (单位: 米)
LAMBDA = 650e-9  # 激光波长 (红色一般是 650nm=650e-9)
# 下面填你刚才算出来的比例
SCALE = 0.000204  # m/像素


# ============ 核心算法  ============
def diffraction_model(x, d, I0, x0, b):
    """单缝衍射模型 I = I0 * (sin(alpha)/alpha)^2 + base"""
    k = (np.pi * d) / (LAMBDA * L)
    alpha = k * (x - x0) + 1e-9  # 防除0
    return I0 * (np.sin(alpha) / alpha) ** 2 + b


def make_loss_fn(x_data, y_data):
    def loss(params):
        d, I0, x0, b = params
        pred = diffraction_model(x_data, d, I0, x0, b)
        residual = pred - y_data
        return np.sum(residual ** 2)

    return loss


# 读取图片
print(f"正在读取 {IMAGE_PATH} ...")
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ 错误：找不到图片，请检查文件名或路径！")
    raise SystemExit(1)

# 转灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 提取中心行的数据
h, w = gray.shape
center_x = w // 2
# 我们取中间 20 列 (Columns) 的平均值
# axis=1 表示把这 20 列横向压缩成 1 列
signal = np.mean(gray[:, center_x - 30:center_x + 30], axis=1)

y_data = signal  # 光强数据
x_data = np.arange(h) * SCALE  # 注意：现在的坐标轴是高度(h)
# AI 拟合（可视化迭代日志）
print("🤖 AI 正在计算头发直径（带迭代日志）...")
# 初始猜测: 直径80微米, 最大亮度255, 中心在图中间, 底噪0
p0 = np.array([80e-6, 255, h / 2 * SCALE, 0], dtype=float)
loss_fn = make_loss_fn(x_data, y_data)

# 可选：给参数加一点合理约束，避免无意义值
bounds = [
    (10e-6, 160e-6),              # d: 10~200 微米
    (0, None),                    # I0: 非负
    (x_data.min(), x_data.max()),  # x0: 落在屏幕范围
    (None, None),                 # b: 可正可负
]

iter_state = {"n": 0, "best": np.inf}


def callback(params):
    iter_state["n"] += 1
    current_loss = loss_fn(params)
    d_um = params[0] * 1e6
    if current_loss < iter_state["best"]:
        iter_state["best"] = current_loss
    print(f"迭代 {iter_state['n']:03d}: 直径 = {d_um:7.2f} um, Loss = {current_loss:.6f}")


try:
    result = minimize(
        loss_fn,
        p0,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={"maxiter": 500, "disp": False},
    )

    if not result.success:
        raise RuntimeError(result.message)

    d_fit, I0_fit, x0_fit, b_fit = result.x
    diameter_um = d_fit * 1e6  # 换算成微米
    final_loss = loss_fn(result.x)

    print("✅ 优化成功！")
    print("========================================")
    print(f"🎯 头发丝直径为: {diameter_um:.2f} 微米 (um)")
    print(f"📉 最终损失: {final_loss:.6f}")
    print(f"🧠 迭代次数: {iter_state['n']}")
    print("========================================")

    # 画图
    plt.figure(figsize=(12, 6))
    plt.scatter(x_data * 100, y_data, s=1, c='gray', alpha=0.5, label='Sensor Data')
    plt.plot(
        x_data * 100,
        diffraction_model(x_data, d_fit, I0_fit, x0_fit, b_fit),
        'r-',
        lw=2,
        label=f'Best Fit (d={diameter_um:.1f} um)',
    )
    plt.title(f'AI-Based Diffraction Analysis (d = {diameter_um:.2f} $\\mu m$)', fontsize=14)
    plt.xlabel('Position on Screen (cm)')
    plt.ylabel('Light Intensity')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('result_plot.png')  # 保存结果图
    plt.show()

except Exception as e:
    print(f"❌ 拟合失败: {e}")
    print("建议：检查SCALE是否算对，或者裁切一下图片只保留中间亮斑区域。")