# 抗光强饱和的高鲁棒性柔性微丝非接触式光学检测系统

本项目面向柔性微丝、毛发、纤维等细线状材料的非接触式线径测量问题，提出一种融合单缝衍射物理模型、非理想光学退化建模与一维卷积神经网络的端到端测径方法。项目重点解决传统接触式测量容易造成柔性材料形变，以及传统数值拟合方法在强光晕、过曝和噪声干扰下容易陷入局部最优的问题。

论文题目：

**Design of a High-Robustness Non-Contact Optical Measurement System for Flexible Micro-Wires with Anti-Saturation Capability**


## 项目亮点

- **非接触测量**：利用激光衍射条纹反演微丝直径，避免螺旋测微器等接触式工具造成的挤压形变。
- **物理先验建模**：在理想夫琅禾费衍射模型基础上，引入高斯光束包络、中心光晕、背景底噪和相机软饱和响应。
- **Sim-to-Real 训练**：通过物理孪生仿真引擎生成大规模训练样本，缓解真实标注数据难以获取的问题。
- **一维 CNN 回归**：使用大感受野一维卷积网络从光强序列中提取高频暗纹特征，实现线径端到端预测。
- **抗初值敏感**：相比 L-BFGS-B 和牛顿迭代等传统优化方法，神经网络推理阶段无需人为给定初始值，避免初值选取导致的假性收敛。

## 方法概述

本系统的核心流程如下：

1. 使用激光照射待测微丝，在接收屏上形成衍射条纹。
2. 通过相机采集条纹图像，并使用 OpenCV 转换为灰度图。
3. 对条纹方向进行行平均或列平均，提取一维光强序列。
4. 使用传统 L-BFGS-B 方法进行非线性曲线拟合，作为对照组。
5. 构建包含高斯包络、光晕、底噪和软饱和响应的仿真数据引擎。
6. 使用仿真生成的光强序列训练一维 CNN。
7. 将真实采集的光强序列输入网络，直接回归微丝直径。

## 物理模型

理想情况下，微丝可根据巴比涅原理等效为单缝衍射问题。若入射光近似为单色平行光，远场光强分布可用 `sinc^2` 形式描述。实际实验中，普通半导体激光器并不满足理想平面波假设，因此本项目进一步加入以下非理想因素：

- 高斯光束径向能量衰减；
- 中心主极大区域过曝和软饱和；
- 低频散射光晕；
- CMOS 采集噪声和背景底噪；
- 光斑边缘处高频暗纹衰减。

这些修正使仿真数据更接近真实工业场景，从而降低纯理想公式训练带来的域偏移。

## 模型结构

深度学习部分采用一维卷积神经网络，输入为归一化后的一维光强序列，输出为连续线径预测值。网络设计包含：

- 大尺寸一维卷积核，用于提取衍射条纹的空间频率特征；
- ReLU 非线性激活；
- 最大池化与自适应平均池化，用于压缩序列维度并提高平移鲁棒性；
- Dropout 正则化，降低对局部噪声像素的依赖；
- 全连接回归头，输出微丝直径。

训练时使用 MAE 损失函数、AdamW 优化器和余弦退火学习率调度。

## 实验结果

在强光晕、中心过曝和背景噪声存在的实验条件下，传统 L-BFGS-B 拟合容易被中央主极大区域主导，出现假性收敛。本文进一步分析了牛顿迭代的局部收敛条件，得到其稳定初值范围约为：

```text
76.3 um < d0 < 84.7 um
```

当初值远离真实线径，例如取 `40 um`、`60 um` 或 `120 um` 时，传统迭代方法容易收敛到由光晕和中心噪声主导的局部极小值。

相比之下，基于物理孪生数据训练的一维 CNN 不依赖人工初值选取，能够从完整光强序列中学习全局映射关系，并稳定提取两侧高频暗纹特征。

实验对比结果：

| 方法 | 测量结果 | 分析 |
| --- | ---: | --- |
| 螺旋测微器 | 80 um | 接触式测量，可能受挤压形变影响 |
| L-BFGS-B 非线性拟合 | 80.46 um | 对初值敏感，受光晕与噪声影响明显 |
| 1D-CNN 深度学习 | 87.11 um | 全局特征提取，无需初值，适合强干扰场景 |

训练验证集中，模型平均绝对误差最低约为：

```text
MAE = 1.79 um
```

## 环境依赖

建议使用 Python 3.9 及以上版本。

```bash
pip install numpy scipy matplotlib opencv-python torch torchvision tqdm
```

主要依赖：

- `numpy`：数值计算；
- `scipy`：传统 L-BFGS-B 曲线拟合；
- `opencv-python`：图像读取、灰度化和光强提取；
- `matplotlib`：光强曲线与拟合结果可视化；
- `torch`：一维卷积神经网络训练与推理。

## 推荐目录结构

```text
pytorch_based_flexibilewires/
|-- data/                  # 实验图像与提取后的光强序列
|-- figures/               # 论文和 README 中使用的图片
|-- src/
|   |-- extract_profile.py # 图像灰度化与光强序列提取
|   |-- simulate_data.py   # 物理孪生数据生成
|   |-- fit_l_bfgs_b.py    # 传统非线性拟合对照
|   |-- model.py           # 1D-CNN 网络结构
|   |-- train.py           # 模型训练
|   `-- predict.py         # 真实样本推理
|-- paper/                 # 论文与实验报告
`-- README.md
```

## 运行流程

下面给出推荐的实验复现流程，具体脚本名称可根据仓库实际代码调整。

### 1. 提取真实衍射条纹光强

```bash
python src/extract_profile.py --image data/raw/diffraction.jpg --output data/profile.npy
```

### 2. 传统 L-BFGS-B 拟合

```bash
python src/fit_l_bfgs_b.py --profile data/profile.npy
```

### 3. 生成仿真训练集

```bash
python src/simulate_data.py --num-samples 100000 --output data/simulated_dataset.npz
```

### 4. 训练一维 CNN

```bash
python src/train.py --dataset data/simulated_dataset.npz --epochs 50
```

### 5. 对真实样本进行预测

```bash
python src/predict.py --profile data/profile.npy --checkpoint checkpoints/best.pt
```

## 应用场景

本项目适用于以下需要无损、非接触和高鲁棒测径的场景：

- 柔性微丝线径检测；
- 纺织纤维直径测量；
- 毛发、碳纤维、生物微丝等细线状样本测量；
- 复杂光照环境下的工业在线计量；
- 光学测量与深度学习结合的教学或竞赛项目。

## 局限与改进方向

- 当前实验数据规模仍有限，后续可采集更多真实样本进行半监督或迁移学习。
- 可加入更精确的相机响应标定，进一步提升仿真到现实的一致性。
- 可将网络轻量化，部署到嵌入式设备或移动端。
- 可扩展到多波长、多距离和多材料线径测量。

## 参考文献

1. Born M, Wolf E. *Principles of Optics*. 2016.
2. Goodman J W. *Introduction to Fourier Optics*. 2011.
3. Byrd R H, Lu P, Nocedal J, et al. A limited memory algorithm for bound constrained optimization. *SIAM Journal on Scientific Computing*, 1995.
4. Zhu C, Byrd R H, Lu P, et al. Algorithm 778: L-BFGS-B. *ACM Transactions on Mathematical Software*, 1997.
5. Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press, 2016.

## 作者

王涵  
东南大学集成电路学院

项目仓库：

```text
https://github.com/wanghan-star/pytorch_based_flexibilewires
```
