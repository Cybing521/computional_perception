# TarDAL 图像融合评估指标

## 实验配置

- **模型**: TarDAL (Target-aware Dual Adversarial Learning)
- **数据集**: MSRS (Multi-Spectral Road Scenarios)
- **测试图像数量**: 361张
- **训练配置**: 100 epochs, batch_size=8, image_size=[480, 640]

## 评估指标汇总

| 指标 | 平均值 | 最小值 | 最大值 | 说明 |
|------|--------|--------|--------|------|
| SSIM_AVG | 0.6693 | 0.4966 | 0.8158 | 平均结构相似性 |
| PSNR_AVG | 18.32 dB | 10.70 dB | 26.94 dB | 平均峰值信噪比 |
| EN | 5.94 | 3.44 | 7.37 | 熵(信息量) |
| SF | 10.14 | 2.77 | 25.70 | 空间频率 |
| AG | 32.76 | 10.49 | 104.81 | 平均梯度 |
| SD | 31.96 | 7.57 | 69.92 | 标准差 |
| MI_AVG | 1.16 | 0.69 | 1.91 | 平均互信息 |
| Qabf | 0.9556 | 0.7313 | 1.0000 | 融合质量指标 |

## 指标说明

### 参考指标（与源图像对比）
- **SSIM (Structural Similarity Index)**: 结构相似性，范围0-1，越高越好
- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，单位dB，越高越好
- **MI (Mutual Information)**: 互信息，越高表示保留的源图像信息越多

### 无参考指标
- **EN (Entropy)**: 熵，表示图像信息量，越高越好
- **SF (Spatial Frequency)**: 空间频率，表示细节丰富程度，越高越好
- **AG (Average Gradient)**: 平均梯度，表示图像清晰度，越高越好
- **SD (Standard Deviation)**: 标准差，表示图像对比度，越高越好
- **EI (Edge Intensity)**: 边缘强度，表示边缘清晰程度，越高越好
- **Qabf**: 基于梯度的融合质量指标，范围0-1，越高越好

## 文件说明

- `tardal_msrs_metrics.json`: JSON格式的详细指标数据
- `tardal_msrs_metrics.csv`: CSV格式的指标数据，便于导入Excel或其他工具
- `README.md`: 本说明文件




