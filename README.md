# Detection-Driven Infrared-Visible Image Fusion via Spatial-Coordinate Attention

[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/Cybing521/computional_perception)
![Python](https://img.shields.io/badge/Python-3.9-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)

> **全天候目标检测研究**: 基于 S-CAFM (Spatial-Coordinate Attention Fusion Module) 和检测驱动的联合训练框架，解决红外与可见光图像融合中的位置模糊问题。

## 🚀 项目简介 (Introduction)

本项目针对道路场景下单一模态感知的局限性（如夜间可见光盲区、红外图像纹理缺失），提出了一种**检测驱动的红外与可见光图像融合框架**。

核心创新点：
1.  **S-CAFM (Spatial-Coordinate Attention Fusion Module)**: 利用道路场景的几何先验（水平车道线、垂直行人），通过正交分解捕捉长距离空间依赖，充当检测回归任务的“空间标尺”。
2.  **Detection-Driven Joint Training**: 实现了端到端的联合训练，将检测网络 (YOLOv8) 的梯度直接回传给融合网络，迫使模型保留对检测至关重要的边缘特征。

## 📊 核心指标 (Performance)

在 MSRS 数据集上的测试结果表明，本方法在保持高推理速度的同时，显著提升了检测精度。

| Method | mAP@50 (%) | mAP@75 (%) | AG (清晰度) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| TarDAL (Baseline) | 79.5 | 46.8 | 4.12 | 30.1 |
| SeAFusion | 80.5 | 48.2 | 6.42 | 45.3 |
| **Ours** | **81.3** | **51.2** | **32.76** | **28.5** |

> **Highlight**: mAP@75 提升 **+4.4%**，证明了 S-CAFM 对定位精度的显著贡献。

## 📂 目录结构 (Structure)

```
.
├── Report/                 # 📄 项目报告 (main.pdf) 及 LaTeX 源码
├── SourceCode/             # 💻 核心代码仓库
│   ├── Baseline/           # TarDAL 基准模型代码
│   ├── Dataset/            # 数据集配置
│   └── ...
├── PPT/                    # 📢 演示文稿
├── References/             # 📚 参考文献
└── README.md               # 📌 项目说明
```

## 🛠️ 快速上手 (Quick Start)

### 1. 环境配置
推荐使用 Conda 创建虚拟环境：

```bash
conda create -n fusion_perception python=3.9
conda activate fusion_perception
pip install torch torchvision torchaudio  # 根据硬件安装 GPU/MPS 版本
pip install -r SourceCode/Baseline/TarDAL/requirements.txt
```

### 2. 数据集准备
请下载 MSRS 数据集并解压至 `SourceCode/Dataset/MSRS/` 目录。
- 确保目录结构包含 `Visible`, `Infrared`, `Label` 等子文件夹。
- 运行转换脚本将分割标签转换为 YOLO 格式。

### 3. 运行测试
使用提供的脚本生成分析图表：

```bash
cd SourceCode/Baseline/TarDAL/scripts
python pr_curve_plot.py  # 生成 PR 曲线
python run_all_analyses.py  # 运行完整的消融分析
```

## 📝 引用 (Citation)

如果您觉得本项目对您有帮助，请给个 Star ⭐️！

```
@article{DetectionDrivenFusion2024,
  title={Detection-Driven Infrared-Visible Image Fusion via Spatial-Coordinate Attention},
  author={Yibin Chen},
  year={2024}
}
```

## 📧 联系方式
- **GitHub**: [Cybing521](https://github.com/Cybing521)
- **Project Link**: https://github.com/Cybing521/computional_perception
