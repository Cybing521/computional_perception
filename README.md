# 基于红外与可见光图像融合的全天候目标检测研究

## 1. 项目简介
本项目旨在利用计算感知技术，通过红外与可见光图像的融合，解决复杂光照条件（如黑夜、雾霾）下的目标检测难题，提升系统鲁棒性。

**项目负责人**: 陈艺彬 (25121360)

## 2. 项目目录结构
- **`Report/`**: 项目报告文档及 LaTeX 源码。
- **`PPT/`**: 演示文稿文件。
- **`SourceCode/`**: 项目源代码（包括 Dataset, Baseline 代码, 改进代码）。
  - **`Dataset/`**: 数据集存放及配置说明。
- **`References/`**: 参考文献及 PDF。
- **`CourseMaterials/`**: 课程相关材料。

## 3. 环境搭建指南

推荐使用 **Conda** 进行环境管理。

### 3.1 创建虚拟环境
```bash
conda create -n fusion_perception python=3.9
conda activate fusion_perception
```

### 3.2 安装核心依赖 (PyTorch)
根据你的硬件选择合适的安装命令：

- **Mac (M1/M2/M3 芯片)** - 支持 MPS 加速:
  ```bash
  pip install torch torchvision torchaudio
  ```

- **NVIDIA RTX 3060 (Windows/Linux)** - 支持 CUDA 11.8:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

### 3.3 安装其他常用库
```bash
pip install opencv-python matplotlib tqdm pyyaml
```

## 4. 开发流程概览

1.  **数据准备**:
    - 将 MSRS 数据集解压至 `SourceCode/Dataset/MSRS/`。
    - 参考 `SourceCode/Dataset/README.md` 进行配置验证。
    - 将分割标签 (Segmentation) 转换为检测框 (Bounding Box) 标签。

2.  **Baseline 运行**:
    - 推荐使用 [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL) 作为基准。
    - 运行测试脚本获取初始指标 (MI, VIF, mAP)。

3.  **算法改进 (创新点)**:
    - **注意力机制**: 在融合层引入 Coordinate Attention。
    - **损失函数**: 增加边缘一致性 Loss。

4.  **实验与报告**:
    - 对比 Baseline 与改进算法的视觉效果和定量指标。
    - 更新 `Report/main.tex` 并编译最终报告。
