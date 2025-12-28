# TarDAL: 全天候目标检测 Baseline (适配 MSRS 数据集)

本项目基于 **TarDAL (Target-aware Dual Adversarial Learning)** 框架，针对 **MSRS (Multi-Spectral Road Scenarios)** 数据集进行了适配。

## 1. 目录结构说明

为了方便理解，我们将代码库进行了精简，保留了核心模块：

| 目录/文件 | 说明 | 关键用途 |
| :--- | :--- | :--- |
| **`config/`** | **配置文件** | 存放训练参数。**`msrs.yaml`** 是我们专为本项目创建的配置（包含路径、参数）。 |
| **`loader/`** | **数据加载** | **`msrs.py`** 是我们编写的数据加载器，用于读取 MSRS 的红外/可见光图像及标签。 |
| **`module/`** | **模型组件** | 包含网络的核心定义。<br>- `fuse/`: 融合网络 (Generator, Discriminator)<br>- `detect/`: 检测网络 (YOLOv5) |
| **`pipeline/`**| **流程控制** | 定义了训练和推理的具体步骤 (Forward, Loss计算等)。 |
| **`scripts/`** | **运行脚本** | - `train_fd.py`: **训练入口** (Fusion & Detection 联合训练)<br>- `infer_fd.py`: **推理入口** |
| **`weights/`** | **权重文件** | 存放预训练模型 (如 `tardal-ct.pth`) 和训练过程中的保存模型。 |
| **`runs/`** | **运行结果** | 存放推理生成的图片和训练日志。 |
| `infer.py` | 推理启动器 | 简易的命令行入口，调用 `scripts/` 下的逻辑。 |

## 2. 权重文件说明

在 `weights/` 目录下，你可能会看到以下几种预训练权重：

*   **`tardal-dt.pth` (Discriminator/Target)**:
    *   仅包含融合网络的预训练参数。
    *   通常用于**热启动**，即在联合训练开始前，先给融合网络一个较好的初始状态。

*   **`tardal-ct.pth` (Cooperative Training)**:
    *   **联合训练权重**。这是 TarDAL 的核心，它是融合网络和检测网络**一起训练**后得到的权重。
    *   在这个模式下，融合网络生成的图像不仅视觉效果好，而且**最适合目标检测**。
    *   **本项目使用此权重**作为 Baseline 的推理基础。

*   **`mask-u2.pth`**:
    *   这是显著性检测网络 (U2Net) 的权重，用于生成红外图像的 Heatmap 掩码，辅助训练。

## 3. 环境配置与使用

### 3.1 激活环境
```bash
conda activate fusion_perception
```

### 3.2 训练 (Training)
如果你想从头训练模型：
```bash
# 标准训练 (使用 config/msrs.yaml 配置)
python scripts/train_fd.py --cfg config/msrs.yaml
```
*训练结果保存在 `cache/` 目录下。*

### 3.3 推理 (Inference / Testing)
如果你想用现有的权重查看效果：
```bash
# 运行推理 (结果保存在 runs/msrs_baseline)
python infer.py --cfg config/msrs.yaml --save_dir runs/msrs_baseline
```

## 4. 常见问题
*   **为什么第一次运行很慢？**
    *   TarDAL 需要先计算数据集中每张图的“显著性掩码”和“互信息权重”。第一次运行时会自动生成并缓存到 `Dataset/MSRS/mask` 和 `iqa` 目录，之后运行就会很快。
