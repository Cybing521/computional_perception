# 数据集配置：MSRS

## 概览
本目录包含用于“计算感知 - 红外与可见光图像融合”项目的 **MSRS (Multi-Spectral Road Scenarios)** 数据集。

## 数据集来源
- **名称**: MSRS (Multi-Spectral Road Scenarios, 多光谱道路场景)
- **作者**: Tang Linfeng 等
- **GitHub 仓库**: [https://github.com/Linfeng-Tang/MSRS](https://github.com/Linfeng-Tang/MSRS)

## 目录结构
数据集应放置在 `MSRS` 子目录中，结构如下：

```
SourceCode/Dataset/MSRS/
├── train/              # 训练数据 (1083 对)
├── test/               # 测试数据 (361 对)
│   ├── ir/             # 红外图像
│   ├── vi/             # 可见光图像
│   ├── Segmentation_labels/ # 语义分割标签
│   └── ...
└── visualize.py        # 用于验证标签的脚本
```

## 注意
由于数据量较大，实际的图像数据文件夹 (`MSRS/`) 已被版本控制 (`.gitignore`) 忽略。请从上述官方仓库链接下载数据集并在此处解压。

## 验证
您可以使用 `MSRS/` 目录中提供的 `visualize.py` 脚本来验证数据集的完整性并可视化语义分割标签。
