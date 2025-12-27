# TarDAL Implementation for "All-Weather Target Detection"

This is the TarDAL baseline adapted for the `MSRS` dataset.

## Setup

1.  **Environment**: Ensure `fusion_perception` environment is activated.
    ```bash
    conda activate fusion_perception
    ```

2.  **Configuration**:
    - Config file: `config/msrs.yaml`
    - Dataset path: `../../Dataset/MSRS` (Already configured in yaml)

## Training

To train the model on MSRS dataset:

```bash
# Debug run (fast, few epochs)
python scripts/train_fd.py --cfg config/msrs.yaml --debug.fast_run True

# Full training
python scripts/train_fd.py --cfg config/msrs.yaml
```

## Dataset Structure

The code expects the dataset to be in `../../Dataset/MSRS` with the following structure (automatically set up):
- `ir/`: Infrared images
- `vi/`: Visible images
- `labels/`: YOLO format labels
- `meta/`: Contains `train.txt` and `val.txt`

## Modifying Configuration

Edit `config/msrs.yaml` to change:
- `train.batch_size`: Adjust based on your GPU vRAM (Default: 8).
- `train.image_size`: Input resolution (Default: [480, 640] resized).
- `dataset.detect.flip_...`: Augmentation settings.
