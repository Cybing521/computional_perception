#!/bin/bash

# Stop on error
set -e

echo "========================================"
echo "   Setting up Development Environment   "
echo "========================================"

# 1. Create Conda Environment
if conda info --envs | grep -q "fusion_perception"; then
    echo "Environment 'fusion_perception' already exists. Skipping creation."
else
    echo "Creating Conda environment 'fusion_perception' with Python 3.9..."
    conda create -n fusion_perception python=3.9 -y
fi

# 2. Install Dependencies via Pip (using conda run to execute in the new env)
echo "Installing PyTorch (Mac MPS support) and dependencies..."
conda run -n fusion_perception pip install torch torchvision torchaudio
conda run -n fusion_perception pip install opencv-python matplotlib tqdm pyyaml

echo "========================================"
echo "              Setup Complete            "
echo "========================================"
echo "To start working, run:"
echo "  conda activate fusion_perception"
