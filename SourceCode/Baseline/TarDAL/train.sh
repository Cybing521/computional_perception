#!/bin/bash
# Set PYTHONPATH to current directory so 'loader' module can be found
export PYTHONPATH=$PYTHONPATH:.

# Run training with unbuffered output
python -u scripts/train_fd.py --cfg config/msrs.yaml
