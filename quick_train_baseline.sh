#!/bin/bash
# This script quickly trains a baseline model on a subset of Cityscapes.

echo "--- Starting Improved Baseline Training ---"

python -m src.training.baseline_train \
    --config configs/cityscapes_config.py \
    --batch_size 4 \
    --num_epochs 15 \
    --dataset_percentage 0.1

echo "--- Improved Baseline Training Finished ---"