#!/bin/bash
set -e

echo "--- Step 1: Generating High-Quality Pseudo-Labels ---"

python main.py generate_pseudo_labels \
    --config configs/idd_config.py \
    --checkpoint checkpoints/daformer/best_model.pth \
    --output_dir pseudo_labels/ \
    --dataset idd \
    --dataset_percentage 0.1

echo "--- Step 2: Enhanced Retraining on Pseudo-Labels ---"

python main.py adapt_train \
    --config configs/idd_config.py \
    --source_config configs/cityscapes_config.py \
    --target_config configs/idd_config.py \
    --pseudo_labels pseudo_labels/ \
    --batch_size 4 \
    --num_epochs 3 \
    --dataset_percentage 0.1

echo "--- Enhanced Refinement Finished ---"