#!/bin/bash
# This script trains a high-quality DAFormer model for domain adaptation.

echo "--- Starting Enhanced Domain Adaptation Training ---"

python main.py adapt_train \
    --config configs/cityscapes_config.py \
    --source_config configs/cityscapes_config.py \
    --target_config configs/idd_config.py \
    --batch_size 4 \
    --num_epochs 10 \
    --dataset_percentage 0.1

echo "--- Enhanced Domain Adaptation Finished ---"