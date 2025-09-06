#!/bin/bash
# This script evaluates a checkpoint.
# Usage: ./scripts/quick_evaluate.sh [model_type] [dataset]
# Example: ./scripts/quick_evaluate.sh baseline idd

MODEL_TYPE=$1
DATASET=$2

# Set the correct config and checkpoint path based on input
CHECKPOINT="checkpoints/${MODEL_TYPE}/best_model.pth"
CONFIG="configs/${DATASET}_config.py"

echo "--- Evaluating ${MODEL_TYPE} model on ${DATASET} dataset ---"
echo "Using checkpoint: ${CHECKPOINT}"
echo "Using config: ${CONFIG}"

# FIX: Removed the unsupported --dataset_percentage argument.
python main.py evaluate \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --dataset ${DATASET} \
    --split val

echo "--- Evaluation Finished ---"