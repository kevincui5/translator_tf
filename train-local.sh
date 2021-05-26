#!/bin/bash
set -ev

echo "Training local ML model"
gcloud ai-platform local train \
        --module-name=trainer.task \
        --package-path="./trainer" \
        -- \
        --batch_size=256 \
        --embedding_dim=256 \
        --hidden_units=1024 \
        --num_epochs=2 \
        --job-dir="./job" \
        --output_dir="./training_checkpoints" \
        --full_data_path="./full-data-1700.csv" \
        --train_data_path="./train-1700.csv"