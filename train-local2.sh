#!/bin/bash
set -ev

export example_limit=1700

echo "Training local ML model"
gcloud ai-platform local train \
        --module-name=trainer2.task \
        --package-path="./trainer2" \
        -- \
        --batch_size=256 \
        --embedding_dim=256 \
        --hidden_units=1024 \
        --num_epochs=20 \
        --job-dir="./job" \
        --output_dir="./trained_model2_"$example_limit \
        --full_data_path="./english-german-"$example_limit".csv" \
        --train_data_path="./english-german-train-"$example_limit".csv"
