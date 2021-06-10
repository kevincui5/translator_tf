#!/bin/bash
set -ev

export example_limit=1700

echo "Training local ML model 5"
gcloud ai-platform local train \
        --module-name=trainer5.task \
        --package-path="./trainer5" \
        -- \
        --batch_size=256 \
        --embedding_dim=256 \
        --hidden_units=1024 \
        --num_epochs=2 \
        --job-dir="./job" \
        --output_dir="./trained_model5_"$example_limit \
        --full_data_path="./english-german-"$example_limit".csv" \
        --train_data_path="./english-german-train-"$example_limit".csv"