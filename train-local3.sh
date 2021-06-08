#!/bin/bash
set -ev

export example_limit=10000

echo "Training local ML model 3"
gcloud ai-platform local train \
        --module-name=trainer3.task \
        --package-path="./trainer3" \
        -- \
        --batch_size=256 \
        --embedding_dim=256 \
        --hidden_units=1024 \
        --num_epochs=32 \
        --job-dir="./job" \
        --output_dir="./trained_model3_"$example_limit \
        --full_data_path="./english-german-"$example_limit".csv" \
        --train_data_path="./english-german-train-"$example_limit".csv" \
        --valid_data_path="./english-german-valid-"$example_limit".csv"