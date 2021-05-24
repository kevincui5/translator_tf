#!/bin/bash
set -ev

echo "Training local ML model"
gcloud ai-platform local train \
        --module-name=trainer.task \
        --package-path="./trainer" \
        -- \
        --batch_size=128 \
        --embedding_dim=100 \
        --num_epochs=1 \
        --job-dir="./job" \
        --output_dir="./trained_model" \
        --complete_data_path="./spa-6624.txt"