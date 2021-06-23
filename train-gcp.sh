#!/bin/bash
set -ev

export example_limit=180000
export BUCKET_NAME="translator-tf1"
export REGION="us-central1"
export JOB_DIR="gs://$BUCKET_NAME/job-dir"
export DATE=$(date '+%Y%m%d_%H%M')
export JOB_NAME="translator_tf"$DATE

gsutil -m cp english-german-$example_limit.csv gs://$BUCKET_NAME/data/
gsutil -m cp english-german-train-$example_limit.csv gs://$BUCKET_NAME/train_data/
gsutil -m cp english-german-valid-$example_limit.csv gs://$BUCKET_NAME/valid_data/

echo "Training AI platform ML model"
gcloud ai-platform jobs submit training $JOB_NAME \
        --module-name=trainer6.task \
        --package-path="./trainer6" \
        --job-dir=$JOB_DIR \
        --python-version=3.7 \
        --runtime-version 2.4 \
        --config config.yaml \
        -- \
        --batch_size 256 \
        --embedding_dim 256 \
        --hidden_units 1024 \
        --num_epochs 26 \
        --output_dir gs://$BUCKET_NAME/training_checkpoints \
        --full_data_path gs://$BUCKET_NAME/data/english-german-$example_limit.csv \
        --train_data_path gs://$BUCKET_NAME/train_data/english-german-train-$example_limit.csv \
        --valid_data_path gs://$BUCKET_NAME/valid_data/english-german-valid-$example_limit.csv 
