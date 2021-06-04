import argparse
import json
import os

from trainer2 import model2

#import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        help="this model ignores this field, but it is required by gcloud",
        default="junk"
    )
    parser.add_argument(
        "--train_data_path",
        help="GCS location of training data",
    )
    parser.add_argument(
        "--valid_data_path",
        help="GCS location of validation data",
    )
    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models",
        required=True
    )
    parser.add_argument(
        "--full_data_path",
        help="GCS location of raw data",
        required=True
    )
    parser.add_argument(
        "--batch_size",
        help="Number of examples to compute gradient over.",
        type=int,
        default=512
    )
    parser.add_argument(
        "--hidden_units",
        help="Hidden layer sizes for LSTM hidden units",
        type=int,
        default=512
    )
    parser.add_argument(
        "--embedding_dim",
        help="Embedding size of encoder and decoder's embedding layers",
        type=int,
        default=256
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of epochs to train the model.",
        type=int,
        default=20
    )
    parser.add_argument(
        "--eval_steps",
        help="""Positive number of steps for which to evaluate model. Default
        to None, which means to evaluate until input_fn raises an end-of-input
        exception""",
        type=int,
        default=None
    )

    # Parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service
    arguments.pop("job_dir", None)
    arguments.pop("job-dir", None)

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    arguments["output_dir"] = os.path.join(
        arguments["output_dir"],
        json.loads(
            os.environ.get("TF_CONFIG", "{}")
        ).get("task", {}).get("trial", "")
    )

    # Run the training job
    model2.train(arguments)