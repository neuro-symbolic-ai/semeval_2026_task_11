"""
Main entry point for BERT fine-tuning on SemEval-2026 Task 11 - Subtask 1.

Usage:
    python bert_main.py train                    # Train the model
    python bert_main.py predict                  # Generate predictions
    python bert_main.py train --no-balance       # Train without balanced sampling
    python bert_main.py predict --model ./path   # Use specific model
"""

import argparse
import sys

from bert_config import Config, PROJECT_ROOT
from bert_train import train
from bert_predict import run_inference


def main():
    parser = argparse.ArgumentParser(
        description="BERT fine-tuning for syllogistic reasoning"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--model", type=str, default="bert-base-uncased",
        help="Model name (default: bert-base-uncased)"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=4,
        help="Number of epochs (default: 4)"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    train_parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    train_parser.add_argument(
        "--no-balance", action="store_true",
        help="Disable balanced sampling"
    )
    train_parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: PROJECT_ROOT/results)"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model (default: ./results/best_model)"
    )
    predict_parser.add_argument(
        "--test", type=str, default=None,
        help="Path to test file"
    )
    predict_parser.add_argument(
        "--output", type=str, default=None,
        help="Output file (default: PROJECT_ROOT/predictions.json)"
    )

    args = parser.parse_args()

    if args.command == "train":
        config = Config(
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            balanced_sampling=not args.no_balance,
        )
        if args.output:
            config.output_dir = args.output
        train(config)

    elif args.command == "predict":
        run_inference(
            model_path=args.model,
            test_file=args.test,
            output_file=args.output
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
