"""
Main entry point for BERT fine-tuning on SemEval-2026 Task 11 - Subtask 1.

Usage:
    python bert_main.py train                       # Train on original data
    python bert_main.py train --polished gemini     # Train on Gemini polished syllogisms
    python bert_main.py train --polished opus       # Train on Opus polished syllogisms
    python bert_main.py train --polished qwen       # Train on Qwen polished syllogisms
    python bert_main.py train --polished all        # Train on all polished versions (comparison)
    python bert_main.py train --no-balance          # Train without balanced sampling
    python bert_main.py search                      # Search across all data sources (original + polished)
    python bert_main.py predict                     # Generate predictions
    python bert_main.py predict --model ./path      # Use specific model
"""

import argparse
import json
import os
import sys
from datetime import datetime

from bert_config import Config, PROJECT_ROOT, POLISHED_FILES
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
        "--polished", type=str, default=None,
        choices=["gemini", "opus", "qwen", "all"],
        help="Use polished syllogisms (gemini, opus, qwen, or all)"
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
        # Determine which polished sources to train on
        if args.polished == "all":
            polished_sources = list(POLISHED_FILES.keys())
        elif args.polished:
            polished_sources = [args.polished]
        else:
            polished_sources = [None]

        all_results = {}
        for polished_source in polished_sources:
            source_name = polished_source if polished_source else "original"
            print("\n" + "=" * 60)
            print(f"TRAINING MODEL: {source_name.upper()}")
            print("=" * 60)

            config = Config(
                model_name=args.model,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                balanced_sampling=not args.no_balance,
                polished_source=polished_source,
            )

            # Set output directory based on polished source
            if args.output:
                base_output = args.output
            else:
                base_output = str(PROJECT_ROOT / "results")

            if polished_source:
                config.output_dir = f"{base_output}_{polished_source}"
            else:
                config.output_dir = base_output

            trainer, results = train(config)
            all_results[source_name] = results

        # Print comparison summary if multiple models were trained
        if len(polished_sources) > 1:
            print("\n" + "=" * 60)
            print("COMPARISON SUMMARY")
            print("=" * 60)
            print(f"{'Source':<15} {'Train Acc':<12} {'Test Acc':<12} {'All Acc':<12}")
            print("-" * 51)
            for source, results in all_results.items():
                train_acc = results['train']['eval_accuracy']
                test_acc = results['test']['eval_accuracy']
                all_acc = results['all']['eval_accuracy']
                print(f"{source:<15} {train_acc:<12.4f} {test_acc:<12.4f} {all_acc:<12.4f}")

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
