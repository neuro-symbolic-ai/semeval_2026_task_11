"""
Data source search script for BERT fine-tuning on SemEval-2026 Task 11.

Tests training on original data vs 3 polished syllogism sources (Gemini, Opus, Qwen).
This treats the data source as a hyperparameter to search over, comparing how
different polishing approaches affect model performance.

Total: 4 experiments (original + 3 polished sources)

Usage:
    python data_source_search.py                    # Run search
    python data_source_search.py --no-balance       # Without balanced sampling
    python data_source_search.py --seed 123         # Custom seed
    python data_source_search.py --model roberta-base  # Use different model
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

from bert_config import Config, PROJECT_ROOT, POLISHED_FILES
from bert_dataset import (
    SyllogismDataset,
    load_data,
    split_data,
    get_balanced_data,
    merge_polished_with_labels,
)


# Data sources to test
DATA_SOURCES = {
    "original": None,  # Use original training data
    "gemini": "gemini",
    "opus": "opus",
    "qwen": "qwen",
}

# Fixed hyperparameters
DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256
EARLY_STOPPING_PATIENCE = 3


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}


def load_data_for_source(
    source_key: Optional[str],
    original_file: str,
) -> List[Dict]:
    """Load data for a given source (original or polished)."""
    if source_key is None:
        return load_data(original_file)
    else:
        polished_file = POLISHED_FILES[source_key]
        return merge_polished_with_labels(polished_file, original_file)


def run_experiment(
    source_name: str,
    source_key: Optional[str],
    model_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    warmup_ratio: float,
    weight_decay: float,
    max_length: int,
    early_stopping_patience: int,
    original_file: str,
    output_dir: str,
    balanced: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single training experiment with given data source."""

    set_seed(seed)

    result = {
        "source_name": source_name,
        "source_key": source_key,
        "model_name": model_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_length": max_length,
        "early_stopping_patience": early_stopping_patience,
        "balanced": balanced,
        "seed": seed,
        "status": "running",
        "error": None,
        # Dataset sizes
        "train_size": None,
        "val_size": None,
        "test_size": None,
        "total_size": None,
        # Accuracy metrics
        "train_accuracy": None,
        "val_accuracy": None,
        "test_accuracy": None,
        "all_accuracy": None,
        # Loss metrics
        "train_loss": None,
        "val_loss": None,
        "test_loss": None,
        "all_loss": None,
        # Training info
        "training_time_seconds": None,
        "best_epoch": None,
        "actual_epochs": None,
    }

    start_time = time.time()

    try:
        # Load data for this source
        data = load_data_for_source(source_key, original_file)
        print(f"  Loaded {len(data)} samples")

        # Balance if requested
        if balanced:
            data = get_balanced_data(data, seed=seed)
            print(f"  Balanced to {len(data)} samples")

        # Split data
        train_data, val_data, test_data = split_data(
            data, val_split=0.1, test_split=0.1, seed=seed
        )

        result["train_size"] = len(train_data)
        result["val_size"] = len(val_data)
        result["test_size"] = len(test_data)
        result["total_size"] = len(train_data) + len(val_data) + len(test_data)

        print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )

        # Create datasets
        train_dataset = SyllogismDataset(train_data, tokenizer, max_length)
        val_dataset = SyllogismDataset(val_data, tokenizer, max_length)
        test_dataset = SyllogismDataset(test_data, tokenizer, max_length)

        # Combined dataset for "all data" evaluation
        all_data = train_data + val_data + test_data
        all_dataset = SyllogismDataset(all_data, tokenizer, max_length)

        # Experiment-specific output directory
        exp_output_dir = os.path.join(output_dir, f"model_{source_name}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=exp_output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            logging_steps=50,
            fp16=True,
            report_to="none",
            seed=seed,
            save_total_limit=1,
        )

        # Initialize trainer with early stopping
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        )

        # Train
        trainer.train()

        # Get actual epochs trained and best epoch
        result["actual_epochs"] = int(trainer.state.epoch)
        if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
            checkpoint_name = os.path.basename(trainer.state.best_model_checkpoint)
            try:
                step = int(checkpoint_name.split('-')[-1])
                steps_per_epoch = len(train_dataset) // batch_size
                result["best_epoch"] = (step // steps_per_epoch) + 1
            except:
                result["best_epoch"] = None

        # Evaluate on all datasets
        train_results = trainer.evaluate(train_dataset)
        val_results = trainer.evaluate(val_dataset)
        test_results = trainer.evaluate(test_dataset)
        all_results = trainer.evaluate(all_dataset)

        # Log accuracy metrics
        result["train_accuracy"] = train_results["eval_accuracy"]
        result["val_accuracy"] = val_results["eval_accuracy"]
        result["test_accuracy"] = test_results["eval_accuracy"]
        result["all_accuracy"] = all_results["eval_accuracy"]

        # Log loss metrics
        result["train_loss"] = train_results["eval_loss"]
        result["val_loss"] = val_results["eval_loss"]
        result["test_loss"] = test_results["eval_loss"]
        result["all_loss"] = all_results["eval_loss"]

        result["status"] = "completed"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    result["training_time_seconds"] = time.time() - start_time

    return result


def save_results(results: List[Dict], output_dir: str, timestamp: str):
    """Save results to CSV and JSON files."""

    # Save to JSON
    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save to CSV
    csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    return json_path, csv_path


def find_best_source(results: List[Dict]) -> Dict:
    """Find the best data source based on test accuracy."""
    completed = [r for r in results if r["status"] == "completed"]
    if not completed:
        return None
    return max(completed, key=lambda x: x["test_accuracy"])


def print_summary(results: List[Dict], best: Dict):
    """Print a summary of the search results."""
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]

    print("\n" + "=" * 70)
    print("DATA SOURCE SEARCH SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {len(results)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")

    if completed:
        # Comparison table
        print(f"\n{'─' * 70}")
        print("COMPARISON TABLE")
        print(f"{'─' * 70}")
        print(f"\n{'Source':<12} {'Train':<10} {'Val':<10} {'Test':<10} {'All':<10} {'Time':<8}")
        print("-" * 60)
        for r in sorted(completed, key=lambda x: x["test_accuracy"], reverse=True):
            print(f"{r['source_name']:<12} "
                  f"{r['train_accuracy']:.4f}    "
                  f"{r['val_accuracy']:.4f}    "
                  f"{r['test_accuracy']:.4f}    "
                  f"{r['all_accuracy']:.4f}    "
                  f"{r['training_time_seconds']:.0f}s")

        # Accuracy statistics
        print(f"\n{'─' * 70}")
        print("ACCURACY STATISTICS")
        print(f"{'─' * 70}")

        for split_name, key in [("Train", "train_accuracy"),
                                 ("Val", "val_accuracy"),
                                 ("Test", "test_accuracy"),
                                 ("All Data", "all_accuracy")]:
            accuracies = [r[key] for r in completed if r[key] is not None]
            if accuracies:
                print(f"\n  {split_name}:")
                print(f"    Mean: {np.mean(accuracies):.4f}  |  "
                      f"Std: {np.std(accuracies):.4f}  |  "
                      f"Min: {np.min(accuracies):.4f}  |  "
                      f"Max: {np.max(accuracies):.4f}")

    if best:
        print(f"\n{'=' * 70}")
        print("BEST DATA SOURCE (by test accuracy)")
        print("=" * 70)
        print(f"\n  Source: {best['source_name'].upper()}")
        print(f"\n  Accuracy:")
        print(f"    Train:    {best['train_accuracy']:.4f}")
        print(f"    Val:      {best['val_accuracy']:.4f}")
        print(f"    Test:     {best['test_accuracy']:.4f}")
        print(f"    All Data: {best['all_accuracy']:.4f}")

        print(f"\n  Loss:")
        print(f"    Train:    {best['train_loss']:.4f}")
        print(f"    Val:      {best['val_loss']:.4f}")
        print(f"    Test:     {best['test_loss']:.4f}")
        print(f"    All Data: {best['all_loss']:.4f}")

        print(f"\n  Dataset Sizes:")
        print(f"    Train: {best['train_size']}  |  Val: {best['val_size']}  |  "
              f"Test: {best['test_size']}  |  Total: {best['total_size']}")

        print(f"\n  Training Time: {best['training_time_seconds']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Data source search for BERT fine-tuning")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Max epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--no-balance", action="store_true", help="Disable balanced sampling")
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or str(PROJECT_ROOT / "data_source_search")
    os.makedirs(output_dir, exist_ok=True)

    config = Config()
    set_seed(args.seed)

    # Print search configuration
    print("=" * 70)
    print("DATA SOURCE SEARCH CONFIGURATION")
    print("=" * 70)
    print(f"Data Sources: {list(DATA_SOURCES.keys())}")
    print(f"Model: {args.model}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs} (with early stopping, patience={EARLY_STOPPING_PATIENCE})")
    print(f"Warmup ratio: {WARMUP_RATIO}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Balanced sampling: {not args.no_balance}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")

    # Run experiments
    results = []
    sources = list(DATA_SOURCES.items())

    for i, (source_name, source_key) in enumerate(sources, 1):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i}/{len(sources)}: {source_name.upper()}")
        print(f"{'=' * 70}")

        result = run_experiment(
            source_name=source_name,
            source_key=source_key,
            model_name=args.model,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            max_length=MAX_LENGTH,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            original_file=config.train_file,
            output_dir=output_dir,
            balanced=not args.no_balance,
            seed=args.seed,
        )

        results.append(result)

        if result["status"] == "completed":
            print(f"\n  Epochs: {result['actual_epochs']}/{args.epochs} "
                  f"(best: {result['best_epoch']})")
            print(f"  Accuracy:")
            print(f"    Train: {result['train_accuracy']:.4f}  |  "
                  f"Val: {result['val_accuracy']:.4f}  |  "
                  f"Test: {result['test_accuracy']:.4f}  |  "
                  f"All: {result['all_accuracy']:.4f}")
            print(f"  Loss:")
            print(f"    Train: {result['train_loss']:.4f}  |  "
                  f"Val: {result['val_loss']:.4f}  |  "
                  f"Test: {result['test_loss']:.4f}  |  "
                  f"All: {result['all_loss']:.4f}")
            print(f"  Time: {result['training_time_seconds']:.1f}s")
        else:
            print(f"\nFailed: {result['error']}")

        # Save intermediate results after each experiment
        save_results(results, output_dir, timestamp)

    # Final save and summary
    json_path, csv_path = save_results(results, output_dir, timestamp)
    best = find_best_source(results)

    print_summary(results, best)

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # Save best config separately
    if best:
        best_path = os.path.join(output_dir, f"best_source_{timestamp}.json")
        with open(best_path, 'w') as f:
            json.dump(best, f, indent=2)
        print(f"  Best: {best_path}")


if __name__ == "__main__":
    main()
