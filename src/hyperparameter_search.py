"""
Hyperparameter search script for BERT fine-tuning on SemEval-2026 Task 11.

Tests 3 models (BERT, RoBERTa, BERT-Large) with 3 learning rates each.
Uses early stopping with patience=3, fixed warmup/weight decay.
Batch sizes optimized for 16GB VRAM.

Total: 9 experiments (3 models × 3 learning rates)

Usage:
    python hyperparameter_search.py                    # Run search
    python hyperparameter_search.py --no-balance       # Without balanced sampling
    python hyperparameter_search.py --seed 123         # Custom seed
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

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

from bert_config import Config, PROJECT_ROOT
from bert_dataset import (
    SyllogismDataset,
    load_data,
    split_data,
    get_balanced_data,
)


# Model configurations with batch sizes optimized for 16GB VRAM
MODEL_CONFIGS = {
    "bert-base-uncased": {
        "batch_size": 16,
        "learning_rates": [1e-5, 2e-5, 3e-5],
    },
    "roberta-base": {
        "batch_size": 16,
        "learning_rates": [1e-5, 2e-5, 3e-5],
    },
    "bert-large-uncased": {
        "batch_size": 8,  # Smaller batch for larger model
        "learning_rates": [5e-6, 1e-5, 2e-5],  # Lower LR for large model
    },
}

# Fixed hyperparameters
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


def run_experiment(
    model_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    warmup_ratio: float,
    weight_decay: float,
    max_length: int,
    early_stopping_patience: int,
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single training experiment with given configuration."""

    set_seed(seed)

    result = {
        "model_name": model_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_length": max_length,
        "early_stopping_patience": early_stopping_patience,
        "seed": seed,
        "status": "running",
        "error": None,
        # Dataset sizes
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "total_size": len(train_data) + len(val_data) + len(test_data),
        # Accuracy metrics
        "train_accuracy": None,
        "val_accuracy": None,
        "test_accuracy": None,
        "all_accuracy": None,  # Combined train + val + test
        # Loss metrics
        "train_loss": None,
        "val_loss": None,
        "test_loss": None,
        "all_loss": None,
        # Training info
        "training_time_seconds": None,
        "best_epoch": None,
        "actual_epochs": None,  # Epochs before early stopping
    }

    start_time = time.time()

    try:
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
        exp_output_dir = os.path.join(
            output_dir,
            f"{model_name.replace('/', '_')}_lr{learning_rate}_bs{batch_size}_ep{num_epochs}"
        )

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
            save_total_limit=1,  # Save disk space
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
        train_output = trainer.train()

        # Get actual epochs trained and best epoch
        result["actual_epochs"] = int(trainer.state.epoch)
        if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
            checkpoint_name = os.path.basename(trainer.state.best_model_checkpoint)
            try:
                # checkpoint format: checkpoint-{step}
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

    result["training_time_seconds"] = time.time() - start_time

    return result


def generate_configurations() -> List[Dict]:
    """
    Generate hyperparameter configurations to test.

    3 models × 3 learning rates = 9 total experiments
    """
    configs = []

    for model_name, model_cfg in MODEL_CONFIGS.items():
        for lr in model_cfg["learning_rates"]:
            configs.append({
                "model_name": model_name,
                "learning_rate": lr,
                "batch_size": model_cfg["batch_size"],
                "num_epochs": NUM_EPOCHS,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
                "max_length": MAX_LENGTH,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            })

    return configs


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


def find_best_config(results: List[Dict]) -> Dict:
    """Find the best configuration based on test accuracy."""
    completed = [r for r in results if r["status"] == "completed"]
    if not completed:
        return None
    return max(completed, key=lambda x: x["test_accuracy"])


def print_summary(results: List[Dict], best: Dict):
    """Print a summary of the search results."""
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] == "failed"]

    print("\n" + "=" * 70)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {len(results)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")

    if completed:
        # Accuracy statistics for all splits
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

        # Loss statistics
        print(f"\n{'─' * 70}")
        print("LOSS STATISTICS")
        print(f"{'─' * 70}")

        for split_name, key in [("Train", "train_loss"),
                                 ("Val", "val_loss"),
                                 ("Test", "test_loss"),
                                 ("All Data", "all_loss")]:
            losses = [r[key] for r in completed if r[key] is not None]
            if losses:
                print(f"\n  {split_name}:")
                print(f"    Mean: {np.mean(losses):.4f}  |  "
                      f"Std: {np.std(losses):.4f}  |  "
                      f"Min: {np.min(losses):.4f}  |  "
                      f"Max: {np.max(losses):.4f}")

    if best:
        print(f"\n{'=' * 70}")
        print("BEST CONFIGURATION (by test accuracy)")
        print("=" * 70)
        print(f"\n  Hyperparameters:")
        print(f"    Model:         {best['model_name']}")
        print(f"    Learning Rate: {best['learning_rate']}")
        print(f"    Batch Size:    {best['batch_size']}")
        print(f"    Epochs:        {best['num_epochs']}")
        print(f"    Warmup Ratio:  {best['warmup_ratio']}")
        print(f"    Weight Decay:  {best['weight_decay']}")
        print(f"    Max Length:    {best['max_length']}")

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

    # Top 5 configurations
    if len(completed) > 1:
        print(f"\n{'=' * 70}")
        print("TOP 5 CONFIGURATIONS (by test accuracy)")
        print("=" * 70)
        top5 = sorted(completed, key=lambda x: x["test_accuracy"], reverse=True)[:5]
        for i, cfg in enumerate(top5, 1):
            print(f"\n{i}. {cfg['model_name']}")
            print(f"   Params: lr={cfg['learning_rate']}, bs={cfg['batch_size']}, ep={cfg['num_epochs']}")
            print(f"   Accuracy: train={cfg['train_accuracy']:.4f}, val={cfg['val_accuracy']:.4f}, "
                  f"test={cfg['test_accuracy']:.4f}, all={cfg['all_accuracy']:.4f}")
            print(f"   Loss: train={cfg['train_loss']:.4f}, val={cfg['val_loss']:.4f}, "
                  f"test={cfg['test_loss']:.4f}, all={cfg['all_loss']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for BERT fine-tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--no-balance", action="store_true", help="Disable balanced sampling")
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or str(PROJECT_ROOT / "hyperparameter_search")
    os.makedirs(output_dir, exist_ok=True)

    config = Config()
    set_seed(args.seed)

    # Print search configuration
    print("=" * 70)
    print("HYPERPARAMETER SEARCH CONFIGURATION")
    print("=" * 70)
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Epochs: {NUM_EPOCHS} (with early stopping, patience={EARLY_STOPPING_PATIENCE})")
    print(f"Warmup ratio: {WARMUP_RATIO}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Seed: {args.seed}")

    # Load data once
    print("\nLoading data...")
    data = load_data(config.train_file)
    print(f"Loaded {len(data)} samples")

    if not args.no_balance:
        print("Balancing data...")
        data = get_balanced_data(data, seed=args.seed)
        print(f"Balanced dataset size: {len(data)}")

    # Split data
    train_data, val_data, test_data = split_data(
        data, val_split=0.1, test_split=0.1, seed=args.seed
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Generate configurations
    configs = generate_configurations()
    print(f"\nTotal configurations to test: {len(configs)}")
    print("(3 models × 3 learning rates = 9 experiments)")

    # Run experiments
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 70}")
        print(f"Experiment {i}/{len(configs)}")
        print(f"{'=' * 70}")
        print(f"Model: {cfg['model_name']}")
        print(f"LR: {cfg['learning_rate']}, BS: {cfg['batch_size']}, "
              f"Epochs: {cfg['num_epochs']}, Warmup: {cfg['warmup_ratio']}, "
              f"WD: {cfg['weight_decay']}, MaxLen: {cfg['max_length']}")

        result = run_experiment(
            model_name=cfg["model_name"],
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            num_epochs=cfg["num_epochs"],
            warmup_ratio=cfg["warmup_ratio"],
            weight_decay=cfg["weight_decay"],
            max_length=cfg["max_length"],
            early_stopping_patience=cfg["early_stopping_patience"],
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            output_dir=output_dir,
            seed=args.seed,
        )

        results.append(result)

        if result["status"] == "completed":
            print(f"\n  Epochs: {result['actual_epochs']}/{cfg['num_epochs']} "
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
    best = find_best_config(results)

    print_summary(results, best)

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # Save best config separately
    if best:
        best_path = os.path.join(output_dir, f"best_config_{timestamp}.json")
        with open(best_path, 'w') as f:
            json.dump(best, f, indent=2)
        print(f"  Best: {best_path}")


if __name__ == "__main__":
    main()
