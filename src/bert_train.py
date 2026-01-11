"""
Training script for BERT fine-tuning on SemEval-2026 Task 11 - Subtask 1.
"""

import os
import numpy as np
from sklearn.metrics import accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from bert_config import Config, POLISHED_FILES
from bert_dataset import (
    SyllogismDataset,
    load_data,
    split_data,
    get_balanced_data,
    analyze_data_distribution,
    merge_polished_with_labels,
)


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}


def train(config: Config = None):
    """Main training function."""
    if config is None:
        config = Config()

    # Set seed for reproducibility
    set_seed(config.seed)

    print(f"Using device: {config.device}")
    print(f"Model: {config.model_name}")

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        problem_type="single_label_classification"
    )

    # Load and prepare data
    if config.polished_source:
        polished_file = POLISHED_FILES[config.polished_source]
        print(f"\nLoading polished data from: {polished_file}")
        print(f"Using labels from: {config.train_file}")
        data = merge_polished_with_labels(polished_file, config.train_file)
        print(f"Loaded {len(data)} polished samples (source: {config.polished_source})")
    else:
        print(f"\nLoading training data from: {config.train_file}")
        data = load_data(config.train_file)
        print(f"Loaded {len(data)} samples")

    # Analyze distribution
    dist = analyze_data_distribution(data)
    print(f"\nData distribution:")
    print(f"  Valid + Plausible:     {dist['valid_plausible']}")
    print(f"  Valid + Implausible:   {dist['valid_implausible']}")
    print(f"  Invalid + Plausible:   {dist['invalid_plausible']}")
    print(f"  Invalid + Implausible: {dist['invalid_implausible']}")

    # Balance data if requested
    if config.balanced_sampling:
        print("\nBalancing data across validity/plausibility groups...")
        data = get_balanced_data(data, seed=config.seed)
        print(f"Balanced dataset size: {len(data)}")

    # Split into train/val/test
    print(f"\nSplitting data (val_split={config.val_split}, test_split={config.test_split})...")
    train_data, val_data, test_data = split_data(
        data,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed
    )
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    # Create datasets
    train_dataset = SyllogismDataset(train_data, tokenizer, config.max_length)
    val_dataset = SyllogismDataset(val_data, tokenizer, config.max_length)
    test_dataset = SyllogismDataset(test_data, tokenizer, config.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=config.logging_steps,
        fp16=config.fp16 and config.device == "cuda",
        report_to="none",  # Disable wandb/tensorboard
        seed=config.seed,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save best model
    best_model_path = os.path.join(config.output_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"\nBest model saved to: {best_model_path}")

    # Evaluation on train set
    print("\n" + "="*50)
    print("Evaluation on TRAIN set:")
    print("="*50)
    train_results = trainer.evaluate(train_dataset)
    print(f"  Train Accuracy: {train_results['eval_accuracy']:.4f}")

    # Evaluation on test set
    print("\n" + "="*50)
    print("Evaluation on TEST set:")
    print("="*50)
    test_results = trainer.evaluate(test_dataset)
    print(f"  Test Accuracy: {test_results['eval_accuracy']:.4f}")

    # Evaluation on all data (train + val + test combined)
    print("\n" + "="*50)
    print("Evaluation on ALL data (train + val + test):")
    print("="*50)
    all_data = train_data + val_data + test_data
    all_dataset = SyllogismDataset(all_data, tokenizer, config.max_length)
    all_results = trainer.evaluate(all_dataset)
    print(f"  All Data Accuracy: {all_results['eval_accuracy']:.4f}")

    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print(f"  Train Accuracy:    {train_results['eval_accuracy']:.4f}")
    print(f"  Test Accuracy:     {test_results['eval_accuracy']:.4f}")
    print(f"  All Data Accuracy: {all_results['eval_accuracy']:.4f}")

    return trainer, {'train': train_results, 'test': test_results, 'all': all_results}


if __name__ == "__main__":
    config = Config()
    trainer, results = train(config)
