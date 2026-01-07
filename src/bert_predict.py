"""
Inference script for generating predictions on test data.
"""

import json
import os
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bert_config import Config, PROJECT_ROOT
from bert_dataset import load_data


def predict(
    model,
    tokenizer,
    test_data: List[Dict],
    max_length: int = 256,
    batch_size: int = 32,
    device: str = "cuda"
) -> List[Dict]:
    """Generate predictions for test data."""
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size), desc="Predicting"):
            batch = test_data[i:i + batch_size]

            # Tokenize batch
            texts = [item['syllogism'] for item in batch]
            encoding = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)

            # Forward pass
            outputs = model(**encoding)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

            # Collect predictions
            for item, pred in zip(batch, preds):
                predictions.append({
                    'id': item['id'],
                    'validity': bool(pred)
                })

    return predictions


def save_predictions(predictions: List[Dict], output_path: str):
    """Save predictions in JSON lines format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print(f"Predictions saved to: {output_path}")


def run_inference(
    model_path: str = None,
    test_file: str = None,
    output_file: str = None,
    config: Config = None
):
    """Run inference on test data."""
    if config is None:
        config = Config()

    if model_path is None:
        model_path = os.path.join(config.output_dir, "best_model")

    if test_file is None:
        test_file = config.test_file

    if output_file is None:
        output_file = str(PROJECT_ROOT / "predictions.json")

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    print(f"Loading test data from: {test_file}")
    test_data = load_data(test_file)
    print(f"Loaded {len(test_data)} test samples")

    print(f"\nGenerating predictions on {config.device}...")
    predictions = predict(
        model,
        tokenizer,
        test_data,
        max_length=config.max_length,
        batch_size=config.eval_batch_size,
        device=config.device
    )

    save_predictions(predictions, output_file)

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate predictions")
    parser.add_argument("--model", type=str, default=None, help="Path to model")
    parser.add_argument("--test", type=str, default=None, help="Path to test file")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        test_file=args.test,
        output_file=args.output
    )
