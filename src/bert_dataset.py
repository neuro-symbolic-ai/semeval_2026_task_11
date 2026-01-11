"""
Dataset class for SemEval-2026 Task 11 - Subtask 1.
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SyllogismDataset(Dataset):
    """PyTorch Dataset for syllogism validity classification."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        encoding = self.tokenizer(
            item['syllogism'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }

        # Labels only present in training data
        if 'validity' in item:
            result['labels'] = torch.tensor(1 if item['validity'] else 0)

        return result


def load_data(file_path: str) -> List[Dict]:
    """Load JSON data from file (supports both JSON array and JSON lines)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Try parsing as JSON array first
    if content.startswith('['):
        return json.loads(content)

    # Fall back to JSON lines format
    data = []
    for line in content.split('\n'):
        line = line.strip()
        if line:
            data.append(json.loads(line))
    return data


def split_data(
    data: List[Dict],
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    stratify: bool = True
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train, validation, and test sets."""
    stratify_labels = [d['validity'] for d in data] if stratify else None

    # First split: separate test set
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_split,
        stratify=stratify_labels,
        random_state=seed
    )

    # Second split: separate validation from training
    # Adjust val_split to account for already removed test data
    adjusted_val_split = val_split / (1 - test_split)
    stratify_labels_train = [d['validity'] for d in train_val_data] if stratify else None

    train_data, val_data = train_test_split(
        train_val_data,
        test_size=adjusted_val_split,
        stratify=stratify_labels_train,
        random_state=seed
    )

    return train_data, val_data, test_data


def get_balanced_data(data: List[Dict], seed: int = 42) -> List[Dict]:
    """
    Balance data across all 4 validity/plausibility combinations.
    Undersamples to the size of the smallest group.
    """
    import random
    random.seed(seed)

    # Group by (validity, plausibility)
    groups = defaultdict(list)
    for item in data:
        key = (item['validity'], item['plausibility'])
        groups[key].append(item)

    # Find minimum group size
    min_size = min(len(g) for g in groups.values())

    # Undersample each group
    balanced_data = []
    for key, items in groups.items():
        sampled = random.sample(items, min_size)
        balanced_data.extend(sampled)

    # Shuffle
    random.shuffle(balanced_data)

    return balanced_data


def analyze_data_distribution(data: List[Dict]) -> Dict:
    """Analyze the distribution of validity and plausibility in the data."""
    groups = defaultdict(int)
    for item in data:
        key = (item['validity'], item['plausibility'])
        groups[key] += 1

    return {
        'total': len(data),
        'valid_plausible': groups[(True, True)],
        'valid_implausible': groups[(True, False)],
        'invalid_plausible': groups[(False, True)],
        'invalid_implausible': groups[(False, False)],
        'total_valid': groups[(True, True)] + groups[(True, False)],
        'total_invalid': groups[(False, True)] + groups[(False, False)],
    }


def merge_polished_with_labels(
    polished_file: str,
    original_file: str
) -> List[Dict]:
    """
    Merge polished syllogisms with labels from original training data.

    Polished files contain abstract syllogisms (A, B, C variables).
    Labels (validity, plausibility) are taken from original data by matching IDs.
    """
    polished_data = load_data(polished_file)
    original_data = load_data(original_file)

    # Create lookup dict from original data
    original_lookup = {item['id']: item for item in original_data}

    # Merge polished syllogisms with original labels
    merged_data = []
    for polished_item in polished_data:
        item_id = polished_item['id']
        if item_id in original_lookup:
            original = original_lookup[item_id]
            merged_item = {
                'id': item_id,
                'syllogism': polished_item['syllogism'],
                'validity': original['validity'],
                'plausibility': original['plausibility'],
            }
            merged_data.append(merged_item)
        else:
            print(f"Warning: ID {item_id} not found in original data")

    return merged_data
