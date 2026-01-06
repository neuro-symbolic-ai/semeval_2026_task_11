"""
Data loading utilities for SemEval 2026 Task 11.
"""
import json
from pathlib import Path
from typing import List, Union

from .models import TrainingExample, TestExample


def load_training_data(file_path: Union[str, Path]) -> List[TrainingExample]:
    """
    Load training data from JSON file.

    Args:
        file_path: Path to the training data JSON file

    Returns:
        List of TrainingExample objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return [TrainingExample(**item) for item in data]


def load_test_data(file_path: Union[str, Path]) -> List[TestExample]:
    """
    Load test data from JSON file.

    Args:
        file_path: Path to the test data JSON file

    Returns:
        List of TestExample objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return [TestExample(**item) for item in data]


def parse_syllogism(syllogism_text: str) -> tuple[str, str, str]:
    """
    Parse a syllogism text into major premise, minor premise, and conclusion.

    Assumes format: "Premise1. Premise2. Therefore, Conclusion."

    Args:
        syllogism_text: The full syllogism text

    Returns:
        Tuple of (major_premise, minor_premise, conclusion)
    """
    # Split by sentences
    sentences = [s.strip() for s in syllogism_text.replace('\n', ' ').split('.') if s.strip()]

    # Look for the conclusion (typically starts with "Therefore" or "Thus" or "So")
    conclusion_idx = None
    conclusion = ""

    for i, sentence in enumerate(sentences):
        lower_sent = sentence.lower()
        if any(marker in lower_sent for marker in ['therefore', 'thus', 'so,', 'hence', 'consequently']):
            # Extract the conclusion part after the marker
            for marker in ['therefore,', 'therefore', 'thus,', 'thus', 'so,', 'hence,', 'hence', 'consequently,', 'consequently']:
                if marker in lower_sent:
                    conclusion = sentence.split(marker, 1)[-1].strip()
                    conclusion_idx = i
                    break
            break

    # If we found a conclusion, the premises are everything before it
    if conclusion_idx is not None and conclusion_idx >= 2:
        major_premise = sentences[0]
        minor_premise = '. '.join(sentences[1:conclusion_idx])
    elif conclusion_idx == 1:
        # Only one premise before conclusion
        major_premise = sentences[0]
        minor_premise = ""
    elif len(sentences) >= 3:
        # Fallback: assume last sentence is conclusion, first two are premises
        major_premise = sentences[0]
        minor_premise = sentences[1]
        conclusion = sentences[2]
    elif len(sentences) == 2:
        # Only two sentences, first is premise, second is conclusion
        major_premise = sentences[0]
        minor_premise = ""
        conclusion = sentences[1]
    else:
        # Can't parse properly, return as-is
        major_premise = syllogism_text
        minor_premise = ""
        conclusion = ""

    return major_premise, minor_premise, conclusion


def save_predictions(predictions: List[dict], output_path: Union[str, Path]):
    """
    Save predictions to JSON file in submission format.

    Args:
        predictions: List of prediction dictionaries with 'id' and 'validity' keys
        output_path: Path to save the predictions
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
