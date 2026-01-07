"""
Configuration for BERT fine-tuning on SemEval-2026 Task 11 - Subtask 1.
"""

from dataclasses import dataclass
from pathlib import Path
import torch

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class Config:
    """Training configuration."""
    # Model
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    max_length: int = 256

    # Data
    train_file: str = str(PROJECT_ROOT / "train_data" / "subtask 1" / "train_data.json")
    test_file: str = str(PROJECT_ROOT / "test_data" / "subtask 1" / "test_data_subtask_1.json")
    val_split: float = 0.1
    test_split: float = 0.1

    # Training
    output_dir: str = str(PROJECT_ROOT / "results")
    num_epochs: int = 4
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    logging_steps: int = 50
    seed: int = 42

    # Bias mitigation
    balanced_sampling: bool = True

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
