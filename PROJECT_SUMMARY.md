# SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in LLMs

## Overview

This repository contains the official materials for **SemEval-2026 Task 11**, a shared task focused on **multilingual syllogistic reasoning**. The goal is to assess whether Large Language Models can determine the formal validity of logical arguments independent of their plausibility (content bias or "content effect").

## Project Structure

```
nlp_semeval_11/
├── README.md                    # Main documentation
├── .gitignore                   # Git ignore patterns
├── evaluation_kit/              # Official evaluation scripts
│   ├── task 1 & 3/              # Binary classification evaluation
│   │   ├── evaluation_script.py
│   │   └── mock_*.json          # Example files
│   └── task 2 & 4/              # Retrieval + classification evaluation
│       ├── evaluation_script.py
│       └── mock_*.json          # Example files
├── train_data/                  # Training datasets
│   └── subtask 1/
│       └── train_data.json      # ~2,880 English syllogisms
├── test_data/                   # Test datasets
│   └── subtask 1/               # English binary classification
├── src/                         # Source code
│   ├── bert_config.py           # BERT training configuration
│   ├── bert_dataset.py          # Dataset class for BERT
│   ├── bert_train.py            # BERT training script
│   ├── bert_predict.py          # BERT inference script
│   └── bert_main.py             # CLI entry point
├── scripts/                     # Helper scripts
└── tests/                       # Unit tests
```

## Task Description (Subtask 1)

### Goal
Determine if a syllogism is formally valid (binary classification).

### Input
Syllogism text

### Output
Boolean validity prediction

## Data Format

### Training Data
```json
{
  "id": "uuid",
  "syllogism": "Full text of the syllogism...",
  "validity": true/false,
  "plausibility": true/false
}
```

### Prediction Format
```json
{
  "id": "uuid",
  "validity": true/false
}
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy (ACC)** | Overall validity prediction accuracy |
| **Total Content Effect (TCE)** | Measures plausibility bias in predictions |
| **Ranking Score** | ACC / (1 + ln(1 + TCE)) |

## Quick Start

```bash
cd nlp_semeval_11/src

# Train BERT model
python bert_main.py train

# Generate predictions
python bert_main.py predict

# Run evaluation
python "../evaluation_kit/task 1 & 3/evaluation_script.py"
```

## Key Insights

The task is designed to expose a critical limitation in LLMs: the tendency to confuse **formal logical validity** with **argument plausibility**. A logically valid argument can have an implausible conclusion, and an invalid argument can have a plausible one. Models that conflate these concepts will show high content effect bias.

The evaluation methodology explicitly penalizes this bias through the logarithmic ranking formula, encouraging development of models that truly understand formal reasoning independent of content.
