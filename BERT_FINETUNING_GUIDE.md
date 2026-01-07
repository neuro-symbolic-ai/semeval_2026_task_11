# BERT Fine-Tuning Guide for SemEval-2026 Task 11 - Subtask 1

## Overview

This guide outlines the steps to fine-tune a BERT model for **Subtask 1: English Syllogistic Reasoning** (binary validity classification).

---

## 1. Environment Setup

### Required Dependencies
```bash
pip install torch transformers datasets scikit-learn pandas tqdm
```

### Recommended Hardware
- GPU with 8GB+ VRAM (e.g., RTX 3060, T4, or better)
- 16GB+ RAM
- CUDA 11.x or 12.x

---

## 2. Model Selection

| Model | Size | Recommendation |
|-------|------|----------------|
| `bert-base-uncased` | 110M | Good starting point |
| `bert-large-uncased` | 340M | Better accuracy, more VRAM |
| `roberta-base` | 125M | Often outperforms BERT |

**Recommendation:** Start with `bert-base-uncased`, then try `roberta-base` if needed.

---

## 3. Quick Start

```bash
cd nlp_semeval_11/src

# Train the model
python bert_main.py train

# Train with custom settings
python bert_main.py train --epochs 5 --batch-size 32 --lr 3e-5

# Generate predictions
python bert_main.py predict
```

---

## 4. File Structure

```
nlp_semeval_11/
├── src/
│   ├── bert_config.py      # Hyperparameters
│   ├── bert_dataset.py     # SyllogismDataset class
│   ├── bert_train.py       # Training script
│   ├── bert_predict.py     # Generate predictions
│   └── bert_main.py        # CLI entry point
├── train_data/
│   └── subtask 1/
│       └── train_data.json
├── test_data/
│   └── subtask 1/
│       └── test_data_subtask_1.json
└── results/                # Output directory (created during training)
    └── best_model/
```

---

## 5. Training Configuration

### Recommended Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 2e-5 | Standard for BERT fine-tuning |
| Batch Size | 16-32 | Adjust based on VRAM |
| Epochs | 3-5 | Monitor validation loss |
| Weight Decay | 0.01 | Regularization |
| Warmup Steps | 10% of total | Gradual LR increase |
| Max Sequence Length | 256 | Syllogisms are typically short |

---

## 6. Addressing Content Effect Bias

The task penalizes models biased by plausibility. The implementation includes:

### Balanced Sampling (Default: Enabled)
Ensures equal representation of all 4 conditions during training:
- valid + plausible
- valid + implausible
- invalid + plausible
- invalid + implausible

To disable: `python bert_main.py train --no-balance`

---

## 7. Evaluation

Run the official evaluation script:
```bash
python "evaluation_kit/task 1 & 3/evaluation_script.py"
```

Key metrics to track:
- **Accuracy (ACC):** Higher is better
- **Total Content Effect (TCE):** Lower is better (0 = no bias)
- **Ranking Score:** ACC / (1 + ln(1 + TCE))

---

## 8. Quick Start Checklist

- [ ] Install dependencies (`torch`, `transformers`, `sklearn`, `tqdm`)
- [ ] Run training: `python bert_main.py train`
- [ ] Check validation accuracy in output
- [ ] Generate predictions: `python bert_main.py predict`
- [ ] Run official evaluation script

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size: `--batch-size 8` |
| Overfitting | Reduce epochs: `--epochs 3` |
| High content effect | Keep balanced sampling enabled (default) |
| Slow training | Ensure GPU is being used, check `config.device` |
