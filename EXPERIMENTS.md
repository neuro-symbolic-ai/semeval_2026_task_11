# Experiment Commands (PowerShell One-Liners)

This file contains ready-to-run experiment commands for testing different models and prompt templates.

## Quick Reference

- **Models**: Qwen (instruct/thinking), Claude, GPT-4o, DeepSeek, Llama
- **Prompts**: `direct` (simple), `cot` (chain-of-thought)
- **Data**: Train data for evaluation, test data for final predictions
- **All commands are one-liners** - just copy and paste!

---

## Core Experiments: Qwen Models

### Qwen Instruct - Direct Prompt
**Cost**: ~$0.08 | **Examples**: 960 | **Best for**: Baseline comparison, cheapest option
Tests the non-thinking Qwen model with simple direct prompt asking for validity judgment.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-instruct" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_instruct_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

### Qwen Instruct - Chain of Thought Prompt
**Cost**: ~$0.12 | **Examples**: 960 | **Best for**: Testing if explicit reasoning helps non-thinking models
Tests the non-thinking Qwen model with step-by-step reasoning prompt.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-instruct" --prompt "cot" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_instruct_cot.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

### Qwen Thinking - Direct Prompt ⭐ RECOMMENDED
**Cost**: ~$1.20 | **Examples**: 960 | **Best for**: High accuracy with internal reasoning
Uses thinking model with simple prompt. Model reasons internally (258 reasoning tokens). Most efficient for thinking models.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-thinking" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_thinking_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

### Qwen Thinking - Chain of Thought Prompt ⚠️ EXPENSIVE
**Cost**: ~$3.60 | **Examples**: 960 | **Best for**: Research only - likely redundant
Uses thinking model with CoT prompt. Generates 3.4x more reasoning tokens (872). Likely unnecessary since model already thinks internally.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-thinking" --prompt "cot" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_thinking_cot.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

---

## Budget-Friendly Models

### DeepSeek Chat - Direct Prompt 💰 CHEAPEST
**Cost**: ~$0.04 | **Examples**: 960 | **Best for**: Quick testing, budget constraints
Extremely cheap option. Good for rapid iteration and testing prompt effectiveness.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "deepseek/deepseek-chat" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/deepseek_chat_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

### DeepSeek Chat - Chain of Thought Prompt
**Cost**: ~$0.08 | **Examples**: 960 | **Best for**: Testing if CoT improves cheap models
Tests whether explicit reasoning steps help budget models perform better.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "deepseek/deepseek-chat" --prompt "cot" --input "train_data/subtask 1/train_data.json" --output "predictions/deepseek_chat_cot.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

### Meta Llama 3.1 70B - Direct Prompt 🎯 BEST VALUE
**Cost**: ~$0.20 | **Examples**: 960 | **Best for**: Good performance/cost ratio
Strong open-source model. Often performs well on reasoning tasks at reasonable cost.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "meta-llama/llama-3.1-70b-instruct" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/llama_70b_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

### Meta Llama 3.1 70B - Chain of Thought Prompt
**Cost**: ~$0.32 | **Examples**: 960 | **Best for**: Testing if CoT helps Llama models
Tests whether step-by-step reasoning improves Llama's logical reasoning performance.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "meta-llama/llama-3.1-70b-instruct" --prompt "cot" --input "train_data/subtask 1/train_data.json" --output "predictions/llama_70b_cot.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

---

## Premium Models (Limited Data to Control Costs)

### Claude 3.5 Sonnet - Direct Prompt (100 examples)
**Cost**: ~$0.10 | **Examples**: 100 | **Best for**: High-quality baseline comparison
Premium model limited to 100 examples for cost control. Known for strong reasoning.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "anthropic/claude-3.5-sonnet" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/claude_sonnet_direct_limited.json" --evaluate --reference "train_data/subtask 1/train_data.json" --limit 100
```

### Claude 3.5 Sonnet - Chain of Thought (100 examples)
**Cost**: ~$0.15 | **Examples**: 100 | **Best for**: Testing if CoT helps Claude
Tests whether explicit reasoning steps improve Claude's already strong reasoning ability.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "anthropic/claude-3.5-sonnet" --prompt "cot" --input "train_data/subtask 1/train_data.json" --output "predictions/claude_sonnet_cot_limited.json" --evaluate --reference "train_data/subtask 1/train_data.json" --limit 100
```

### GPT-4o - Direct Prompt (100 examples)
**Cost**: ~$0.08 | **Examples**: 100 | **Best for**: OpenAI's flagship model comparison
Premium OpenAI model limited to 100 examples. Strong general reasoning capabilities.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "openai/gpt-4o" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/gpt4o_direct_limited.json" --evaluate --reference "train_data/subtask 1/train_data.json" --limit 100
```

### GPT-4o - Chain of Thought (100 examples)
**Cost**: ~$0.12 | **Examples**: 100 | **Best for**: Testing CoT with GPT-4o
Tests whether step-by-step reasoning improves GPT-4o's logical reasoning.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "openai/gpt-4o" --prompt "cot" --input "train_data/subtask 1/train_data.json" --output "predictions/gpt4o_cot_limited.json" --evaluate --reference "train_data/subtask 1/train_data.json" --limit 100
```

---

## Additional Experiments

### DeepSeek R1 - Thinking Model (Direct Prompt)
**Cost**: ~$1.60 | **Examples**: 960 | **Best for**: Alternative thinking model comparison
DeepSeek's reasoning model. Use direct prompt only (no CoT needed for thinking models).

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "deepseek/deepseek-r1" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/deepseek_r1_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

### Google Gemini Pro - Direct Prompt
**Cost**: ~$0.12 | **Examples**: 960 | **Best for**: Google model comparison
Google's capable model at reasonable cost. Good for diversifying model coverage.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "google/gemini-pro" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/gemini_pro_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

---

## Testing on Test Data (Final Predictions)

### Qwen Thinking - Direct Prompt on Test Data
**Cost**: ~$1.20+ | **Examples**: Unknown (test set) | **Best for**: Final submission
Once you've identified the best model/prompt, run on actual test data. No evaluation since test data has no labels.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-thinking" --prompt "direct" --input "test_data/subtask 1/test_data_subtask_1.json" --output "predictions/final_test_predictions.json"
```

### Best Model - Direct Prompt on Test Data (Template)
**Cost**: Varies | **Examples**: Unknown (test set) | **Best for**: Final submission
Replace `{best_model}` and `{best_prompt}` with your highest-performing configuration from training experiments.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "{best_model}" --prompt "{best_prompt}" --input "test_data/subtask 1/test_data_subtask_1.json" --output "predictions/final_test_predictions.json"
```

---

## Custom Results CSV Path

### All Experiments to Single CSV
**Best for**: Centralized results tracking
All experiments append to one CSV file instead of separate files per model/prompt combination.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-instruct" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_instruct_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json" --results-csv "experiments/all_results.csv"
```

---

## Cost Comparison Table

Based on 960 training examples:

| Model | Prompt | Cost/Run | Speed | Accuracy (Est.) | Value Rating |
|-------|--------|----------|-------|-----------------|--------------|
| DeepSeek Chat | Direct | $0.04 | Fast | Medium | ⭐⭐⭐⭐⭐ |
| Qwen Instruct | Direct | $0.08 | Fast | Medium-High | ⭐⭐⭐⭐⭐ |
| DeepSeek Chat | CoT | $0.08 | Fast | Medium | ⭐⭐⭐⭐ |
| Gemini Pro | Direct | $0.12 | Fast | Medium-High | ⭐⭐⭐⭐ |
| Qwen Instruct | CoT | $0.12 | Medium | Medium-High | ⭐⭐⭐⭐ |
| Llama 3.1 70B | Direct | $0.20 | Fast | High | ⭐⭐⭐⭐ |
| Llama 3.1 70B | CoT | $0.32 | Medium | High | ⭐⭐⭐ |
| Claude 3.5 (100) | Direct | $0.10 | Medium | Very High | ⭐⭐⭐ |
| GPT-4o (100) | Direct | $0.08 | Medium | Very High | ⭐⭐⭐ |
| Qwen Thinking | Direct | $1.20 | Slow | Very High | ⭐⭐ |
| DeepSeek R1 | Direct | $1.60 | Slow | Very High | ⭐⭐ |
| Qwen Thinking | CoT | $3.60 | Very Slow | Very High | ⭐ (Redundant) |

**Value Rating**: Performance per dollar (⭐⭐⭐⭐⭐ = best value)

---

## Results Files

All experiments with `--evaluate` automatically create/append to CSV files:

**Default naming**: `experiments/results_{model}_{prompt}.csv`

Examples:
- `experiments/results_qwen_qwen3-vl-235b-a22b-instruct_direct.csv`
- `experiments/results_qwen_qwen3-vl-235b-a22b-instruct_cot.csv`
- `experiments/results_qwen_qwen3-vl-235b-a22b-thinking_direct.csv`
- `experiments/results_meta-llama_llama-3.1-70b-instruct_direct.csv`

**CSV Columns**:
- Experiment metadata: timestamp, model, prompt_template, input/output files, num_examples
- Main metrics: accuracy, content_effect, combined_score
- Subgroup accuracies: plausible_valid, implausible_valid, plausible_invalid, implausible_invalid
- Content effect breakdown: intra_validity, inter_validity

---

## Recommended Experiment Sequence

### Phase 1: Budget Exploration (Total: ~$1.00)
```
1. DeepSeek Chat - Direct ($0.04)
2. DeepSeek Chat - CoT ($0.08)
3. Qwen Instruct - Direct ($0.08)
4. Qwen Instruct - CoT ($0.12)
5. Gemini Pro - Direct ($0.12)
6. Llama 3.1 70B - Direct ($0.20)
7. Llama 3.1 70B - CoT ($0.32)
```

**Goal**: Identify if CoT helps, establish baseline performance

### Phase 2: Thinking Models (Total: ~$1.20-2.80)
```
8. Qwen Thinking - Direct ($1.20) ⭐ RECOMMENDED
9. (Optional) DeepSeek R1 - Direct ($1.60)
```

**Goal**: Test if thinking models significantly outperform instruct models

### Phase 3: Premium Comparison (Total: ~$0.18)
```
10. Claude 3.5 Sonnet - Direct, 100 examples ($0.10)
11. GPT-4o - Direct, 100 examples ($0.08)
```

**Goal**: Benchmark against top-tier models to see if extra cost is justified

### Phase 4: Final Prediction
```
12. Best model from above on test data (~$0.04-1.20+)
```

**Total estimated cost for all phases**: ~$2.38-4.18

---

## Key Insights from Test Results

### Thinking vs Instruct Models
- **Thinking models** reason internally (see `message.reasoning` field)
- **Direct prompt** is sufficient for thinking models
- **CoT prompt** with thinking models is redundant and 3x more expensive

### OpenRouter Handling
- Reasoning tokens are separated from content
- `message.content` contains only JSON response
- `message.reasoning` contains the internal reasoning process
- Current code works perfectly with both model types

### Prompt Strategy
- **Instruct models**: Try both direct and CoT prompts
- **Thinking models**: Use direct prompt only
- CoT might help non-thinking models organize their reasoning

---

## Quick Copy Commands (Most Important)

**Fastest baseline** (3 seconds):
```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "deepseek/deepseek-chat" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/deepseek_chat_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

**Best value** (good performance/cost):
```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "meta-llama/llama-3.1-70b-instruct" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/llama_70b_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

**Highest accuracy** (recommended for final):
```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-thinking" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_thinking_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

---

## Notes

- 💡 **Thinking models**: Always use `direct` prompt - they already reason internally
- 💰 **Cost control**: Use `--limit` flag to test expensive models on subset first
- 📊 **Results tracking**: All runs automatically append to CSV for easy comparison
- ⚡ **Speed**: Budget models (DeepSeek, Llama) are fastest; thinking models are slowest
- 🎯 **Accuracy vs Cost**: Thinking models likely best accuracy but 15-45x more expensive
