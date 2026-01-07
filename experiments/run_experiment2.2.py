"""
Normalization runner (VARIABLE-REPLACED) for SemEval 2026 Task 11 – Subtask 1.

This script rewrites syllogistic arguments into a purely FORMAL normalized form
using abstract variables (A, B, C, ...) instead of concrete terms.

The JSON schema is IDENTICAL to the input:
[
  {
    "id": "...",
    "syllogism": "..."
  }
]

Usage:
    poetry run python experiments/run_experiment2.2.py --model "qwen/qwen3-vl-235b-a22b-instruct" --input "train_data/subtask 1/train_data.json" --output "data/polished/polished_syllogisms_variables.json"

Optional:
    --limit 100
"""

import argparse
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import OpenRouterClient
from src.data_loader import load_test_data
from src.prompts import get_prompt_template
from src.normalizer import SyllogismNormalizer


PROMPT_NAME = "normalization_replace"


def main():
    parser = argparse.ArgumentParser(
        description="Normalize syllogisms into variable-based canonical form using an LLM"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen2.5-vl-235b-a22b-instruct",
        help="OpenRouter model identifier"
    )

    # Input / Output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file (raw syllogisms)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save variable-normalized syllogisms JSON"
    )

    # Optional limit
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)"
    )

    # API key
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenRouter API key (optional, can use .env file)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    print("=" * 60)
    print("SemEval 2026 Task 11 – Syllogism Normalization (VARIABLE FORM)")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Prompt:  {PROMPT_NAME}")
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    if args.limit:
        print(f"Limit:   {args.limit}")
    print("=" * 60 + "\n")

    # ---------------------------------------------------------
    # Initialize LLM client
    # ---------------------------------------------------------
    print("Initializing OpenRouter client...")
    try:
        llm_client = OpenRouterClient(
            api_key=args.api_key,
            model=args.model
        )
        print("Client initialized successfully.\n")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENROUTER_API_KEY in .env file or use --api-key")
        sys.exit(1)

    # ---------------------------------------------------------
    # Load prompt template (FIXED)
    # ---------------------------------------------------------
    print(f"Loading prompt template: {PROMPT_NAME}...")
    prompt_template = get_prompt_template(PROMPT_NAME)
    print("Prompt template loaded.\n")

    # ---------------------------------------------------------
    # Load input data
    # ---------------------------------------------------------
    print(f"Loading raw syllogisms from: {args.input}...")
    try:
        examples = load_test_data(args.input)
        if args.limit:
            examples = examples[:args.limit]
        print(f"Loaded {len(examples)} examples.\n")
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading input data: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Initialize normalizer
    # ---------------------------------------------------------
    normalizer = SyllogismNormalizer(
        llm_client=llm_client,
        prompt_template=prompt_template
    )

    # ---------------------------------------------------------
    # Normalize syllogisms
    # ---------------------------------------------------------
    print("Normalizing syllogisms into variable-based form...")
    print("This may take a while depending on the number of examples...\n")

    try:
        normalized = normalizer.normalize_batch_as_dicts(
            test_examples=examples,
            show_progress=True
        )
    except Exception as e:
        print(f"Error during normalization: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Save output (SCHEMA IDENTICAL TO INPUT)
    # ---------------------------------------------------------
    print(f"\nSaving normalized syllogisms to: {args.output}...")
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)

        print("Variable-normalized JSON saved successfully.\n")
    except Exception as e:
        print(f"Error saving output JSON: {e}")
        sys.exit(1)

    print("Normalization experiment (variable form) completed successfully!")


if __name__ == "__main__":
    main()
